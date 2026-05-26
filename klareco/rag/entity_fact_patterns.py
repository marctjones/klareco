"""
Entity-fact extraction patterns (#745).

Generalizes the pattern KB tables (capital_of, lingvo) to every
biographical/factual relation the AST reveals. Each pattern is a
function (row, ast) → list[Fact]. The extractor (in
scripts/index/extract_entity_facts.py) iterates the corpus once,
runs every pattern per row, and writes the resulting Fact records
to the entity_facts table.

Design:

    Each FactPattern subclass implements .extract(row, ast) → [Fact].
    Patterns are pure functions over a candidate sentence; no DB
    state required at extraction time. This keeps the framework
    debuggable: you can run a single pattern over a list of
    handcrafted sentences and inspect what it would emit.

    Fact rows are inserted by the extractor (not by the pattern)
    so the pattern stays declarative.

The 'row' dict expected by patterns:

    {
        'sid':              int,
        'text':             str,
        'subj_radiko':      str | None,
        'subj_vortspeco':   str | None,
        'subj_propranoma_kat': str | None,
        'subj_kazo':        str | None,
        'verb_radiko':      str | None,
        'verb_tempo':       str | None,
        'verb_klaso':       str | None,
        'verb_negated':     bool | None,
        'obj_radiko':       str | None,
        'obj_kazo':         str | None,
        'aliaj_json':       str | None,
    }

The 'ast' is the full parsed AST as a dict, OR None if the parser
failed for this sentence.

Last Updated: 2026-05-26
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional

# Same year-regex the AST-aware reranker uses (see ast_aware_reranker.py).
_YEAR_RE = re.compile(r'\b(1[0-9]{3}|20[0-9]{2}|2100)\b')

# Place-implying prepositions (locative). Mirrors the loko detector.
_PLACE_PREPS = {
    'en', 'ĉe', 'al', 'el', 'ekde', 'ĝis', 'tra', 'super', 'sur', 'sub',
    'apud', 'trans', 'kontraŭ', 'antaŭ', 'malantaŭ', 'inter',
}

# Verb radikos by event class. Validated against the live DuckDB store
# (data/indexes/duckdb_store.db, 5.4M sentences):
#
#   verb_klaso is too sparse to filter on (only ~38k rows have it set,
#   and 'fond' isn't in kreado-26). Use verb_radiko lists instead.

# Founder/creator/inventor verbs.
_FOUNDER_RADIKOJ = {
    'fond',       # fondi (found)
    'kunfond',    # kunfondi (co-found)
    'kre',        # krei (create)
    'invent',     # inventi (invent)
    'eltrov',     # eltrovi (discover)
    'malkovr',    # malkovri (discover, alt)
    'establ',     # establi (establish)
    'iniciat',    # iniciati (initiate)
    'organiz',    # organizi (organize)
}
# Birth verbs (validated: 'nask' is the radiko, ~48k occurrences).
_BIRTH_RADIKOJ = {'nask'}
# Death verbs.
_DEATH_RADIKOJ = {'mort'}

# Profession-suffix heuristic: nouns ending in -ist, -isto, -anto, etc.
_PROFESSION_SUFFIXES = ('ist', 'anto', 'into', 'onto')

# Kat values that signal "this is a named entity" — accept any non-null
# variant since the kat column doesn't distinguish persono/loko/etc.
_PROPER_KATS = {'propranomo', 'propranomo_esperantigita', 'neologismo'}


def _is_proper(row: dict) -> bool:
    """Heuristic: this row's subject is a named entity?"""
    if (row.get('subj_vortspeco') or '').lower() != 'propra_nomo':
        return False
    kat = (row.get('subj_propranoma_kat') or '').lower()
    # Either an explicit propranomo* kat, or kat=NULL but vortspeco says
    # propra_nomo — accept both (kat coverage is partial).
    return kat == '' or kat in _PROPER_KATS


@dataclass(frozen=True)
class Fact:
    """A single extracted (entity, slot, value, source) tuple."""
    entity_radiko: str
    slot:          str
    value:         str
    value_radiko:  str
    source_sid:    int
    confidence:    float
    pattern_name:  str
    # Optional: a small dict of debug bits (e.g., which AST node)
    debug:         dict = field(default_factory=dict)


# =============================================================================
# Pattern base
# =============================================================================

class FactPattern:
    name: str = '???'

    def extract(self, row: dict, ast: Optional[dict]) -> list[Fact]:
        raise NotImplementedError

    # Helpers shared by many patterns
    @staticmethod
    def _aliaj_iter(row: dict):
        """Iterate the candidate sentence's aliaj as list of dict items."""
        aliaj_json = row.get('aliaj_json')
        if not aliaj_json:
            return
        try:
            arr = json.loads(aliaj_json) if isinstance(aliaj_json, str) else aliaj_json
        except Exception:
            return
        if isinstance(arr, list):
            for item in arr:
                yield item

    @staticmethod
    def _find_prep_objects(row: dict, prep_radikoj: set, accept_year=False
                            ) -> list[tuple[str, str, str]]:
        """Walk aliaj and return ALL `[prep in prep_radikoj] + [propra_nomo|
        substantivo|year]` pairs as (value, value_radiko, vortspeco) tuples.

        Returns the matches in the order they appear in aliaj. Callers
        usually take the LAST one (city > street, country > region) or
        emit all of them so frequency aggregation across the corpus
        ranks the correct value first.

        accept_year=True allows matches whose object is a 4-digit year.
        """
        aliaj = list(FactPattern._aliaj_iter(row))
        out: list[tuple[str, str, str]] = []
        for i, item in enumerate(aliaj):
            if not isinstance(item, dict):
                continue
            kerno = item.get('kerno') if item.get('tipo') == 'vortgrupo' else item
            if not isinstance(kerno, dict):
                continue
            radiko = (kerno.get('radiko') or '').lower()
            vs = (kerno.get('vortspeco') or '').lower()
            if vs != 'prepozicio' or radiko not in prep_radikoj:
                continue
            for j in range(i + 1, min(i + 3, len(aliaj))):
                nxt = aliaj[j]
                if not isinstance(nxt, dict):
                    continue
                nxt_k = nxt.get('kerno') if nxt.get('tipo') == 'vortgrupo' else nxt
                if not isinstance(nxt_k, dict):
                    continue
                nxt_vs = (nxt_k.get('vortspeco') or '').lower()
                nxt_pv = str(nxt_k.get('plena_vorto') or '')
                nxt_r = (nxt_k.get('radiko') or '').lower()
                if accept_year and _YEAR_RE.fullmatch(nxt_pv):
                    out.append((nxt_pv, nxt_pv, 'jaro'))
                    break
                if nxt_vs == 'propra_nomo' and not _YEAR_RE.fullmatch(nxt_pv):
                    out.append((nxt_pv, nxt_r or nxt_pv.lower(), 'propra_nomo'))
                    break
                if nxt_vs == 'substantivo' and radiko == 'en':
                    out.append((nxt_pv, nxt_r or nxt_pv.lower(), 'substantivo'))
                    break
        return out

    @staticmethod
    def _find_prep_object(row: dict, prep_radikoj: set, accept_year=False
                          ) -> Optional[tuple[str, str, str]]:
        """Backwards-compatible single-result helper. Returns the LAST
        match (city > street heuristic) or None.

        New callers should prefer _find_prep_objects() and emit all."""
        matches = FactPattern._find_prep_objects(row, prep_radikoj, accept_year)
        return matches[-1] if matches else None

    @staticmethod
    def _all_years_in_text(text: Optional[str]) -> list[str]:
        if not text:
            return []
        return _YEAR_RE.findall(text)


# =============================================================================
# Patterns
# =============================================================================

class FounderOfPattern(FactPattern):
    """[Named subj] [founder verb] [Object] → (Object, founder, Subject).

    Captures `Zamenhof fondis Esperanton`, `Bell inventis la telefonon`,
    `Gates fondis Microsoft`, etc.

    Uses a hard-coded verb_radiko set (verb_klaso is too sparse —
    only 38k of 5.4M rows have it set, and 'fond' isn't in kreado-26).
    """
    name = 'founder_of'

    def extract(self, row, ast):
        if not _is_proper(row):
            return []
        verb_r = (row.get('verb_radiko') or '').lower()
        if verb_r not in _FOUNDER_RADIKOJ:
            return []
        if row.get('verb_negated'):
            return []  # "ne fondis" → can't extract a founder
        if not row.get('obj_radiko') or not row.get('subj_radiko'):
            return []
        if row['subj_radiko'] == row['obj_radiko']:
            return []  # self-reference (rare; defensive)
        # Confidence: higher when the subj radiko looks person-like (no
        # great signal for that in the shredded columns, so flat 0.75).
        return [Fact(
            entity_radiko=row['obj_radiko'].lower(),
            slot='founder',
            value=row['subj_radiko'],
            value_radiko=row['subj_radiko'].lower(),
            source_sid=int(row['sid']),
            confidence=0.75,
            pattern_name=self.name,
        )]


class BirthPlacePattern(FactPattern):
    """[Named subj] [nask verb] aliaj=[en/ĉe + place-propra_nomo]
    → (Subject, birth_place, Place).

    Emits ALL `en/ĉe + propra_nomo` matches so frequency aggregation
    across the corpus picks the correct value. For
    `Zamenhof naskiĝis en Ulica Zielona 6, en la urbo Bjalistoko`, this
    yields TWO facts:
      (zamenhof, birth_place, Ulica)
      (zamenhof, birth_place, Bjalistoko)
    Across the full corpus, Bjalistoko will accumulate many more
    sources than Ulica, so the retriever ranks it first.
    """
    name = 'birth_place'

    def extract(self, row, ast):
        if not _is_proper(row):
            return []
        verb_r = (row.get('verb_radiko') or '').lower()
        if verb_r not in _BIRTH_RADIKOJ:
            return []
        if row.get('verb_negated'):
            return []
        if not row.get('subj_radiko'):
            return []
        matches = self._find_prep_objects(row, {'en', 'ĉe'}, accept_year=False)
        out: list[Fact] = []
        for value, value_r, vs in matches:
            if vs == 'jaro':
                continue
            conf = 0.85 if vs == 'propra_nomo' else 0.55
            out.append(Fact(
                entity_radiko=row['subj_radiko'].lower(),
                slot='birth_place',
                value=value,
                value_radiko=value_r,
                source_sid=int(row['sid']),
                confidence=conf,
                pattern_name=self.name,
            ))
        return out


class BirthYearPattern(FactPattern):
    """[Named subj] [nask verb] aliaj=[en + YEAR] OR text contains single YEAR
    → (Subject, birth_year, YEAR)."""
    name = 'birth_year'

    def extract(self, row, ast):
        if not _is_proper(row):
            return []
        verb_r = (row.get('verb_radiko') or '').lower()
        if verb_r not in _BIRTH_RADIKOJ:
            return []
        if row.get('verb_negated'):
            return []
        if not row.get('subj_radiko'):
            return []
        # Prefer an explicit "en YEAR" in aliaj; fall back to text-level
        # year regex when exactly one year appears.
        found = self._find_prep_object(row, {'en'}, accept_year=True)
        year_value = None
        from_aliaj = False
        if found and found[2] == 'jaro':
            year_value = found[0]
            from_aliaj = True
        else:
            yrs = self._all_years_in_text(row.get('text'))
            if len(yrs) == 1:
                year_value = yrs[0]
        if not year_value:
            return []
        conf = 0.80 if from_aliaj else 0.60
        return [Fact(
            entity_radiko=row['subj_radiko'].lower(),
            slot='birth_year',
            value=year_value,
            value_radiko=year_value,
            source_sid=int(row['sid']),
            confidence=conf,
            pattern_name=self.name,
        )]


class DeathYearPattern(FactPattern):
    """[Named subj] [mort verb] aliaj=[en + YEAR]
    → (Subject, death_year, YEAR)."""
    name = 'death_year'

    def extract(self, row, ast):
        if not _is_proper(row):
            return []
        verb_r = (row.get('verb_radiko') or '').lower()
        if verb_r not in _DEATH_RADIKOJ:
            return []
        if row.get('verb_negated'):
            return []
        if not row.get('subj_radiko'):
            return []
        found = self._find_prep_object(row, {'en'}, accept_year=True)
        year_value = None
        from_aliaj = False
        if found and found[2] == 'jaro':
            year_value = found[0]
            from_aliaj = True
        else:
            yrs = self._all_years_in_text(row.get('text'))
            if len(yrs) == 1:
                year_value = yrs[0]
        if not year_value:
            return []
        conf = 0.80 if from_aliaj else 0.60
        return [Fact(
            entity_radiko=row['subj_radiko'].lower(),
            slot='death_year',
            value=year_value,
            value_radiko=year_value,
            source_sid=int(row['sid']),
            confidence=conf,
            pattern_name=self.name,
        )]


class ProfessionPattern(FactPattern):
    """[Named subj] estis [profession-noun]
    → (Subject, profession, profession-noun).

    Heuristic: object's radiko ends in a profession suffix (-ist, -anto, ...).
    """
    name = 'profession'

    def extract(self, row, ast):
        if not _is_proper(row):
            return []
        if row.get('verb_radiko') != 'est':
            return []
        if row.get('verb_negated'):
            return []
        obj_r = (row.get('obj_radiko') or '').lower()
        if not obj_r or not row.get('subj_radiko'):
            return []
        if not any(obj_r.endswith(suffix) for suffix in _PROFESSION_SUFFIXES):
            return []
        return [Fact(
            entity_radiko=row['subj_radiko'].lower(),
            slot='profession',
            value=obj_r,
            value_radiko=obj_r,
            source_sid=int(row['sid']),
            confidence=0.70,
            pattern_name=self.name,
        )]


class DefinitionPattern(FactPattern):
    """[Named X] estas [predicate] → (X, definition, predicate-snippet).

    Captures definitional sentences like 'Esperanto estas internacia lingvo'.
    Restricted to PROPER-NOUN subjects so we don't index every "la hundo
    estas bruna" as a definition fact (huge noise). Already-restricted
    common-noun cases (e.g. 'Esperanto') still trigger when shredded
    as propra_nomo.

    Confidence: low — even with the proper-noun gate, definitions are
    inherently noisier than role-specific patterns. The Profession
    pattern fires alongside this one when the predicate is a profession
    noun (and at higher confidence), so use this as a fallback for
    questions where Profession misses.
    """
    name = 'definition'

    def extract(self, row, ast):
        if not _is_proper(row):
            return []
        if row.get('verb_radiko') != 'est':
            return []
        if row.get('verb_negated'):
            return []
        if not row.get('subj_radiko'):
            return []
        predicate = row.get('obj_radiko')
        if not predicate:
            return []
        return [Fact(
            entity_radiko=row['subj_radiko'].lower(),
            slot='definition',
            value=predicate,
            value_radiko=predicate.lower(),
            source_sid=int(row['sid']),
            confidence=0.50,
            pattern_name=self.name,
        )]


class EventLocationPattern(FactPattern):
    """[Event] okazis aliaj=[en + Loko] (+ optional year)
    → (Event, location_of_event, Loko)
    +  (Event, year_of_event, Year) if year present.

    Emits ALL `place_prep + propra_nomo` matches (corpus frequency picks
    the right city when multiple are mentioned)."""
    name = 'event_location'

    def extract(self, row, ast):
        verb_r = (row.get('verb_radiko') or '').lower()
        if verb_r != 'okaz':
            return []
        if row.get('verb_negated'):
            return []
        if not row.get('subj_radiko'):
            return []
        out: list[Fact] = []
        for val, val_r, vs in self._find_prep_objects(
            row, _PLACE_PREPS, accept_year=False
        ):
            if vs == 'jaro':
                continue
            conf = 0.80 if vs == 'propra_nomo' else 0.50
            out.append(Fact(
                entity_radiko=row['subj_radiko'].lower(),
                slot='location_of_event',
                value=val,
                value_radiko=val_r,
                source_sid=int(row['sid']),
                confidence=conf,
                pattern_name=self.name,
            ))
        # Year: text-level (so "Olimpikoj de 1936" works whether the year
        # is in aliaj as "en 1936" or just appears in the surrounding text)
        years = self._all_years_in_text(row.get('text'))
        if len(years) >= 1:
            yr = years[0]
            out.append(Fact(
                entity_radiko=row['subj_radiko'].lower(),
                slot='year_of_event',
                value=yr,
                value_radiko=yr,
                source_sid=int(row['sid']),
                confidence=0.75 if len(years) == 1 else 0.60,
                pattern_name=self.name,
            ))
        return out


# =============================================================================
# Registry
# =============================================================================

ALL_PATTERNS: list[FactPattern] = [
    FounderOfPattern(),
    BirthPlacePattern(),
    BirthYearPattern(),
    DeathYearPattern(),
    ProfessionPattern(),
    EventLocationPattern(),
    DefinitionPattern(),
]


def extract_facts_from_row(row: dict,
                            ast: Optional[dict] = None,
                            patterns: Optional[list[FactPattern]] = None
                            ) -> list[Fact]:
    """Run all patterns against a single shredded sentence row.
    Returns the union of facts emitted."""
    if patterns is None:
        patterns = ALL_PATTERNS
    facts: list[Fact] = []
    for p in patterns:
        try:
            facts.extend(p.extract(row, ast))
        except Exception:
            # A buggy pattern shouldn't kill the extractor — log and skip
            import logging
            logging.warning(f'Pattern {p.name} crashed on sid={row.get("sid")}',
                            exc_info=True)
    return facts
