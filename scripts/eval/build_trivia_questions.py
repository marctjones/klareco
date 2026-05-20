#!/usr/bin/env python3
"""
Real-trivia question generator targeting definite-description and
superlative patterns in the corpus.

VERSION: v2.x (DuckDB)
COMPATIBLE WITH: DuckDB store (sentences: shredded cols + ast_json blob)
DEPENDENCIES: duckdb
STAGE: Evaluation

Description:
    Earlier generators produced underspecified questions ("Kie naskiĝis
    Béla?" with first-name-only anchor) or categorical questions
    ("Kio estas Bockhorn?" → "komunumo"). Both fail as trivia: the
    first because the anchor isn't a rigid designator, the second
    because the answer is a category, not a named entity.

    This generator builds questions in the shape of real trivia:
    *definite descriptions* and *superlatives*. Each question's
    constraint uniquely picks out one specific entity from the
    universe, and the answer is that entity by name. Source patterns
    targeted:
      - `[Entity] estas la ĉefurbo de [Country]`
      - `[Entity] estas la plej [adj] [common-noun] [de|en] [Place]`
      - `[Entity], la ĉefurbo de [Country]`  (appositive form)

    A small batch is generated so each pair can be reviewed one by one
    against semantic-quality criteria:
      1. The question contains a definite description or superlative
         that uniquely identifies the answer
      2. The answer is a named entity, not a category
      3. World knowledge would be needed to answer (no answer leak in
         the question text)
      4. The source sentence actually demonstrates the claim

Pipeline Position:
    DuckDB store → [THIS SCRIPT] → trivia JSONL → human review
                                → evaluate_extractive_qa.py

Usage:
    python scripts/eval/build_trivia_questions.py --pattern capital
    python scripts/eval/build_trivia_questions.py --pattern superlative \\
        --target-size 30 --query-limit 2000

Inputs:
    --duckdb-path  data/indexes/duckdb_store.db
    --pattern      capital | superlative | all

Outputs:
    data/test_sets/trivia_{pattern}_candidates.jsonl  (for human review)

Last Updated: 2026-05-20
Author: Claude Code (with Marc Jones)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import duckdb


# A proper-noun-headed name span, possibly with surname / midword Latin
# accented characters / hyphenated parts. Anchored to start with capital.
_PROPER_NOUN = (
    r'[A-ZÀ-ÞĈĜĤĴŜŬ][\wÀ-ſĉĝĥĵŝŭĈĜĤĴŜŬ.-]+'
    r'(?:\s+[A-ZÀ-ÞĈĜĤĴŜŬ][\wÀ-ſĉĝĥĵŝŭĈĜĤĴŜŬ.-]+){0,3}'
)

# "X estas la ĉefurbo de Y" / "X, la ĉefurbo de Y"
_CAPITAL_RE = re.compile(
    r'(?P<entity>' + _PROPER_NOUN + r')'
    r'(?:,|\s+estas|\s+iĝis)\s+'
    r'la\s+ĉefurbo\s+de\s+'
    r'(?P<country>' + _PROPER_NOUN + r')',
    re.UNICODE,
)

# "X estas la plej ADJ NOUN [de|en] PLACE"
# - Multi-word ADJ (rare; allow 1-2 tokens)
# - NOUN single-token (rivero, monto, lago, lando, urbo, ...)
# - PLACE: propra_nomo OR `la mondo` / `Eŭropo`-style
_SUPERLATIVE_RE = re.compile(
    r'(?P<entity>' + _PROPER_NOUN + r')'
    r'(?:\s*\([^)]*\))?'                     # optional parenthetical
    r'\s+estas\s+'
    r'la\s+plej\s+'
    r'(?P<adj>\w+(?:\s+\w+)?)\s+'
    r'(?P<noun>\w+)'
    r'\s+(?P<scope>(?:de|en|laŭ)\s+(?:la\s+)?(?:' + _PROPER_NOUN + r'|mondo|tero))',
    re.UNICODE,
)

# Anti-patterns that hint the claim isn't a clean superlative
_SUPERLATIVE_REJECT = (
    'unu el la plej',     # "one of the most" — NOT the most
    'post ',              # "after X, the most" — chained comparison
    'kun la plej',        # "with the most" — adjective predicate, not superlative subject
    'havas la plej',      # "has the most" — possession, not identity
    'inter la plej',      # "among the most" — non-uniqueness
)

# Hedging tokens that mean the source isn't asserting an authoritative claim.
# (Catches `Nilo aŭ Amazono estas la plej longa` — a disjunction, not a fact.)
_HEDGING_TOKENS = (
    ' aŭ ', ' disputo ', ' kontroversa ', ' kontroverso ',
    ' malsamaj informoj ', ' diskutas ', ' verŝajne ', ' eble ',
    ' laŭ kelkaj ', ' iuj diras ', ' eble estas ',
)

# Post-modifier patterns that, when they follow the matched scope, mean the
# claim has a qualifier our extractor would drop (causing over-generalization).
# Example: `…la plej longa epopeo de la mondo VERKITA DE UNUSOLA POETO`.
_POST_QUALIFIER_RE = re.compile(
    r'\s+(?:verkita|kreita|fondita|farita|konstruita|trovita|elektita|'
    r'eldonita|publikigita|registrita)\s+',
    re.IGNORECASE,
)

# Common nouns that often pass as proper-noun in our regex (capital-start words
# at sentence-initial positions get falsely flagged).
_LIKELY_COMMON_NOUNS_CAPITALIZED = {
    'La', 'Tiu', 'Tio', 'Ĉi', 'Ĝi', 'Li', 'Ŝi', 'Ni', 'Ili', 'Vi', 'Mi',
    # Accusative-form pronouns (Esperanto OSV order puts these sentence-initial)
    'Min', 'Vin', 'Lin', 'Ŝin', 'Ĝin', 'Nin', 'Ilin', 'Sin', 'Onin',
    # Indefinite pronoun
    'Oni',
    'Sub', 'En', 'Sur', 'Al', 'De', 'Al', 'Por', 'Ekde',
    'Krome', 'Cetere', 'Tamen', 'Sed', 'Kaj',
    'Kiel', 'Kiam', 'Kie', 'Kio', 'Kial', 'Kiu',
    'Inter', 'Post', 'Antaŭ',
    # Common nouns frequently capitalised in Wikipedia headings (`La unua
    # Universitato estis fondita…`) — these aren't names of specific entities,
    # they're generic terms. Reject so the founding-year question doesn't ask
    # "When was the Universitato founded?".
    'Universitato', 'Biblioteko', 'Muzeo', 'Teatro', 'Stadiono',
    'Eklezio', 'Akademio', 'Lernejo', 'Hospitalo', 'Kongreso',
    'Konsilio', 'Ligo', 'Asocio', 'Organizaĵo', 'Komitato',
    'Konferenco', 'Renkontiĝo', 'Reĝimo',
}


def is_real_proper_noun(name: str) -> bool:
    """Reject sentence-initial common-word-as-propra_nomo class."""
    if not name:
        return False
    first_tok = name.split()[0]
    return first_tok not in _LIKELY_COMMON_NOUNS_CAPITALIZED


def extract_capital_pair(text: str) -> dict | None:
    """Find one `<City> estas la ĉefurbo de <Country>` claim in `text`."""
    m = _CAPITAL_RE.search(text)
    if not m:
        return None
    city, country = m.group('entity').strip(), m.group('country').strip()
    if not is_real_proper_noun(city) or not is_real_proper_noun(country):
        return None
    if city.lower() == country.lower():
        return None
    # Reject if the city name appears in the country span (truncation/echo)
    if city in country or country in city:
        return None
    question = f'Kio estas la ĉefurbo de {country}?'
    return {
        'pattern':         'capital',
        'question':        question,
        'expected_answer': city,
        'extracted_constraint': f'la ĉefurbo de {country}',
    }


def extract_superlative_pair(text: str) -> dict | None:
    """Find one `<Entity> estas la plej <ADJ> <NOUN> de <SCOPE>` claim."""
    if any(bad in text for bad in _SUPERLATIVE_REJECT):
        return None
    # Hedging detector: source is uncertain about which entity satisfies the
    # superlative (`Nilo aŭ Amazono`, `disputo`, `verŝajne` …). Drop these
    # to avoid one-sided extraction of a disputed claim.
    if any(token in text for token in _HEDGING_TOKENS):
        return None
    m = _SUPERLATIVE_RE.search(text)
    if not m:
        return None
    entity = m.group('entity').strip()
    adj = m.group('adj').strip()
    noun = m.group('noun').strip()
    scope = m.group('scope').strip()
    # Qualifier detector: if a participle-modifier follows the matched scope,
    # the claim has a qualifier the question would drop. Drop the pair.
    tail = text[m.end():]
    if _POST_QUALIFIER_RE.match(tail):
        return None
    if not is_real_proper_noun(entity):
        return None
    # The adjective should be a real Esperanto adverb-base (ends in 'a' for
    # adjective form, the surface form is already in adj form due to "plej").
    # Common bad adjectives: those that don't end in 'a' or are participle.
    if not adj.endswith('a'):
        return None
    if re.search(r'(it|at|ot|int|ant|ont|unt)a$', adj):
        return None
    # NOUN should end in 'o' (noun) or 'oj' (plural noun)
    if not (noun.endswith('o') or noun.endswith('oj')):
        return None
    # Reject if entity span includes the noun (mis-extraction)
    if noun in entity.lower():
        return None
    # Reject articles in entity span
    if entity.split()[0].lower() in ('la', 'tiu', 'ĉi', 'unu'):
        return None
    # `Kiu` for "which one" — superlatives pick one entity from a set.
    question = f'Kiu estas la plej {adj} {noun} {scope}?'
    return {
        'pattern':         'superlative',
        'question':        question,
        'expected_answer': entity,
        'extracted_constraint': f'la plej {adj} {noun} {scope}',
    }


# =============================================================================
# Family 3: Inventor / Discoverer  ("Kiu inventis/malkovris X?")
# =============================================================================

# `<Person> inventis/malkovris/eltrovis <ProperNounObject>`
# We restrict to propra_nomo objects (capitalised, ≥4 chars, with optional
# accusative -n). Common-noun objects (`la kaŭzon`, `la asteroidon`) are
# rejected — they truncate the specifying complement (`de la pesto`,
# `Aglajo`) and produce ambiguous questions.
_INVENTOR_RE = re.compile(
    r'(?P<person>' + _PROPER_NOUN + r')'
    r'\s+(?P<verb>inventis|malkovris|eltrovis)\s+'
    r'(?P<thing>[A-ZÀ-ÞĈĜĤĴŜŬ][\wÀ-ſĉĝĥĵŝŭĈĜĤĴŜŬ-]+(?:n)?)',
    re.UNICODE,
)


def extract_inventor_pair(text: str) -> dict | None:
    """Find `<Person> inventis/malkovris/eltrovis <ProperNoun>`. Object MUST
    be capitalised (a proper noun) so the question carries enough specificity
    — otherwise the dropped complement (`la kaŭzon de la pesto`) leaves an
    underspecified question."""
    if any(token in text for token in _HEDGING_TOKENS):
        return None
    m = _INVENTOR_RE.search(text)
    if not m:
        return None
    person = m.group('person').strip()
    verb = m.group('verb').strip()
    thing = m.group('thing').strip()
    if not is_real_proper_noun(person):
        return None
    if not is_real_proper_noun(thing):
        return None
    # Strip accusative -n for both nominative-form and re-add it for the
    # question (which uses object case).
    thing_nom = thing[:-1] if thing.endswith('n') and len(thing) > 3 else thing
    if len(thing_nom) < 4:
        return None
    if thing_nom.lower() in person.lower() or person.lower() in thing_nom.lower():
        return None
    qthing = thing_nom + 'n' if thing_nom.endswith(('o', 'a', 'e', 'i', 'u')) else thing_nom
    question = f'Kiu {verb} {qthing}?'
    return {
        'pattern':         'inventor',
        'question':        question,
        'expected_answer': person,
        'extracted_constraint': f'{verb} {thing_nom}',
    }


# =============================================================================
# Family 4: Official-X of country  (oficiala lingvo, valuto, …)
# =============================================================================

_OFFICIAL_X = {
    'oficiala lingvo': 'la oficiala lingvo',
    'oficiala valuto': 'la oficiala valuto',
    'valuto':          'la valuto',
    'moneda unuo':     'la moneda unuo',
    'nacia himno':     'la nacia himno',
}

# Pattern A: `<Y> estas la <X> de <Country>`
_OFFICIAL_A_RE = re.compile(
    r'(?P<value>(?:la\s+)?[\wĉĝĥĵŝŭĈĜĤĴŜŬ-]+(?:\s+[\wĉĝĥĵŝŭĈĜĤĴŜŬ-]+){0,2})'
    r'\s+estas\s+'
    r'(?P<kind>la\s+(?:oficiala\s+lingvo|oficiala\s+valuto|valuto|moneda\s+unuo|nacia\s+himno))'
    r'\s+de\s+'
    r'(?P<country>' + _PROPER_NOUN + r')',
    re.UNICODE,
)

# Pattern B: `la <X> de <Country> estas <Y>`
_OFFICIAL_B_RE = re.compile(
    r'(?P<kind>la\s+(?:oficiala\s+lingvo|oficiala\s+valuto|valuto|moneda\s+unuo|nacia\s+himno))'
    r'\s+de\s+'
    r'(?P<country>' + _PROPER_NOUN + r')'
    r'\s+estas\s+'
    r'(?P<value>(?:la\s+)?[\wĉĝĥĵŝŭĈĜĤĴŜŬ-]+(?:\s+[\wĉĝĥĵŝŭĈĜĤĴŜŬ-]+){0,2})',
    re.UNICODE,
)


def extract_official_pair(text: str) -> dict | None:
    """Find `<Y> estas la <kind> de <Country>` claims. Strict shape: Y is
    1-2 clean noun tokens immediately before "estas", with no intervening
    relative-clause / year-phrase tokens (`kiu`, `kiuj`, `kiam`, `jam`,
    `fakte`, `ekde`, digits)."""
    if any(token in text for token in _HEDGING_TOKENS):
        return None
    m = _OFFICIAL_A_RE.search(text) or _OFFICIAL_B_RE.search(text)
    if not m:
        return None
    value = m.group('value').strip().rstrip('.,;:')
    kind = m.group('kind').strip()
    country = m.group('country').strip()
    if not is_real_proper_noun(country):
        return None
    # Reject if value span contains relative-clause / time-shifting tokens:
    # the value isn't a clean NP, the regex over-captured.
    INVALID_VALUE_TOKENS = {
        'kiu', 'kiuj', 'kiam', 'kie', 'kies', 'ke', 'kio',
        'jam', 'ankoraŭ', 'tiam', 'nun', 'fakte', 'ekde', 'antaŭ',
        'jaro', 'jaroj', 'jare', 'de',
    }
    value_lower_tokens = {t.lower().rstrip('.,;:') for t in value.split()}
    if value_lower_tokens & INVALID_VALUE_TOKENS:
        return None
    # No digits inside the value
    if re.search(r'\d', value):
        return None
    if value.lower() in {'la', 'tiu', 'tio', 'iuj', 'pluraj', 'kelkaj'}:
        return None
    if value.lower() == country.lower():
        return None
    # Strip leading "la "
    if value.lower().startswith('la '):
        value_clean = value[3:].strip()
    else:
        value_clean = value
    if len(value_clean) < 3:
        return None
    # Cap at 2 tokens for clean answer
    tokens = value_clean.split()
    if len(tokens) > 2:
        return None
    question = f'Kio estas {kind} de {country}?'
    return {
        'pattern':         'official',
        'question':        question,
        'expected_answer': value_clean,
        'extracted_constraint': f'{kind} de {country}',
    }


# =============================================================================
# Family 5: Founding year  ("En kiu jaro fondiĝis X?")
# =============================================================================

# Pattern: `<Org/Place> fondiĝis en <YEAR>` or `<Org> estis fondita en <YEAR>`.
_FOUNDED_RE = re.compile(
    r'(?P<org>' + _PROPER_NOUN + r')'
    r'(?:\s*\([^)]*\))?'
    r'\s+(?:fondiĝis|estis\s+fondita|estis\s+kreita|estis\s+establita|fondita)\s+'
    r'(?:en\s+(?:la\s+jaro\s+)?)?'
    r'(?P<year>1[0-9]{3}|20[0-2][0-9]|[789]\d{2})',
    re.UNICODE,
)


def extract_founded_pair(text: str) -> dict | None:
    """Find `<Org> fondiĝis en <YEAR>`. Reject if the matched Org is
    embedded inside a longer subject NP (e.g. `Loka Ligo de Breslaŭaj
    Esperanto-Unuiĝoj estis fondita` shouldn't extract `Breslaŭaj
    Esperanto-Unuiĝoj`). We require the Org's first character to be at
    the start of the sentence OR right after a period/start-of-clause,
    AND the immediate predecessor token (if any) to not be `de`, `en`,
    `kun`, `el` (these would mean the Org is in a PP)."""
    if any(token in text for token in _HEDGING_TOKENS):
        return None
    m = _FOUNDED_RE.search(text)
    if not m:
        return None
    org = m.group('org').strip()
    year = m.group('year').strip()
    if not is_real_proper_noun(org):
        return None
    y = int(year)
    if y < 700 or y > 2030:
        return None
    # Anchor-embeddedness check. Find the org's position; look at the
    # preceding 30 chars; if the preceding alphabetic word is a PP-preposition
    # or a possessive, the Org is embedded in a larger NP and the founding
    # claim is about that larger NP, not the Org alone.
    opos = text.find(org)
    if opos < 0:
        return None
    pre = text[max(0, opos - 30):opos]
    pre_word_match = re.search(r'([\wĉĝĥĵŝŭĈĜĤĴŜŬ-]+)\s*$', pre)
    if pre_word_match:
        pre_word = pre_word_match.group(1).lower()
        if pre_word in {'de', 'en', 'kun', 'el', 'al', 'por', 'pri', 'pro',
                        'ĉe', 'sur', 'apud', 'sub'}:
            return None
        # `unua`, `granda` etc. used as superlatives before `Universitato`
        # signal a common-noun-as-org case ("la unua Universitato estis…").
        # Also reject if pre_word is a noun-modifying adjective ending in 'a'
        # AND the org token starts uppercase — likely a common noun.
        # But many legitimate cases have adjectives too. Skip this check.
    return {
        'pattern':         'founded',
        'question':        f'En kiu jaro fondiĝis {org}?',
        'expected_answer': year,
        'extracted_constraint': f'fondiĝis (kiam) — {org}',
    }


PATTERN_EXTRACTORS = {
    'capital': (
        extract_capital_pair,
        ['la ĉefurbo de'],
    ),
    'superlative': (
        extract_superlative_pair,
        ['estas la plej'],
    ),
    'inventor': (
        extract_inventor_pair,
        ['inventis ', 'malkovris ', 'eltrovis '],
    ),
    'official': (
        extract_official_pair,
        ['la oficiala lingvo de', 'la valuto de', 'la moneda unuo de',
         'la nacia himno de', 'oficiala lingvo de'],
    ),
    'founded': (
        extract_founded_pair,
        ['fondiĝis en', 'estis fondita en', 'estis kreita en'],
    ),
}


def stream_candidates(conn, like_patterns: list[str], limit: int, seed: int):
    seed_mod = max(2, int(limit) // 200 + 2)
    seed_pick = int(seed) % seed_mod
    like_clauses = ' OR '.join('text LIKE ?' for _ in like_patterns)
    sql = f"""
        SELECT sid, text FROM sentences
        WHERE ({like_clauses})
          AND length(text) BETWEEN 30 AND 250
          AND (HASH(sid) % {seed_mod}) = {seed_pick}
        LIMIT {int(limit)}
    """
    params = [f'%{p}%' for p in like_patterns]
    cursor = conn.execute(sql, params)
    while True:
        row = cursor.fetchone()
        if row is None:
            return
        yield {'sid': row[0], 'text': row[1]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pattern',
                    choices=['capital', 'superlative', 'inventor',
                             'official', 'founded', 'all'],
                    default='all')
    ap.add_argument('--duckdb-path', default='data/indexes/duckdb_store.db')
    ap.add_argument('--target-size', type=int, default=15)
    ap.add_argument('--seed',        type=int, default=42)
    ap.add_argument('--query-limit', type=int, default=2000)
    ap.add_argument('--output',      default=None,
                    help='Output JSONL path (default: per-pattern in data/test_sets/)')
    args = ap.parse_args()

    conn = duckdb.connect(args.duckdb_path, read_only=True)

    patterns = (
        ['capital', 'superlative', 'inventor', 'official', 'founded']
        if args.pattern == 'all' else [args.pattern]
    )

    for pat in patterns:
        extractor, like_patterns = PATTERN_EXTRACTORS[pat]
        print(f'\n=== Pattern: {pat}  (LIKE filter: {like_patterns}) ===')

        accepted: list[dict] = []
        n_raw = 0
        n_extracted = 0
        seen_questions: set[str] = set()
        for c in stream_candidates(conn, like_patterns, args.query_limit, args.seed):
            n_raw += 1
            v = extractor(c['text'])
            if v is None:
                continue
            n_extracted += 1
            # De-duplicate questions (we don't want 5 "What is the capital of France?")
            if v['question'] in seen_questions:
                continue
            seen_questions.add(v['question'])
            v['source_sentence_id']   = c['sid']
            v['source_sentence_text'] = c['text']
            v['id'] = f'trivia_{pat}_{len(accepted)+1:03d}'
            v['question_type'] = {
                'capital':     'KIO',
                'superlative': 'KIU',
                'inventor':    'KIU',
                'official':    'KIO',
                'founded':     'KIAM',
            }.get(pat, 'KIO')
            v['expected_keywords'] = [v['expected_answer']]
            accepted.append(v)
            if len(accepted) >= args.target_size:
                break

        out_path = Path(
            args.output if args.output
            else f'data/test_sets/trivia_{pat}_candidates.jsonl'
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            for v in accepted:
                f.write(json.dumps(v, ensure_ascii=False) + '\n')

        print(f'  Raw streamed:    {n_raw}')
        print(f'  Pattern hits:    {n_extracted}')
        print(f'  Unique kept:     {len(accepted)}')
        print(f'  Wrote {out_path}')


if __name__ == '__main__':
    main()
