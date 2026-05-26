"""
Multi-sentence discourse generation from entity_facts (#766 #775 Stage 1).

Hybrid strategy: rather than realizing new sentences from scratch (which
risks ungrammatical output if a template is wrong), we:

  1. Look up the source sentences that supported each slot's best fact.
  2. Order those sentences by a discourse template (intro → birth →
     realization → death, etc.).
  3. Deduplicate when one sentence already covers multiple slots.
  4. Apply cheap pronoun reduction between consecutive sentences that
     share the same subject.

The deparser remains the grammatical guarantee for any NEW sentences
we synthesize (currently only the intro). Surface-derived sentences
inherit their grammaticality from the corpus.

V1 covers: biography(), define(), compare().
Future: events, places, summarization, more template families.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import duckdb

logger = logging.getLogger(__name__)

DB = 'data/indexes/duckdb_store.db'


# ---------------------------------------------------------------------------
# Connection / lookup helpers
# ---------------------------------------------------------------------------

_DEFAULT_CONN: Optional[duckdb.DuckDBPyConnection] = None


def _conn() -> duckdb.DuckDBPyConnection:
    global _DEFAULT_CONN
    if _DEFAULT_CONN is None:
        _DEFAULT_CONN = duckdb.connect(DB, read_only=True)
        _DEFAULT_CONN.execute("SET memory_limit = '1GB'")
    return _DEFAULT_CONN


def _best_fact_for_slot(entity_radiko: str, slot: str,
                         conn=None) -> Optional[dict]:
    """Return the most-supported fact for (entity, slot), or None.

    Ranking matches EntityFactRetriever: per-value source-count DESC,
    then confidence DESC.
    """
    conn = conn or _conn()
    rows = conn.execute("""
        WITH cnt AS (
          SELECT value_radiko, COUNT(*) AS n
          FROM entity_facts
          WHERE entity_radiko = ? AND slot = ?
          GROUP BY value_radiko
        )
        SELECT ef.value, ef.value_radiko, ef.source_sid, s.text, ef.confidence
        FROM entity_facts ef
        JOIN sentences s ON s.sid = ef.source_sid
        JOIN cnt c ON c.value_radiko = ef.value_radiko
        WHERE ef.entity_radiko = ? AND ef.slot = ?
        ORDER BY c.n DESC, ef.confidence DESC, ef.source_sid
        LIMIT 1
    """, [entity_radiko, slot, entity_radiko, slot]).fetchone()
    if rows is None:
        return None
    value, value_radiko, sid, text, conf = rows
    return {'slot': slot, 'value': value, 'value_radiko': value_radiko,
            'source_sid': int(sid), 'text': text, 'confidence': float(conf)}


def _reverse_lookup_one(slot: str, target_value_radiko: str,
                         conn=None) -> Optional[dict]:
    """Reverse: which entity has (entity, slot, target)? Best match only."""
    conn = conn or _conn()
    rows = conn.execute("""
        SELECT ef.entity_radiko, ef.source_sid, s.text, ef.confidence
        FROM entity_facts ef
        JOIN sentences s ON s.sid = ef.source_sid
        WHERE ef.slot = ? AND ef.value_radiko = ?
        ORDER BY ef.confidence DESC, ef.source_sid
        LIMIT 1
    """, [slot, target_value_radiko.lower()]).fetchone()
    if rows is None:
        return None
    ent, sid, text, conf = rows
    return {'slot': slot, 'entity_radiko': ent,
            'source_sid': int(sid), 'text': text, 'confidence': float(conf)}


def _surface_form(entity_radiko: str, conn=None) -> str:
    """Find the canonical surface form for an entity radiko by looking
    in `sentences` for a propra_nomo with this radiko. Falls back to
    capitalizing the radiko itself."""
    conn = conn or _conn()
    row = conn.execute("""
        SELECT subj_radiko FROM sentences
        WHERE LOWER(subj_radiko) = ?
          AND subj_vortspeco = 'propra_nomo'
        LIMIT 1
    """, [entity_radiko.lower()]).fetchone()
    if row and row[0]:
        return row[0]
    return entity_radiko.capitalize()


def _entity_is_person(entity_radiko: str, conn=None) -> bool:
    """Heuristic: does this entity have person-like facts?"""
    conn = conn or _conn()
    n = conn.execute("""
        SELECT COUNT(*) FROM entity_facts
        WHERE entity_radiko = ?
          AND slot IN ('birth_year','birth_place','death_year','profession')
    """, [entity_radiko.lower()]).fetchone()[0]
    return n > 0


def _is_biographable(entity_radiko: str, conn=None) -> bool:
    """An entity is biographable if it predominantly appears as a
    propra_nomo (named entity) rather than a substantivo (common noun).

    Catches the case where a radiko like `land` (country) accidentally
    pattern-matches biographical facts ("La Lando Oz naskiĝis...")
    despite being a common noun in 5000+ other sentences.

    Returns True iff propra_nomo / (propra_nomo + substantivo) > 0.5
    AND at least one biographical fact exists.
    """
    conn = conn or _conn()
    er = entity_radiko.lower()
    n_pn, n_sub = conn.execute("""
        SELECT
          SUM(CASE WHEN subj_vortspeco = 'propra_nomo' THEN 1 ELSE 0 END),
          SUM(CASE WHEN subj_vortspeco = 'substantivo' THEN 1 ELSE 0 END)
        FROM sentences WHERE LOWER(subj_radiko) = ?
    """, [er]).fetchone()
    n_pn = n_pn or 0
    n_sub = n_sub or 0
    if n_pn + n_sub == 0:
        return False
    return (n_pn / (n_pn + n_sub)) > 0.5


# ---------------------------------------------------------------------------
# Cohesion: pronoun reduction
# ---------------------------------------------------------------------------

# Subject-name → pronoun substitution. Esperanto distinguishes
# li (male) / ŝi (female) / ĝi (non-person), but we don't store
# gender. Default to "li" for persono (most biographical sources
# are male-skewed in the corpus); fall back to "ĝi" otherwise.

def _pronoun_for(entity_radiko: str, is_person: bool) -> str:
    return 'Li' if is_person else 'Ĝi'


def _apply_pronoun_reduction(sentences: list[str], surface_name: str,
                              pronoun: str) -> list[str]:
    """For each sentence after the first that starts with surface_name,
    replace the leading occurrence with the pronoun. Preserves capitalization."""
    if not sentences:
        return sentences
    out = [sentences[0]]
    for s in sentences[1:]:
        # Strip leading whitespace then check for the name
        stripped = s.lstrip()
        if stripped.startswith(surface_name + ' '):
            # Replace just the first occurrence
            replacement = pronoun + ' ' + stripped[len(surface_name) + 1:]
            out.append(replacement)
        else:
            out.append(s)
    return out


def _trim_sentence(text: str, max_len: int = 220) -> str:
    """Source sentences can be long. Trim at the first sensible boundary
    if past max_len; preserve a terminating period."""
    text = text.strip()
    if len(text) <= max_len:
        # Ensure period
        if text and text[-1] not in '.!?':
            text += '.'
        return text
    # Find first sensible break (period, comma) within bounds
    cutoff = text.rfind(',', 0, max_len)
    if cutoff < max_len // 2:
        cutoff = max_len
    return text[:cutoff].rstrip(' ,;') + '.'


# ---------------------------------------------------------------------------
# Public generators
# ---------------------------------------------------------------------------

def biography(entity_radiko: str, conn=None) -> str:
    """Produce a multi-sentence biography from facts in entity_facts.

    Discourse plan:
        1. Introduction (name + profession if known)
        2. Birth (place and/or year)
        3. Notable realization (what they founded / wrote / discovered)
        4. Death (year)

    Returns a paragraph; returns a 'no info' message if the entity has
    no facts.
    """
    entity_radiko = entity_radiko.lower()
    conn = conn or _conn()
    surface = _surface_form(entity_radiko, conn)
    is_person = _entity_is_person(entity_radiko, conn)
    pronoun = _pronoun_for(entity_radiko, is_person)

    # Gather facts
    facts: dict[str, Optional[dict]] = {
        'profession':   _best_fact_for_slot(entity_radiko, 'profession', conn),
        'birth_year':   _best_fact_for_slot(entity_radiko, 'birth_year', conn),
        'birth_place':  _best_fact_for_slot(entity_radiko, 'birth_place', conn),
        'death_year':   _best_fact_for_slot(entity_radiko, 'death_year', conn),
        'definition':   _best_fact_for_slot(entity_radiko, 'definition', conn),
    }
    # Reverse lookup: what did this entity found?
    founded = conn.execute("""
        SELECT ef.entity_radiko, ef.source_sid, s.text, ef.confidence
        FROM entity_facts ef
        JOIN sentences s ON s.sid = ef.source_sid
        WHERE ef.slot = 'founder' AND ef.value_radiko = ?
        ORDER BY ef.confidence DESC, ef.source_sid
        LIMIT 1
    """, [entity_radiko]).fetchone()
    if founded:
        ent_r, sid, text, conf = founded
        facts['founded'] = {'slot': 'founded', 'value': ent_r,
                            'source_sid': int(sid), 'text': text,
                            'confidence': float(conf)}

    n_facts = sum(1 for f in facts.values() if f is not None)
    if n_facts == 0:
        return f"Mi ne havas faktojn pri {surface}."

    # If only the 'definition' slot fired, this is probably a common
    # noun or category, not a biographical subject. Bail to define().
    if n_facts == 1 and facts.get('definition') is not None:
        return define(entity_radiko, conn=conn)
    # If the entity has no biographical slots at all (no birth/death/
    # profession/founded), it's probably not a person; return its
    # definition only.
    biographical_slots = ('profession', 'birth_year', 'birth_place',
                          'death_year', 'founded')
    if not any(facts.get(s) is not None for s in biographical_slots):
        return define(entity_radiko, conn=conn)
    # The entity has biographical-looking facts but only because pattern
    # extraction misfired on a common-noun radiko. If propra_nomo
    # mentions are outnumbered by substantivo mentions, bail.
    if not _is_biographable(entity_radiko, conn):
        return define(entity_radiko, conn=conn)

    # Order by discourse plan, dedupe source sids (one source can cover
    # multiple slots; emit each sentence only once)
    plan = ['profession', 'birth_year', 'birth_place', 'founded',
            'death_year', 'definition']
    seen_sids: set[int] = set()
    sentences: list[str] = []
    for slot in plan:
        f = facts.get(slot)
        if f is None:
            continue
        sid = f['source_sid']
        if sid in seen_sids:
            continue
        seen_sids.add(sid)
        sentences.append(_trim_sentence(f['text']))

    # Cohesion: replace repeated leading-name with pronoun
    sentences = _apply_pronoun_reduction(sentences, surface, pronoun)
    return ' '.join(sentences)


def define(entity_radiko: str, conn=None) -> str:
    """Produce a definition. Prefers the 'definition' slot if present;
    falls back to assembling from type + key properties."""
    entity_radiko = entity_radiko.lower()
    conn = conn or _conn()
    surface = _surface_form(entity_radiko, conn)
    defn = _best_fact_for_slot(entity_radiko, 'definition', conn)
    if defn is not None:
        return _trim_sentence(defn['text'])

    # No explicit definition — fall back to whatever's in the
    # 'category' / 'profession' / 'founded' slots that says what X IS.
    for slot in ('profession', 'birth_place', 'birth_year'):
        f = _best_fact_for_slot(entity_radiko, slot, conn)
        if f is not None:
            return _trim_sentence(f['text'])
    return f"Mi ne havas difinon de {surface}."


def compare(a_radiko: str, b_radiko: str, conn=None) -> str:
    """Three-sentence comparison: shared category, divergent A, divergent B."""
    a_radiko = a_radiko.lower()
    b_radiko = b_radiko.lower()
    conn = conn or _conn()
    a_surf = _surface_form(a_radiko, conn)
    b_surf = _surface_form(b_radiko, conn)

    # Gather a few headline facts per entity, prefer the same slots so
    # the comparison aligns
    sentences: list[str] = []
    for slot in ('profession', 'birth_year', 'birth_place', 'death_year'):
        a_fact = _best_fact_for_slot(a_radiko, slot, conn)
        b_fact = _best_fact_for_slot(b_radiko, slot, conn)
        if a_fact is not None and b_fact is not None:
            # Pick the more compact phrasing for the comparison
            sentences.append(_trim_sentence(a_fact['text'], 140))
            sentences.append(_trim_sentence(b_fact['text'], 140))
            break
    if not sentences:
        # No shared slot — return what we have
        a_intro = _best_fact_for_slot(a_radiko, 'profession', conn) or \
                  _best_fact_for_slot(a_radiko, 'definition', conn)
        b_intro = _best_fact_for_slot(b_radiko, 'profession', conn) or \
                  _best_fact_for_slot(b_radiko, 'definition', conn)
        if a_intro:
            sentences.append(_trim_sentence(a_intro['text']))
        if b_intro:
            sentences.append(_trim_sentence(b_intro['text']))

    if not sentences:
        return f"Mi ne havas sufiĉe da informoj pri {a_surf} kaj {b_surf} por kompari ilin."
    return ' '.join(sentences)
