"""
Forward-chaining inference + constraint propagation (#749 #759).

Walk the entity_facts table, apply declarative rules to derive new
facts, write them back. Idempotent: re-runs produce the same KB.

A rule is a function that takes a dict of {(entity, slot): value} and
returns a list of new Fact tuples. Rules are pure functions over the
fact table — no DB writes inside rules; the orchestrator persists.

The classic example:
    (X, birth_year, Y) ∧ (X, death_year, Z)  ⊢  (X, age_at_death, Z-Y)

Implementation note: 5,199 facts × ~7 rules per iteration × a few
iterations to saturate is sub-second in pure Python. No need for
PyDatalog complexity at this scale, though we could swap it in later
if rules grow combinatorial.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, Optional

import duckdb

from klareco.knowledge.entity_facts import SLOTS

logger = logging.getLogger(__name__)


# A simplified Fact tuple for in-memory work. Insertion to DuckDB
# uses the full entity_facts row including source_sid + confidence.
@dataclass(frozen=True)
class DerivedFact:
    entity_radiko: str
    slot:          str
    value:         str
    value_radiko:  str
    source_sid:    int       # source sid of the most-supporting parent fact
    confidence:    float
    rule_name:     str


@dataclass
class Rule:
    """A forward-chaining rule. The body is a callable that takes
    `facts_index` (a dict keyed by (entity_radiko, slot) for fast
    lookup) and yields DerivedFact instances.

    Rules are pure: they read but never write the index. The driver
    handles persistence."""
    name:   str
    desc:   str
    body:   Callable
    iterates: bool = False   # True = needs saturation; False = one pass suffices


# ---------------------------------------------------------------------------
# Rule library
# ---------------------------------------------------------------------------

def _r_age_at_death(idx: dict) -> list[DerivedFact]:
    """(X, birth_year, Y) ∧ (X, death_year, Z) ⊢ (X, age_at_death, Z-Y)"""
    out = []
    # idx is keyed by (entity, slot) → list of (value, value_radiko, sid, conf)
    entities = {e for (e, s) in idx.keys()}
    for e in entities:
        births = idx.get((e, 'birth_year'))
        deaths = idx.get((e, 'death_year'))
        if not (births and deaths):
            continue
        try:
            by = int(births[0][0])  # best birth year
            dy = int(deaths[0][0])  # best death year
        except (ValueError, IndexError):
            continue
        age = dy - by
        if 0 < age < 150:  # sanity
            out.append(DerivedFact(
                entity_radiko=e, slot='age_at_death',
                value=str(age), value_radiko=str(age),
                source_sid=deaths[0][2],  # cite the death-year source
                confidence=min(births[0][3], deaths[0][3]) * 0.95,
                rule_name='age_at_death',
            ))
    return out


def _r_lifespan(idx: dict) -> list[DerivedFact]:
    """(X, birth_year, Y) ∧ (X, death_year, Z) ⊢ (X, lifespan, "Y-Z")"""
    out = []
    entities = {e for (e, s) in idx.keys()}
    for e in entities:
        births = idx.get((e, 'birth_year'))
        deaths = idx.get((e, 'death_year'))
        if not (births and deaths):
            continue
        by, dy = births[0][0], deaths[0][0]
        span = f'{by}–{dy}'
        out.append(DerivedFact(
            entity_radiko=e, slot='lifespan',
            value=span, value_radiko=span,
            source_sid=births[0][2],
            confidence=min(births[0][3], deaths[0][3]) * 0.95,
            rule_name='lifespan',
        ))
    return out


def _r_alive_in(idx: dict, sample_years=(1800, 1850, 1900, 1950, 2000)
                ) -> list[DerivedFact]:
    """For a few canonical sample years, derive (X, alive_in, YEAR)
    when birth_year <= YEAR <= death_year. Useful for contemporary
    queries."""
    out = []
    entities = {e for (e, s) in idx.keys()}
    for e in entities:
        births = idx.get((e, 'birth_year'))
        deaths = idx.get((e, 'death_year'))
        if not (births and deaths):
            continue
        try:
            by = int(births[0][0])
            dy = int(deaths[0][0])
        except ValueError:
            continue
        for yr in sample_years:
            if by <= yr <= dy:
                out.append(DerivedFact(
                    entity_radiko=e, slot='alive_in',
                    value=str(yr), value_radiko=str(yr),
                    source_sid=births[0][2],
                    confidence=min(births[0][3], deaths[0][3]) * 0.9,
                    rule_name='alive_in',
                ))
    return out


def _r_founder_of_implies_creator(idx: dict) -> list[DerivedFact]:
    """(X, founder, Y) ⊢ (Y, has_creator, X). Inverse-relation indexing
    so reverse lookups don't require a separate query."""
    out = []
    for (entity, slot), facts in idx.items():
        if slot != 'founder':
            continue
        for value, value_r, sid, conf in facts:
            out.append(DerivedFact(
                entity_radiko=value_r,
                slot='has_creator',
                value=entity,
                value_radiko=entity,
                source_sid=sid,
                confidence=conf,
                rule_name='inverse_founder',
            ))
    return out


def _r_transitive_event_loko(idx: dict) -> list[DerivedFact]:
    """(EVENT, year_of_event, YEAR) ∧ (EVENT, location_of_event, LOKO)
    ⊢ (LOKO, hosted_event_in_year, "EVENT YEAR")

    Useful for KIE-de-YEAR-Olimpikoj-style questions: given the year
    and event, derive the place.
    """
    out = []
    entities = {e for (e, s) in idx.keys()}
    for e in entities:
        years = idx.get((e, 'year_of_event'))
        lokoj = idx.get((e, 'location_of_event'))
        if not (years and lokoj):
            continue
        for y_val, y_r, y_sid, y_conf in years[:3]:
            for l_val, l_r, l_sid, l_conf in lokoj[:3]:
                out.append(DerivedFact(
                    entity_radiko=l_r,
                    slot='hosted_event_in_year',
                    value=f'{e}:{y_val}',
                    value_radiko=f'{e}:{y_val}',
                    source_sid=l_sid,
                    confidence=min(y_conf, l_conf) * 0.85,
                    rule_name='loko_hosted_event_in_year',
                ))
    return out


ALL_RULES: list[Rule] = [
    Rule('age_at_death',  'birth_year + death_year → age',
         _r_age_at_death),
    Rule('lifespan',      'birth_year + death_year → lifespan string',
         _r_lifespan),
    Rule('alive_in',      'sample years between birth and death',
         _r_alive_in),
    Rule('inverse_founder', 'founder relation → has_creator inverse',
         _r_founder_of_implies_creator),
    Rule('loko_hosted_event', 'event + year + loko → loko hosted event-in-year',
         _r_transitive_event_loko),
]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _load_facts_index(conn) -> dict:
    """Load entity_facts into a dict keyed by (entity_radiko, slot),
    value = list of (value, value_radiko, source_sid, confidence) tuples
    sorted by confidence DESC then sid."""
    idx: dict = {}
    rows = conn.execute(f"""
        SELECT entity_radiko, slot, value, value_radiko,
               source_sid, confidence
        FROM {SLOTS} ef
        ORDER BY entity_radiko, slot, confidence DESC, source_sid
    """).fetchall()
    for er, slot, val, val_r, sid, conf in rows:
        idx.setdefault((er, slot), []).append((val, val_r, int(sid), float(conf)))
    return idx


def apply_rules(conn, rules: Optional[list[Rule]] = None,
                 dry_run: bool = False) -> dict:
    """Run all rules over entity_facts. Persists derived rows with
    pattern_name='inferred:<rule_name>'.

    Returns per-rule counts."""
    if rules is None:
        rules = ALL_RULES

    t0 = time.time()
    idx = _load_facts_index(conn)
    n_base = sum(len(v) for v in idx.values())
    logger.info(f'inference: loaded {n_base:,} base facts')

    # First pass: apply each rule once
    derived: list[DerivedFact] = []
    per_rule: dict[str, int] = {}
    for rule in rules:
        before = len(derived)
        new_facts = rule.body(idx) or []
        derived.extend(new_facts)
        per_rule[rule.name] = len(new_facts)
        logger.info(f'  rule {rule.name}: +{len(new_facts)} facts')

    if dry_run:
        return {'derived_total': len(derived), 'per_rule': per_rule,
                'elapsed_s': time.time() - t0}

    # Persist
    if not derived:
        return {'derived_total': 0, 'per_rule': per_rule,
                'elapsed_s': time.time() - t0}

    # ⚠️ #881: the WRITE-BACK path below still assumes the old SLOT schema
    # (pattern_name, entity_radiko, ...) — the live table is the TRIPLE schema
    # and has no `pattern_name` column, so this DELETE/INSERT would raise. The
    # READ path is adapted (via SLOTS); persisting inferred facts is deferred
    # until facts flow as FactFragments (MVP-1) or the triple extractor gains a
    # provenance column. apply_rules(dry_run=True) is the supported mode today.
    conn.execute("DELETE FROM entity_facts "
                  "WHERE pattern_name LIKE 'inferred:%'")

    rows = [(f.entity_radiko, f.slot, f.value, f.value_radiko,
             f.source_sid, f.confidence, f'inferred:{f.rule_name}')
            for f in derived]
    conn.executemany(
        "INSERT INTO entity_facts (entity_radiko, slot, value, value_radiko, "
        "source_sid, confidence, pattern_name) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.execute("CHECKPOINT")
    elapsed = time.time() - t0
    logger.info(f'inference: wrote {len(derived)} derived facts in {elapsed:.2f}s')
    return {'derived_total': len(derived), 'per_rule': per_rule,
            'elapsed_s': elapsed}
