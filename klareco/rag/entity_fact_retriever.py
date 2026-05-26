"""
EntityFactRetriever — rank-1 lookup over the entity_facts table (#745).

When a question shape matches a known slot template, this retriever
returns the supporting source sentences directly, at the highest
possible confidence (analogous to how pattern_capital_of works but
generalized to every relation extracted by entity_fact_patterns).

Usage:
    retriever = EntityFactRetriever(duckdb_path=...)
    # Returns a list of sentence dicts (id, text, score=100, source='entity_fact')
    # if the question matches a known slot template; empty list otherwise.
    candidates = retriever.lookup(question_ast)

This is intended to slot in *before* the BM25 retriever in the
orchestrator: if it returns hits, use them; if not, fall back to the
BM25+AST-roles pipeline.

Question-shape → slot mapping (initial):

  KIE naskiĝis X?        → entity_facts(X, birth_place, ?)
  KIAM naskiĝis X?       → entity_facts(X, birth_year, ?)
  KIAM mortis X?         → entity_facts(X, death_year, ?)
  Kiu profesie estis X?  → entity_facts(X, profession, ?)
  Kiu fondis/inventis Y? → entity_facts(Y, founder, ?)
  Kie okazis X?          → entity_facts(X, location_of_event, ?)
  Kiam okazis X?         → entity_facts(X, year_of_event, ?)
  Kio estas X?           → entity_facts(X, definition, ?)
                            (only as fallback; definition is noisy)

Last Updated: 2026-05-26
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import duckdb

from klareco.rag.ast_aware_reranker import detect_question_type

logger = logging.getLogger(__name__)


# Question-type + verb-radiko → fact-slot mapping.
# These define which question shapes the entity-fact lookup can answer.
_QTYPE_SLOTS: dict[tuple[str, Optional[str]], str] = {
    # KIE (where) + birth verb → birth_place
    ('KIE',  'nask'):  'birth_place',
    ('KIAM', 'nask'):  'birth_year',
    ('KIAM', 'mort'):  'death_year',
    # KIE (where) + okazi verb → location of event
    ('KIE',  'okaz'):  'location_of_event',
    ('KIAM', 'okaz'):  'year_of_event',
    # KIO_DEF (definition) → definition fallback
    ('KIO_DEF', None): 'definition',
}

# KIU questions (who) need a special case: the slot depends on the verb.
# "Kiu fondis X?" → entity_facts(X, founder, ?). The question's *object* is
# the entity; the answer is the value.
_KIU_VERB_SLOTS: dict[str, str] = {
    'fond':     'founder',
    'kunfond':  'founder',
    'kre':      'founder',
    'invent':   'founder',
    'eltrov':   'founder',
    'malkovr':  'founder',
    'establ':   'founder',
    'iniciat':  'founder',
}

# Profession questions: "Kiu profesie estis X?" / "Kio estis la profesio de X?"
# detect via a marker in the question text. We don't try to be exhaustive —
# this is a recall booster, not a primary route.


class EntityFactRetriever:
    """Look up entity_facts directly when the question matches a known
    slot template. Returns [] when no template fits (so callers can
    fall back to BM25+AST)."""

    def __init__(self,
                 duckdb_path: str | Path = 'data/indexes/duckdb_store.db'):
        self.conn = duckdb.connect(str(duckdb_path), read_only=True)
        self.conn.execute("SET memory_limit = '2GB'")
        self.have_entity_facts = self._table_exists('entity_facts')
        if not self.have_entity_facts:
            logger.info(
                'EntityFactRetriever: entity_facts table not present; '
                'will return [] for every query. Run '
                'scripts/index/extract_entity_facts.py --apply to build it.'
            )

    def _table_exists(self, name: str) -> bool:
        try:
            self.conn.execute(f"SELECT * FROM {name} LIMIT 1").fetchone()
            return True
        except Exception:
            return False

    # ------ Public API ------

    def lookup(self, question_ast: dict) -> list[dict]:
        """Return candidate sentence dicts if the question maps to a known
        slot template. Empty list otherwise.

        Each dict has: id, text, score (=100), source ('entity_fact'),
        ef_slot, ef_value, ef_confidence (for debugging/scoring).
        """
        if not self.have_entity_facts or not isinstance(question_ast, dict):
            return []

        qtype = detect_question_type(question_ast)
        q_verb = (question_ast.get('verbo') or {}).get('radiko')
        q_verb_l = q_verb.lower() if q_verb else None

        # Identify (entity, slot) pair from the question's structure.
        entity_radiko, slot = self._route(qtype, q_verb_l, question_ast)
        if not entity_radiko or not slot:
            return []

        rows = self._lookup_entity_facts(entity_radiko, slot)
        if not rows:
            logger.debug(
                f'EntityFactRetriever miss: qtype={qtype} verb={q_verb_l!r} '
                f'entity_radiko={entity_radiko!r} slot={slot!r}')
            return []

        # Convert to retriever output format.
        out = []
        for sid, text, value, confidence in rows:
            out.append({
                'id':            sid,
                'text':          text,
                'score':         100.0 + 10.0 * confidence,  # KB lookups
                'source':        'entity_fact',
                'ast':           None,
                'ef_slot':       slot,
                'ef_value':      value,
                'ef_confidence': confidence,
            })
        return out

    # ------ Internal ------

    def _route(self, qtype: str, q_verb: Optional[str],
               question_ast: dict) -> tuple[Optional[str], Optional[str]]:
        """Return (entity_radiko, slot) the question is asking about,
        or (None, None) if no template matches."""
        # KIU questions: the answer is who did the action; the entity
        # is the verb's *object*.
        if qtype in ('KIU', 'KIU_OBJ') and q_verb and q_verb in _KIU_VERB_SLOTS:
            slot = _KIU_VERB_SLOTS[q_verb]
            obj_radiko = self._radiko_of(question_ast.get('objekto'))
            if obj_radiko:
                return obj_radiko.lower(), slot

        # KIE / KIAM questions: the answer is a place/year; the entity is
        # the *subject* (e.g. "Kie naskiĝis Zamenhof?" — subj=Zamenhof).
        if (qtype, q_verb) in _QTYPE_SLOTS:
            slot = _QTYPE_SLOTS[(qtype, q_verb)]
            subj_radiko = self._radiko_of(question_ast.get('subjekto'))
            if subj_radiko:
                return subj_radiko.lower(), slot

        # KIO_DEF (definition) → entity is the subject
        if qtype == 'KIO_DEF':
            slot = _QTYPE_SLOTS.get((qtype, None))
            subj_radiko = self._radiko_of(question_ast.get('subjekto'))
            if subj_radiko and slot:
                return subj_radiko.lower(), slot

        return None, None

    @staticmethod
    def _radiko_of(node) -> Optional[str]:
        if not isinstance(node, dict):
            return None
        kerno = node.get('kerno') if node.get('tipo') == 'vortgrupo' else node
        if not isinstance(kerno, dict):
            return None
        r = kerno.get('radiko')
        if isinstance(r, str) and r and r.lower() != 'ki':
            return r
        return None

    def _lookup_entity_facts(self, entity_radiko: str, slot: str
                              ) -> list[tuple[int, str, str, float]]:
        """Return (source_sid, text, value, confidence) tuples.

        Ordering:
          1. Source-count per value (the value with the most supporting
             sentences ranks first). For "Zamenhof birth_place" this
             ensures Bjalistoko (many mentions) outranks Ulica (1).
          2. Within a value, confidence DESC then sid ASC.

        Joins facts to sentences for the text. Limit 50."""
        try:
            rows = self.conn.execute("""
                WITH cnt AS (
                  SELECT value_radiko, COUNT(*) AS n
                  FROM entity_facts
                  WHERE entity_radiko = ? AND slot = ?
                  GROUP BY value_radiko
                )
                SELECT ef.source_sid, s.text, ef.value, ef.confidence
                FROM entity_facts ef
                JOIN sentences s ON s.sid = ef.source_sid
                JOIN cnt c ON c.value_radiko = ef.value_radiko
                WHERE ef.entity_radiko = ?
                  AND ef.slot = ?
                ORDER BY c.n DESC, ef.confidence DESC, ef.source_sid
                LIMIT 50
            """, [entity_radiko, slot, entity_radiko, slot]).fetchall()
            return rows
        except Exception as e:
            logger.warning(f'entity_facts lookup failed: {e}')
            return []

    def reverse_lookup(self, slot: str, value_radiko: str
                       ) -> list[tuple[int, str, str, float]]:
        """`(slot, value)` → entities. Used for "who founded X?" where X
        is known and the answer is the entity."""
        if not self.have_entity_facts:
            return []
        try:
            return self.conn.execute("""
                SELECT ef.source_sid, s.text, ef.entity_radiko, ef.confidence
                FROM entity_facts ef
                JOIN sentences s ON s.sid = ef.source_sid
                WHERE ef.slot = ?
                  AND ef.value_radiko = ?
                ORDER BY ef.confidence DESC, ef.source_sid
                LIMIT 50
            """, [slot, value_radiko.lower()]).fetchall()
        except Exception as e:
            logger.warning(f'reverse entity_facts lookup failed: {e}')
            return []
