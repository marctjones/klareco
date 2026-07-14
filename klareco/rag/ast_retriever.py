"""
AST-aware retriever: routes by question shape to structural lookups,
falling back to BM25 only for unstructured queries.

VERSION: v2.x
COMPATIBLE WITH: post-bug-fix parser + DuckDB store with
                 entity_postings, verb_klaso, verb_negated columns
DEPENDENCIES: duckdb, klareco.parser, klareco.rag.question_shape
STAGE: Retrieval

Description:
    Bypasses BM25 for question shapes we can answer structurally.

    Routing:
      ┌─ capital_of, founded_year_of, official_language_of, currency_of
      │      → KB lookup (pattern_capital_of, etc.) when those tables exist
      │
      ├─ who_agent_of_work, who_agent, who_invented_discovered
      │      → entity_postings lookup on the anchor entity
      │        + optional verb_klaso filter
      │
      ├─ where_born, where_located, where_occurred
      │      → entity_postings lookup + locative-PP requirement
      │
      ├─ when_born, when_founded, when_occurred
      │      → entity_postings lookup + temporal-PP requirement
      │
      ├─ what_is
      │      → entity_postings lookup (loose)
      │
      └─ unstructured / generic
              → WhooshRetriever fallback (BM25)

    For each structural route, the retriever:
      1. Queries DuckDB for sids matching the anchor + filters
      2. Fetches text + AST for those sids in one batch
      3. Optionally augments with a small BM25 pool to catch borderline
         lexical-only matches
      4. Returns a candidate pool the reranker can score

Pipeline Position:
    Question → ParseStage → ASTRetriever (this module) → reranker → extractor

Last Updated: 2026-05-20
Author: Claude Code (with Marc Jones)
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import duckdb

from klareco.rag.question_shape import classify as classify_shape, Shape

logger = logging.getLogger(__name__)


class ASTRetriever:
    """Routes a question to a retrieval strategy based on its AST shape.

    Falls back to a provided BM25 retriever (e.g. WhooshRetriever) when
    the question doesn't fit a structural shape, or when the structural
    lookup returns too few candidates.
    """

    def __init__(self,
                 duckdb_path: str | Path = 'data/indexes/duckdb_store.db',
                 bm25_fallback=None,
                 min_candidates: int = 10):
        """
        Parameters
        ----------
        duckdb_path : path to DuckDB store with entity_postings + shredded cols
        bm25_fallback : an object with .retrieve_with_ast_roles(ast, top_k)
                        method (e.g. WhooshRetriever or DuckDBRetriever)
        min_candidates : if structural lookup returns fewer than this,
                         augment with BM25 fallback.
        """
        self.conn = duckdb.connect(str(duckdb_path), read_only=True)
        # OOM safety: cap each ASTRetriever's working memory
        self.conn.execute("SET memory_limit = '2GB'")
        self.conn.execute("SET threads = 4")
        self.bm25_fallback = bm25_fallback
        self.min_candidates = min_candidates

        # Detect which assets are available — graceful degradation
        self.have_entity_postings = self._table_exists('entity_postings')
        self.have_pattern_capital = self._table_exists('pattern_capital_of')
        self.have_pattern_founded = self._table_exists('pattern_founded_year_of')
        self.have_pattern_lingvo = self._table_exists('pattern_official_language_of')

        # the frame lives on `clauses` (#836), not on `sentences`
        self.have_verb_klaso = self._column_exists('clauses', 'verb_klaso')
        self.have_verb_negated = self._column_exists('clauses', 'verb_negated')

        logger.info(
            f"ASTRetriever asset summary: "
            f"entity_postings={self.have_entity_postings} | "
            f"pattern_capital_of={self.have_pattern_capital} | "
            f"verb_klaso={self.have_verb_klaso} | "
            f"verb_negated={self.have_verb_negated}"
        )

    def _table_exists(self, name: str) -> bool:
        try:
            self.conn.execute(f"SELECT * FROM {name} LIMIT 1").fetchone()
            return True
        except Exception:
            return False

    def _column_exists(self, table: str, col: str) -> bool:
        rows = self.conn.execute(
            f"SELECT column_name FROM information_schema.columns "
            f"WHERE table_name = '{table}'"
        ).fetchall()
        return any(r[0] == col for r in rows)

    # =========================================================================
    # Per-shape retrieval strategies
    # =========================================================================

    def _retrieve_kb(self, table: str, lookup_col: str, lookup_val: str,
                    select_col: str) -> list[dict]:
        """KB lookup: SELECT select_col, sid FROM table WHERE lookup_col = ?"""
        try:
            rows = self.conn.execute(
                f"SELECT {select_col}, sid FROM {table} "
                f"WHERE {lookup_col} = ?",
                [lookup_val]
            ).fetchall()
        except Exception as e:
            logger.warning(f"KB lookup on {table} failed: {e}")
            return []
        if not rows:
            return []
        # Pull the underlying sentence rows for each hit
        sids = [r[1] for r in rows]
        if not sids:
            return []
        # Use IN clause for batch fetch
        placeholders = ','.join(['?'] * len(sids))
        sentence_rows = self.conn.execute(
            f"SELECT sid, text FROM sentences WHERE sid IN ({placeholders})",
            sids
        ).fetchall()
        # Return in KB-hit order
        text_by_sid = {sid: text for sid, text in sentence_rows}
        return [
            {'id':    sid,
             'text':  text_by_sid.get(sid, ''),
             'score': 100.0,                  # very high — direct KB hit
             'ast':   None,                    # ast not needed at this stage
             'source':       'kb_lookup',
             'kb_table':     table,
             'kb_answer':    answer}
            for answer, sid in rows
        ]

    def _retrieve_entity_postings(self, entity: str,
                                   verb_klaso: Optional[str] = None,
                                   verb_radiko: Optional[str] = None,
                                   require_negated: Optional[bool] = None,
                                   top_k: int = 100) -> list[dict]:
        """Lookup sentences mentioning the entity, optionally filtered by
        verb class / radiko / negation polarity."""
        if not self.have_entity_postings:
            return []

        # Build the WHERE clause progressively
        sql = (
            "SELECT s.sid, s.text, s.subj_radiko, s.verb_radiko, s.obj_radiko "
            "FROM entity_postings ep "
            "JOIN sentences s ON s.sid = ep.sid "
            "WHERE ep.entity_normalized = ?"
        )
        params: list = [self._fold(entity)]

        if verb_klaso and self.have_verb_klaso:
            sql += " AND s.verb_klaso = ?"
            params.append(verb_klaso)
        elif verb_radiko:
            sql += " AND s.verb_radiko = ?"
            params.append(verb_radiko)

        if require_negated is not None and self.have_verb_negated:
            if require_negated:
                sql += " AND s.verb_negated = TRUE"
            else:
                sql += " AND (s.verb_negated IS NULL OR s.verb_negated = FALSE)"

        sql += f" LIMIT {int(top_k)}"

        try:
            rows = self.conn.execute(sql, params).fetchall()
        except Exception as e:
            logger.warning(f"entity_postings lookup failed: {e}")
            return []
        return [
            {'id':           sid,
             'text':         text,
             'score':        10.0,  # base score; reranker refines
             'ast':          None,
             'source':       'entity_postings',
             'subj_radiko':  subj_r,
             'verb_radiko':  verb_r,
             'obj_radiko':   obj_r}
            for sid, text, subj_r, verb_r, obj_r in rows
        ]

    @staticmethod
    def _fold(s: str) -> str:
        """Diacritic-fold + lowercase to match entity_normalized."""
        import unicodedata
        decomposed = unicodedata.normalize('NFKD', s or '')
        return ''.join(c for c in decomposed if not unicodedata.combining(c)).lower()

    def _verb_klaso_for_radiko(self, radiko: Optional[str]) -> Optional[str]:
        """Look up the VerbaKlaso for a verb radiko."""
        if not radiko:
            return None
        try:
            row = self.conn.execute(
                "SELECT class_id FROM ontology_edges "
                "WHERE rel = 'APARTENAS_AL_VERBA_KLASO' AND radiko = ?",
                [radiko]
            ).fetchone()
            return row[0] if row else None
        except Exception:
            return None

    # =========================================================================
    # Top-level retrieve()
    # =========================================================================

    def retrieve_with_ast_roles(self, question_ast: dict, top_k: int = 100
                                 ) -> list[dict]:
        """Implements the interface used by RetrieveStage. Backward-compatible
        with the BM25 retriever signature so this can be swapped in directly.

        Returns a list of {id, text, score, ast, source, ...} dicts.
        """
        # Extract question text from the AST's frazo_teksto / surface
        # (parsers return original text in different fields; try a few)
        question_text = (
            (question_ast or {}).get('frazo_teksto')
            or (question_ast or {}).get('teksto')
            or (question_ast or {}).get('original_text')
            or ''
        )
        if not question_text:
            # Last resort: assemble from word ASTs
            words = []
            for role in ('subjekto', 'verbo', 'objekto'):
                k = (question_ast or {}).get(role)
                if isinstance(k, dict):
                    if k.get('tipo') == 'vortgrupo':
                        k = k.get('kerno') or {}
                    pv = k.get('plena_vorto')
                    if pv:
                        words.append(pv)
            for x in (question_ast or {}).get('aliaj') or []:
                if isinstance(x, dict):
                    k = x.get('kerno') if x.get('tipo') == 'vortgrupo' else x
                    if isinstance(k, dict):
                        pv = k.get('plena_vorto')
                        if pv:
                            words.append(pv)
            question_text = ' '.join(words)

        shape_info = classify_shape(question_text, question_ast or {})
        logger.debug(f"Question shape: {shape_info.shape.value}, "
                     f"anchor={shape_info.anchor_entity}, "
                     f"y={shape_info.constraint_y}")

        # --- Route by shape ---
        candidates: list[dict] = []
        route = 'fallback_bm25'

        if shape_info.shape == Shape.CAPITAL_OF and self.have_pattern_capital and shape_info.constraint_y:
            candidates = self._retrieve_kb(
                'pattern_capital_of', 'country', shape_info.constraint_y, 'city'
            )
            route = 'kb_capital_of'
        elif shape_info.shape == Shape.FOUNDED_YEAR_OF and self.have_pattern_founded and shape_info.anchor_entity:
            candidates = self._retrieve_kb(
                'pattern_founded_year_of', 'org', shape_info.anchor_entity, 'year'
            )
            route = 'kb_founded'
        elif shape_info.shape == Shape.OFFICIAL_LANGUAGE_OF and self.have_pattern_lingvo and shape_info.constraint_y:
            candidates = self._retrieve_kb(
                'pattern_official_language_of', 'country', shape_info.constraint_y, 'language'
            )
            route = 'kb_lingvo'
        elif shape_info.shape in (
            Shape.WHO_AGENT_OF_WORK, Shape.WHO_AGENT,
            Shape.WHO_INVENTED_DISCOVERED, Shape.WHAT_IS,
            Shape.WHERE_BORN, Shape.WHERE_LOCATED, Shape.WHERE_OCCURRED,
            Shape.WHEN_BORN, Shape.WHEN_OCCURRED, Shape.WHEN_FOUNDED,
        ) and shape_info.anchor_entity:
            # Multi-route (#741/#744 Phase B option #3): always run BOTH
            # the strict (verb-filtered) and loose (entity-only) variants
            # and union them, instead of strict-then-fallback-when-narrow.
            # Strict candidates rank first (they're more specific); loose
            # candidates fill in. Closes the R@100 gap where strict's
            # over-filtering was dropping correct candidates entirely.
            verb_klaso = self._verb_klaso_for_radiko(shape_info.verb_radiko)
            strict_candidates = self._retrieve_entity_postings(
                entity=shape_info.anchor_entity,
                verb_klaso=verb_klaso,
                verb_radiko=shape_info.verb_radiko if not verb_klaso else None,
                top_k=top_k,
            )
            loose_candidates: list[dict] = []
            if verb_klaso or shape_info.verb_radiko:
                loose_candidates = self._retrieve_entity_postings(
                    entity=shape_info.anchor_entity, top_k=top_k,
                )
            # Union: strict first (preserves high-confidence ordering),
            # loose adds new sids not already in strict.
            seen = {c['id'] for c in strict_candidates}
            candidates = list(strict_candidates)
            for c in loose_candidates:
                if c['id'] not in seen:
                    candidates.append(c)
                    seen.add(c['id'])
            route = f'entity_postings_{shape_info.shape.value}'
            if loose_candidates:
                route += '_multi'

        # Always-on BM25 supplement (#741/#744 Phase B option #1).
        # The R@100 audit (commit e836350 bench) revealed that
        # structural routing was dropping 15-17 of 120 answers from the
        # top-100 candidate pool entirely. No reranker can recover what
        # the retriever loses. To restore retrieval recall to BM25's
        # ceiling, ALWAYS run BM25 alongside the structural route and
        # union the candidate sets. Structural items keep their high
        # scores (KB lookup = 100, entity_postings = 10) and rank above
        # BM25 candidates by default; the reranker still does the final
        # ordering.
        if self.bm25_fallback is not None:
            try:
                bm25_results = self.bm25_fallback.retrieve_with_ast_roles(
                    question_ast, top_k
                )
            except Exception as e:
                logger.warning(f"BM25 supplement failed: {e}")
                bm25_results = []
            seen = {c['id'] for c in candidates}
            for r in bm25_results:
                if r.get('id') not in seen:
                    r = dict(r)
                    r['source'] = (r.get('source') or 'bm25') + '_supplement'
                    candidates.append(r)
                    seen.add(r.get('id'))
                if len(candidates) >= top_k:
                    break
            if route == 'fallback_bm25':
                route = 'bm25_only'
            elif bm25_results:
                route += '+bm25_supp'

        logger.debug(f"Retrieved {len(candidates)} candidates via {route}")
        # Annotate the route for downstream introspection
        for c in candidates:
            c.setdefault('retriever_route', route)
        return candidates
