"""
ASTAwareRerankStage: structural-score reranking over BM25 candidates.

Wraps `klareco.rag.ast_aware_reranker.ASTAwareScorer` as a pipeline
stage. Sits between RetrieveStage and the (still-stub) RerankStage so
the AST-aware structural scoring runs as part of the default pipeline
without needing a neural reranker.

Bench (capability_candidates_v1, n=120, top-K=10):

  Reranker        R@1  R@5  R@10  MRR    Ans %
  B_phrase_query   54   91   105  0.586  54.2
  G_ast_aware      57   92   103  0.590  59.2   ← this stage
  H_hybrid         53   92   108  0.585  55.0

Adopted as the default reranker in `factory.py` after Stage 3
(`must_have_anchors`) lifted G_ast_aware past B_phrase_query on R@1,
MRR, and Ans%.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import duckdb

from klareco.orchestrator.context import (
    ContextDelta, ParsedPassage, QueryContext, StageMetrics,
)
from klareco.orchestrator.stage import PipelineStage
from klareco.rag.ast_aware_reranker import ASTAwareScorer

logger = logging.getLogger(__name__)


# Mirrors `ASTAwareRanker._COLS_*` in scripts/eval/multi_reranker_bench.py
_COLS_CORE = ('sid', 'text', 'subj_radiko', 'subj_vortspeco',
              'subj_propranoma_kat', 'subj_kazo',
              'verb_radiko', 'verb_tempo', 'verb_klaso', 'verb_negated',
              'obj_radiko', 'obj_kazo', 'aliaj_json')
_COLS_STAGE2 = ('aliaj_has_loko', 'aliaj_has_jaro', 'aliaj_has_kvant')


class ASTAwareRerankStage(PipelineStage):
    """Reorder the BM25 candidate pool by AST-aware structural score."""

    name = 'ast_aware_rerank'

    def __init__(self, duckdb_path: str = 'data/indexes/duckdb_store.db'):
        self.duckdb_path = duckdb_path
        self._conn: Optional[duckdb.DuckDBPyConnection] = None
        self._scorer: Optional[ASTAwareScorer] = None
        self._stage2_available: Optional[bool] = None

    def _ensure(self) -> None:
        if self._conn is None:
            self._conn = duckdb.connect(self.duckdb_path, read_only=True)
            self._conn.execute("SET memory_limit = '2GB'")
            self._conn.execute("SET threads = 4")
            self._scorer = ASTAwareScorer(self._conn)
            try:
                self._conn.execute(
                    "SELECT aliaj_has_loko, aliaj_has_jaro, aliaj_has_kvant "
                    "FROM sentences LIMIT 1").fetchone()
                self._stage2_available = True
            except Exception:
                self._stage2_available = False
            logger.info(
                f"ASTAwareRerankStage ready (Stage 2 columns: "
                f"{self._stage2_available})")

    def should_skip(self, ctx: QueryContext) -> bool:
        return (not ctx.symbolic.passage_asts
                or not ctx.symbolic.question_ast
                or ctx.flag('retrieval_empty'))

    def run(self, ctx: QueryContext) -> ContextDelta:
        self._ensure()
        passages = ctx.symbolic.passage_asts
        question_ast = ctx.symbolic.question_ast
        question_text = ctx.question
        t0 = time.time()

        cols = tuple(_COLS_CORE) + (_COLS_STAGE2 if self._stage2_available
                                    else ())
        col_list = ', '.join(cols)
        sids = [int(p.sentence_id) for p in passages]
        placeholders = ','.join('?' for _ in sids)
        rows = self._conn.execute(
            f"SELECT {col_list} FROM sentences "
            f"WHERE sid IN ({placeholders})",
            sids,
        ).fetchall()
        row_by_sid = {int(r[0]): dict(zip(cols, r)) for r in rows}

        bm25_scores = {int(p.sentence_id): float(p.score) for p in passages}
        ast_by_sid = {int(p.sentence_id): p.ast for p in passages}
        cand_dicts = []
        for sid in sids:
            r = row_by_sid.get(sid)
            if r is None:
                continue
            d = dict(r)
            d['ast'] = ast_by_sid.get(sid)
            cand_dicts.append(d)

        scored = self._scorer.score_batch(
            question_ast, cand_dicts, bm25_scores,
            question_text=question_text,
        )

        pp_by_sid = {int(p.sentence_id): p for p in passages}
        reranked: list[ParsedPassage] = []
        for new_score, c in scored:
            pp = pp_by_sid.get(int(c['sid']))
            if pp is None:
                continue
            reranked.append(ParsedPassage(
                sentence_id=pp.sentence_id, text=pp.text, ast=pp.ast,
                score=float(new_score),
                source_doc=pp.source_doc, source_type=pp.source_type,
            ))
        # Preserve any candidates DuckDB didn't return (shouldn't happen
        # in practice but defensive)
        if len(reranked) < len(passages):
            seen = {p.sentence_id for p in reranked}
            for p in passages:
                if p.sentence_id not in seen:
                    reranked.append(p)

        elapsed_ms = (time.time() - t0) * 1000.0
        return ContextDelta(
            symbolic={'passage_asts': tuple(reranked)},
            metrics=StageMetrics(
                stage_name=self.name,
                timing_ms=elapsed_ms,
                confidence_before=ctx.confidence,
                confidence_after=ctx.confidence + 0.1,
                symbolic_coverage=1.0,
                stage_specific={
                    'reranked':    len(reranked),
                    'stage2_cols': self._stage2_available,
                },
            ),
        )

    def on_failure(self, ctx: QueryContext, exc: Exception) -> ContextDelta:
        logger.warning(
            f"[ast_aware_rerank] failed ({exc}), continuing with prior order")
        return ContextDelta()
