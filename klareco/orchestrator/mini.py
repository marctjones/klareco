"""
build_mini_pipeline: the same orchestrator, wired for a tiny store.

This exists so the contract/golden-trace suite (#883, #886) can test the REAL
orchestrator and REAL stages end-to-end without the production indexes or the
artifact preflight (which, correctly, refuses a 12-row store). It mirrors
build_default_pipeline's stage list and order — minus the artifact preflight
and minus the default-off modules — so a drift between this and the factory
would itself show up as a test failure.

It still runs the STAGE-dependency preflight (#884): the mini stages must
declare and satisfy their deps just like the real ones.
"""
from __future__ import annotations

from pathlib import Path

from klareco.orchestrator.dependencies import preflight_stages
from klareco.orchestrator.pipeline import Orchestrator
from klareco.orchestrator.stage import ModelRegistry
from klareco.orchestrator.stages.parse_question import ParseQuestionStage
from klareco.orchestrator.stages.math_tool import MathToolStage
from klareco.orchestrator.stages.retrieve import RetrieveStage
from klareco.orchestrator.stages.deterministic_rerank import DeterministicRerankStage
from klareco.orchestrator.stages.ast_aware_rerank import ASTAwareRerankStage
from klareco.orchestrator.stages.extract_generate import ExtractAndGenerateStage
from klareco.orchestrator.stages.format_output import FormatOutputStage
from klareco.rag.duckdb_retriever import DuckDBRetriever
from klareco.rag.extractive_answering import ExtractiveAnswerGenerator
from klareco.orchestrator.store_view import StoreView


def build_mini_pipeline(whoosh_dir: Path | str,
                        duckdb_path: Path | str,
                        top_k: int = 10) -> Orchestrator:
    """A real Orchestrator over a tiny store — the contract-suite target.

    A contract-COVERAGE pipeline (not a mirror of the shipping default):
      parse → math → retrieve → deterministic_rerank → ast_aware_rerank
      → extract → format
    It includes ast_aware_rerank so the contract suite still covers it even
    though it was DEMOTED from build_default_pipeline (#895: it hurts the number
    on rebaseline_210). A demoted stage must still honor the contract — that is
    exactly what this harness verifies.
    """
    models = ModelRegistry()
    retriever = DuckDBRetriever(whoosh_index_dir=Path(whoosh_dir),
                                duckdb_path=Path(duckdb_path))
    store = StoreView(duckdb_path)
    stages = [
        ParseQuestionStage(),
        MathToolStage(),
        RetrieveStage(retriever=retriever, models=models, top_k=top_k),
        DeterministicRerankStage(),
        ASTAwareRerankStage(store=store),
        ExtractAndGenerateStage(generator=ExtractiveAnswerGenerator()),
        FormatOutputStage(),
    ]
    preflight_stages(stages, duckdb_path=duckdb_path)
    return Orchestrator(stages=stages, models=models)
