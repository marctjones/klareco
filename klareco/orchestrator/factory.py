"""
build_default_pipeline: construct the standard RAG orchestrator.

All expensive objects (retriever, generator, index connections) are
instantiated here, once, at startup.  The returned Orchestrator is safe
to call repeatedly from any thread as long as the underlying retrievers
are thread-safe (WhooshRetriever's Whoosh searchers are).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from klareco.orchestrator.pipeline import Orchestrator
from klareco.orchestrator.stage import ModelRegistry
from klareco.orchestrator.stages.parse_question import ParseQuestionStage
from klareco.orchestrator.stages.retrieve import RetrieveStage
from klareco.orchestrator.stages.deterministic_rerank import DeterministicRerankStage
from klareco.orchestrator.stages.ast_aware_rerank import ASTAwareRerankStage
from klareco.orchestrator.stages.rerank import RerankStage
from klareco.orchestrator.stages.extract_generate import ExtractAndGenerateStage
from klareco.orchestrator.stages.format_output import FormatOutputStage
from klareco.orchestrator.stages.dialog import DialogStage
from klareco.orchestrator.stages.math_tool import MathToolStage
from klareco.orchestrator.stages.planner import PlannerStage
from klareco.orchestrator.stages.biography_format import BiographyFormatStage
from klareco.rag.extractive_answering import ExtractiveAnswerGenerator
from klareco.rag.duckdb_retriever import DuckDBRetriever


def build_default_pipeline(
    whoosh_index_dir: Path | str,
    duckdb_path: Path | str = 'data/indexes/duckdb_store.db',
    top_k: int = 20,
    models: Optional[ModelRegistry] = None,
    debug: bool = False,
    kuzu_db_path: Path | str | None = None,  # deprecated alias, ignored
    enable_dialog: bool = False,
    enable_math_tool: bool = True,
    enable_planner: bool = True,
    enable_biography: bool = True,
) -> Orchestrator:
    """
    Build the standard Klareco RAG pipeline.

    Stages in order:
      1. ParseQuestion         — deterministic; always runs
      2. Retrieve              — BM25 + AST-role matching via WhooshRetriever
      3. DeterministicRerank   — boosts passages whose AST matches the
                                 question type's expected answer shape
                                 (propra_nomo for WHO/WHERE, numero for
                                 WHEN/HOW_MANY). No-op for other types.
      4. Rerank                — stub today; activates when models.reranker is set
      5. ExtractAndGenerate    — fact extraction + discourse-planned answer
      6. FormatOutput          — assemble final_text with citation list

    Parameters
    ----------
    whoosh_index_dir : path to the Whoosh FTS index directory
    kuzu_db_path     : path to the Kuzu graph database
    top_k            : number of passages to retrieve
    models           : optional pre-loaded ModelRegistry (neural models)
    debug            : enable delta field validation on every stage call
    """
    whoosh_index_dir = Path(whoosh_index_dir)
    duckdb_path = Path(duckdb_path)

    if models is None:
        models = ModelRegistry()

    # Kuzu retired 2026-05; DuckDB store (shredded cols + ast_json blob)
    # is the backend. kuzu_db_path kept only as an ignored alias so
    # older callers don't break during the migration.
    retriever = DuckDBRetriever(
        whoosh_index_dir=whoosh_index_dir,
        duckdb_path=duckdb_path,
    )
    generator = ExtractiveAnswerGenerator()

    # Build the stage list. Order matters:
    #   parse → dialog (resolve pronouns) → math/planner (short-circuit)
    #   → retrieve → reranks → extract → biography format → format_output
    # Each new stage is opt-in via factory flags; existing callers see
    # the same pipeline as before with no surprises (math/planner/
    # biography default-on because they only fire on matching inputs).
    stages: list = [ParseQuestionStage()]
    if enable_dialog:
        # DialogStage is OFF by default because it holds per-conversation
        # state; callers should construct one pipeline per conversation.
        stages.append(DialogStage())
    if enable_math_tool:
        # MathToolStage: short-circuits when the question is a
        # math expression. No-op on non-math questions.
        stages.append(MathToolStage())
    if enable_planner:
        # PlannerStage: decomposes nested questions. No-op on simple ones.
        stages.append(PlannerStage(duckdb_path=str(duckdb_path)))
    stages.extend([
        RetrieveStage(retriever=retriever, models=models, top_k=top_k),
        DeterministicRerankStage(),
        # AST-aware structural reranker (#741 Stage 3). Beats
        # B_phrase_query on R@1, R@5, MRR, and answer accuracy on
        # capability_candidates_v1. Drops in between
        # DeterministicRerank and the (still-stub) neural RerankStage.
        ASTAwareRerankStage(duckdb_path=str(duckdb_path)),
        RerankStage(models=models),
        ExtractAndGenerateStage(generator=generator),
    ])
    if enable_biography:
        # BiographyFormatStage: for "diru pri X" / "kio estas X?", replace
        # one-sentence extractor output with a multi-sentence paragraph
        # from klareco.generation. No-op when the question doesn't match.
        stages.append(BiographyFormatStage())
    stages.append(FormatOutputStage())

    return Orchestrator(stages=stages, models=models, debug=debug)
