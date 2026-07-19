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
from klareco.orchestrator.stages.rerank import RerankStage
from klareco.orchestrator.stages.extract_generate import ExtractAndGenerateStage
from klareco.orchestrator.stages.format_output import FormatOutputStage
from klareco.orchestrator.stages.dialog import DialogStage
from klareco.orchestrator.stages.math_tool import MathToolStage
from klareco.orchestrator.stages.planner import PlannerStage
from klareco.orchestrator.stages.biography_format import BiographyFormatStage
from klareco.orchestrator.dependencies import preflight_stages
from klareco.orchestrator.store_view import StoreView
from klareco.rag.extractive_answering import ExtractiveAnswerGenerator
from klareco.rag.duckdb_retriever import DuckDBRetriever
from klareco.preflight import preflight


def build_default_pipeline(
    whoosh_index_dir: Path | str,
    duckdb_path: Path | str = 'data/indexes/duckdb_store.db',
    top_k: int = 20,
    models: Optional[ModelRegistry] = None,
    debug: bool = False,
    kuzu_db_path: Path | str | None = None,  # deprecated alias, ignored
    enable_dialog: bool = False,
    enable_math_tool: bool = True,
    # Default-OFF (#888): planner and biography are silently dead against the
    # live entity_facts schema (#881 — BinderException swallowed per question,
    # probed 2026-07-18). Re-enabling is EARNED: contract suite green + a
    # number that moved. Enabling them explicitly triggers a LOUD
    # stage-dependency preflight failure until #881 lands.
    enable_planner: bool = False,
    enable_biography: bool = False,
    allow_degraded: Optional[bool] = None,
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

    # Fail loudly if the artifacts we depend on are missing or empty (#779).
    # A silently-degrading dependency is a bug: the June 2026 migration cost a
    # month of invisible quality loss because every loader logged a warning and
    # carried on. You may run degraded — you may not do so by accident.
    preflight(duckdb_path=duckdb_path,
              whoosh_index_dir=whoosh_index_dir,
              allow_degraded=allow_degraded)

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
    # #885: ONE StoreView, injected into every stage that reads the store.
    store = StoreView(duckdb_path)

    # Build the stage list. Order matters:
    #   parse → dialog (resolve pronouns) → math/planner (short-circuit)
    #   → retrieve → reranks → extract → biography format → format_output
    # Optional modules are opt-in via factory flags and run default-OFF
    # until they pass the contract suite and carry a number (#888;
    # DESIGN.md → "The orchestration contract"). Math is on: 5/5 smoke, live.
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
        stages.append(PlannerStage(store=store))
    stages.extend([
        RetrieveStage(retriever=retriever, models=models, top_k=top_k),
        DeterministicRerankStage(),
        # AST-aware structural reranker (#741) — DEMOTED from the default
        # pipeline 2026-07-18 (#895). Its old "beats B_phrase_query on
        # capability_candidates_v1" claim was on a likely-circular set; once the
        # dead verb_klaso SELECT was fixed so it actually RAN, a rebaseline_210
        # A/B showed it HURTS: MRR 0.3619 -> 0.3446 (R@1 -4, R@5 -3, R@20 ±0 —
        # it only reorders, and reorders worse). "Ship what wins, drop what
        # hurts" (#26). The stage still passes the contract suite and is
        # available for Reranker-v2 work; it just does not ship on by default.
        RerankStage(models=models),
        ExtractAndGenerateStage(generator=generator),
    ])
    if enable_biography:
        # BiographyFormatStage: for "diru pri X" / "kio estas X?", replace
        # one-sentence extractor output with a multi-sentence paragraph
        # from klareco.generation. No-op when the question doesn't match.
        stages.append(BiographyFormatStage())
    stages.append(FormatOutputStage())

    # Stage-level dependency preflight (#884): each stage declares the exact
    # tables/columns/artifacts its run() queries (PipelineStage.REQUIRES);
    # construction fails LOUDLY — itemized, with issue numbers — if the live
    # environment can't satisfy them. This is what turns an #881-class schema
    # drift from a silent per-question no-op into a build-time crash.
    preflight_stages(stages, duckdb_path=duckdb_path,
                     allow_degraded=allow_degraded)

    return Orchestrator(stages=stages, models=models, debug=debug)
