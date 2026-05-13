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
from klareco.rag.extractive_answering import ExtractiveAnswerGenerator
from klareco.rag.whoosh_retriever import WhooshRetriever


def build_default_pipeline(
    whoosh_index_dir: Path | str,
    kuzu_db_path: Path | str,
    top_k: int = 20,
    models: Optional[ModelRegistry] = None,
    debug: bool = False,
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
    kuzu_db_path = Path(kuzu_db_path)

    if models is None:
        models = ModelRegistry()

    retriever = WhooshRetriever(
        whoosh_index_dir=whoosh_index_dir,
        kuzu_db_path=kuzu_db_path,
    )
    generator = ExtractiveAnswerGenerator()

    stages = [
        ParseQuestionStage(),
        RetrieveStage(retriever=retriever, models=models, top_k=top_k),
        DeterministicRerankStage(),
        RerankStage(models=models),
        ExtractAndGenerateStage(generator=generator),
        FormatOutputStage(),
    ]

    return Orchestrator(stages=stages, models=models, debug=debug)
