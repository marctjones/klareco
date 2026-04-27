"""
ExtractAndGenerateStage: passages → facts + answer segments + citations.

Wraps ExtractiveAnswerGenerator, which internally runs:
  extract facts → score importance → plan discourse → generate text.

The stage decomposes the Answer object into the orchestrator's typed
structures (FactFragment, Segment, CitationRecord) so every downstream
stage has access to structured thought rather than opaque strings.

Known limitation: ExtractiveAnswerGenerator re-parses the query internally.
The parsed AST is already in ctx.symbolic.question_ast.  Eliminating this
double-parse requires a refactor of ExtractiveAnswerGenerator to accept a
pre-parsed AST — tracked as tech debt.
"""
from __future__ import annotations

import logging
from typing import Optional

from klareco.orchestrator.context import (
    QueryContext, ContextDelta, StageMetrics,
    FactFragment, Segment, CitationRecord,
)
from klareco.orchestrator.stage import PipelineStage
from klareco.rag.extractive_answering import ExtractiveAnswerGenerator
from klareco.rag.importance_scorer import QuestionType

logger = logging.getLogger(__name__)

_TYPE_MAP: dict[str, QuestionType] = {
    'who':      QuestionType.WHO,
    'what':     QuestionType.WHAT,
    'where':    QuestionType.WHERE,
    'when':     QuestionType.WHEN,
    'how':      QuestionType.HOW,
    'why':      QuestionType.WHY,
}
# All unmapped types (how_many, boolean, unknown) fall back to OTHER


class ExtractAndGenerateStage(PipelineStage):
    name = 'extract_generate'

    def __init__(self, generator: Optional[ExtractiveAnswerGenerator] = None):
        self.generator = generator or ExtractiveAnswerGenerator()

    def should_skip(self, ctx: QueryContext) -> bool:
        return bool(ctx.flag('retrieval_empty')) or not ctx.symbolic.passage_asts

    def run(self, ctx: QueryContext) -> ContextDelta:
        passages = ctx.symbolic.passage_asts
        question_type = _TYPE_MAP.get(ctx.symbolic.question_type, QuestionType.OTHER)
        query_entity = _extract_query_entity(ctx.symbolic.question_ast)

        sentences = [
            {
                'id':        p.sentence_id,
                'text':      p.text,
                'ast':       p.ast,
                'score':     p.score,
                'doc_title': p.source_doc,
                'source':    p.source_type,
            }
            for p in passages
        ]

        answer = self.generator.generate(
            sentences=sentences,
            query=ctx.question,
            question_type=question_type,
            query_entity=query_entity,
        )

        fact_fragments = _build_fact_fragments(answer.facts_used)
        citations, segments = _build_citations_and_segments(answer)

        facts_found = len(fact_fragments)
        confidence_gain = min(0.4, facts_found * 0.1)

        return ContextDelta(
            symbolic={
                'fact_fragments':  fact_fragments,
                'answer_segments': segments,
                'citations':       citations,
            },
            metrics=StageMetrics(
                stage_name=self.name,
                timing_ms=0.0,
                confidence_before=ctx.confidence,
                confidence_after=ctx.confidence + confidence_gain,
                symbolic_coverage=1.0,
                stage_specific={
                    'facts_extracted': answer.num_facts_extracted,
                    'facts_selected':  answer.num_facts_selected,
                    'citations':       len(citations),
                    'query_entity':    query_entity or '',
                },
            ),
        )

    def on_failure(self, ctx: QueryContext, exc: Exception) -> ContextDelta:
        logger.error(f"[extract_generate] failed: {exc}")
        return ContextDelta(flags={'no_answer': True})


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def _extract_query_entity(ast: Optional[dict]) -> Optional[str]:
    """Extract the main entity root being asked about from the question AST."""
    if not ast:
        return None
    obj = ast.get('objekto')
    if obj and isinstance(obj, dict):
        kerno = obj.get('kerno') if obj.get('tipo') == 'vortgrupo' else obj
        if kerno:
            root = kerno.get('radiko')
            if root:
                return root.lower()
    subj = ast.get('subjekto')
    if subj and isinstance(subj, dict):
        kerno = subj.get('kerno') if subj.get('tipo') == 'vortgrupo' else subj
        if kerno and kerno.get('vortspeco') not in ('korelativo', 'pronomo'):
            root = kerno.get('radiko')
            if root:
                return root.lower()
    return None


def _build_fact_fragments(facts_used) -> tuple:
    if not facts_used:
        return ()
    fragments = []
    for f in facts_used:
        relation = f.relation.value if hasattr(f.relation, 'value') else str(f.relation)
        arguments = tuple(sorted(
            (k, str(v)) for k, v in (f.arguments or {}).items()
        ))
        fragments.append(FactFragment(
            relation=relation,
            entity=f.entity or '',
            arguments=arguments,
            confidence=getattr(f, 'confidence', 0.5),
            source_passage_id=f.sentence_id or '',
            ast_node=f.source_ast,
        ))
    return tuple(fragments)


def _build_citations_and_segments(answer) -> tuple[tuple, tuple]:
    """Convert Answer → (CitationRecord tuple, Segment tuple)."""
    citation_records = []
    citation_id_set: set[str] = set()

    for c in (answer.citations or []):
        cid = str(c.id)
        citation_records.append(CitationRecord(
            id=cid,
            sentence_id=c.sentence_id or '',
            snippet=c.sentence_text or '',
            doc_title=c.doc_title or '',
            doc_source=c.doc_source or '',
        ))
        citation_id_set.add(cid)

    all_ids = tuple(c.id for c in citation_records)
    text = answer.text or ''

    # Represent the full answer as a single Segment carrying all citation ids.
    # Sentence-level decomposition is deferred to a future discourse stage.
    segments = (Segment(text=text, citation_ids=all_ids),) if text else ()

    return tuple(citation_records), segments
