"""
ParseQuestionStage: text → question AST + question type.

Runs the deterministic Esperanto parser once and classifies the question type
from the resulting AST.  Both results are stored in SymbolicLayer so every
downstream stage reads from the context rather than re-parsing.
"""
from __future__ import annotations

import logging

from klareco.orchestrator.context import QueryContext, ContextDelta, StageMetrics
from klareco.orchestrator.stage import PipelineStage
from klareco.parser import parse

logger = logging.getLogger(__name__)

_CORRELATIVE_TO_TYPE = {
    'KIU':  'who',
    'KIO':  'what',
    'KIE':  'where',
    'KIAM': 'when',
    'KIEL': 'how',
    'KIAL': 'why',
    'KIOM': 'how_many',
    'CXU':  'boolean',
    'ĈU':   'boolean',
}


class ParseQuestionStage(PipelineStage):
    name = 'parse_question'

    def run(self, ctx: QueryContext) -> ContextDelta:
        question = ctx.question.strip()
        ast = parse(question)

        question_type = _classify_from_ast(ast)
        stats = ast.get('parse_statistics', {})
        # success_rate only exists for multi-sentence corpus parses; for a single
        # question we treat a non-empty AST with a verb or subject as full coverage.
        parse_rate = stats.get('success_rate', None)
        if parse_rate is not None:
            symbolic_coverage = 1.0 if parse_rate >= 0.5 else 0.4
        else:
            has_content = bool(ast.get('verbo') or ast.get('subjekto') or ast.get('objekto'))
            symbolic_coverage = 1.0 if has_content else 0.3

        confidence_after = 0.1 if ast else 0.0

        return ContextDelta(
            symbolic={
                'question_ast': ast,
                'question_type': question_type,
            },
            metrics=StageMetrics(
                stage_name=self.name,
                timing_ms=0.0,
                confidence_before=ctx.confidence,
                confidence_after=confidence_after,
                symbolic_coverage=symbolic_coverage,
                stage_specific={
                    'parse_rate': parse_rate,
                    'question_type': question_type,
                },
            ),
        )


def _classify_from_ast(ast: dict) -> str:
    """
    Detect question type from the ki-correlative in the parsed AST.

    Checks subjekto first (Kiu fondis…?), then objekto (…kiun mi vidas?),
    then aliaj (for adverbial ki-words like kie/kiam).
    """
    for slot in ('subjekto', 'objekto'):
        node = ast.get(slot)
        if not node:
            continue
        kerno = node.get('kerno') if node.get('tipo') == 'vortgrupo' else node
        if kerno and kerno.get('vortspeco') == 'korelativo':
            radiko = kerno.get('radiko', '').upper()
            if radiko in _CORRELATIVE_TO_TYPE:
                return _CORRELATIVE_TO_TYPE[radiko]

    for alia in ast.get('aliaj', []):
        if isinstance(alia, dict) and alia.get('vortspeco') == 'korelativo':
            radiko = alia.get('radiko', '').upper()
            if radiko in _CORRELATIVE_TO_TYPE:
                return _CORRELATIVE_TO_TYPE[radiko]

    return 'unknown'
