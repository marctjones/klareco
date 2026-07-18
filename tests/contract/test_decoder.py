"""
Contract tests (#882): the universal thought decoder.

The decoder is a TEST ORACLE (DESIGN.md → orchestration contract, rule 4):
every symbolic enrichment must render; a value it cannot decode must surface
as a loud `nedekodebla` marker — never as an exception, never silently.
"""
from __future__ import annotations

import pytest

from klareco.orchestrator.context import (
    CitationRecord, ContextDelta, FactFragment, ParsedPassage, QueryContext,
    Segment,
)
from klareco.orchestrator.decoder import (
    decode_context, decode_delta, decode_result, decode_trace,
)
from klareco.orchestrator.metrics import StageTrace


def _rich_context() -> QueryContext:
    """A hand-built thought exercising every SymbolicLayer leaf type."""
    from klareco.parser import parse
    ctx = QueryContext(question='Kiu fondis Esperanton?')
    delta = ContextDelta(
        symbolic={
            'question_ast': parse('Kiu fondis Esperanton?'),
            'question_type': 'kiu',
            'passage_asts': (
                ParsedPassage(sentence_id='42', text='Zamenhof kreis Esperanton.',
                              ast=None, score=0.91, source_doc='vikipedio',
                              source_type='wiki'),
            ),
            'fact_fragments': (
                FactFragment(relation='kre', entity='zamenhof',
                             arguments=(('objekto', 'esperanto'),),
                             confidence=0.9, source_passage_id='42'),
            ),
            'answer_segments': (
                Segment(text='Zamenhof fondis Esperanton.', citation_ids=('1',)),
            ),
            'citations': (
                CitationRecord(id='1', sentence_id='42',
                               snippet='Zamenhof kreis Esperanton.',
                               doc_title='Esperanto', doc_source='vikipedio'),
            ),
            'final_text': 'Zamenhof fondis Esperanton. [1]',
        },
        flags={'qe_weight': 0.3},
    )
    return ctx.apply(delta)


def test_full_thought_renders_every_leaf_type():
    out = decode_context(_rich_context())
    assert 'PENSO' in out
    assert 'Kiu fondis Esperanton?' in out           # the question
    assert 'tipo=kiu' in out
    assert '[sid 42]' in out                          # passage
    assert 'zamenhof —kre→' in out                    # fact triple, Esperanto-glossed
    assert 'objekto=esperanto' in out
    assert '[1] sid 42' in out                        # citation
    assert 'Zamenhof fondis Esperanton. [1]' in out   # final text
    assert '[regulo]' in out                          # provenance default
    assert 'latenta tavolo' in out
    assert 'nedekodebla' not in out                   # nothing failed to render


def test_decoder_never_raises_on_garbage():
    """Undecodable values surface as loud markers, not exceptions."""
    ctx = QueryContext(question='Kio?').apply(ContextDelta(symbolic={
        'question_ast': 12345,                         # not an AST at all
        'passage_asts': (
            ParsedPassage(sentence_id='1', text='', ast=('ne', 'dict'),
                          score=0.0, source_doc='', source_type=''),
        ),
        'fact_fragments': (
            FactFragment(relation='x', entity='y', arguments=None,   # type: ignore[arg-type]
                         confidence=0.0, source_passage_id=''),
        ),
    }))
    out = decode_context(ctx)                          # must not raise
    assert 'nedekodebla' in out                        # and must SAY it failed


def test_recovered_stage_failure_is_prominent():
    ctx = QueryContext(question='Kiu?').apply(ContextDelta(
        flags={'stage_failed:planner': 'BinderException: column "slot"'}))
    out = decode_context(ctx)
    assert '⚠ FALO planner' in out
    assert 'BinderException' in out


def test_decode_trace_renders_skipped_and_run_stages():
    ctx0 = QueryContext(question='Kiu?')
    delta = ContextDelta(symbolic={'question_type': 'kiu'})
    trace = [
        StageTrace(stage_name='math_tool', ctx_before=ctx0,
                   delta=None, metrics=None, skipped=True),
        StageTrace(stage_name='parse_question', ctx_before=ctx0,
                   delta=delta, metrics=None),
    ]
    out = decode_trace(trace)
    assert '[math_tool] (preterlasita)' in out
    assert '[parse_question]' in out
    assert 'question_type' in out


def test_decode_result_end_to_end_shape():
    """decode_result stitches per-stage evolution + final thought."""
    ctx0 = QueryContext(question='Kiu fondis Esperanton?')
    delta = ContextDelta(symbolic={'final_text': 'Zamenhof.'})

    class _R:                                          # minimal AnswerResult shape
        question = 'Kiu fondis Esperanton?'
        text = 'Zamenhof.'
        trace = [StageTrace(stage_name='s', ctx_before=ctx0, delta=delta,
                            metrics=None)]

    out = decode_result(_R())
    assert 'EVOLUO DE LA PENSO' in out
    assert 'PENSO' in out
    assert 'Zamenhof.' in out
