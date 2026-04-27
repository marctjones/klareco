"""
Tests for the orchestration pipeline.

Organised into three layers:
  1. Context immutability — frozen dataclasses, MappingProxyType flags,
     numpy array locking, copy-on-write semantics.
  2. Stage interface — PipelineStage ABC, should_skip(), on_failure(),
     ContextDelta field validation.
  3. Orchestrator integration — trace accumulation, skip recording,
     abort_pipeline flag, ParseQuestionStage round-trip.
"""
import dataclasses
import types
import unittest

from klareco.orchestrator.context import (
    QueryContext, ContextDelta,
    SymbolicLayer, LatentLayer,
    StageMetrics, ParsedPassage, FactFragment, Segment, CitationRecord,
)
from klareco.orchestrator.metrics import StageTrace
from klareco.orchestrator.pipeline import Orchestrator, AnswerResult
from klareco.orchestrator.stage import PipelineStage, ModelRegistry
from klareco.orchestrator.stages.parse_question import ParseQuestionStage, _classify_from_ast


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ctx(**kwargs) -> QueryContext:
    return QueryContext(question='Kiu fondis Esperanton?', **kwargs)


def _minimal_metrics(stage_name='test') -> StageMetrics:
    return StageMetrics(
        stage_name=stage_name,
        timing_ms=1.0,
        confidence_before=0.0,
        confidence_after=0.2,
        symbolic_coverage=1.0,
    )


class EchoStage(PipelineStage):
    """Stage that echoes a fixed delta, used for testing the orchestrator loop."""
    name = 'echo'

    def __init__(self, delta: ContextDelta):
        self._delta = delta

    def run(self, ctx: QueryContext) -> ContextDelta:
        return self._delta


class SkipAlwaysStage(PipelineStage):
    name = 'skip_always'

    def should_skip(self, ctx: QueryContext) -> bool:
        return True

    def run(self, ctx: QueryContext) -> ContextDelta:
        raise AssertionError("run() must not be called when should_skip() returns True")


class AbortStage(PipelineStage):
    name = 'abort'

    def run(self, ctx: QueryContext) -> ContextDelta:
        return ContextDelta(flags={'abort_pipeline': True})


class CrashStage(PipelineStage):
    name = 'crash'

    def run(self, ctx: QueryContext) -> ContextDelta:
        raise RuntimeError("deliberate crash")

    def on_failure(self, ctx: QueryContext, exc: Exception) -> ContextDelta:
        return ContextDelta(flags={'crashed': True})


# ---------------------------------------------------------------------------
# 1. Context immutability
# ---------------------------------------------------------------------------

class TestContextImmutability(unittest.TestCase):

    def test_query_context_is_frozen(self):
        ctx = _make_ctx()
        with self.assertRaises((dataclasses.FrozenInstanceError, AttributeError)):
            ctx.confidence = 0.9  # type: ignore[misc]

    def test_symbolic_layer_is_frozen(self):
        sym = SymbolicLayer()
        with self.assertRaises((dataclasses.FrozenInstanceError, AttributeError)):
            sym.question_type = 'kiu'  # type: ignore[misc]

    def test_latent_layer_is_frozen(self):
        lat = LatentLayer()
        with self.assertRaises((dataclasses.FrozenInstanceError, AttributeError)):
            lat.question_embedding = object()  # type: ignore[misc]

    def test_flags_is_mapping_proxy(self):
        ctx = _make_ctx()
        self.assertIsInstance(ctx.flags, types.MappingProxyType)

    def test_flags_is_read_only(self):
        ctx = _make_ctx()
        with self.assertRaises(TypeError):
            ctx.flags['new_key'] = True  # type: ignore[index]

    def test_parsed_passage_is_frozen(self):
        p = ParsedPassage('id1', 'text', None, 1.0, 'doc', 'wiki')
        with self.assertRaises((dataclasses.FrozenInstanceError, AttributeError)):
            p.score = 999.0  # type: ignore[misc]

    def test_fact_fragment_is_frozen(self):
        f = FactFragment('IS-A', 'esperant', (('type', 'lingvo'),), 0.9, 'sent1')
        with self.assertRaises((dataclasses.FrozenInstanceError, AttributeError)):
            f.confidence = 0.0  # type: ignore[misc]

    def test_segment_is_frozen(self):
        s = Segment('Zamenhof kreis Esperanton.', ('1',))
        with self.assertRaises((dataclasses.FrozenInstanceError, AttributeError)):
            s.text = 'other'  # type: ignore[misc]

    def test_citation_record_is_frozen(self):
        c = CitationRecord('1', 'sent1', 'snippet', 'doc', 'wiki')
        with self.assertRaises((dataclasses.FrozenInstanceError, AttributeError)):
            c.id = '99'  # type: ignore[misc]


class TestNumpyLocking(unittest.TestCase):

    def test_numpy_arrays_locked_after_apply(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest('numpy not available')

        arr = np.zeros(8)
        self.assertTrue(arr.flags.writeable)

        ctx = _make_ctx()
        delta = ContextDelta(
            latent={'question_embedding': arr},
            metrics=_minimal_metrics(),
        )
        new_ctx = ctx.apply(delta)

        self.assertFalse(new_ctx.latent.question_embedding.flags.writeable)

    def test_tuple_arrays_locked_after_apply(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest('numpy not available')

        a1, a2 = np.ones(4), np.ones(4)
        ctx = _make_ctx()
        delta = ContextDelta(
            latent={'passage_embeddings': (a1, a2)},
            metrics=_minimal_metrics(),
        )
        new_ctx = ctx.apply(delta)
        for arr in new_ctx.latent.passage_embeddings:
            self.assertFalse(arr.flags.writeable)


class TestApplyDelta(unittest.TestCase):

    def test_apply_updates_symbolic(self):
        ctx = _make_ctx()
        ast = {'tipo': 'frazo', 'subjekto': None, 'verbo': None, 'objekto': None, 'aliaj': []}
        delta = ContextDelta(
            symbolic={'question_ast': ast, 'question_type': 'kiu'},
            metrics=_minimal_metrics(),
        )
        new_ctx = ctx.apply(delta)
        self.assertEqual(new_ctx.symbolic.question_ast, ast)
        self.assertEqual(new_ctx.symbolic.question_type, 'kiu')

    def test_apply_does_not_mutate_original(self):
        ctx = _make_ctx()
        original_type = ctx.symbolic.question_type
        delta = ContextDelta(
            symbolic={'question_type': 'what'},
            metrics=_minimal_metrics(),
        )
        new_ctx = ctx.apply(delta)
        self.assertEqual(ctx.symbolic.question_type, original_type)
        self.assertEqual(new_ctx.symbolic.question_type, 'what')

    def test_apply_merges_flags(self):
        ctx = QueryContext(
            question='test',
            flags=types.MappingProxyType({'existing': True}),
        )
        delta = ContextDelta(flags={'new_flag': 42}, metrics=_minimal_metrics())
        new_ctx = ctx.apply(delta)
        self.assertTrue(new_ctx.flag('existing'))
        self.assertEqual(new_ctx.flag('new_flag'), 42)

    def test_apply_updates_confidence(self):
        ctx = _make_ctx()
        delta = ContextDelta(
            metrics=StageMetrics('test', 1.0, 0.0, 0.35, 1.0)
        )
        new_ctx = ctx.apply(delta)
        self.assertAlmostEqual(new_ctx.confidence, 0.35)

    def test_unchanged_symbolic_shares_reference(self):
        passages = (ParsedPassage('id1', 'txt', None, 1.0, 'doc', 'wiki'),)
        sym = SymbolicLayer(passage_asts=passages)
        ctx = QueryContext(question='test', symbolic=sym)
        delta = ContextDelta(
            symbolic={'question_type': 'kiu'},
            metrics=_minimal_metrics(),
        )
        new_ctx = ctx.apply(delta)
        self.assertIs(new_ctx.symbolic.passage_asts, passages)

    def test_flag_helper(self):
        ctx = QueryContext(
            question='test',
            flags=types.MappingProxyType({'abort_pipeline': True}),
        )
        self.assertTrue(ctx.flag('abort_pipeline'))
        self.assertIsNone(ctx.flag('missing'))
        self.assertEqual(ctx.flag('missing', 'default'), 'default')


# ---------------------------------------------------------------------------
# 2. Stage interface
# ---------------------------------------------------------------------------

class TestStageInterface(unittest.TestCase):

    def test_should_skip_defaults_false(self):
        stage = EchoStage(ContextDelta())
        ctx = _make_ctx()
        self.assertFalse(stage.should_skip(ctx))

    def test_on_failure_default_reraises(self):
        stage = EchoStage(ContextDelta())
        ctx = _make_ctx()
        exc = ValueError("boom")
        with self.assertRaises(ValueError):
            stage.on_failure(ctx, exc)

    def test_model_registry_has(self):
        reg = ModelRegistry()
        self.assertFalse(reg.has('reranker'))
        reg.reranker = object()
        self.assertTrue(reg.has('reranker'))

    def test_context_delta_symbolic_dict_mutable(self):
        delta = ContextDelta()
        delta.symbolic['question_type'] = 'kiu'
        self.assertEqual(delta.symbolic['question_type'], 'kiu')


# ---------------------------------------------------------------------------
# 3. Orchestrator integration
# ---------------------------------------------------------------------------

class TestOrchestrator(unittest.TestCase):

    def test_single_stage_applies_delta(self):
        metrics = _minimal_metrics('echo')
        delta = ContextDelta(
            symbolic={'question_type': 'kiu'},
            metrics=metrics,
        )
        orch = Orchestrator(stages=[EchoStage(delta)])
        result = orch.answer("Kiu fondis Esperanton?")
        self.assertIsInstance(result, AnswerResult)
        self.assertEqual(len(result.trace), 1)

    def test_skipped_stage_recorded_in_trace(self):
        orch = Orchestrator(stages=[SkipAlwaysStage()])
        result = orch.answer("test")
        self.assertEqual(len(result.trace), 1)
        self.assertTrue(result.trace[0].skipped)

    def test_abort_flag_stops_pipeline(self):
        after_abort = EchoStage(ContextDelta(
            symbolic={'question_type': 'never_reached'},
            metrics=_minimal_metrics('after'),
        ))
        orch = Orchestrator(stages=[AbortStage(), after_abort])
        result = orch.answer("test")
        stage_names = [t.stage_name for t in result.trace]
        self.assertIn('abort', stage_names)
        self.assertNotIn('echo', stage_names)

    def test_on_failure_graceful_degradation(self):
        orch = Orchestrator(stages=[CrashStage()])
        result = orch.answer("test")
        self.assertEqual(len(result.trace), 1)
        self.assertTrue(result.trace[0].ctx_after.flag('crashed'))

    def test_trace_ctx_before_is_immutable_snapshot(self):
        metrics1 = _minimal_metrics('s1')
        metrics2 = _minimal_metrics('s2')
        s1 = EchoStage(ContextDelta(
            symbolic={'question_type': 'kiu'},
            metrics=metrics1,
        ))
        s2 = EchoStage(ContextDelta(
            symbolic={'question_type': 'kio'},
            metrics=metrics2,
        ))
        orch = Orchestrator(stages=[s1, s2])
        result = orch.answer("test")
        # ctx_before of s2 should reflect s1's output
        self.assertEqual(result.trace[1].ctx_before.symbolic.question_type, 'kiu')
        # ctx_before of s1 should be the initial context
        self.assertEqual(result.trace[0].ctx_before.symbolic.question_type, 'nekonata')

    def test_stage_trace_ctx_after_property(self):
        metrics = _minimal_metrics('echo')
        delta = ContextDelta(
            symbolic={'question_type': 'where'},
            metrics=metrics,
        )
        orch = Orchestrator(stages=[EchoStage(delta)])
        result = orch.answer("test")
        ctx_after = result.trace[0].ctx_after
        self.assertEqual(ctx_after.symbolic.question_type, 'where')

    def test_orchestrator_fills_timing_when_stage_omits_metrics(self):
        class NoMetricsStage(PipelineStage):
            name = 'no_metrics'
            def run(self, ctx):
                return ContextDelta()

        orch = Orchestrator(stages=[NoMetricsStage()])
        result = orch.answer("test")
        m = result.trace[0].metrics
        self.assertIsNotNone(m)
        self.assertGreaterEqual(m.timing_ms, 0.0)


class TestDebugValidation(unittest.TestCase):

    def test_unknown_symbolic_field_raises_in_debug(self):
        class BadStage(PipelineStage):
            name = 'bad'
            def run(self, ctx):
                return ContextDelta(symbolic={'nonexistent_field': 'x'})

        orch = Orchestrator(stages=[BadStage()], debug=True)
        with self.assertRaises(ValueError):
            orch.answer("test")

    def test_unknown_latent_field_raises_in_debug(self):
        class BadStage(PipelineStage):
            name = 'bad'
            def run(self, ctx):
                return ContextDelta(latent={'nonexistent': None})

        orch = Orchestrator(stages=[BadStage()], debug=True)
        with self.assertRaises(ValueError):
            orch.answer("test")


# ---------------------------------------------------------------------------
# 4. ParseQuestionStage (deterministic, no external index needed)
# ---------------------------------------------------------------------------

class TestParseQuestionStage(unittest.TestCase):

    def _run(self, question: str) -> QueryContext:
        stage = ParseQuestionStage()
        ctx = QueryContext(question=question)
        delta = stage.run(ctx)
        return ctx.apply(delta)

    def test_produces_question_ast(self):
        new_ctx = self._run("Kiu fondis Esperanton?")
        self.assertIsNotNone(new_ctx.symbolic.question_ast)
        self.assertEqual(new_ctx.symbolic.question_ast.get('tipo'), 'frazo')

    def test_who_question_classified_correctly(self):
        new_ctx = self._run("Kiu fondis Esperanton?")
        self.assertEqual(new_ctx.symbolic.question_type, 'kiu')

    def test_what_question_classified_correctly(self):
        new_ctx = self._run("Kio estas Esperanto?")
        self.assertEqual(new_ctx.symbolic.question_type, 'kio')

    def test_where_question_classified_correctly(self):
        new_ctx = self._run("Kie loĝas Zamenhof?")
        self.assertEqual(new_ctx.symbolic.question_type, 'kie')

    def test_when_question_classified_correctly(self):
        new_ctx = self._run("Kiam naskiĝis Zamenhof?")
        self.assertEqual(new_ctx.symbolic.question_type, 'kiam')

    def test_confidence_increases(self):
        stage = ParseQuestionStage()
        ctx = QueryContext(question="Kiu fondis Esperanton?")
        delta = stage.run(ctx)
        self.assertGreater(delta.metrics.confidence_after, 0.0)

    def test_metrics_populated(self):
        stage = ParseQuestionStage()
        ctx = QueryContext(question="Kiu fondis Esperanton?")
        delta = stage.run(ctx)
        self.assertIsNotNone(delta.metrics)
        self.assertEqual(delta.metrics.stage_name, 'parse_question')
        self.assertIn('question_type', delta.metrics.stage_specific)

    def test_stage_trace_summary_not_skipped(self):
        stage = ParseQuestionStage()
        ctx = QueryContext(question="Kiu fondis Esperanton?")
        delta = stage.run(ctx)
        delta.metrics.timing_ms = 5.2
        trace = StageTrace('parse_question', ctx, delta, delta.metrics)
        summary = trace.summary()
        self.assertIn('parse_question', summary)
        self.assertNotIn('SKIPPED', summary)


if __name__ == '__main__':
    unittest.main()
