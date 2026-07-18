"""
Stage conformance (#883): the contract every stage must honor, parameterized
over EVERY stage in the real pipeline. A new stage inherits this whole suite
for free the moment it is added to the mini pipeline — that is the point.

The four invariants (DESIGN.md → "The orchestration contract"):
  1. Immutability   — run() must not mutate the context it was given.
  2. Delta discipline — enrichments land only in known SymbolicLayer /
                        LatentLayer fields (no side channels).
  3. Decodability   — every produced enrichment renders through the thought
                        decoder without raising (the decoder is the oracle).
  4. Attribution    — metrics carry the stage name; nothing is anonymous.

These run against the REAL stages over a tiny real store (see conftest), so
they are fast and CI-safe while still exercising true stage behavior.
"""
from __future__ import annotations

import dataclasses

import pytest

from klareco.orchestrator.context import QueryContext, ContextDelta
from klareco.orchestrator.decoder import decode_context, decode_delta
from klareco.orchestrator.pipeline import _SYMBOLIC_FIELDS, _LATENT_FIELDS
from tests.contract.conftest import CANONICAL_QUESTIONS


def _run_capturing_traces(pipeline, question):
    """Run one question; return [(stage, ctx_before, delta), ...] for real
    (non-skipped) stages, plus the final context."""
    result = pipeline.answer(question)
    steps = []
    for entry in result.trace:
        if not entry.skipped and entry.delta is not None:
            steps.append((entry.stage_name, entry.ctx_before, entry.delta))
    final_ctx = result.trace[-1].ctx_after if result.trace else None
    return steps, final_ctx


# One (question, stage_name, ctx_before, delta) case per real stage execution,
# collected across the canonical questions. Parameterization = "every stage".
def _all_stage_executions(pipeline):
    cases = []
    for q in CANONICAL_QUESTIONS:
        steps, _ = _run_capturing_traces(pipeline, q)
        for stage_name, ctx_before, delta in steps:
            cases.append(pytest.param(stage_name, ctx_before, delta,
                                      id=f"{stage_name}::{q[:18]}"))
    return cases


@pytest.fixture(scope="module")
def stage_executions(mini_pipeline):
    return _all_stage_executions(mini_pipeline)


# ---- collect once, then assert per-invariant ------------------------------

def test_at_least_every_core_stage_ran(stage_executions):
    """Sanity: the mini pipeline actually exercised the core stages."""
    ran = {name for name, _, _ in [(c.values[0], None, None)
                                   for c in stage_executions]}
    assert {'parse_question', 'retrieve', 'extract_generate',
            'format_output'} <= ran, f"only saw {ran}"


def test_context_is_frozen():
    """The thought is immutable by construction — you cannot reassign a field."""
    ctx = QueryContext(question="Kiu?")
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.question = "changed"          # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.confidence = 0.9              # type: ignore[misc]


def test_stages_do_not_mutate_the_input_thought(mini_pipeline):
    """apply(delta) returns a NEW context; the pre-stage snapshot is unchanged.

    Copy-on-write: unchanged layers keep their identity; changed ones are new
    objects. Either way the stage never wrote through the context it was given.
    """
    for q in CANONICAL_QUESTIONS:
        for stage_name, ctx_before, delta in _run_capturing_traces(
                mini_pipeline, q)[0]:
            before_sym = ctx_before.symbolic
            before_flags = dict(ctx_before.flags)
            after = ctx_before.apply(delta)
            # The stage's delta produced a distinct post-context...
            assert after is not ctx_before, f"[{stage_name}] returned same obj"
            # ...without mutating the snapshot the stage was handed.
            assert ctx_before.symbolic is before_sym, (
                f"[{stage_name}] mutated ctx_before.symbolic identity")
            assert dict(ctx_before.flags) == before_flags, (
                f"[{stage_name}] mutated ctx_before.flags")


def test_delta_discipline(mini_pipeline):
    """Every delta touches only known SymbolicLayer / LatentLayer fields."""
    for q in CANONICAL_QUESTIONS:
        for stage_name, _cb, delta in _run_capturing_traces(mini_pipeline, q)[0]:
            bad_sym = set(delta.symbolic) - _SYMBOLIC_FIELDS
            bad_lat = set(delta.latent) - _LATENT_FIELDS
            assert not bad_sym, f"[{stage_name}] unknown symbolic fields: {bad_sym}"
            assert not bad_lat, f"[{stage_name}] unknown latent fields: {bad_lat}"


def test_decodability_of_every_delta(mini_pipeline):
    """The decoder must render every stage's delta without raising, and never
    emit a `nedekodebla` marker on legitimate pipeline output."""
    for q in CANONICAL_QUESTIONS:
        result = mini_pipeline.answer(q)
        for entry in result.trace:
            rendered = decode_delta(entry)         # must not raise
            assert 'nedekodebla' not in rendered, (
                f"[{entry.stage_name}] produced an undecodable enrichment "
                f"on {q!r}:\n{rendered}")


def test_decodability_of_every_final_thought(mini_pipeline):
    for q in CANONICAL_QUESTIONS:
        result = mini_pipeline.answer(q)
        rendered = decode_context(result.trace[-1].ctx_after)
        assert 'nedekodebla' not in rendered, rendered


def test_attribution_metrics_name_their_stage(mini_pipeline):
    """Every non-skipped stage's metrics carry its own name (no anonymity)."""
    for q in CANONICAL_QUESTIONS:
        result = mini_pipeline.answer(q)
        for entry in result.trace:
            if entry.skipped or entry.metrics is None:
                continue
            assert entry.metrics.stage_name == entry.stage_name


def test_no_silent_stage_failures_on_the_happy_path(mini_pipeline):
    """The whole reason this suite exists: a stage_failed stamp on a canonical
    question means a stage is silently broken (this is how #895 shows up)."""
    for q in CANONICAL_QUESTIONS:
        result = mini_pipeline.answer(q)
        final = result.trace[-1].ctx_after
        failed = [k for k in final.flags if k.startswith('stage_failed:')]
        assert not failed, (
            f"silent stage failure(s) on {q!r}: "
            f"{[(k, final.flag(k)) for k in failed]}")
