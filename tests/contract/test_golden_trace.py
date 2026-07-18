"""
Golden traces (#886): end-to-end orchestration snapshots over the mini store.

We assert on the STRUCTURE of the thought's evolution — which stages ran, which
fields each enriched, coarse counts, key invariants — rather than exact
Esperanto surface text (which can shift with parser/generator tweaks without a
regression). The human-readable decoded trace is ALSO written to a committed
artifact (regenerate with KLARECO_UPDATE_GOLDEN=1) so drift is eyeball-reviewable
in a diff.

This is the test that exercises the orchestrator as a whole: parse → math →
retrieve → rerank → extract → format, threading one immutable thought.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from klareco.orchestrator.decoder import decode_result
from tests.contract.conftest import CANONICAL_QUESTIONS

GOLDEN_DIR = Path(__file__).parent / "golden"


def _structural_trace(result) -> list[dict]:
    """A stable, text-light structural summary of the thought's evolution."""
    steps = []
    for entry in result.trace:
        if entry.skipped:
            steps.append({"stage": entry.stage_name, "skipped": True})
            continue
        d = entry.delta
        sym = {}
        for k, v in (d.symbolic or {}).items():
            sym[k] = len(v) if isinstance(v, tuple) else "set"
        steps.append({
            "stage": entry.stage_name,
            "symbolic": sym,
            "flags": sorted(d.flags or {}),
        })
    return steps


def test_pipeline_answers_every_canonical_question(mini_pipeline):
    for q in CANONICAL_QUESTIONS:
        result = mini_pipeline.answer(q)
        assert result.trace, f"no trace for {q!r}"
        # every question yields a non-empty final thought OR a clean skip chain
        final = result.trace[-1].ctx_after
        assert final.symbolic.final_text or result.text is not None


def test_who_created_esperanto_retrieves_and_extracts(mini_pipeline):
    result = mini_pipeline.answer("Kiu kreis Esperanton?")
    final = result.trace[-1].ctx_after
    assert final.symbolic.question_type == "kiu"
    assert len(final.symbolic.passage_asts) > 0, "retrieval produced nothing"
    # the Zamenhof sentence should be retrievable in a 12-doc store
    texts = " ".join(p.text for p in final.symbolic.passage_asts)
    assert "Zamenhof" in texts


def test_math_question_short_circuits_before_retrieval(mini_pipeline):
    result = mini_pipeline.answer("Kiom estas du plus tri?")
    names = [e.stage_name for e in result.trace]
    # math_tool ran; retrieve should have been skipped by the short-circuit flag
    math_idx = names.index("math_tool")
    retrieve_entry = next(e for e in result.trace if e.stage_name == "retrieve")
    assert retrieve_entry.skipped, "retrieve should skip after a math short-circuit"
    assert "5" in result.text


def test_stage_order_is_stable(mini_pipeline):
    result = mini_pipeline.answer("Kio estas Esperanto?")
    order = [e.stage_name for e in result.trace]
    assert order == [
        "parse_question", "math_tool", "retrieve",
        "deterministic_rerank", "extract_generate", "format_output",
    ], order


@pytest.mark.parametrize("question", CANONICAL_QUESTIONS,
                         ids=lambda q: q[:20])
def test_golden_structural_trace(mini_pipeline, question):
    """Snapshot the structural trace; regenerate with KLARECO_UPDATE_GOLDEN=1."""
    result = mini_pipeline.answer(question)
    structural = _structural_trace(result)

    GOLDEN_DIR.mkdir(exist_ok=True)
    slug = "".join(c if c.isalnum() else "_" for c in question)[:32]
    struct_path = GOLDEN_DIR / f"{slug}.json"
    human_path = GOLDEN_DIR / f"{slug}.txt"

    if os.environ.get("KLARECO_UPDATE_GOLDEN") == "1":
        struct_path.write_text(json.dumps(structural, indent=2, ensure_ascii=False))
        human_path.write_text(decode_result(result))
        pytest.skip(f"golden updated: {struct_path.name}")

    assert struct_path.exists(), (
        f"no golden for {question!r} — regenerate with KLARECO_UPDATE_GOLDEN=1")
    expected = json.loads(struct_path.read_text())
    assert structural == expected, (
        f"structural trace drift for {question!r}.\n"
        f"expected {expected}\n     got {structural}\n"
        f"If this change is intended, regenerate with KLARECO_UPDATE_GOLDEN=1")
