"""Tests for the construction-time LLM grammaticality triage (#795).

No network. These test the parts that must be right regardless of which model
answers: field resolution, verdict coercion, and provenance.

The governing rule (docs/QA_TEST_SET_PIPELINE.md):

    ML may BUILD a test set. ML may never SCORE one.

So the model's output is treated as untrusted input to a frozen artifact — it is
coerced, bounded, and stamped with provenance. It is never believed as-is.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "eval"))

from triage_language_quality_llm import (  # noqa: E402
    TRIAGE_VERSION,
    answer_span,
    build_user_prompt,
    normalize_verdict,
)


class TestAnswerSpanResolution:
    """`build_gold_review_queue.py` stores the answer SPAN in `short_answer` and
    the whole SOURCE SENTENCE in `expected_answer`. Reading the wrong one hands
    the model a paragraph as the 'answer' — the exact retrieval/extraction
    conflation that #783 exists to eliminate."""

    def test_prefers_gold_answer_span(self):
        row = {"gold_answer_span": "Zamenhof", "short_answer": "X",
               "expected_answer": "whole sentence"}
        assert answer_span(row) == "Zamenhof"

    def test_falls_back_to_short_answer_not_expected_answer(self):
        row = {"short_answer": "Musaburo Makita",
               "expected_answer": "En marto 1915 Musaburo Makita (naskita en 1893) fondis la ..."}
        assert answer_span(row) == "Musaburo Makita"

    def test_last_resort_expected_answer(self):
        assert answer_span({"expected_answer": "Zamenhof"}) == "Zamenhof"

    def test_missing_everything_is_empty_not_crash(self):
        assert answer_span({}) == ""


class TestPromptConstruction:
    def test_question_and_context_are_included(self):
        p = build_user_prompt({
            "question": "Kiu fondis Esperanton?",
            "short_answer": "Zamenhof",
            "source_sentence_text": "Zamenhof fondis Esperanton en 1887.",
        })
        assert "Kiu fondis Esperanton?" in p
        assert "Zamenhof" in p
        assert "1887" in p

    def test_works_without_context(self):
        p = build_user_prompt({"question": "Kiu fondis Esperanton?"})
        assert "Kiu fondis Esperanton?" in p


class TestVerdictNormalization:
    """A model's output is untrusted input. Coerce it; never believe it."""

    def test_wellformed_verdict_passes_through(self):
        v = normalize_verdict({"grammatical": True, "score": 0.95,
                               "issues": [], "reason": "clean"})
        assert v["llm_grammatical"] is True
        assert v["llm_score"] == 0.95

    def test_score_is_clamped_to_unit_interval(self):
        assert normalize_verdict({"score": 4.2})["llm_score"] == 1.0
        assert normalize_verdict({"score": -3})["llm_score"] == 0.0

    def test_garbage_score_does_not_crash(self):
        assert normalize_verdict({"score": "very good"})["llm_score"] == 0.0

    def test_missing_fields_default_to_the_SAFE_direction(self):
        """A verdict we could not read must default to 'ungrammatical', so a
        broken pair is sent to a human rather than silently admitted."""
        v = normalize_verdict({})
        assert v["llm_grammatical"] is False
        assert v["llm_score"] == 0.0

    def test_non_list_issues_are_coerced(self):
        assert normalize_verdict({"issues": "calque"})["llm_issues"] == ["calque"]

    def test_reason_is_bounded(self):
        v = normalize_verdict({"reason": "x" * 5000})
        assert len(v["llm_reason"]) <= 400


class TestProvenanceContract:
    def test_triage_version_exists(self):
        """A verdict is only comparable to another verdict from the same prompt.
        Bump TRIAGE_VERSION whenever the prompt or schema changes."""
        assert TRIAGE_VERSION
