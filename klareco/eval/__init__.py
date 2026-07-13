"""
Klareco evaluation utilities.

Shared helpers for QA-style evaluation that can be reused by local scripts
(scripts/eval/evaluate_extractive_qa.py) and remote runners (scripts/eval/modal_eval.py).
"""
from klareco.eval.qa_metrics import (
    evaluate_question,
    summarize,
    print_summary,
    aggregate_stage_timings,
    aggregate_phase_timings,
)
from klareco.eval.answer_scoring import (
    aggregate_extraction,
    exact_match,
    normalize,
    score_extraction,
    token_f1,
)

__all__ = [
    "evaluate_question",
    "summarize",
    "print_summary",
    "aggregate_stage_timings",
    "aggregate_phase_timings",
    # Extraction scoring (R17 / #783) — deterministic, no LLM judge.
    "score_extraction",
    "aggregate_extraction",
    "exact_match",
    "token_f1",
    "normalize",
]
