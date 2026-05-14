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

__all__ = [
    "evaluate_question",
    "summarize",
    "print_summary",
    "aggregate_stage_timings",
    "aggregate_phase_timings",
]
