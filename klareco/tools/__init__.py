"""
klareco.tools — deterministic-tool dispatch (#770 #772).

Tools are pure functions invoked when a question matches a tool's
trigger patterns. Each tool returns a dict; the orchestrator can
surface the result through the deparser.

Public API:
    math.evaluate(expression_eo) → number or string
    math.year_diff(year_a, year_b) → integer years between two years
    math.detect_and_evaluate(question_text, question_ast) → result or None
"""
from klareco.tools import math

__all__ = ['math']
