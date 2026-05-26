"""
klareco.planning — task decomposition for nested questions (#771).

For multi-step questions like "Kiu fondis la organizajxon kies sidejo
estas en Roterdamo?" — find the org via its location, then look up
its founder — we want explicit plan steps rather than ad-hoc routing.

Public API:
    decompose(question_ast, conn) → Plan
        Returns a sequence of PlanSteps the orchestrator can execute.
    execute(plan, conn) → result
        Runs the steps sequentially, threading variable bindings.
"""
from klareco.planning.planner import (
    PlanStep, Plan, decompose, execute,
)

__all__ = ['PlanStep', 'Plan', 'decompose', 'execute']
