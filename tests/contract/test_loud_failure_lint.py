"""
Contract lint (#884): failure must be loud.

Two enforcement mechanisms:

1. STATIC — an `on_failure` override that can swallow an exception (i.e. its
   body contains a `return`) must be explicitly waived below, with an issue
   reference. A new stage that silently swallows fails this test. This is the
   pattern that hid #881 for weeks: three stages caught BinderException on
   every question and returned an empty delta.

2. RUNTIME — even a WAIVED graceful degradation may not be invisible: the
   orchestrator stamps a `stage_failed:<name>` flag on every recovered
   failure, so it shows in the trace, the thought decoder, and downstream.
"""
from __future__ import annotations

import ast as pyast
from pathlib import Path

import pytest

STAGES_DIR = Path(__file__).resolve().parents[2] / 'klareco' / 'orchestrator' / 'stages'

# Stages allowed to degrade gracefully. Every entry MUST carry an issue
# reference; adding one is a reviewed decision, not a default.
FALLBACK_WAIVERS = {
    'retrieve.py':         'sets retrieval_empty flag; per-stage review in #812',
    'extract_generate.py': 'sets no_answer flag; per-stage review in #812',
    'rerank.py':           'neural-stub fallback to BM25 order by design; formalized by #893',
    'ast_aware_rerank.py': 'reranker fallback to prior order by design; formalized by #893',
    'math_tool.py':        'sympy hiccup must not kill QA; per-stage review in #812',
    'dialog.py':           'default-off; defects tracked in #890/#891',
}


def _on_failure_swallows(tree: pyast.AST) -> bool:
    """True if any on_failure def in the module contains a return statement."""
    for node in pyast.walk(tree):
        if isinstance(node, pyast.FunctionDef) and node.name == 'on_failure':
            for sub in pyast.walk(node):
                if isinstance(sub, pyast.Return):
                    return True
    return False


def _stage_files():
    return sorted(p for p in STAGES_DIR.glob('*.py') if p.name != '__init__.py')


def test_stages_dir_exists_and_nonempty():
    files = _stage_files()
    assert files, f'no stage modules found under {STAGES_DIR}'


@pytest.mark.parametrize('path', _stage_files(), ids=lambda p: p.name)
def test_no_unwaived_swallowing_on_failure(path):
    tree = pyast.parse(path.read_text())
    if _on_failure_swallows(tree):
        assert path.name in FALLBACK_WAIVERS, (
            f'{path.name} overrides on_failure with a swallowing fallback '
            f'but is NOT waived. A silently-degrading dependency is a bug '
            f'(#884). Either re-raise, or add a waiver WITH an issue ref.')


def test_every_waiver_carries_an_issue_reference():
    for name, reason in FALLBACK_WAIVERS.items():
        assert '#' in reason, f'waiver for {name} has no issue reference'


def test_waivers_list_matches_reality():
    """A waiver for a stage that no longer swallows is stale — remove it."""
    actual = {p.name for p in _stage_files()
              if _on_failure_swallows(pyast.parse(p.read_text()))}
    stale = set(FALLBACK_WAIVERS) - actual
    assert not stale, f'stale waivers (stage no longer swallows): {stale}'
    # And the two #881 offenders must NOT have crept back in:
    assert 'planner.py' not in actual
    assert 'biography_format.py' not in actual


def test_recovered_failure_is_stamped_on_the_thought():
    """RUNTIME half: orchestrator stamps stage_failed:<name> on recovery."""
    from klareco.orchestrator.context import QueryContext, ContextDelta
    from klareco.orchestrator.pipeline import Orchestrator
    from klareco.orchestrator.stage import PipelineStage

    class Boom(PipelineStage):
        name = 'boom'

        def run(self, ctx):
            raise ValueError('kaboom')

        def on_failure(self, ctx, exc):
            return ContextDelta()          # graceful … but may not be silent

    result = Orchestrator(stages=[Boom()]).answer('Kiu?')
    final_ctx = result.trace[-1].ctx_after
    stamped = final_ctx.flag('stage_failed:boom')
    assert stamped and 'ValueError' in stamped
