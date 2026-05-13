"""DeterministicRerankStage: question-type-aware boost on retrieved passages.

Inserts between Retrieve and the (currently stub) neural Rerank. Walks
each passage's AST once and multiplies its score by a fixed boost when
its features match the question type's expected answer shape:

  WHO (kiu)      subject's kerno is propra_nomo, OR `aliaj` contains
                 'de' followed by a propra_nomo (passive agent slot)
  WHERE (kie)    `aliaj` contains a location preposition next to a
                 propra_nomo
  WHEN (kiam)    any `aliaj` token has vortspeco='numero'
  HOW_MANY       any `aliaj` token has vortspeco='numero'
  (kiom)

For unmapped question types (kio/WHAT, kiel/HOW, kial/WHY, nekonata)
this stage is a no-op — the BM25 + importance ranking from Retrieve
stays as-is.

Why deterministic boost instead of a new retrieval path: the new paths
we tried added candidates and diluted the top-k. This stage *reorders*
existing candidates without growing the set, so it can't dilute. It
composes cleanly with a future neural reranker (run before, neural
after).
"""
from __future__ import annotations

from dataclasses import replace
from typing import Optional

from klareco.orchestrator.context import (
    ContextDelta, ParsedPassage, QueryContext, StageMetrics,
)
from klareco.orchestrator.stage import PipelineStage

# Multiplicative boost when a passage matches the expected answer shape.
# 1.5 is small enough that a high-BM25 mismatch can still beat a
# low-BM25 match, but large enough to surface answer-shaped sentences
# that BM25 would otherwise tie or rank below noise.
BOOST_FACTOR = 1.5

LOCATION_PREPOSITIONS = frozenset({'en', 'al', 'ĉe', 'sur', 'ĝis', 'sub', 'super', 'inter'})


def _kerno(node: Optional[dict]) -> Optional[dict]:
    """Return the head Vorto of a subjekto/objekto slot, whether it's
    a Vortgrupo (use kerno) or already a flat Vorto."""
    if not isinstance(node, dict):
        return None
    if node.get('tipo') == 'vortgrupo':
        return node.get('kerno')
    return node


def _all_vortoj_in_aliaj(ast: dict):
    """Yield each Vorto in `aliaj`, including kerno of any Vortgrupo."""
    for item in ast.get('aliaj') or ():
        if not isinstance(item, dict):
            continue
        if item.get('tipo') == 'vorto':
            yield item
        elif item.get('tipo') == 'vortgrupo':
            kerno = item.get('kerno')
            if isinstance(kerno, dict):
                yield kerno
            for d in item.get('priskriboj') or ():
                if isinstance(d, dict) and d.get('tipo') == 'vorto':
                    yield d


def _has_propra_subject_or_de_agent(ast: dict) -> bool:
    """WHO match: subject is a proper noun, OR 'de' + propra_nomo in aliaj."""
    subj = _kerno(ast.get('subjekto'))
    if isinstance(subj, dict) and subj.get('vortspeco') == 'propra_nomo':
        return True
    aliaj = list(_all_vortoj_in_aliaj(ast))
    has_de = any(v.get('radiko') == 'de' for v in aliaj)
    has_propra = any(v.get('vortspeco') == 'propra_nomo' for v in aliaj)
    return has_de and has_propra


def _has_propra_after_location_prep(ast: dict) -> bool:
    """WHERE match: a location preposition immediately followed by a
    proper noun (or with one between them) in `aliaj`. We use
    'within 2 positions of the prep' since `aliaj` is a flat list and
    the prepositional phrase may include articles/adjectives between
    the prep and its noun."""
    aliaj = list(_all_vortoj_in_aliaj(ast))
    for i, v in enumerate(aliaj):
        if v.get('radiko') in LOCATION_PREPOSITIONS:
            for j in range(i + 1, min(i + 4, len(aliaj))):
                if aliaj[j].get('vortspeco') == 'propra_nomo':
                    return True
    return False


def _has_numero(ast: dict) -> bool:
    """WHEN/HOW_MANY match: any token in `aliaj` is a numero."""
    return any(v.get('vortspeco') == 'numero'
               for v in _all_vortoj_in_aliaj(ast))


# Map ctx.symbolic.question_type → predicate(ast) -> bool. Question
# types not in this map are no-ops (no boost applied).
_BOOST_RULES = {
    'kiu':  _has_propra_subject_or_de_agent,
    'kie':  _has_propra_after_location_prep,
    'kiam': _has_numero,
    'kiom': _has_numero,
}


class DeterministicRerankStage(PipelineStage):
    """Reorder passages by AST-feature match against the question type."""

    name = 'deterministic_rerank'

    def should_skip(self, ctx: QueryContext) -> bool:
        return (
            ctx.symbolic.question_type not in _BOOST_RULES
            or not ctx.symbolic.passage_asts
            or ctx.flag('retrieval_empty')
        )

    def run(self, ctx: QueryContext) -> ContextDelta:
        rule = _BOOST_RULES[ctx.symbolic.question_type]

        boosted = 0
        rescored = []
        for p in ctx.symbolic.passage_asts:
            if p.ast and rule(p.ast):
                rescored.append(replace(p, score=p.score * BOOST_FACTOR))
                boosted += 1
            else:
                rescored.append(p)

        rescored.sort(key=lambda p: p.score, reverse=True)

        return ContextDelta(
            symbolic={'passage_asts': tuple(rescored)},
            metrics=StageMetrics(
                stage_name=self.name,
                timing_ms=0.0,
                confidence_before=ctx.confidence,
                confidence_after=ctx.confidence,
                symbolic_coverage=1.0,
                stage_specific={
                    'boosted': boosted,
                    'total':   len(rescored),
                    'rule':    ctx.symbolic.question_type,
                },
            ),
        )
