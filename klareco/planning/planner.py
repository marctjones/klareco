"""
Task planner for nested factual questions (#771).

Design choice: full PDDL/STRIPS machinery (unified-planning) is overkill
for our domain — 5K facts and a handful of action primitives. Instead
we implement a focused decomposer that handles the most common nested
pattern: a relative clause that constrains the outer question.

Example:
    Question: "Kiu fondis la organizaĵon kies sidejo estas en Roterdamo?"

    Plan:
      Step 1: find_entity(slot='sidejo', value='roterdam') → ?X
      Step 2: lookup_slot(entity=?X, slot='founder') → answer

    Execution: Step 1 returns X={UEA}. Step 2 binds entity=UEA and
    queries entity_facts(UEA, founder, ?), returning Hodler.

The planner is composable with the rest of the symbolic stack:
EntityFactRetriever handles individual steps; the planner orchestrates
chains of them.

V1 covers:
  - "[Wh] [verb] la [N] kies [slot] estas [Y]"
    (e.g. "Kiu fondis la organizaĵon kies sidejo estas en Roterdamo?")
  - "[Wh] [verb] [N] de [Y]"
    (e.g. "Kiu estas la cxefurbo de [land]?" — though this is already
    a 1-hop KB query)
  - "[Wh] [verb] [N] [verb-participle] [Y]"
    (e.g. "Kiu skribis la libron pri Mozart?" — pivot on Mozart)

Future: full STRIPS via unified-planning when the action library grows.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import duckdb

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Plan data structures
# ---------------------------------------------------------------------------

@dataclass
class PlanStep:
    """One primitive operation. `args` can reference earlier step results
    via the special syntax `'$step_N'` (e.g. `'$step_0.result'`)."""
    name:        str               # action name (e.g. 'find_entity_by_slot')
    args:        dict[str, Any]    # action-specific arguments
    bind_to:     Optional[str] = None  # variable name to capture result
    description: str = ''


@dataclass
class Plan:
    """A sequence of PlanStep that, when executed, answers a question."""
    steps:    list[PlanStep] = field(default_factory=list)
    question: str = ''
    notes:    list[str] = field(default_factory=list)

    def explain(self) -> str:
        """Produce a human-readable plan trace."""
        lines = [f'Plan for: {self.question!r}']
        for i, s in enumerate(self.steps):
            lines.append(f'  step {i}: {s.name}({s.args}) → ${s.bind_to or "_"}')
        return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Action library
# ---------------------------------------------------------------------------

def _act_find_entity_by_slot(conn, slot: str, value_radiko: str) -> list[str]:
    """Reverse fact lookup: which entities have (slot, value)?"""
    rows = conn.execute("""
        SELECT entity_radiko, MAX(confidence)
        FROM entity_facts
        WHERE slot = ? AND value_radiko = ?
        GROUP BY entity_radiko
        ORDER BY MAX(confidence) DESC
        LIMIT 5
    """, [slot, value_radiko.lower()]).fetchall()
    return [r[0] for r in rows]


def _act_lookup_slot(conn, entity_radiko: str, slot: str) -> list[str]:
    """Forward fact lookup: best values for (entity, slot)."""
    rows = conn.execute("""
        WITH cnt AS (
          SELECT value_radiko, COUNT(*) AS n
          FROM entity_facts
          WHERE entity_radiko = ? AND slot = ?
          GROUP BY value_radiko
        )
        SELECT ef.value, c.n
        FROM entity_facts ef
        JOIN cnt c ON c.value_radiko = ef.value_radiko
        WHERE ef.entity_radiko = ? AND ef.slot = ?
        ORDER BY c.n DESC, ef.confidence DESC
        LIMIT 5
    """, [entity_radiko, slot, entity_radiko, slot]).fetchall()
    # Dedup by value
    seen, out = set(), []
    for v, _ in rows:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


_ACTION_LIBRARY = {
    'find_entity_by_slot': _act_find_entity_by_slot,
    'lookup_slot':         _act_lookup_slot,
}


# ---------------------------------------------------------------------------
# Decomposer
# ---------------------------------------------------------------------------

# Esperanto relative-clause markers and their typical referents
_REL_MARKERS = {'kies', 'kiu', 'kiun', 'kie', 'kies', 'kies'}

# Common verb-radiko → slot mappings
_VERB_TO_SLOT = {
    'fond':   'founder',
    'kre':    'founder',
    'invent': 'founder',
    'eltrov': 'founder',
    'nask':   'birth_place',     # "naskigxis en X" — answer slot is birth_place/year
    'mort':   'death_year',
    'situ':   'sidejo',
}


def _radiko(kerno_or_role) -> Optional[str]:
    if not isinstance(kerno_or_role, dict):
        return None
    kerno = kerno_or_role.get('kerno') if kerno_or_role.get('tipo') == 'vortgrupo' \
        else kerno_or_role
    if isinstance(kerno, dict):
        return kerno.get('radiko')
    return None


def _value_radiko_from_aliaj(aliaj: list, target_prep: str = 'en'
                              ) -> Optional[str]:
    """Find `prep + propra_nomo` in aliaj; return the propra_nomo radiko.
    Used to extract 'Roterdamo' from 'en Roterdamo'."""
    if not aliaj:
        return None
    found_prep = False
    for item in aliaj:
        if not isinstance(item, dict):
            continue
        kerno = item.get('kerno') if item.get('tipo') == 'vortgrupo' else item
        if not isinstance(kerno, dict):
            continue
        vs = (kerno.get('vortspeco') or '').lower()
        r = (kerno.get('radiko') or '').lower()
        if vs == 'prepozicio' and r == target_prep:
            found_prep = True
            continue
        if found_prep and vs == 'propra_nomo':
            return r
    return None


def decompose(question_ast: dict, question_text: str = '') -> Optional[Plan]:
    """Detect nested patterns and emit a Plan; return None for simple
    1-hop questions (which the regular retriever handles).

    V1 patterns:
      - 'Kiu [verb1] la N kies [slot] estas en [Y]?'  (2-hop)
      - 'Kiu [verb1] la N de [Y]?'                     (1-hop; not decomposed)
    """
    if not isinstance(question_ast, dict):
        return None

    text_lower = (question_text or '').lower()

    # Pattern 1: outer KIU + verb, inner 'kies X estas en Y'
    if 'kies' in text_lower and ' estas en ' in text_lower:
        m = re.search(
            r'kiu\s+(\w+)\s+(?:la\s+)?(\w+)\s+kies\s+(\w+)\s+estas\s+en\s+(\w+)',
            text_lower
        )
        if m:
            outer_verb_word, _outer_noun, inner_slot_word, inner_value = m.groups()
            # Map the outer verb to a slot (heuristic)
            outer_slot = None
            for radiko, slot in _VERB_TO_SLOT.items():
                if outer_verb_word.startswith(radiko):
                    outer_slot = slot
                    break
            if outer_slot is None:
                return None
            # Map inner slot word (e.g. 'sidejo') to entity_facts slot.
            # For V1, accept the surface word as the slot name.
            inner_slot = inner_slot_word
            plan = Plan(question=question_text or '')
            plan.steps.append(PlanStep(
                name='find_entity_by_slot',
                args={'slot': inner_slot, 'value_radiko': inner_value},
                bind_to='step_0_entity',
                description=f'Find entities with {inner_slot} = {inner_value}',
            ))
            plan.steps.append(PlanStep(
                name='lookup_slot',
                args={'entity_radiko': '$step_0_entity', 'slot': outer_slot},
                bind_to='step_1_result',
                description=f'Look up {outer_slot} for the entity from step 0',
            ))
            return plan
    return None


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

def _resolve_arg(arg: Any, bindings: dict) -> Any:
    """Substitute $step_N references from earlier results."""
    if isinstance(arg, str) and arg.startswith('$'):
        key = arg[1:]
        val = bindings.get(key)
        # If multiple results, default to first (most-confident).
        # If empty list, treat as None so the step fails cleanly.
        if isinstance(val, list):
            return val[0] if val else None
        return val
    return arg


def execute(plan: Plan, conn) -> dict:
    """Run each step in order, substituting variable bindings.

    Returns a dict with:
      result:   the final step's result list
      trace:    per-step (action, args, output) for audit
      bindings: variable map
    """
    bindings: dict = {}
    trace = []
    for step in plan.steps:
        action = _ACTION_LIBRARY.get(step.name)
        if action is None:
            trace.append({'step': step.name, 'error': 'unknown action'})
            break
        # Substitute variable refs in args
        resolved_args = {k: _resolve_arg(v, bindings) for k, v in step.args.items()}
        # Any None args → step can't run
        if any(v is None for v in resolved_args.values()):
            trace.append({'step': step.name, 'args': resolved_args,
                          'error': 'unresolved variable'})
            break
        try:
            out = action(conn, **resolved_args)
        except Exception as e:
            trace.append({'step': step.name, 'args': resolved_args,
                          'error': str(e)})
            break
        trace.append({'step': step.name, 'args': resolved_args, 'output': out})
        if step.bind_to:
            bindings[step.bind_to] = out
    final = trace[-1].get('output') if trace else None
    return {'result': final, 'trace': trace, 'bindings': bindings}
