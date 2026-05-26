"""
Conversational dialog state (#767).

Tracks entities mentioned across turns and resolves Esperanto pronouns
to specific entity radikos before retrieval. Solves the
"Kiu fondis Esperanton?" → "Kaj kiam li mortis?" follow-up case
without an LLM.

Esperanto personal pronouns we handle:
    li   — male persono (3sg.m)
    ŝi   — female persono (3sg.f)
    ĝi   — non-person (3sg.n)
    ili  — plural (3pl)
    tiu  — that one (demonstrative referring to a previously-mentioned entity)
    tio  — that thing (event/concept)

We don't track gender in entity_facts, so li/ŝi/ĝi disambiguation
uses a coarse "is this a persono?" check.

The state machine itself is just a dataclass + a handful of update
rules — no actual FSM library needed for V1, though `transitions`
becomes useful when we add multi-step intents.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Esperanto pronouns and their types.
_PERSONAL_PRONOUNS = {
    'li':  'persono',
    'ŝi':  'persono',
    'gi':  'non_person',    # diacritic-folded variant
    'ĝi':  'non_person',
    'ili': 'plural',
}
_DEMONSTRATIVES = {'tiu', 'tio', 'tiuj'}


@dataclass
class EntityRef:
    """One mention of an entity in the conversation history."""
    radiko:    str
    surface:   str
    type_hint: str   # 'persono' / 'organizajxo' / 'loko' / 'evento' / 'unknown'
    turn:      int   # which turn introduced or last referenced it


@dataclass
class DialogState:
    """Per-conversation state. Stays in memory between turns; not
    persisted to disk (V1)."""
    turn:           int = 0
    entity_stack:   list[EntityRef] = field(default_factory=list)
    intent_history: list[str] = field(default_factory=list)
    topic:          Optional[str] = None

    def push_entity(self, ref: EntityRef) -> None:
        """Push onto the stack; if already present, refresh its turn."""
        for existing in self.entity_stack:
            if existing.radiko == ref.radiko:
                existing.turn = ref.turn
                return
        self.entity_stack.append(ref)
        # Cap stack at 8 entries — anaphora doesn't reach further back
        if len(self.entity_stack) > 8:
            # Drop oldest by turn
            self.entity_stack.sort(key=lambda e: -e.turn)
            self.entity_stack = self.entity_stack[:8]


# ---------------------------------------------------------------------------
# Anaphora resolution
# ---------------------------------------------------------------------------

def _kerno(node) -> dict:
    if not isinstance(node, dict):
        return {}
    if node.get('tipo') == 'vortgrupo':
        return node.get('kerno') or {}
    return node


def _is_pronoun_node(kerno: dict) -> Optional[str]:
    """Return the pronoun's resolved type expectation, or None."""
    if not isinstance(kerno, dict):
        return None
    pv = (kerno.get('plena_vorto') or '').lower()
    radiko = (kerno.get('radiko') or '').lower()
    # Normalise diacritics for matching (the parser may emit ĝ or gx)
    pv_fold = pv.replace('ĝ', 'g').replace('ŝ', 's')
    radiko_fold = radiko.replace('ĝ', 'g').replace('ŝ', 's')
    if pv in _PERSONAL_PRONOUNS:
        return _PERSONAL_PRONOUNS[pv]
    if pv_fold in _PERSONAL_PRONOUNS:
        return _PERSONAL_PRONOUNS[pv_fold]
    if radiko in _PERSONAL_PRONOUNS:
        return _PERSONAL_PRONOUNS[radiko]
    if radiko_fold in _PERSONAL_PRONOUNS:
        return _PERSONAL_PRONOUNS[radiko_fold]
    if pv in _DEMONSTRATIVES or radiko in _DEMONSTRATIVES:
        return 'demonstrative'
    return None


def _find_compatible_entity(state: DialogState,
                             type_hint: str) -> Optional[EntityRef]:
    """Walk the entity_stack newest-to-oldest, return the first
    type-compatible match. Returns None if nothing fits."""
    # Newest first (highest turn)
    candidates = sorted(state.entity_stack, key=lambda e: -e.turn)
    if type_hint == 'persono':
        for e in candidates:
            if e.type_hint == 'persono':
                return e
    elif type_hint == 'non_person':
        for e in candidates:
            if e.type_hint in ('organizajxo', 'loko', 'evento', 'unknown'):
                return e
    elif type_hint == 'plural':
        # No tracking of plurality; return most recent
        return candidates[0] if candidates else None
    elif type_hint == 'demonstrative':
        # Most recent of any type
        return candidates[0] if candidates else None
    return None


def _build_replacement_kerno(kerno: dict, resolved: EntityRef) -> dict:
    return {
        'tipo':          'vorto',
        'radiko':        resolved.radiko,
        'vortspeco':     'propra_nomo',
        'plena_vorto':   resolved.surface,
        'kazo':          kerno.get('kazo'),
        'propranoma_kat': 'propranomo',
        '_resolved_from': kerno.get('plena_vorto'),
    }


def resolve_anaphora(state: DialogState, question_ast: dict) -> dict:
    """Return a (shallow-cloned) question_ast where pronominal subject /
    object / aliaj slots are replaced with the resolved entity radiko.

    When the question has an interrogative subject (`Kiam`, `Kie`...),
    the actual subject pronoun ends up in `aliaj` — we check there too.
    """
    if not isinstance(question_ast, dict):
        return question_ast
    new_ast = dict(question_ast)
    # Roles subjekto / objekto
    for role in ('subjekto', 'objekto'):
        node = new_ast.get(role)
        if not isinstance(node, dict):
            continue
        kerno = _kerno(node)
        ptype = _is_pronoun_node(kerno)
        if ptype is None:
            continue
        resolved = _find_compatible_entity(state, ptype)
        if resolved is None:
            continue
        new_kerno = _build_replacement_kerno(kerno, resolved)
        if node.get('tipo') == 'vortgrupo':
            new_node = dict(node)
            new_node['kerno'] = new_kerno
            new_ast[role] = new_node
        else:
            new_ast[role] = new_kerno
        logger.debug(f'anaphora: {role} pronoun {kerno.get("plena_vorto")!r} '
                     f'→ {resolved.radiko}')
    # aliaj — common when subjekto is occupied by an interrogative
    aliaj = new_ast.get('aliaj') or []
    if aliaj:
        new_aliaj = []
        for item in aliaj:
            if not isinstance(item, dict):
                new_aliaj.append(item)
                continue
            kerno = _kerno(item)
            ptype = _is_pronoun_node(kerno)
            if ptype is None:
                new_aliaj.append(item)
                continue
            resolved = _find_compatible_entity(state, ptype)
            if resolved is None:
                new_aliaj.append(item)
                continue
            new_kerno = _build_replacement_kerno(kerno, resolved)
            if item.get('tipo') == 'vortgrupo':
                new_item = dict(item)
                new_item['kerno'] = new_kerno
                new_aliaj.append(new_item)
            else:
                new_aliaj.append(new_kerno)
            logger.debug(f'anaphora: aliaj pronoun {kerno.get("plena_vorto")!r} '
                         f'→ {resolved.radiko}')
        new_ast['aliaj'] = new_aliaj
    return new_ast


# ---------------------------------------------------------------------------
# State update
# ---------------------------------------------------------------------------

def update_state(state: DialogState, question_ast: dict,
                  answer_entities: Optional[list[EntityRef]] = None
                  ) -> DialogState:
    """Advance the turn counter; push the question's named entities and
    any answer entities onto the stack.

    Mutates and returns the state."""
    state.turn += 1
    # Push named-entity subjects/objects from the question
    for role in ('subjekto', 'objekto'):
        node = question_ast.get(role)
        if not isinstance(node, dict):
            continue
        kerno = _kerno(node)
        if not isinstance(kerno, dict):
            continue
        vs = (kerno.get('vortspeco') or '').lower()
        if vs != 'propra_nomo':
            continue
        radiko = kerno.get('radiko')
        if not radiko:
            continue
        surface = kerno.get('plena_vorto') or radiko
        kat = (kerno.get('propranoma_kat') or '').lower()
        # We don't have great type signals; heuristically:
        type_hint = 'unknown'
        # Could enrich via entity_facts lookup later
        state.push_entity(EntityRef(
            radiko=radiko, surface=surface, type_hint=type_hint,
            turn=state.turn,
        ))
    # Push any explicitly-passed answer entities
    if answer_entities:
        for ref in answer_entities:
            ref.turn = state.turn
            state.push_entity(ref)
    return state
