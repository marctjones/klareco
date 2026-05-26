"""
klareco.dialog — multi-turn conversational state (#767).

Public API:
    DialogState         — per-conversation state
    resolve_anaphora(state, question_ast) → new question_ast
        Replace pronouns with their resolved entity radiko.
    update_state(state, question_ast, answer_facts) → new state
        After answering, push the answer's entities onto the stack.
"""
from klareco.dialog.state import (
    DialogState, EntityRef,
    resolve_anaphora, update_state,
)

__all__ = ['DialogState', 'EntityRef', 'resolve_anaphora', 'update_state']
