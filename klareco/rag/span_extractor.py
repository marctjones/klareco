"""
Conservative deterministic answer-span extraction (#869).

The multi-sentence extractor returns whole passages (token_f1 0.0148). The old
AST cascade returned tight spans but regressed answer recall (contains_gold
0.312 -> 0.113) by replacing good passages with wrong short spans.

This extractor is CONSERVATIVE by construction: it reads the answer role off the
TOP passage AST using the interrogative's own position (CLAUDE.md rule 5), and
returns a span ONLY when it cleanly finds the matching constituent. Otherwise it
returns None and the caller keeps the full passage — so contains_gold can only
be held or improved, never traded away for a guess.

  Kiu / Kio / Kies  → the answer fills the SAME grammatical role (subjekto /
                      objekto) the interrogative occupies in the question.
  Kie               → a locative adjunct in the passage (prep 'en/ĉe/apud…' + loko).
  Kiam              → a temporal adjunct (a year / number, or 'en <jaro>').
  Kiom              → a number / quantity.

STATUS (#869, measured 2026-07-18/19): NOT wired into the default pipeline —
BLOCKED on retrieval@1, not a tuning problem. Three gate variants on the
qa_gold_v2 verbatim stratum, contains_gold regression vs fire-rate:
  - aggressive cascade  (fire ~all): token_f1 +0.034, contains -0.200
  - conservative role   (fire 26%):  token_f1 +0.019, contains -0.069
  - strict proper-noun  (fire 7%):   token_f1 +0.0055, contains -0.019
The regression asymptotes toward 0 only as the fire-rate does — it NEVER reaches
contains-neutral. Property of the problem: recall@1 is ~28%, so whenever the top
passage isn't the gold, extracting its span can only hurt; no gate fixes that
without the gold at rank-1. So this ships once retrieval@1 improves (deep-band
recall #25 / reranker #26 incl. the #877 proper-noun boost), NOT before.
"""
from __future__ import annotations

from typing import Optional

_LOC_PREPS = {'en', 'ĉe', 'apud', 'sur', 'sub', 'trans', 'ĝis', 'de', 'el'}
_SAME_ROLE_INTERROG = {'kiu', 'kio', 'kies'}


def _kerno(node) -> Optional[dict]:
    if not isinstance(node, dict):
        return None
    if node.get('tipo') == 'vortgrupo':
        return node.get('kerno')
    return node


def _span_text(node) -> str:
    """Reconstruct the surface span of a role node (vortgrupo or word)."""
    if not isinstance(node, dict):
        return ''
    if node.get('tipo') == 'vortgrupo':
        parts = []
        if node.get('artikolo'):
            parts.append(node['artikolo'])
        for pr in node.get('priskriboj') or []:
            k = _kerno(pr)
            if k and k.get('plena_vorto'):
                parts.append(k['plena_vorto'])
        k = node.get('kerno') or {}
        if k.get('plena_vorto'):
            parts.append(k['plena_vorto'])
        return ' '.join(parts).strip()
    return (node.get('plena_vorto') or '').strip()


def _find_interrogative(qast: dict):
    """Return (role, radiko) where the interrogative sits in the question."""
    for role in ('subjekto', 'objekto'):
        k = _kerno(qast.get(role))
        if k and k.get('vortspeco') == 'korelativo':
            return role, (k.get('radiko') or '').lower()
    for x in qast.get('aliaj') or []:
        k = _kerno(x)
        if k and k.get('vortspeco') == 'korelativo':
            return 'aliaj', (k.get('radiko') or '').lower()
    return None, None


def _is_number_word(k: dict) -> bool:
    if not k:
        return False
    if k.get('vortspeco') in ('numeralo', 'nombro'):
        return True
    pv = (k.get('plena_vorto') or '')
    return pv.isdigit() or any(ch.isdigit() for ch in pv)


def _locative_span(passage_ast: dict) -> Optional[str]:
    """A locative adjunct: a passage `aliaj` group headed by a place prep."""
    for x in passage_ast.get('aliaj') or []:
        if not isinstance(x, dict):
            continue
        prep = (x.get('prepozicio') or x.get('rolvorto') or '').lower()
        k = _kerno(x)
        if prep in _LOC_PREPS and k and k.get('vortspeco') in ('substantivo', 'propra_nomo'):
            span = _span_text(x)
            if span:
                return f"{prep} {span}" if not span.startswith(prep) else span
    return None


def _temporal_span(passage_ast: dict) -> Optional[str]:
    """A temporal adjunct: a year/number in the passage (subj/obj/aliaj)."""
    for x in passage_ast.get('aliaj') or []:
        k = _kerno(x)
        if _is_number_word(k):
            prep = (x.get('prepozicio') or '').lower() if isinstance(x, dict) else ''
            span = _span_text(x)
            return f"{prep} {span}".strip() if prep else span
    return None


def role_span(question_ast: Optional[dict],
              passage_ast: Optional[dict]) -> Optional[str]:
    """Extract the answer span from the top passage, or None (keep passage).

    Deterministic and conservative: returns a span only when the matching
    constituent is cleanly present; otherwise None.
    """
    if not isinstance(question_ast, dict) or not isinstance(passage_ast, dict):
        return None
    role, radiko = _find_interrogative(question_ast)
    if radiko is None:
        return None

    if radiko in _SAME_ROLE_INTERROG and role in ('subjekto', 'objekto'):
        span = _span_text(passage_ast.get(role))
        return span or None
    if radiko == 'kie':
        return _locative_span(passage_ast)
    if radiko in ('kiam', 'kiom'):
        return _temporal_span(passage_ast)
    return None
