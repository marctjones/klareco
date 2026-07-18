"""
Proper-noun / rare-term boost reranker (#877).

The common_terms_competition deep-band bucket fails because the discriminating
PROPER NOUN is buried under sentences sharing the common terms. The #877 probe
showed re-ranking by proper-noun overlap lifts 5/35 in-pool golds into top-5
(from 0) — the first positive deterministic lever on that bucket, and the mirror
of the oracle-negative role-match reranker (#866).

Design: an ADDITIVE boost on top of the existing score, so ordering is unchanged
when the question has no proper noun or a candidate contains none — it can only
promote candidates that share the question's discriminating entity, never
demote on noise. `alpha` is the per-match weight (tuned on rebaseline_210).

STATUS (#877, measured 2026-07-18): a POSITIVE, no-regression lever, but NOT yet
wired into the default pipeline. On rebaseline_210 (n=210, alpha=4): recall@5
93->100 (+7), MRR 0.3619->0.3702 (+0.0083), 19 better vs 9 worse, r@20 unchanged
(never loses recall); per band r@5 rerankable +6, trivial +1, deep +0. BUT the
paired-bootstrap MRR CI is [-0.0020,+0.0194] — it just includes 0, so it does
NOT clear the strict aggregate merge gate (same call as the alias bridge #865:
real but under-powered on the frozen benchmark). Research-track building block;
confirm on the higher-powered gold set (#847/#848) before shipping default-on.
"""
from __future__ import annotations

import re
from typing import Optional

_TOK = re.compile(r"[\wĉĝĥĵŝŭĈĜĤĴŜŬ-]+")
_INTERROG = {'kiu', 'kio', 'kie', 'kiam', 'kiel', 'kial', 'kies', 'kiom', 'kia'}


def _kerno(node) -> Optional[dict]:
    if not isinstance(node, dict):
        return None
    return node.get('kerno') if node.get('tipo') == 'vortgrupo' else node


def proper_noun_terms(question_ast: Optional[dict], question_text: str = '') -> set:
    """Discriminating proper-noun surface terms from the question."""
    out = set()
    if isinstance(question_ast, dict):
        for role in ('subjekto', 'objekto'):
            k = _kerno(question_ast.get(role))
            if k and k.get('vortspeco') == 'propra_nomo' and k.get('plena_vorto'):
                out.add(k['plena_vorto'].lower().strip('.,;:?!'))
        for x in question_ast.get('aliaj') or []:
            k = _kerno(x)
            if k and k.get('vortspeco') == 'propra_nomo' and k.get('plena_vorto'):
                out.add(k['plena_vorto'].lower().strip('.,;:?!'))
    # fallback: capitalized non-interrogative content tokens
    for t in _TOK.findall(question_text or ''):
        if t[:1].isupper() and t.lower() not in _INTERROG:
            out.add(t.lower())
    return {t for t in out if len(t) > 2}


def boost_scores(question_ast: Optional[dict], question_text: str,
                 candidates: list, *, alpha: float = 4.0) -> list:
    """Return candidates reordered by (score + alpha * proper_noun_matches).

    `candidates`: list of objects with `.score` and `.text` (e.g. ParsedPassage).
    Stable: ties keep original order. No-op when the question has no proper noun.
    """
    pn = proper_noun_terms(question_ast, question_text)
    if not pn:
        return list(candidates)

    def matches(text: str) -> int:
        tl = (text or '').lower()
        return sum(1 for p in pn if p in tl)

    scored = [
        (c.score + alpha * matches(getattr(c, 'text', '')), i, c)
        for i, c in enumerate(candidates)
    ]
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [c for _, _, c in scored]
