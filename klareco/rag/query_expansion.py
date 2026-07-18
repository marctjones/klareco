"""
Deterministic query-term variant expansion (#855) — recall for the deep band.

VERSION: v1.0
STAGE: Retrieval
DEPENDENCIES: stdlib only

The #854 taxonomy measured 82 deep-band misses caused by surface-variant blocks.
This module generates ADDITIVE query variants with pure string/morphology rules —
no lists, no lookups (the translated-title alias TABLE is #865, separate):

  compound split    'Enigma-ĉifro'  -> + 'enigma', 'ĉifro'
  dotted acronym    'T.A.R.D.I.S'   -> + 'TARDIS'
  ordinal digits    '40-a'          -> + 'kvardeka'
  de-inflection     'literon'       -> + 'litero'   (strip -n, -j, -jn)
  a/o final swap    'kanada'        -> + 'kanado'   (adjective<->noun derivation)

Variants are ADDED to an OR-query — they can only widen the candidate net. The
merge-gate measurement (recall on the 82, no-regression on trivial/rerankable
controls) decides whether this ships in the production path.
"""

from __future__ import annotations

import re
from typing import List

_UNITS = ['', 'unu', 'du', 'tri', 'kvar', 'kvin', 'ses', 'sep', 'ok', 'naŭ']
_ACRO_RE = re.compile(r'^(?:[A-Za-zĈĜĤĴŜŬĉĝĥĵŝŭ]\.){2,}[A-Za-zĈĜĤĴŜŬĉĝĥĵŝŭ]?\.?$')
_ORD_RE = re.compile(r'^(\d+)-?aj?n?$')   # 40-a, 3-an, 19-aj


def _num_to_eo(n: int) -> str:
    """Cardinal number word, 1..9999 (deterministic composition)."""
    if n == 0:
        return 'nul'
    parts = []
    th, n = divmod(n, 1000)
    if th:
        parts.append(('' if th == 1 else _UNITS[th]) + 'mil')
    h, n = divmod(n, 100)
    if h:
        parts.append(('' if h == 1 else _UNITS[h]) + 'cent')
    t, u = divmod(n, 10)
    if t:
        parts.append(('' if t == 1 else _UNITS[t]) + 'dek')
    if u:
        parts.append(_UNITS[u])
    return ' '.join(parts) if len(parts) > 1 and parts[0].endswith('mil') \
        else ''.join(parts)


def variants(term: str) -> List[str]:
    """Additive surface variants for one query term (may be empty)."""
    out: List[str] = []
    t = term.strip()

    # dotted acronym -> collapsed form
    if _ACRO_RE.match(t):
        out.append(t.replace('.', ''))

    # digit ordinal -> Esperanto ordinal word ('40-a' -> 'kvardeka')
    m = _ORD_RE.match(t)
    if m:
        try:
            word = _num_to_eo(int(m.group(1))).replace(' ', '')
            if word:
                out.append(word + 'a')
        except ValueError:
            pass

    # hyphenated compound -> components (each a real token the index may hold)
    if '-' in t and not _ORD_RE.match(t):
        out.extend(p for p in t.split('-') if len(p) > 2)

    # de-inflection: strip accusative/plural endings (surface forms the index holds)
    low = t.lower()
    for suf, repl in (('ojn', 'o'), ('ajn', 'a'), ('oj', 'o'), ('aj', 'a'),
                      ('on', 'o'), ('an', 'a')):
        if low.endswith(suf) and len(low) > len(suf) + 2:
            out.append(t[: -len(suf)] + repl)
            break

    # adjective<->noun final-vowel swap (kanada <-> kanado) for longer stems
    if len(low) > 4:
        if low.endswith('a'):
            out.append(t[:-1] + 'o')
        elif low.endswith('o'):
            out.append(t[:-1] + 'a')

    # dedupe, drop the original and trivial shorts
    seen, res = {low}, []
    for v in out:
        vl = v.lower()
        if vl not in seen and len(vl) > 2:
            seen.add(vl)
            res.append(v)
    return res


def expand_terms(terms: List[str], cap_ratio: float = 2.0) -> List[str]:
    """Original terms + variants, capped at cap_ratio x the original count so the
    OR-query cannot blow up (variants are appended in term order until the cap)."""
    out = list(terms)
    budget = max(0, int(len(terms) * cap_ratio) - len(terms))
    for t in terms:
        if budget <= 0:
            break
        for v in variants(t):
            if budget <= 0:
                break
            if v.lower() not in (x.lower() for x in out):
                out.append(v)
                budget -= 1
    return out


# ── Gate-driven refinement (first measurement failed its controls) ────────────
# Blanket expansion recovered +6.1% target recall but regressed control MRR
# (trivial 1.000->0.864): variants on COMMON words dilute BM25 and competing
# sentences outrank gold. Two fixes, measured separately:
#   1. expand only RARE-LOOKING terms (hyphen/acronym/ordinal, or capitalized in
#      the raw question — likely proper nouns), never common vocabulary;
#   2. DOWN-WEIGHT variants in the query (whoosh boost syntax `term^w`) so they
#      widen the net without outranking the original terms.

def selective_variants(term: str, raw_question: str = '') -> List[str]:
    """Variants only for rare-looking terms; [] for common vocabulary."""
    rare = ('-' in term or _ACRO_RE.match(term) or _ORD_RE.match(term))
    if not rare and raw_question:
        # capitalized mid-question occurrence = likely proper noun
        import re as _re
        for m in _re.finditer(_re.escape(term), raw_question, _re.IGNORECASE):
            s = m.start()
            if s > 0 and raw_question[m.start()].isupper() \
               and not raw_question[max(0, s - 2):s].strip().endswith(('.', '?', '!')):
                rare = True
                break
    return variants(term) if rare else []


_LEAD_FUNC = frozenset(
    'en la kiu kiun kio kion kie kiam kiom kial kiel kiuj ĉu de al post antaŭ '
    'dum per pri sur sub el ke kaj'.split())
_QUOTE_RE = re.compile(r'[«"\'“]([^»"\'”]{3,60})[»"\'”]')
_CAPRUN_RE = re.compile(
    r'\b([A-ZĈĜĤĴŜŬ][\w\-]*(?:\s+(?:[A-Z0-9ĈĜĤĴŜŬ][\w\-]*|de|la))*)')


def question_anchors(question: str) -> List[str]:
    """Deterministic anchor spans (#870): quoted strings + capitalized runs, with
    leading function words stripped even sentence-initially ('En Minecraft' ->
    'Minecraft'). These identify the entity/title the question is ABOUT."""
    out = [m.group(1) for m in _QUOTE_RE.finditer(question)]
    for m in _CAPRUN_RE.finditer(question):
        toks = m.group(1).split()
        while toks and toks[0].lower() in _LEAD_FUNC:
            toks.pop(0)
        if toks:
            out.append(' '.join(toks))
    seen, res = set(), []
    for a in out:
        a = a.strip()
        if len(a) > 2 and a.lower() not in seen:
            seen.add(a.lower())
            res.append(a)
    return res


def build_expanded_query(terms: List[str], raw_question: str = '',
                         weight: float = 0.3, cap: int = 8) -> str:
    """OR-query string: original terms at full weight + selected variants
    down-weighted (`v^weight`). Feed to whoosh QueryParser."""
    parts = list(terms)
    added = 0
    seen = {t.lower() for t in terms}
    for t in terms:
        if added >= cap:
            break
        for v in selective_variants(t, raw_question):
            if added >= cap:
                break
            if v.lower() not in seen:
                seen.add(v.lower())
                parts.append(f'{v}^{weight}')
                added += 1
    return ' OR '.join(parts)
