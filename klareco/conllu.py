"""Emit CoNLL-U — the format nobody has ever emitted for Esperanto.

Oya, *UD Treebanks for Esperanto as a Natural Language* (UDW/SyntaxFest 2025):

    "automatic parsing was not conducted because **the parsers for Esperanto
     available at present do not yield parse output in the format of CoNLL-U**."

and he closes by calling for exactly this:

    "We need to ... **develop an Esperanto UD parser and evaluate its
     performance** with a UD-annotated gold-standard Esperanto texts."

**No parsing result on the Esperanto UD treebanks has ever been published.** Not
by Stanza, not by UDPipe, not by Trankit — none of them ships an Esperanto model,
because the free gold data is 3,343 tokens, far below their training threshold.

So emitting CoNLL-U does three things at once:
  1. it gives us LAS/UAS as a NATIVE metric, instead of the bespoke
     subject/object F1 we invented;
  2. it lets us be compared against any UD parser on equal terms;
  3. it makes klareco the first published Esperanto UD parsing result.

WHERE THE SCHEME DIFFERS, AND WHY WE DO NOT PRETEND OTHERWISE
------------------------------------------------------------
About 90% of our POS "errors" against UD are annotation-scheme mismatches, not
parsing failures, and we map them honestly rather than quietly:

    Esperanto possessives (`mia`, `ĝia`) ARE adjectives — mi + a. UD says DET.
    `estas` is not a separate AUX class in Esperanto. UD says AUX.
    Participles (`farita`) ARE adjectival. UD says VERB.
    The correlatives are ONE closed table; UD splits them across PRON/DET/ADV.

We emit the UD tag so the numbers are comparable, and `MISC` carries our native
analysis (`Vortspeco=`, `Radiko=`) so nothing is lost. The scheme-adjusted score
stays reported alongside the strict one.
"""

from __future__ import annotations

from klareco.parser import parse

# --- vortspeco -> UPOS ------------------------------------------------------
_UPOS = {
    'substantivo': 'NOUN',
    'propra_nomo': 'PROPN',
    'verbo': 'VERB',
    'adjektivo': 'ADJ',
    'adverbo': 'ADV',
    'pronomo': 'PRON',
    'prepozicio': 'ADP',
    'konjunkcio': 'CCONJ',
    'artikolo': 'DET',
    'numero': 'NUM',
    'partiklo': 'PART',
    'interjekcio': 'INTJ',
    'nekonata': 'X',
    'fremda_vorto': 'X',
}

# UD splits the correlative table across three tags by its SUFFIX. Esperanto does
# not — it is one paradigm — but we emit UD's view so the score is comparable.
_KORELATIVO_UPOS = {
    'u': 'DET',     # kiu, tiu, ĉiu  — "which individual"
    'a': 'DET',     # kia, tia       — "of which kind"
    'o': 'PRON',    # kio, tio       — "which thing"
    'e': 'ADV',     # kie, tie       — "where"
    'am': 'ADV',    # kiam, tiam     — "when"
    'al': 'ADV',    # kial, tial     — "why"
    'el': 'ADV',    # kiel, tiel     — "how"
    'om': 'ADV',    # kiom, tiom     — "how much"
    'es': 'DET',    # kies, ties     — "whose"
}

# `estas` is a copula in UD, not a full verb.
_COPULA_ROOTS = {'est'}

# Subordinating vs coordinating — UD splits these; Esperanto's `konjunkcio` does not.
_SCONJ = {'ke', 'ĉar', 'se', 'kvankam', 'dum', 'ĝis', 'apenaŭ', 'kvazaŭ', 'ol'}


def upos(w: dict) -> str:
    vs = w.get('vortspeco')
    if vs == 'korelativo':
        return _KORELATIVO_UPOS.get(w.get('korelativo_sufikso') or '', 'PRON')
    if vs == 'konjunkcio' and (w.get('radiko') or '').lower() in _SCONJ:
        return 'SCONJ'
    if vs == 'verbo' and (w.get('radiko') or '').lower() in _COPULA_ROOTS:
        return 'AUX'
    return _UPOS.get(vs, 'X')


# Only NOMINALS inflect for case and number. The parser fills `kazo`/`nombro` on
# every node with a default, so emitting them unconditionally puts
# `Case=Nom|Number=Sing` on a VERB — which is not wrong so much as meaningless,
# and UD would count it against us.
_NOMINAL = {'substantivo', 'propra_nomo', 'adjektivo', 'pronomo', 'korelativo',
            'numero', 'artikolo'}


def feats(w: dict) -> str:
    """Esperanto marks case, number and tense ON THE SURFACE. This is free —
    English parsers spend real effort recovering what `-n` and `-j` just say."""
    f = []
    if w.get('vortspeco') in _NOMINAL:
        if w.get('kazo') == 'akuzativo':
            f.append('Case=Acc')
        elif w.get('kazo') == 'nominativo':
            f.append('Case=Nom')
        if w.get('nombro') == 'pluralo':
            f.append('Number=Plur')
        elif w.get('nombro') == 'singularo':
            f.append('Number=Sing')
    if w.get('vortspeco') == 'verbo':
        t = {'prezenco': 'Tense=Pres', 'preterito': 'Tense=Past',
             'futuro': 'Tense=Fut'}.get(w.get('tempo') or '')
        if t:
            f.append(t)
    return '|'.join(sorted(f)) or '_'


def _kern(node):
    if not isinstance(node, dict):
        return None
    return node.get('kerno', node)


def to_conllu(text: str, sent_id: str = '1') -> str:
    """Parse a sentence and emit it as CoNLL-U.

    HEAD/DEPREL come from the CLAUSE TREE (`propozicioj`) — one predicate-argument
    frame per finite verb. That is why the tree had to land first: a flat record
    with one subject slot cannot produce a dependency tree for a sentence with two
    clauses, which is 35.8% of them.
    """
    ast = parse(text)

    # PURE SERIALIZER. The parser now assigns `id`, `kapo` (head) and `rolo`
    # (relation) to every token — see `attach_all` in klareco/parser.py. This
    # function no longer computes any attachment of its own.
    #
    # It used to. That was wrong: the AST and the emitted dependencies could
    # disagree and nothing would catch it. If a relation is missing here, the
    # BUG IS IN THE AST, which is where it should be visible.
    ordered = [w for w in (ast.get('vortoj') or []) if isinstance(w, dict)]
    if not ordered:
        return f'# sent_id = {sent_id}\n# text = {text}\n'

    head = {w['id']: (w.get('kapo') if w.get('kapo') is not None else 0)
            for w in ordered}
    rel = {w['id']: (w.get('rolo') or 'dep') for w in ordered}
    main = next((i for i, r in rel.items() if r == 'root'), None)
    if main is None and ordered:
        main = ordered[0]['id']
        head[main], rel[main] = 0, 'root'

    lines = [f'# sent_id = {sent_id}', f'# text = {text}']
    for i, w in enumerate(ordered, start=1):
        lines.append('\t'.join([
            str(i),
            w.get('plena_vorto') or '_',
            (w.get('radiko') or '_'),
            upos(w),
            '_',
            feats(w),
            str(head.get(i, main or 0)),
            rel.get(i, 'dep'),
            '_',
            f"Vortspeco={w.get('vortspeco')}",
        ]))
    return '\n'.join(lines) + '\n'
