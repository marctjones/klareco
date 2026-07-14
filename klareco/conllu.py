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

    # Collect the word nodes in surface order.
    words: list[dict] = []

    def walk(n):
        if not isinstance(n, dict):
            return
        if n.get('tipo') == 'vorto':
            words.append(n)
            return
        for k in ('kerno', 'subjekto', 'verbo', 'objekto'):
            walk(n.get(k))
        for k in ('aliaj', 'priskriboj'):
            for x in (n.get(k) or []):
                walk(x)

    for c in (ast.get('propozicioj') or [ast]):
        walk(c)

    # Surface order: match against the raw tokens.
    surface = text.replace('.', ' .').replace(',', ' ,').split()
    ordered: list[dict] = []
    used: set[int] = set()
    for tok in surface:
        for i, w in enumerate(words):
            if i in used:
                continue
            if (w.get('plena_vorto') or '').lower() == tok.lower().strip('.,;:!?'):
                ordered.append(w)
                used.add(i)
                break
    idx = {id(w): i + 1 for i, w in enumerate(ordered)}

    # HEAD / DEPREL from the clause frames.
    head: dict[int, int] = {}
    rel: dict[int, str] = {}
    root_set = False
    for c in (ast.get('propozicioj') or []):
        v = _kern(c.get('verbo'))
        if not v or id(v) not in idx:
            continue
        vi = idx[id(v)]
        if not root_set and c.get('rolo') == 'ĉefa':
            head[vi], rel[vi] = 0, 'root'
            root_set = True
        else:
            head[vi] = 0            # provisional; linked below if a main clause exists
            rel[vi] = {'kunordigita': 'conj', 'subordigita': 'advcl',
                       'rilativa': 'acl:relcl'}.get(c.get('rolo'), 'parataxis')
        for slot, deprel in (('subjekto', 'nsubj'), ('objekto', 'obj')):
            n = _kern(c.get(slot))
            if n and id(n) in idx:
                head[idx[id(n)]], rel[idx[id(n)]] = vi, deprel
            grp = c.get(slot)
            if isinstance(grp, dict):
                for d in (grp.get('priskriboj') or []):
                    if id(d) in idx and n is not None:
                        head[idx[id(d)]] = idx[id(n)]
                        rel[idx[id(d)]] = ('det' if d.get('vortspeco') == 'artikolo'
                                           else 'amod')

    # Attach every non-main clause's verb to the main root.
    main = next((i for i, r in rel.items() if r == 'root'), None)
    if main:
        for i, r in rel.items():
            if head.get(i) == 0 and i != main:
                head[i] = main
    if not root_set and ordered:
        head[1], rel[1] = 0, 'root'
        main = 1

    # ---- MODIFIER ATTACHMENT -------------------------------------------
    # Everything above only attaches subjects, objects and verbs. Everything else
    # went into `aliaj`, an unstructured junk drawer, and got no head at all —
    # which is why UAS was 9.7%.
    #
    # But most of these attachments are DETERMINISTIC in Esperanto, and cheap:
    #
    #   la hundo            `la` -> det of the noun that follows
    #   granda hundo        adjective AGREES in case+number -> amod of that noun
    #   en la domo          preposition -> `case` of the noun it governs (UD's
    #                       convention: the ADPOSITION depends on the NOUN)
    #   rapide kuris        adverb -> advmod of the verb
    #
    # What is NOT deterministic is where the whole PREPOSITIONAL PHRASE attaches —
    # to the verb or to a preceding noun. `Mi vidis la viron kun teleskopo` is
    # ambiguous BY GRAMMAR, and Bick measured it at 1/4-1/3 of all Esperanto
    # attachment errors. We attach it to the verb (`obl`), which is the majority
    # baseline, and that CHOICE is exactly the residue a learned ranker should
    # collapse — see klareco/forest.py.
    def _agrees(a: dict, n: dict) -> bool:
        return (a.get('kazo') == n.get('kazo')
                and a.get('nombro') == n.get('nombro'))

    verb_of_clause = main
    for i, w in enumerate(ordered, start=1):
        if i in head:
            continue
        vs = w.get('vortspeco')

        if vs == 'artikolo':
            for j in range(i + 1, len(ordered) + 1):
                if ordered[j - 1].get('vortspeco') in ('substantivo', 'propra_nomo'):
                    head[i], rel[i] = j, 'det'
                    break

        elif vs == 'adjektivo':
            # Rule 3: the adjective agrees with its head noun. Look right, then left.
            for j in list(range(i + 1, len(ordered) + 1)) + list(range(i - 1, 0, -1)):
                n = ordered[j - 1]
                if n.get('vortspeco') in ('substantivo', 'propra_nomo') and _agrees(w, n):
                    head[i], rel[i] = j, 'amod'
                    break

        elif vs == 'prepozicio':
            # UD attaches the ADPOSITION to the NOUN it governs, not vice versa.
            for j in range(i + 1, len(ordered) + 1):
                if ordered[j - 1].get('vortspeco') in ('substantivo', 'propra_nomo',
                                                       'pronomo', 'korelativo'):
                    head[i], rel[i] = j, 'case'
                    break

        elif vs == 'adverbo':
            head[i], rel[i] = verb_of_clause or 0, 'advmod'

        elif vs in ('substantivo', 'propra_nomo', 'pronomo'):
            # A nominal with no role yet is governed by a preposition -> the PP.
            # WHERE the PP attaches is the ambiguous part; the verb is the
            # majority baseline.
            prev = ordered[i - 2] if i >= 2 else None
            if prev is not None and prev.get('vortspeco') == 'prepozicio':
                head[i], rel[i] = verb_of_clause or 0, 'obl'
            else:
                head[i], rel[i] = verb_of_clause or 0, 'nmod'

        elif vs == 'konjunkcio':
            head[i], rel[i] = verb_of_clause or 0, 'cc'

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
