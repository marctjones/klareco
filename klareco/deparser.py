"""
The De-parser (AST -> Text).

Converts a detailed, Esperanto-native AST back into a readable Esperanto sentence.
This module is the inverse of the parser.

VERSION: v2.1
COMPATIBLE WITH: v2.1 parser output (rilata_subfrazo nodes, kunmetitaj_radikoj,
                 all vortspeco types, nested frazo objects, fraztipo punctuation)
"""

# ---------------------------------------------------------------------------
# vortspeco values where morpheme-level reconstruction is unreliable.
# For these, the deparser uses plena_vorto (the original surface form) directly.
# ---------------------------------------------------------------------------
_USE_PLENA_VORTO = frozenset({
    'pronomo',      # mi, vi, li, ŝi, ĝi, ni, ili, oni, si
    'korelativo',   # kiu, kio, kie, kiam, kiel, kiom, kies, tiu, …
    'artikolo',     # la
    'konjunkcio',   # kaj, sed, aŭ, nek, do, tamen, …
    'prepozicio',   # de, en, al, por, el, sur, sub, kun, pri, …
    'partiklo',     # ne, ĉu, jen, nu, …
    'numero',       # 1887, 42, … (digits stored verbatim)
    'propra_nomo',  # Zamenhof, Esperanto, … (arbitrary capitalised forms)
    'nekonata',     # unrecognised words
    'fremda_vorto', # foreign-language words
})

# Verb tense (indicative mood) → Esperanto ending
_TEMPO_ENDINGS = {
    'prezenco':  'as',
    'pasinteco': 'is',
    'futuro':    'os',
    # Esperanto-native tempo names also accepted
    'estanteco': 'as',
    'estonteco': 'os',
}

# Verb mood (non-indicative) → Esperanto ending
_MODO_ENDINGS = {
    'kondicionalo': 'us',
    'imperativo':   'u',
    'infinitivo':   'i',
    # Legacy values that appeared in early parser versions — kept for safety
    'kondiĉa': 'us',
    'vola':    'u',
}


# ---------------------------------------------------------------------------
# Word reconstruction
# ---------------------------------------------------------------------------

def _reconstruct_word(word_ast: dict) -> str:
    """
    Reconstruct a single Esperanto word from its morpheme AST.

    For inflectable content words (substantivo, adjektivo, adverbo, verbo)
    the word is assembled from prefix(es) + root + suffix(es) + ending +
    plural/accusative inflections.

    For function words, proper nouns, numbers, and unknowns the original
    surface form (plena_vorto) is returned directly, because reconstruction
    from morphemes is unreliable for these types.
    """
    if not isinstance(word_ast, dict):
        return ''

    vortspeco = word_ast.get('vortspeco', '')

    if vortspeco in _USE_PLENA_VORTO:
        return word_ast.get('plena_vorto') or word_ast.get('radiko') or ''

    # Unknown tipo — fall back gracefully rather than crash
    if word_ast.get('tipo') != 'vorto':
        return word_ast.get('plena_vorto') or word_ast.get('radiko') or ''

    # --- Stem: prefix(es) + root + suffix(es) ---
    prefiksoj = word_ast.get('prefiksoj') or []
    if not prefiksoj:
        # Backwards-compatible: old 'prefikso' (string) field
        old = word_ast.get('prefikso')
        prefiksoj = [old] if old else []
    prefix = ''.join(prefiksoj)

    # Compound words: reconstruct the full compound, not just the head root.
    #   kunmetitaj_radikoj = ['libr', 'vend']
    #   → 'libr' + linking 'o' + 'vend' → stem 'librovend'
    kunmetitaj = word_ast.get('kunmetitaj_radikoj') or []
    if len(kunmetitaj) >= 2:
        # Non-head roots each get the linking vowel 'o';
        # the head root (last element) receives the normal morphological ending.
        root = 'o'.join(kunmetitaj[:-1]) + 'o' + kunmetitaj[-1]
    else:
        root = word_ast.get('radiko') or ''

    suffixes = ''.join(word_ast.get('sufiksoj') or [])
    stem = f'{prefix}{root}{suffixes}'

    # --- Part-of-speech ending ---
    pos_ending = ''
    if vortspeco == 'substantivo':
        pos_ending = 'o'
    elif vortspeco == 'adjektivo':
        pos_ending = 'a'
    elif vortspeco == 'adverbo':
        pos_ending = 'e'
    elif vortspeco == 'verbo':
        tempo = word_ast.get('tempo')
        modo  = word_ast.get('modo')
        if tempo:
            pos_ending = _TEMPO_ENDINGS.get(tempo, 'i')
        elif modo:
            pos_ending = _MODO_ENDINGS.get(modo, 'i')
        else:
            pos_ending = 'i'   # infinitive as safe default

    # --- Grammatical inflections ---
    plural = 'j' if word_ast.get('nombro') == 'pluralo' else ''
    case   = 'n' if word_ast.get('kazo')   == 'akuzativo' else ''

    return f'{stem}{pos_ending}{plural}{case}'


# ---------------------------------------------------------------------------
# Node dispatcher
# ---------------------------------------------------------------------------

def _deparse_node(node) -> str:
    """
    Dispatch any AST node to the appropriate deparser fragment.
    Returns a plain-text string (no capitalisation, no terminal punctuation).
    """
    if not isinstance(node, dict):
        return str(node) if node else ''

    tipo = node.get('tipo', '')

    if tipo == 'vorto':
        return _reconstruct_word(node)
    elif tipo == 'vortgrupo':
        return _deparse_vortgrupo(node)
    elif tipo == 'frazo':
        return _deparse_frazo_body(node)
    elif tipo == 'rilata_subfrazo':
        return _deparse_rilata_subfrazo(node)
    else:
        # Unknown tipo — best-effort reconstruction as a word
        return _reconstruct_word(node)


# ---------------------------------------------------------------------------
# Vortgrupo (noun phrase)
# ---------------------------------------------------------------------------

def _deparse_vortgrupo(vortgrupo_ast: dict) -> str:
    """
    Deparse a vortgrupo (noun phrase) node.

    Output order:
        [article]  [adjective priskriboj]  kerno  [rilata_subfrazo priskriboj]

    Adjectives and other non-relative priskriboj are placed before the head
    noun; relative clauses are placed after it (standard Esperanto word order).
    """
    if not isinstance(vortgrupo_ast, dict):
        return ''

    # Nested frazo (ke-clause stored as objekto) — delegate to frazo body
    if vortgrupo_ast.get('tipo') == 'frazo':
        return _deparse_frazo_body(vortgrupo_ast)

    parts = []

    # Article (la) — written by parser as vortgrupo['artikolo'] = 'la'
    if 'artikolo' in vortgrupo_ast:
        parts.append(vortgrupo_ast['artikolo'])

    # Pre-nominal priskriboj: adjectives and other non-relative modifiers
    for prisk in vortgrupo_ast.get('priskriboj', []):
        if isinstance(prisk, dict) and prisk.get('tipo') != 'rilata_subfrazo':
            text = _deparse_node(prisk)
            if text:
                parts.append(text)

    # Head noun
    if 'kerno' in vortgrupo_ast:
        text = _deparse_node(vortgrupo_ast['kerno'])
        if text:
            parts.append(text)

    # Post-nominal priskriboj: relative clauses
    for prisk in vortgrupo_ast.get('priskriboj', []):
        if isinstance(prisk, dict) and prisk.get('tipo') == 'rilata_subfrazo':
            text = _deparse_rilata_subfrazo(prisk)
            if text:
                parts.append(text)

    return ' '.join(parts)


# ---------------------------------------------------------------------------
# Rilata subfrazo (relative clause)
# ---------------------------------------------------------------------------

def _deparse_rilata_subfrazo(rilata: dict) -> str:
    """
    Deparse a rilata_subfrazo (relative clause) node.

    The relative pronoun always comes first (kiu / kiun / kie / kiam / …),
    followed by the rest of the clause with the pronoun's own slot removed
    so it is not output twice.

    Examples:
        kiu vidas la hundon    (kiu is nominative subject)
        kiun mi vidas          (kiun is accusative object)
    """
    parts = []
    pronomo = rilata.get('rilata_pronomo')

    # Relative pronoun leads the clause
    if pronomo:
        parts.append(_reconstruct_word(pronomo))

    subj = rilata.get('subjekto')
    verbo = rilata.get('verbo')
    obj   = rilata.get('objekto')

    # Subject — skip if it IS the relative pronoun (e.g. nominative kiu)
    if subj and isinstance(subj, dict):
        if subj.get('kerno') is not pronomo:
            text = _deparse_vortgrupo(subj)
            if text:
                parts.append(text)

    # Verb
    if verbo:
        parts.append(_reconstruct_word(verbo))

    # Object — skip if it IS the relative pronoun (e.g. accusative kiun)
    if obj and isinstance(obj, dict):
        if obj.get('kerno') is not pronomo:
            text = _deparse_vortgrupo(obj)
            if text:
                parts.append(text)

    # Remaining modifiers / adverbials
    for alia in rilata.get('aliaj', []):
        text = _deparse_node(alia)
        if text:
            parts.append(text)

    return ' '.join(parts)


# ---------------------------------------------------------------------------
# Frazo body (shared by top-level and recursive calls)
# ---------------------------------------------------------------------------

def _deparse_frazo_body(ast: dict) -> str:
    """
    Produce the raw text body of a frazo node — no capitalisation, no
    terminal punctuation.  Used by deparse() and by _deparse_node() when
    recursing into nested clauses.

    Output order: subjekto  verbo  objekto  aliaj
    """
    parts = []

    if ast.get('subjekto'):
        text = _deparse_vortgrupo(ast['subjekto'])
        if text:
            parts.append(text)

    if ast.get('verbo'):
        parts.append(_reconstruct_word(ast['verbo']))

    if ast.get('objekto'):
        obj = ast['objekto']
        if isinstance(obj, dict) and obj.get('tipo') == 'frazo':
            # ke-clause: the complementiser "ke" was consumed by the parser,
            # so we restore it here before the nested clause body.
            parts.append('ke')
            parts.append(_deparse_frazo_body(obj))
        else:
            text = _deparse_vortgrupo(obj)
            if text:
                parts.append(text)

    for alia in ast.get('aliaj', []):
        text = _deparse_node(alia)
        if text:
            parts.append(text)

    return ' '.join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def deparse(ast: dict) -> str:
    """
    Convert a morpheme-based sentence AST back into an Esperanto string.

    Handles:
    - All word types: content words, function words, proper nouns, correlatives
    - Compound words via kunmetitaj_radikoj
    - Vortgrupo with article, adjective priskriboj, and relative clause priskriboj
    - rilata_subfrazo (relative clause) nodes anywhere in the tree
    - Nested frazo nodes (ke-clauses stored as objekto)
    - Sentence-type punctuation: '.' for deklaro, '?' for demando, '!' for ordono
    """
    if not isinstance(ast, dict) or ast.get('tipo') != 'frazo':
        raise ValueError("Nevalida AST-formato. Atendis tipo 'frazo'.")

    body = _deparse_frazo_body(ast)
    if not body:
        return ''

    fraztipo = ast.get('fraztipo', 'deklaro')
    if fraztipo == 'demando':
        punct = '?'
    elif fraztipo == 'ordono':
        punct = '!'
    else:
        punct = '.'

    return body[0].upper() + body[1:] + punct


def deparse_from_tokens(tokens: list) -> str:
    """Join a list of surface-form tokens into a sentence string."""
    return ' '.join(tokens)


if __name__ == '__main__':
    from klareco.parser import parse
    import json

    def pretty_print(data):
        print(json.dumps(data, indent=2, ensure_ascii=False))

    tests = [
        ("Simple sentence",          "mi amas la grandan katon"),
        ("Plural subject",           "Malgrandaj hundoj vidas la grandan katon"),
        ("Proper nouns",             "Zamenhof fondis Esperanton"),
        ("Relative clause (kiu)",    "La homo kiu vidas la hundon estas mia amiko"),
        ("Relative clause (kiun)",   "La hundo kiun mi vidas estas bela"),
        ("Multi-level nesting",      "La homo kiu fondis la asocion kiu helpas homojn estas fama"),
        ("Question",                 "Kiu fondis Esperanton"),
        ("Conditional",              "Se vi venus mi estus feliĉa"),
    ]

    for label, sentence in tests:
        ast = parse(sentence)
        result = deparse(ast)
        match = '✓' if result.lower().rstrip('?.!') == sentence.lower() else '△'
        print(f"{match} [{label}]")
        print(f"  In:  {sentence}")
        print(f"  Out: {result}")
