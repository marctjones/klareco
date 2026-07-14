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
    pv = word_ast.get('plena_vorto') or ''

    # A token containing a DIGIT is not made of morphemes. `1-a` is the ordinal
    # "unua" (1st) and the parser was giving it radiko='a' — the digit dropped
    # entirely, so it came back out as bare `a`. Digits are surface, not structure.
    if any(c.isdigit() for c in pv):
        return pv

    if vortspeco in _USE_PLENA_VORTO:
        return pv or word_ast.get('radiko') or ''

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

    # THE STEM, EXACTLY AS IT APPEARED. `tigo` is written by the morphology layer
    # and is the concatenation of the actual morphemes — including the LINKING
    # VOWEL, which is OPTIONAL in Esperanto and which the old AST did not record.
    #
    # The old code GUESSED: it joined compound roots with a hard-coded 'o'.
    #     jarcento    -> jarocento      (there is no linking vowel)
    #     enhavas     -> enohavas
    #     devenas     -> dedeovenas     (and it double-counted the prefix)
    # 40% of corpus sentences failed to round-trip and nothing noticed, because
    # there was no round-trip test.
    tigo = word_ast.get('tigo')
    if tigo:
        stem = tigo
    else:
        kunmetitaj = word_ast.get('kunmetitaj_radikoj') or []
        if len(kunmetitaj) >= 2:
            # WE CANNOT REBUILD THIS. The linking vowel is OPTIONAL in Esperanto
            # (`hundodomo` has one, `dufoje` does not) and this AST did not record
            # it — `tigo` is only written where klareco.morphology owns the word.
            #
            # The old code GUESSED, always inserting an `o`, and produced
            # `duofoje`, `ĉeoesto`, `jarocento`. **Do not fabricate.** Return the
            # surface and let the round-trip test say the morphology is
            # incomplete here, rather than silently emitting a word that does not
            # exist.
            return pv or ''
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
    """AST -> Esperanto. Reconstructs the SURFACE, in order, from `vortoj`.

    WHY THIS CHANGED (#833)
    -----------------------
    The old deparser walked the FLAT FRAME — subjekto + priskriboj + verbo +
    objekto + aliaj — and so had no idea what order the words came in. It
    produced:

        in : Kvankam Esperanto ne estas perfekta, ĝi funkcias bone.
        out: Perfekta perfekta Esperanto estas ne ĝi funkcias bone.

    Duplicated, reordered, and `Kvankam` silently deleted. It had 21 passing
    tests.

    The AST now carries `vortoj` — every token, in surface order, with `id`,
    `kapo` and `rolo` (#825). So deparsing is a walk, and the ROUND TRIP becomes
    a real test of two things at once:

      1. TOKEN COMPLETENESS. If a token is missing from the AST it cannot come
         back out. This alone would have caught `la` being ABSORBED into the
         vortgrupo and lost — 9.1% of LAS — the moment it was introduced. We had
         no such test, so it went unnoticed for months.

      2. MORPHOLOGICAL INVERTIBILITY. `_reconstruct_word` rebuilds each surface
         form from radiko + prefiksoj + sufiksoj + ending. If the decomposition
         is wrong, the word comes back wrong. That is a free, corpus-wide check on
         the morphology.

    For GENERATION from a tree (VISION.md's "grammatically correct by
    construction"), see `deparse_structural`. Note that Esperanto has FREE WORD
    ORDER, so a structural deparse legitimately produces a different string — it
    cannot be tested by string equality, and pretending otherwise is how the old
    tests passed while the deparser was broken.
    """
    if not isinstance(ast, dict) or ast.get('tipo') != 'frazo':
        raise ValueError("Nevalida AST-formato. Atendis tipo 'frazo'.")

    vortoj = ast.get('vortoj')
    if not vortoj:
        return deparse_structural(ast)          # pre-#825 AST: fall back

    # PUNCTUATION IS IN `vortoj` NOW (#836), so it is REPLAYED, not GUESSED.
    #
    # This used to join every token with a space and then APPEND a terminal mark
    # inferred from `fraztipo` — a guess, and the reason `deparse` only reproduced
    # 93.3% of its input. The marks are real tokens now, in their real positions.
    #
    # They just need SPACING, which is a typographic convention and not something
    # the AST should carry: a closing mark binds LEFT (`domo,` not `domo ,`), an
    # opening mark binds RIGHT (`«Faust»` not `« Faust »`).
    _BIND_LEFT = set('.,!?;:…)»]”’%°′″‰\u00ad')
    _BIND_RIGHT = set('(«[“‘')

    out: list[str] = []
    bind_next = False
    for w in vortoj:
        if not isinstance(w, dict):
            continue
        s = _reconstruct_word(w)
        if not s:
            continue
        # CASE is a property of the SURFACE, not of the morphology. `Akademio` and
        # `akademio` have the same radiko, so a morpheme-rebuild cannot know which
        # one was written — 331 of the round-trip failures were exactly this.
        # Replay the recorded case, but ONLY when the rebuild agrees with the
        # surface otherwise: a genuinely wrong rebuild must still show as wrong.
        orig = w.get('plena_vorto') or ''
        if orig and orig.lower() == s.lower() and orig != s:
            s = orig
        is_mark = w.get('vortspeco') == 'interpunkcio'
        if not out:
            out.append(s)
        elif bind_next or (is_mark and s[0] in _BIND_LEFT):
            out[-1] += s                    # no space
        else:
            out.append(s)
        bind_next = bool(is_mark and s[-1] in _BIND_RIGHT)

    body = ' '.join(out)
    if not body:
        return ''
    return body[0].upper() + body[1:]


def deparse_structural(ast: dict) -> str:
    """GENERATE a sentence from the clause tree, rather than replay the surface.

    This is the path VISION.md's "grammatically correct by construction" claim
    rests on. It legitimately produces a DIFFERENT word order from the input —
    Esperanto's word order is free, and the case endings carry the roles — so it
    must NOT be tested by string equality against the source. Testing it that way
    is precisely how the old deparser kept 21 tests green while emitting
    "Perfekta perfekta Esperanto estas ne ĝi funkcias bone."
    """
    if not isinstance(ast, dict) or ast.get('tipo') != 'frazo':
        raise ValueError("Nevalida AST-formato. Atendis tipo 'frazo'.")

    body = _deparse_frazo_body(ast)
    if not body:
        return ''

    punct = {'demando': '?', 'ordono': '!'}.get(ast.get('fraztipo', 'deklaro'), '.')
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
