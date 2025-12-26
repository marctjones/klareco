"""
The De-parser (AST -> Text).

Converts a detailed, Esperanto-native AST back into a readable Esperanto sentence.
This module is the inverse of the parser.
"""

def _reconstruct_word(word_ast: dict) -> str:
    """
    Reconstructs an Esperanto word from its detailed morpheme AST.
    This function works by assembling parts from left-to-right.
    """
    if not isinstance(word_ast, dict) or word_ast.get('tipo') != 'vorto':
        raise ValueError(f"Nevalida vorto-AST: {word_ast}")

    # --- Assemble the word stem ---
    # Handle both new 'prefiksoj' (list) and legacy 'prefikso' (string) formats
    prefiksoj = word_ast.get('prefiksoj', [])
    if not prefiksoj:
        # Backwards compatibility: check for old 'prefikso' field
        old_prefix = word_ast.get('prefikso')
        prefiksoj = [old_prefix] if old_prefix else []
    prefix = "".join(prefiksoj)
    root = word_ast.get('radiko') or ''
    suffixes = "".join(word_ast.get('sufiksoj', []))
    stem = f"{prefix}{root}{suffixes}"

    # --- Determine the main Part-of-Speech ending ---
    pos_ending = ''
    vortspeco = word_ast.get('vortspeco')
    if vortspeco == 'substantivo':
        pos_ending = 'o'
    elif vortspeco == 'adjektivo':
        pos_ending = 'a'
    elif vortspeco == 'adverbo':
        pos_ending = 'e'
    elif vortspeco == 'verbo':
        if 'tempo' in word_ast:
            tempo = word_ast['tempo']
            if tempo == 'prezenco': pos_ending = 'as'
            elif tempo == 'pasinteco': pos_ending = 'is'
            elif tempo == 'futuro': pos_ending = 'os'
        elif 'modo' in word_ast:
            modo = word_ast['modo']
            if modo == 'kondiĉa': pos_ending = 'us'
            elif modo == 'vola': pos_ending = 'u'
            elif modo == 'infinitivo': pos_ending = 'i'

    # --- Add grammatical inflections (right-to-left logic) ---
    plural_ending = 'j' if word_ast.get('nombro') == 'pluralo' else ''
    case_ending = 'n' if word_ast.get('kazo') == 'akuzativo' else ''

    return f"{stem}{pos_ending}{plural_ending}{case_ending}"


def _deparse_vortgrupo(vortgrupo_ast: dict) -> str:
    """
    Deparses a 'vortgrupo' (noun phrase) AST.
    """
    parts = []
    if "artikolo" in vortgrupo_ast:
        parts.append(vortgrupo_ast["artikolo"])
    
    if "priskriboj" in vortgrupo_ast:
        for adj_ast in vortgrupo_ast["priskriboj"]:
            parts.append(_reconstruct_word(adj_ast))
            
    if "kerno" in vortgrupo_ast:
        parts.append(_reconstruct_word(vortgrupo_ast["kerno"]))
        
    return " ".join(parts)


def deparse(ast: dict) -> str:
    """
    Converts a morpheme-based sentence AST back into an Esperanto string.
    """
    if not isinstance(ast, dict) or ast.get('tipo') != 'frazo':
        raise ValueError("Nevalida AST-formato. Atendis tipo 'frazo'.")

    parts = []
    
    # The order of reconstruction can be simple for now
    if "subjekto" in ast and ast["subjekto"]:
        parts.append(_deparse_vortgrupo(ast["subjekto"]))
        
    if "verbo" in ast and ast["verbo"]:
        parts.append(_reconstruct_word(ast["verbo"]))
        
    if "objekto" in ast and ast["objekto"]:
        parts.append(_deparse_vortgrupo(ast["objekto"]))

    # Add other parts that were not classified
    if "aliaj" in ast:
        for alia_ast in ast["aliaj"]:
            parts.append(_reconstruct_word(alia_ast))

    # Capitalize first word and add a period.
    sentence = " ".join(parts)
    if not sentence:
        return ""
        
    return sentence[0].upper() + sentence[1:] + "."


if __name__ == '__main__':
    # Example Usage with the new parser and deparser
    from klareco.parser import parse
    import json

    def pretty_print(data):
        print(json.dumps(data, indent=2, ensure_ascii=False))

    sentence = "La malgrandaj hundoj vidas la grandan katon"
    print(f"--- Originala frazo: '{sentence}' ---")
    
    ast = parse(sentence)
    print("\n--- Generita AST ---")
    pretty_print(ast)
    
    deparsed_text = deparse(ast)
    print(f"\n--- Rekonstruita frazo ---\n'{deparsed_text}'")
    
    # Verification
    print(f"\nĈu rekonstruo egalas originalon? {deparsed_text.lower().startswith(sentence.lower())}")


def deparse_from_tokens(tokens: list[str]) -> str:
    """A simple deparser that joins a list of tokens."""
    return " ".join(tokens)

