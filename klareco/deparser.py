"""
The De-parser.

Converts a morpheme-based Abstract Syntax Tree (AST) back into an Esperanto sentence.
"""

def _reconstruct_word(word_ast: dict) -> str:
    """
    Reconstructs an Esperanto word from its morpheme-based AST.
    """
    if word_ast.get('type') != 'word':
        raise ValueError(f"Invalid word AST: {word_ast}")

    prefix = word_ast.get('prefix', '')
    root = word_ast.get('root', '')
    suffixes = "".join(word_ast.get('suffixes', []))
    endings = "".join(word_ast.get('endings', []))

    return f"{prefix}{root}{suffixes}{endings}"

def deparse(ast: dict) -> str:
    """
    Converts a morpheme-based sentence AST back into an Esperanto string.

    Args:
        ast: The Abstract Syntax Tree to deparse.

    Returns:
        A grammatically correct Esperanto sentence.
    """
    if not isinstance(ast, dict) or ast.get('type') != 'sentence':
        raise ValueError("Invalid AST format for deparsing. Expected type 'sentence'.")

    subject_ast = ast.get('subject')
    verb_ast = ast.get('verb')
    object_ast = ast.get('object')

    if not (subject_ast and verb_ast and object_ast):
        raise ValueError("Incomplete sentence AST for deparsing.")

    # Reconstruct subject
    subject_word = _reconstruct_word(subject_ast)
    # Only add 'la' if the subject is not a pronoun like 'mi'
    if subject_ast.get('root') == 'mi':
        deparsed_subject = subject_word
    else:
        deparsed_subject = f"La {subject_word}"

    # Reconstruct verb
    deparsed_verb = _reconstruct_word(verb_ast)

    # Reconstruct object
    object_word = _reconstruct_word(object_ast)
    deparsed_object = f"la {object_word}"

    sentence = f"{deparsed_subject} {deparsed_verb} {deparsed_object}."
    
    return sentence.strip()

if __name__ == '__main__':
    # Example Usage with the new AST structure
    from klareco.parser import parse
    import json

    text_1 = "La hundo amas la katon."
    ast_1 = parse(text_1)
    deparsed_text_1 = deparse(ast_1)
    print(f"AST: {json.dumps(ast_1, indent=2, ensure_ascii=False)}")
    print(f"Deparsed Text: '{deparsed_text_1}'\n")

    text_2 = "mi vidas la hundon."
    ast_2 = parse(text_2)
    deparsed_text_2 = deparse(ast_2)
    print(f"AST: {json.dumps(ast_2, indent=2, ensure_ascii=False)}")
    print(f"Deparsed Text: '{deparsed_text_2}'\n")