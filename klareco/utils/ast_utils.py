"""
AST utility functions for extracting word structures.

This module provides helper functions for extracting morphological information
from AST nodes, with support for case normalization.
"""

from typing import Dict, List


def normalize_case(ending: str) -> str:
    """
    Normalize case ending to nominative (strip accusative -n).

    Keep semantic features (number, tense), strip grammatical case.
    This is done because case is redundant with encoder position in M1
    (subject_encoder vs object_encoder), and we want the model to learn
    SEMANTIC plausibility (e.g., food doesn't eat dogs), not grammatical
    rules (e.g., accusative in wrong position is bad).

    Args:
        ending: Original ending ('o', 'on', 'oj', 'ojn', 'as', etc.)

    Returns:
        Normalized ending ('o', 'oj', 'as', etc.)

    Examples:
        >>> normalize_case('on')
        'o'
        >>> normalize_case('ojn')
        'oj'
        >>> normalize_case('as')
        'as'
        >>> normalize_case('en')
        'en'
    """
    # Strip accusative marker (just the 'n')
    # Don't strip 'en' (adverb) or 'an' (part of correlatives)
    if ending.endswith('n') and ending not in ['en', 'an']:
        return ending[:-1]
    return ending


def extract_ending_from_ast(word_ast: Dict) -> str:
    """
    Extract grammatical ending string from AST word node.

    Reconstructs ending by stripping prefixes+root+suffixes from full word.

    Args:
        word_ast: AST node for a word (tipo='vorto')

    Returns:
        Ending string ('o', 'on', 'as', etc.) or '<NONE>'

    Examples:
        >>> word = {'plena_vorto': 'hundoj', 'radiko': 'hund', 'prefiksoj': [], 'sufiksoj': []}
        >>> extract_ending_from_ast(word)
        'oj'
    """
    plena = word_ast.get('plena_vorto', '').lower()
    radiko = word_ast.get('radiko', '').lower()
    prefiksoj = word_ast.get('prefiksoj', [])
    sufiksoj = word_ast.get('sufiksoj', [])

    # Reconstruct stem: prefixes + root + suffixes
    stem = ''.join(prefiksoj) + radiko + ''.join(sufiksoj)

    # Ending is what remains after stem
    if plena.startswith(stem):
        ending = plena[len(stem):]
        # Valid endings from CompositionalEmbedding.ENDINGS
        valid_endings = [
            'o', 'on', 'oj', 'ojn',  # Nouns
            'a', 'an', 'aj', 'ajn',  # Adjectives
            'e', 'en',               # Adverbs
            'i',                     # Infinitive
            'as', 'is', 'os', 'us', 'u'  # Verbs
        ]
        if ending in valid_endings:
            return ending

    return '<NONE>'


def extract_word_structure(word_ast: Dict, strip_case: bool = True) -> Dict:
    """
    Extract full word structure for CompositionalEmbedding.

    This extracts all morphological information from an AST word node,
    suitable for encoding with CompositionalEmbedding.

    Args:
        word_ast: AST word node (tipo='vorto')
        strip_case: If True, normalize case to nominative (default: True)
                   This removes redundant grammatical info since role is
                   encoded by position (subject_encoder vs object_encoder).
                   Keeps semantic features like number and tense.

    Returns:
        Dictionary with keys:
        - 'root': str - Root word (e.g., 'hund')
        - 'prefixes': List[str] - Prefixes (e.g., ['mal', 're'])
        - 'suffixes': List[str] - Suffixes (e.g., ['ej', 'et'])
        - 'ending': str - Grammatical ending (e.g., 'o', 'oj', 'as')

    Examples:
        >>> word = parse("hundo")['subjekto']['kerno']
        >>> extract_word_structure(word, strip_case=True)
        {'root': 'hund', 'prefixes': [], 'suffixes': [], 'ending': 'o'}

        >>> word = parse("hundon")['objekto']['kerno']
        >>> extract_word_structure(word, strip_case=True)
        {'root': 'hund', 'prefixes': [], 'suffixes': [], 'ending': 'o'}  # -n stripped

        >>> word = parse("hundojn")['objekto']['kerno']
        >>> extract_word_structure(word, strip_case=True)
        {'root': 'hund', 'prefixes': [], 'suffixes': [], 'ending': 'oj'}  # -n stripped, plural kept
    """
    ending = extract_ending_from_ast(word_ast)

    if strip_case:
        ending = normalize_case(ending)

    return {
        'root': word_ast.get('radiko', '').lower(),
        'prefixes': word_ast.get('prefiksoj', []),
        'suffixes': word_ast.get('sufiksoj', []),
        'ending': ending
    }
