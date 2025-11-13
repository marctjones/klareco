"""
Gating Network for intent classification.

The Gating Network analyzes parsed ASTs to determine query intent,
which is used by the Orchestrator to route queries to appropriate experts.

Level 1 (Phase 4): Symbolic rules based on AST structure
Level 2 (Phase 5): Neural classifier learned from execution traces
"""

from typing import Dict, Any, Optional


# Esperanto question words
QUESTION_WORDS = {
    # Who/What/Which
    'kiu': 'factoid_question',      # who, which (one)
    'kio': 'factoid_question',      # what

    # When/Where
    'kiam': 'factoid_question',     # when
    'kie': 'factoid_question',      # where

    # Why/How
    'kial': 'factoid_question',     # why
    'kiel': 'factoid_question',     # how

    # How much/many
    'kiom': 'calculation_request',  # how much/many (often numerical)
}

# Command verbs for dictionary lookup
DICTIONARY_VERBS = {
    'difin': 'dictionary_query',    # define
    'klarig': 'dictionary_query',   # explain/clarify
    'signif': 'dictionary_query',   # mean/signify
}

# Mathematical operators
MATH_OPERATORS = {
    'plus', 'minus', 'foje', 'divide', 'kvadrato', 'radiko',
    'pli', 'malpli', 'multipliki', 'dividi'
}

# Temporal keywords
TEMPORAL_KEYWORDS = {
    'hodiaŭ', 'hieraŭ', 'morgaŭ', 'tago', 'dato', 'horo',
    'tempo', 'jaro', 'monato', 'semajno', 'minuto', 'sekundo'
}

# Grammar keywords
GRAMMAR_KEYWORDS = {
    'gramatik', 'eksplik', 'analiz', 'vortspec', 'kazo',
    'nombro', 'tempo', 'modo', 'strukt'
}


def has_question_word(ast: Dict[str, Any]) -> Optional[str]:
    """
    Check if AST contains a question word.

    Args:
        ast: Parsed query AST

    Returns:
        Question word if found, None otherwise
    """
    if ast.get('tipo') != 'frazo':
        return None

    # Check all words in the sentence
    aliaj = ast.get('aliaj', [])
    subjekto = ast.get('subjekto')
    verbo = ast.get('verbo')
    objekto = ast.get('objekto')

    # Helper to get root from word/vortgrupo
    def get_root(item):
        if not item:
            return None
        if item.get('tipo') == 'vorto':
            return item.get('radiko')
        elif item.get('tipo') == 'vortgrupo':
            kerno = item.get('kerno')
            if kerno:
                return kerno.get('radiko')
        return None

    # Check subject
    root = get_root(subjekto)
    if root and root.lower() in QUESTION_WORDS:
        return root.lower()

    # Check verb
    root = get_root(verbo)
    if root and root.lower() in QUESTION_WORDS:
        return root.lower()

    # Check object
    root = get_root(objekto)
    if root and root.lower() in QUESTION_WORDS:
        return root.lower()

    # Check other words
    for item in aliaj:
        root = get_root(item)
        if root and root.lower() in QUESTION_WORDS:
            return root.lower()

    return None


def has_numbers(ast: Dict[str, Any]) -> bool:
    """
    Check if AST contains numeric values.

    Args:
        ast: Parsed query AST

    Returns:
        True if AST contains numbers
    """
    if ast.get('tipo') != 'frazo':
        return False

    # Helper to check if word is a number
    def is_number(item):
        if not item:
            return False
        if item.get('tipo') == 'vorto':
            root = item.get('radiko', '')
            # Check if root is numeric or common number words
            if root.isdigit():
                return True
            # Esperanto number words
            number_roots = {
                'unu', 'du', 'tri', 'kvar', 'kvin', 'ses', 'sep', 'ok', 'naŭ', 'dek',
                'cent', 'mil', 'nul'
            }
            return root.lower() in number_roots
        return False

    # Check all parts of sentence
    aliaj = ast.get('aliaj', [])
    subjekto = ast.get('subjekto')
    objekto = ast.get('objekto')

    for item in [subjekto, objekto] + aliaj:
        if is_number(item):
            return True
        # Check vortgrupo members
        if item and item.get('tipo') == 'vortgrupo':
            kerno = item.get('kerno')
            if is_number(kerno):
                return True
            for mod in item.get('modifiloj', []):
                if is_number(mod):
                    return True

    return False


def has_math_operators(ast: Dict[str, Any]) -> bool:
    """
    Check if AST contains mathematical operators.

    Args:
        ast: Parsed query AST

    Returns:
        True if AST contains math operators
    """
    if ast.get('tipo') != 'frazo':
        return False

    # Helper to get root and full word from word/vortgrupo
    def get_word_forms(item):
        if not item:
            return None, None
        if item.get('tipo') == 'vorto':
            return item.get('radiko'), item.get('plena_vorto')
        elif item.get('tipo') == 'vortgrupo':
            kerno = item.get('kerno')
            if kerno:
                return kerno.get('radiko'), kerno.get('plena_vorto')
        return None, None

    # Check all parts
    aliaj = ast.get('aliaj', [])
    subjekto = ast.get('subjekto')
    verbo = ast.get('verbo')
    objekto = ast.get('objekto')

    for item in [subjekto, verbo, objekto] + aliaj:
        root, full_word = get_word_forms(item)
        # Check both root and full word against MATH_OPERATORS
        if (root and root.lower() in MATH_OPERATORS) or \
           (full_word and full_word.lower() in MATH_OPERATORS):
            return True

    return False


def has_dictionary_verb(ast: Dict[str, Any]) -> bool:
    """
    Check if AST contains a dictionary lookup command.

    Args:
        ast: Parsed query AST

    Returns:
        True if AST is a dictionary lookup request
    """
    if ast.get('tipo') != 'frazo':
        return False

    verbo = ast.get('verbo')
    if not verbo or verbo.get('tipo') != 'vorto':
        return False

    root = verbo.get('radiko', '').lower()
    return root in DICTIONARY_VERBS


def is_imperative(ast: Dict[str, Any]) -> bool:
    """
    Check if AST is in imperative mood (command).

    Args:
        ast: Parsed query AST

    Returns:
        True if AST is imperative
    """
    if ast.get('tipo') != 'frazo':
        return False

    verbo = ast.get('verbo')
    if not verbo or verbo.get('tipo') != 'vorto':
        return False

    return verbo.get('modo') == 'imperativo'


def has_temporal_keywords(ast: Dict[str, Any]) -> bool:
    """
    Check if AST contains temporal keywords.

    Args:
        ast: Parsed query AST

    Returns:
        True if AST contains temporal keywords
    """
    if ast.get('tipo') != 'frazo':
        return False

    # Helper to get root and full word from word/vortgrupo
    def get_word_forms(item):
        if not item:
            return None, None
        if item.get('tipo') == 'vorto':
            return item.get('radiko'), item.get('plena_vorto')
        elif item.get('tipo') == 'vortgrupo':
            kerno = item.get('kerno')
            if kerno:
                return kerno.get('radiko'), kerno.get('plena_vorto')
        return None, None

    # Check all parts
    aliaj = ast.get('aliaj', [])
    subjekto = ast.get('subjekto')
    verbo = ast.get('verbo')
    objekto = ast.get('objekto')

    for item in [subjekto, verbo, objekto] + aliaj:
        root, full_word = get_word_forms(item)
        # Check both root and full word against TEMPORAL_KEYWORDS
        if (root and root.lower() in TEMPORAL_KEYWORDS) or \
           (full_word and full_word.lower() in TEMPORAL_KEYWORDS):
            return True

    return False


def has_grammar_keywords(ast: Dict[str, Any]) -> bool:
    """
    Check if AST contains grammar-related keywords.

    Args:
        ast: Parsed query AST

    Returns:
        True if AST contains grammar keywords
    """
    if ast.get('tipo') != 'frazo':
        return False

    # Helper to get root from word/vortgrupo
    def get_root(item):
        if not item:
            return None
        if item.get('tipo') == 'vorto':
            return item.get('radiko')
        elif item.get('tipo') == 'vortgrupo':
            kerno = item.get('kerno')
            if kerno:
                return kerno.get('radiko')
        return None

    # Check all parts
    aliaj = ast.get('aliaj', [])
    subjekto = ast.get('subjekto')
    verbo = ast.get('verbo')
    objekto = ast.get('objekto')

    for item in [subjekto, verbo, objekto] + aliaj:
        root = get_root(item)
        if root and root.lower() in GRAMMAR_KEYWORDS:
            return True

    return False


def classify_intent_symbolic(ast: Dict[str, Any]) -> str:
    """
    Classify query intent using symbolic rules (Level 1).

    This is a pure rule-based classifier that analyzes AST structure
    to determine intent. No machine learning involved.

    Args:
        ast: Parsed query AST

    Returns:
        Intent string, one of:
        - 'factoid_question': Factual question (who/what/when/where/why/how)
        - 'calculation_request': Mathematical calculation
        - 'temporal_query': Date/time related question
        - 'grammar_query': Grammar explanation request
        - 'dictionary_query': Word definition request
        - 'command_intent': General command (imperative mood)
        - 'general_query': Fallback for unclassified queries
    """
    # Check for grammar query first (specific)
    if has_grammar_keywords(ast):
        return 'grammar_query'

    # Check for temporal query
    if has_temporal_keywords(ast):
        return 'temporal_query'

    # Check for dictionary lookup command
    if has_dictionary_verb(ast):
        return 'dictionary_query'

    # Check for mathematical calculation
    if has_numbers(ast) and has_math_operators(ast):
        return 'calculation_request'

    # Check for factual question
    question_word = has_question_word(ast)
    if question_word:
        # Override: 'kiom' (how much/many) often indicates calculation
        if question_word == 'kiom' and has_numbers(ast):
            return 'calculation_request'
        return QUESTION_WORDS.get(question_word, 'factoid_question')

    # Check for imperative mood
    if is_imperative(ast):
        return 'command_intent'

    # Default fallback
    return 'general_query'


class GatingNetwork:
    """
    Gating Network for intent classification.

    Routes queries to appropriate experts based on intent.
    Currently uses symbolic rules (Level 1), will add neural
    classifier in Phase 5 (Level 2).
    """

    def __init__(self, mode: str = 'symbolic'):
        """
        Initialize Gating Network.

        Args:
            mode: Classification mode ('symbolic' or 'neural')
                  'symbolic': Rule-based (Phase 4)
                  'neural': Learned from traces (Phase 5)
        """
        self.mode = mode

        if mode == 'neural':
            raise NotImplementedError("Neural gating network coming in Phase 5")

    def classify(self, ast: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify query intent.

        Args:
            ast: Parsed query AST

        Returns:
            Classification result with:
            - 'intent': Intent string
            - 'confidence': Confidence score (0.0-1.0)
            - 'method': Classification method used
        """
        if self.mode == 'symbolic':
            intent = classify_intent_symbolic(ast)

            # Symbolic rules are deterministic, so confidence is 1.0
            # (we're confident in the rule-based classification)
            # In Phase 5, neural classifier will provide real confidence scores
            return {
                'intent': intent,
                'confidence': 1.0,
                'method': 'symbolic'
            }
        else:
            raise NotImplementedError(f"Unsupported mode: {self.mode}")

    def __repr__(self) -> str:
        return f"GatingNetwork(mode='{self.mode}')"


# Convenience function for direct usage
def classify_intent(ast: Dict[str, Any]) -> str:
    """
    Classify query intent (convenience function).

    Args:
        ast: Parsed query AST

    Returns:
        Intent string
    """
    return classify_intent_symbolic(ast)
