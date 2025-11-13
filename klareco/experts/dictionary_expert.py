"""
Dictionary Expert - Word definition and translation lookup

Provides definitions and translations for Esperanto words.
Uses vocabulary databases and can be extended with external APIs.

Part of Phase 8: External Tools
"""

from typing import Dict, Any, Optional, List
import logging
from ..experts.base import Expert

logger = logging.getLogger(__name__)


class DictionaryExpert(Expert):
    """
    Expert for dictionary lookups and word definitions.

    Capabilities:
    - Word definitions (Esperanto → English)
    - Reverse lookup (English → Esperanto)
    - Root analysis
    - Usage examples
    """

    def __init__(self, vocabulary: Optional[Dict[str, str]] = None):
        """
        Initialize Dictionary Expert.

        Args:
            vocabulary: Optional custom vocabulary dict
        """
        super().__init__(name="Dictionary_Expert")
        self.capabilities = ["word_lookup", "definition", "translation"]

        # Basic vocabulary (can be loaded from file in production)
        self.vocabulary = vocabulary or self._load_default_vocabulary()

        logger.info(f"{self.name} initialized with {len(self.vocabulary)} entries")

    def _load_default_vocabulary(self) -> Dict[str, str]:
        """
        Load default vocabulary.

        In production, this would load from vocabulary files.
        For now, provides essential Esperanto words.
        """
        return {
            # Basics
            'hund': 'dog',
            'kat': 'cat',
            'dom': 'house',
            'libr': 'book',
            'akv': 'water',
            'manĝ': 'eat',
            'drink': 'drink',
            'vid': 'see',
            'aŭd': 'hear',
            'parol': 'speak',

            # Common words
            'bon': 'good',
            'mal': 'opposite/bad',
            'grand': 'big',
            'bel': 'beautiful',
            'nov': 'new',
            'malnov': 'old',

            # Question words
            'kio': 'what',
            'kiu': 'who/which',
            'kie': 'where',
            'kiam': 'when',
            'kial': 'why',
            'kiel': 'how',
            'kiom': 'how much/many',

            # Grammar
            'esperant': 'Esperanto (language)',
            'lingv': 'language',
            'vort': 'word',
            'fraз': 'sentence',
            'gramatik': 'grammar',

            # Actions
            'ir': 'go',
            'ven': 'come',
            'far': 'do/make',
            'hav': 'have',
            'est': 'be',
            'vol': 'want',
            'dev': 'must',
            'pov': 'can/be able',

            # Time
            'hodiaŭ': 'today',
            'hieraŭ': 'yesterday',
            'morgaŭ': 'tomorrow',
            'nun': 'now',
            'tuj': 'immediately',

            # Numbers
            'unu': 'one',
            'du': 'two',
            'tri': 'three',
            'kvar': 'four',
            'kvin': 'five',
        }

    def can_handle(self, ast: Dict[str, Any]) -> bool:
        """
        Check if this expert can handle the query.

        Dictionary queries contain:
        - "kio estas" (what is)
        - "difinu" (define)
        - "signifi" (mean)
        - "traduku" (translate)

        Args:
            ast: Parsed query AST

        Returns:
            True if dictionary query
        """
        # Check for dictionary keywords
        dict_keywords = {'difin', 'signif', 'traduk', 'klarig'}

        if self._contains_any_root(ast, dict_keywords):
            return True

        # Check for "Kio estas X?" pattern (What is X?)
        if self._is_definition_question(ast):
            return True

        return False

    def execute(self, ast: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute dictionary lookup.

        Args:
            ast: Parsed query AST
            context: Optional context

        Returns:
            Response with definition
        """
        logger.info(f"{self.name} executing dictionary lookup")

        # Extract word to look up
        word = self._extract_target_word(ast, context)

        if not word:
            return {
                'answer': "Mi ne trovis vorton por serĉi. (I didn't find a word to look up.)",
                'confidence': 0.1,
                'expert': self.name,
                'error': 'no_word_found'
            }

        # Look up word
        definition = self._lookup_word(word)

        if definition:
            answer = f"'{word}' signifas '{definition}' en la angla. ('{word}' means '{definition}' in English.)"
            confidence = 0.95
        else:
            answer = f"Mi ne trovis la vorton '{word}' en mia vortaro. (I didn't find '{word}' in my dictionary.)"
            confidence = 0.3

        return {
            'answer': answer,
            'confidence': confidence,
            'expert': self.name,
            'word': word,
            'definition': definition,
            'metadata': {
                'vocabulary_size': len(self.vocabulary)
            }
        }

    def _extract_target_word(self, ast: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Optional[str]:
        """
        Extract the word to look up from AST.

        Args:
            ast: Parsed query AST
            context: Optional context

        Returns:
            Word to look up
        """
        # Check context first
        if context and 'target_word' in context:
            return context['target_word']

        # Extract from AST
        # Look for the object of the query
        if ast.get('tipo') == 'frazo':
            objekto = ast.get('objekto')
            if objekto:
                return self._extract_word_text(objekto)

            # Check for words in "aliaj"
            aliaj = ast.get('aliaj', [])
            for item in aliaj:
                if item.get('tipo') == 'vorto':
                    radiko = item.get('radiko', '')
                    # Skip question words and verbs
                    if radiko not in ['kio', 'estas', 'difin', 'signif']:
                        return radiko

        return None

    def _extract_word_text(self, node: Dict[str, Any]) -> Optional[str]:
        """Extract text from AST node"""
        if node.get('tipo') == 'vorto':
            return node.get('radiko') or node.get('plena_vorto')
        elif node.get('tipo') == 'vortgrupo':
            kerno = node.get('kerno')
            if kerno:
                return kerno.get('radiko') or kerno.get('plena_vorto')
        return None

    def _lookup_word(self, word: str) -> Optional[str]:
        """
        Look up word in vocabulary.

        Args:
            word: Word or root to look up

        Returns:
            Definition if found
        """
        # Normalize
        word_lower = word.lower().strip()

        # Direct lookup
        if word_lower in self.vocabulary:
            return self.vocabulary[word_lower]

        # Try without endings (basic stemming)
        # Remove common endings: -o, -a, -e, -as, -is, -os, -us, -u, -i, -n, -j
        for ending in ['oj', 'ojn', 'on', 'o', 'aj', 'ajn', 'an', 'a', 'e',
                       'as', 'is', 'os', 'us', 'u', 'i', 'n', 'j']:
            if word_lower.endswith(ending):
                stem = word_lower[:-len(ending)]
                if stem in self.vocabulary:
                    return self.vocabulary[stem]

        return None

    def _is_definition_question(self, ast: Dict[str, Any]) -> bool:
        """Check if query is asking 'What is X?'"""
        if ast.get('tipo') != 'frazo':
            return False

        # Check for "Kio estas" pattern
        has_kio = False
        has_estas = False

        def check_node(node):
            nonlocal has_kio, has_estas
            if not node:
                return

            if node.get('tipo') == 'vorto':
                radiko = node.get('radiko', '').lower()
                if radiko == 'kio':
                    has_kio = True
                elif radiko == 'est':
                    has_estas = True

        check_node(ast.get('subjekto'))
        check_node(ast.get('verbo'))

        for item in ast.get('aliaj', []):
            check_node(item)

        return has_kio and has_estas

    def _contains_any_root(self, ast: Dict[str, Any], roots: set) -> bool:
        """Check if AST contains any of the specified roots"""
        if ast.get('tipo') == 'vorto':
            radiko = ast.get('radiko', '').lower()
            return any(radiko.startswith(root) for root in roots)
        elif ast.get('tipo') == 'vortgrupo':
            return any(self._contains_any_root(v, roots) for v in ast.get('vortoj', []))
        elif ast.get('tipo') == 'frazo':
            for key in ['subjekto', 'verbo', 'objekto']:
                if ast.get(key) and self._contains_any_root(ast[key], roots):
                    return True
            return any(self._contains_any_root(v, roots) for v in ast.get('aliaj', []))
        return False

    def estimate_confidence(self, ast: Dict[str, Any]) -> float:
        """Estimate confidence for handling this query"""
        if self.can_handle(ast):
            return 0.9
        return 0.0

    def __repr__(self) -> str:
        return f"{self.name}({len(self.vocabulary)} words)"


# Factory function
def create_dictionary_expert() -> DictionaryExpert:
    """Create and return a DictionaryExpert instance"""
    return DictionaryExpert()


if __name__ == "__main__":
    # Test dictionary expert
    print("Testing Dictionary Expert")
    print("=" * 80)

    expert = create_dictionary_expert()
    print(f"\n{expert}\n")

    # Test queries
    test_cases = [
        {
            'ast': {
                'tipo': 'frazo',
                'subjekto': {'tipo': 'vorto', 'radiko': 'kio'},
                'verbo': {'tipo': 'vorto', 'radiko': 'est'},
                'aliaj': [{'tipo': 'vorto', 'radiko': 'hund'}]
            },
            'query': "Kio estas 'hundo'?"
        },
        {
            'ast': {
                'tipo': 'frazo',
                'verbo': {'tipo': 'vorto', 'radiko': 'difin'},
                'objekto': {'tipo': 'vorto', 'radiko': 'kat'}
            },
            'query': "Difinu 'kato'"
        }
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"Test {i}: {test['query']}")
        ast = test['ast']

        can_handle = expert.can_handle(ast)
        print(f"  Can handle: {can_handle}")

        if can_handle:
            result = expert.execute(ast)
            print(f"  Answer: {result['answer']}")
            print(f"  Confidence: {result['confidence']:.2f}")

        print()

    print("✅ Dictionary Expert test complete!")
