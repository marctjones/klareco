"""
Dictionary Expert - Provides word definitions and morphological analysis.

This expert handles queries about word meanings, structure, and usage:
- "Kio signifas 'bela'?" → What does 'bela' mean?
- "Kiel oni konstruas 'malbona'?" → How is 'malbona' constructed?
- "Difinu 'amikino'" → Define 'amikino'
- "Kio estas 'hundo'?" → What is 'hundo'?

Uses pure symbolic processing (no neural networks) by analyzing morphemes.
Combines morphological analysis with comprehensive vocabulary lookup.
"""

from typing import Dict, Any, List, Optional
import logging

from .base import Expert
from klareco.parser import parse_word

logger = logging.getLogger(__name__)


class DictionaryExpert(Expert):
    """
    Expert for word definitions and morphological analysis.

    Provides symbolic word definitions by analyzing morpheme structure
    and looking up root meanings in comprehensive vocabulary.
    """

    # Comprehensive root meanings (merged from both versions)
    ROOT_DEFINITIONS = {
        # Core vocabulary
        'bel': 'beautiful',
        'bon': 'good',
        'grand': 'big, large',
        'mal': 'opposite/bad',
        'plen': 'full',
        'san': 'healthy',
        'nov': 'new',
        'malnov': 'old',

        # Animals and nature
        'hund': 'dog',
        'kat': 'cat',

        # Actions - seeing/perceiving
        'vid': 'see',
        'aŭd': 'hear',
        'parol': 'speak',
        'dir': 'say',

        # Actions - movement
        'ir': 'go',
        'ven': 'come',
        'kur': 'run',
        'salt': 'jump',

        # Actions - basic
        'far': 'do, make',
        'est': 'be',
        'hav': 'have',
        'don': 'give',
        'manĝ': 'eat',
        'drink': 'drink',

        # Actions - mental
        'am': 'love',
        'vol': 'want',
        'dev': 'must',
        'pov': 'can/be able',

        # Actions - communication
        'kant': 'sing',
        'leg': 'read',
        'skrib': 'write',
        'labor': 'work',
        'help': 'help',

        # People
        'hom': 'human, person',
        'vir': 'man (male)',
        'amik': 'friend',
        'patr': 'father',
        'matr': 'mother',
        'frat': 'brother',
        'infan': 'child',

        # Places
        'dom': 'house',
        'urb': 'city',
        'land': 'land, country',
        'mond': 'world',

        # Time
        'tag': 'day',
        'nokt': 'night',
        'jar': 'year',
        'temp': 'time',
        'hodiaŭ': 'today',
        'hieraŭ': 'yesterday',
        'morgaŭ': 'tomorrow',
        'nun': 'now',
        'tuj': 'immediately',

        # Objects
        'libr': 'book',
        'akv': 'water',

        # Question words
        'kio': 'what',
        'kiu': 'who/which',
        'kie': 'where',
        'kiam': 'when',
        'kial': 'why',
        'kiel': 'how',
        'kiom': 'how much/many',

        # Grammar/language terms
        'esperant': 'Esperanto (language)',
        'lingv': 'language',
        'vort': 'word',
        'fraz': 'sentence',
        'gramatik': 'grammar',
        'signif': 'mean, signify',
        'difin': 'define',
        'traduk': 'translate',
        'klarig': 'explain',
        'konstru': 'construct',

        # Numbers
        'unu': 'one',
        'du': 'two',
        'tri': 'three',
        'kvar': 'four',
        'kvin': 'five',
    }

    # Suffix meanings
    SUFFIX_DEFINITIONS = {
        'ul': 'person characterized by',
        'in': 'female',
        'et': 'small, diminutive',
        'eg': 'large, augmentative',
        'ej': 'place for',
        'il': 'tool for',
        'ar': 'collection of',
        'ad': 'continuous action',
        'aĉ': 'bad quality',
        'an': 'member of',
        'ebl': 'able to be',
        'end': 'must be',
        'ind': 'worthy of',
        'ig': 'cause to be',
        'iĝ': 'become',
    }

    def __init__(self):
        """Initialize Dictionary Expert."""
        super().__init__("Dictionary Expert")
        logger.info(f"{self.name} initialized with {len(self.ROOT_DEFINITIONS)} root definitions")

    def can_handle(self, ast: Dict[str, Any]) -> bool:
        """
        Check if this is a dictionary query.

        Looks for:
        - "Kio signifas...?" (What does ... mean?)
        - "Difinu..." (Define...)
        - "Kiel oni konstruas...?" (How is ... constructed?)
        - "Kio estas...?" (What is...?)

        Args:
            ast: Parsed query AST

        Returns:
            True if this is a dictionary query
        """
        if not ast or ast.get('tipo') != 'frazo':
            return False

        # Extract all words
        words = self._extract_all_words(ast)
        words_lower = [w.lower() for w in words]

        # Check for definition keywords
        definition_keywords = {
            'signif',  # signifas = means
            'difin',   # difinu = define
            'konstru', # konstruas = constructs
            'traduk',  # traduku = translate
            'klarig',  # klarigu = explain
        }

        has_definition_keyword = any(
            any(keyword in word for keyword in definition_keywords)
            for word in words_lower
        )

        if has_definition_keyword:
            return True

        # Check for "Kio estas X?" pattern (What is X?)
        if self._is_definition_question(ast):
            return True

        return False

    def estimate_confidence(self, ast: Dict[str, Any]) -> float:
        """
        Estimate confidence in handling this query.

        Args:
            ast: Parsed query AST

        Returns:
            Confidence score 0.0-1.0
        """
        if not self.can_handle(ast):
            return 0.0

        words = self._extract_all_words(ast)
        words_lower = [w.lower() for w in words]

        # High confidence for explicit definition requests
        if 'signifas' in words_lower or 'difinu' in words_lower:
            return 0.95

        # High confidence for "What is X?" pattern
        if self._is_definition_question(ast):
            return 0.90

        # Moderate confidence for construction questions
        if 'konstruas' in words_lower or 'konstruita' in words_lower:
            return 0.85

        return 0.7

    def execute(self, ast: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute dictionary query.

        Args:
            ast: Parsed query AST
            context: Optional execution context

        Returns:
            Response with word definition and analysis
        """
        # Validate AST
        if not ast or ast.get('tipo') != 'frazo':
            return {
                'answer': 'Mi ne povas procezi malplenan aŭ malvalidan demandon.',
                'confidence': 0.0,
                'expert': self.name,
                'error': 'Invalid or empty AST'
            }

        try:
            # Extract target word from query (check context first)
            target_word = None
            if context and 'target_word' in context:
                target_word = context['target_word']
            else:
                target_word = self._extract_target_word(ast)

            if not target_word:
                return {
                    'answer': 'Mi ne povas trovi la vorton por difini.',
                    'confidence': 0.0,
                    'expert': self.name,
                    'error': 'No target word found'
                }

            # Analyze the word
            analysis = self._analyze_word(target_word)

            # Format answer
            answer = self._format_definition(target_word, analysis)

            return {
                'answer': answer,
                'confidence': 0.9,
                'expert': self.name,
                'word': target_word,
                'analysis': analysis,
                'explanation': f'Morfologia analizo de "{target_word}"',
                'metadata': {
                    'vocabulary_size': len(self.ROOT_DEFINITIONS)
                }
            }

        except Exception as e:
            logger.error(f"Dictionary lookup failed: {e}", exc_info=True)
            return {
                'answer': f'Eraro dum vort-analizo: {str(e)}',
                'confidence': 0.0,
                'expert': self.name,
                'error': str(e)
            }

    def _extract_all_words(self, ast: Dict[str, Any]) -> List[str]:
        """Extract all words from AST recursively."""
        words = []

        if isinstance(ast, dict):
            if ast.get('tipo') == 'vorto':
                word = ast.get('plena_vorto', '') or ast.get('radiko', '')
                if word:
                    words.append(word)

            for value in ast.values():
                if isinstance(value, (dict, list)):
                    words.extend(self._extract_all_words(value))

        elif isinstance(ast, list):
            for item in ast:
                words.extend(self._extract_all_words(item))

        return words

    def _extract_target_word(self, ast: Dict[str, Any]) -> Optional[str]:
        """
        Extract the word being asked about.

        Looks for quoted words or words after definition keywords.
        """
        # Simple heuristic: get last noun or proper noun
        words = []

        def extract(node):
            if isinstance(node, dict):
                if node.get('tipo') == 'vorto':
                    vortspeco = node.get('vortspeco', '')
                    radiko = node.get('radiko', '')
                    # Include all meaningful words except query words
                    if radiko and radiko not in ['kio', 'kiu', 'difin', 'signif', 'est', 'estas']:
                        if vortspeco in ['substantivo', 'nomo', 'adjektivo', 'verbo']:
                            words.append(node.get('plena_vorto', radiko))
                        # Also include words from "aliaj" that might be the target
                        elif radiko not in ['kiel', 'oni', 'konstruas']:
                            words.append(node.get('plena_vorto', radiko))

                for value in node.values():
                    if isinstance(value, (dict, list)):
                        extract(value)
            elif isinstance(node, list):
                for item in node:
                    extract(item)

        extract(ast)

        # Return last meaningful word
        return words[-1] if words else None

    def _is_definition_question(self, ast: Dict[str, Any]) -> bool:
        """
        Check if query is asking 'What is X?' (Kio estas X?)

        Args:
            ast: Parsed query AST

        Returns:
            True if this is a definition question
        """
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

    def _analyze_word(self, word: str) -> Dict[str, Any]:
        """
        Analyze word morphology.

        Args:
            word: Esperanto word to analyze

        Returns:
            Dict with morphological analysis
        """
        try:
            # Parse the word
            word_ast = parse_word(word)

            # Extract components
            radiko = word_ast.get('radiko', '')
            prefikso = word_ast.get('prefikso')
            sufiksoj = word_ast.get('sufiksoj', [])
            vortspeco = word_ast.get('vortspeco', 'unknown')
            nombro = word_ast.get('nombro', 'singularo')
            kazo = word_ast.get('kazo', 'nominativo')

            # Look up meanings
            root_meaning = self.ROOT_DEFINITIONS.get(radiko.lower(), f"[root: {radiko}]")

            suffix_meanings = []
            for suffix in sufiksoj:
                meaning = self.SUFFIX_DEFINITIONS.get(suffix, f"[suffix: {suffix}]")
                suffix_meanings.append((suffix, meaning))

            return {
                'radiko': radiko,
                'radiko_meaning': root_meaning,
                'prefikso': prefikso,
                'sufiksoj': sufiksoj,
                'suffix_meanings': suffix_meanings,
                'vortspeco': vortspeco,
                'nombro': nombro,
                'kazo': kazo,
                'parse_success': True
            }

        except Exception as e:
            logger.warning(f"Could not parse word '{word}': {e}")
            # Fallback: try direct lookup
            direct_def = self._lookup_word_direct(word)
            if direct_def:
                return {
                    'radiko': word,
                    'radiko_meaning': direct_def,
                    'prefikso': None,
                    'sufiksoj': [],
                    'suffix_meanings': [],
                    'vortspeco': 'unknown',
                    'nombro': 'singularo',
                    'kazo': 'nominativo',
                    'parse_success': True,
                    'fallback': True
                }
            return {
                'parse_success': False,
                'error': str(e)
            }

    def _lookup_word_direct(self, word: str) -> Optional[str]:
        """
        Direct vocabulary lookup with stemming (fallback method).

        Args:
            word: Word or root to look up

        Returns:
            Definition if found
        """
        # Normalize
        word_lower = word.lower().strip()

        # Direct lookup
        if word_lower in self.ROOT_DEFINITIONS:
            return self.ROOT_DEFINITIONS[word_lower]

        # Try without endings (basic stemming)
        for ending in ['oj', 'ojn', 'on', 'o', 'aj', 'ajn', 'an', 'a', 'e',
                       'as', 'is', 'os', 'us', 'u', 'i', 'n', 'j']:
            if word_lower.endswith(ending):
                stem = word_lower[:-len(ending)]
                if stem in self.ROOT_DEFINITIONS:
                    return self.ROOT_DEFINITIONS[stem]

        return None

    def _format_definition(self, word: str, analysis: Dict[str, Any]) -> str:
        """
        Format word definition from analysis.

        Args:
            word: Original word
            analysis: Morphological analysis

        Returns:
            Formatted definition string
        """
        if not analysis.get('parse_success'):
            return f"Mi ne povas analizi la vorton '{word}'."

        parts = []

        # Word and part of speech
        vortspeco = analysis.get('vortspeco', 'unknown')
        vortspeco_map = {
            'substantivo': 'substantivo (noun)',
            'adjektivo': 'adjektivo (adjective)',
            'verbo': 'verbo (verb)',
            'adverbo': 'adverbo (adverb)'
        }
        vortspeco_str = vortspeco_map.get(vortspeco, vortspeco)

        if vortspeco != 'unknown':
            parts.append(f"'{word}' estas {vortspeco_str}.")
        else:
            parts.append(f"'{word}':")

        # Root meaning
        radiko = analysis.get('radiko', '')
        radiko_meaning = analysis.get('radiko_meaning', '')
        if radiko:
            parts.append(f"Radiko: '{radiko}' = {radiko_meaning}")

        # Prefix
        prefikso = analysis.get('prefikso')
        if prefikso:
            parts.append(f"Prefikso: {prefikso}-")

        # Suffixes
        suffix_meanings = analysis.get('suffix_meanings', [])
        if suffix_meanings:
            suffix_strs = [f"-{suf} ({meaning})" for suf, meaning in suffix_meanings]
            parts.append(f"Sufiksoj: {', '.join(suffix_strs)}")

        # Grammatical info
        nombro = analysis.get('nombro', '')
        kazo = analysis.get('kazo', '')
        if nombro == 'pluralo':
            parts.append("Numero: pluralo (plural)")
        if kazo == 'akuzativo':
            parts.append("Kazo: akuzativo (accusative)")

        return " ".join(parts)


# Export
__all__ = ['DictionaryExpert']


# Factory function for consistency with other experts
def create_dictionary_expert() -> DictionaryExpert:
    """Create and return a DictionaryExpert instance"""
    return DictionaryExpert()


if __name__ == "__main__":
    # Test dictionary expert
    print("Testing Dictionary Expert")
    print("=" * 80)

    expert = create_dictionary_expert()
    print(f"\n{expert}\n")
    print(f"Vocabulary size: {len(expert.ROOT_DEFINITIONS)} roots\n")

    # Test queries
    test_cases = [
        {
            'ast': {
                'tipo': 'frazo',
                'subjekto': {'tipo': 'vorto', 'radiko': 'kio'},
                'verbo': {'tipo': 'vorto', 'radiko': 'est'},
                'aliaj': [{'tipo': 'vorto', 'radiko': 'hund', 'plena_vorto': 'hundo'}]
            },
            'query': "Kio estas 'hundo'?"
        },
        {
            'ast': {
                'tipo': 'frazo',
                'verbo': {'tipo': 'vorto', 'radiko': 'difin'},
                'objekto': {'tipo': 'vorto', 'radiko': 'kat', 'plena_vorto': 'kato'}
            },
            'query': "Difinu 'kato'"
        },
        {
            'ast': {
                'tipo': 'frazo',
                'verbo': {'tipo': 'vorto', 'radiko': 'signif'},
                'aliaj': [{'tipo': 'vorto', 'radiko': 'bel', 'plena_vorto': 'bela', 'vortspeco': 'adjektivo'}]
            },
            'query': "Kio signifas 'bela'?"
        }
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"Test {i}: {test['query']}")
        ast = test['ast']

        can_handle = expert.can_handle(ast)
        confidence = expert.estimate_confidence(ast)
        print(f"  Can handle: {can_handle}")
        print(f"  Confidence: {confidence:.2f}")

        if can_handle:
            result = expert.execute(ast)
            print(f"  Answer: {result['answer']}")
            print(f"  Word analyzed: {result.get('word', 'N/A')}")

        print()

    print("✅ Dictionary Expert test complete!")
