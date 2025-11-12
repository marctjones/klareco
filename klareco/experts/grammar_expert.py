"""
Grammar Tool Expert - Explains grammatical structure of sentences.

This expert can:
- Explain the grammatical structure of sentences from AST
- Identify parts of speech, case, number, tense
- Explain grammar rules being applied
- Help users learn Esperanto grammar

Examples:
- "Eksplik la gramatikon de 'La hundo vidas la katon'"
  → Explains subject (hundo), verb (vidas), object (katon), accusative case
- "Kio estas la vortspeco de 'rapide'?"
  → Identifies as adverb (adverbo)
"""

from typing import Dict, Any, List
from .base import Expert


class GrammarExpert(Expert):
    """
    Expert for explaining grammatical structure.

    Uses the detailed AST to provide precise grammatical analysis.
    This is pure symbolic processing - no ML needed!
    """

    # Esperanto grammar terms
    PART_OF_SPEECH_EO = {
        'substantivo': 'substantivo (noun)',
        'adjektivo': 'adjektivo (adjective)',
        'verbo': 'verbo (verb)',
        'adverbo': 'adverbo (adverb)',
        'prepozicio': 'prepozicio (preposition)',
        'konjunkcio': 'konjunkcio (conjunction)',
        'pronomo': 'pronomo (pronoun)',
        'korelativo': 'korelativo (correlative)',
        'partiklo': 'partiklo (particle)',
        'interjection': 'interjekcio (interjection)',
    }

    CASE_EO = {
        'nominativo': 'nominativo (subject/nominative)',
        'akuzativo': 'akuzativo (object/accusative)',
    }

    NUMBER_EO = {
        'singularo': 'singularo (singular)',
        'pluralo': 'pluralo (plural)',
    }

    TENSE_EO = {
        'prezenco': 'prezenco (present tense)',
        'pasinteco': 'pasinteco (past tense)',
        'futuro': 'futuro (future tense)',
    }

    MOOD_EO = {
        'infinitivo': 'infinitivo (infinitive)',
        'vola': 'vola (volitive/imperative)',
        'kondiĉa': 'kondiĉa (conditional)',
    }

    # Grammar keywords
    GRAMMAR_KEYWORDS = [
        'gramatik',  # grammar
        'eksplik',   # explain
        'analiz',    # analyze
        'vortspec',  # part of speech
        'kazo',      # case
        'nombro',    # number
        'tempo',     # tense
        'modo',      # mood
        'strukt',    # structure
    ]

    def __init__(self):
        """Initialize Grammar Expert."""
        super().__init__("Grammar Tool Expert")

    def can_handle(self, ast: Dict[str, Any]) -> bool:
        """
        Check if this is a grammar question.

        Looks for:
        - Grammar-related keywords
        - Questions about sentence structure
        - Requests for grammatical analysis

        Args:
            ast: Parsed query AST

        Returns:
            True if this appears to be a grammar query
        """
        if not ast or ast.get('tipo') != 'frazo':
            return False

        words = self._extract_all_words(ast)
        text = ' '.join(words).lower()

        # Check for grammar keywords
        has_grammar_keyword = any(
            keyword in text
            for keyword in self.GRAMMAR_KEYWORDS
        )

        # Check for meta-linguistic questions
        has_metalinguistic = any(word in text for word in ['kio estas', 'kiu estas', 'kiaspeca'])

        return has_grammar_keyword or has_metalinguistic

    def estimate_confidence(self, ast: Dict[str, Any]) -> float:
        """
        Estimate confidence in handling this query.

        High confidence if:
        - Explicit grammar keywords present
        - Metalinguistic questions about language structure

        Args:
            ast: Parsed query AST

        Returns:
            Confidence score 0.0-1.0
        """
        if not self.can_handle(ast):
            return 0.0

        words = self._extract_all_words(ast)
        text = ' '.join(words).lower()

        # Very high confidence: explicit grammar requests
        if 'gramatik' in text or 'eksplik' in text or 'analiz' in text:
            return 0.95

        # High confidence: asks about word properties
        if 'vortspec' in text or 'kazo' in text or 'tempo' in text:
            return 0.85

        # Medium confidence: general structure questions
        return 0.65

    def execute(self, ast: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute grammatical analysis.

        Args:
            ast: Parsed query AST

        Returns:
            Response with grammatical explanation
        """
        try:
            # Analyze the sentence structure
            analysis = self._analyze_structure(ast)

            # Format explanation in Esperanto
            explanation = self._format_explanation(analysis)

            return {
                'answer': explanation,
                'analysis': analysis,
                'confidence': 0.95,
                'expert': self.name,
                'explanation': "Gramatika analizo de la frazo"
            }

        except Exception as e:
            return {
                'answer': f"Eraro dum gramatika analizo: {str(e)}",
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

            # Recursively extract from all fields
            for value in ast.values():
                if isinstance(value, (dict, list)):
                    words.extend(self._extract_all_words(value))

        elif isinstance(ast, list):
            for item in ast:
                words.extend(self._extract_all_words(item))

        return words

    def _analyze_structure(self, ast: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the grammatical structure of the sentence.

        Args:
            ast: Sentence AST

        Returns:
            Analysis dictionary with subject, verb, object, etc.
        """
        analysis = {
            'sentence_type': 'frazo',
            'components': []
        }

        if ast.get('tipo') != 'frazo':
            return analysis

        # Analyze subject
        if 'subjekto' in ast and ast['subjekto']:
            subject_info = self._analyze_noun_phrase(ast['subjekto'], 'subjekto')
            if subject_info:
                analysis['components'].append(subject_info)

        # Analyze verb
        if 'verbo' in ast and ast['verbo']:
            verb_info = self._analyze_verb(ast['verbo'])
            if verb_info:
                analysis['components'].append(verb_info)

        # Analyze object
        if 'objekto' in ast and ast['objekto']:
            object_info = self._analyze_noun_phrase(ast['objekto'], 'objekto')
            if object_info:
                analysis['components'].append(object_info)

        # Analyze other elements
        if 'aliaj' in ast and ast['aliaj']:
            for word in ast['aliaj']:
                if isinstance(word, dict) and word.get('tipo') == 'vorto':
                    other_info = self._analyze_word(word, 'alia elemento')
                    if other_info:
                        analysis['components'].append(other_info)

        return analysis

    def _analyze_noun_phrase(self, phrase: Dict[str, Any], role: str) -> Dict[str, Any]:
        """Analyze a noun phrase (vortgrupo)."""
        if not phrase:
            return None

        info = {
            'role': role,
            'type': 'vortgrupo'
        }

        # Get the head noun
        if 'kerno' in phrase and phrase['kerno']:
            kernel = phrase['kerno']
            if isinstance(kernel, dict):
                word = kernel.get('plena_vorto', '')
                pos = kernel.get('vortspeco', '')
                case = kernel.get('kazo', 'nominativo')
                number = kernel.get('nombro', 'singularo')

                info['word'] = word
                info['part_of_speech'] = self.PART_OF_SPEECH_EO.get(pos, pos)
                info['case'] = self.CASE_EO.get(case, case)
                info['number'] = self.NUMBER_EO.get(number, number)

        # Get modifiers
        if 'priskriboj' in phrase and phrase['priskriboj']:
            info['modifiers'] = [
                m.get('plena_vorto', '')
                for m in phrase['priskriboj']
                if isinstance(m, dict)
            ]

        return info

    def _analyze_verb(self, verb: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a verb."""
        if not verb or not isinstance(verb, dict):
            return None

        info = {
            'role': 'verbo',
            'type': 'vorto',
            'word': verb.get('plena_vorto', ''),
            'part_of_speech': 'verbo (verb)',
        }

        # Tense
        if 'tempo' in verb:
            info['tense'] = self.TENSE_EO.get(verb['tempo'], verb['tempo'])

        # Mood
        if 'modo' in verb:
            info['mood'] = self.MOOD_EO.get(verb['modo'], verb['modo'])

        return info

    def _analyze_word(self, word: Dict[str, Any], role: str) -> Dict[str, Any]:
        """Analyze a single word."""
        if not word or not isinstance(word, dict):
            return None

        info = {
            'role': role,
            'type': 'vorto',
            'word': word.get('plena_vorto', ''),
        }

        if 'vortspeco' in word:
            pos = word['vortspeco']
            info['part_of_speech'] = self.PART_OF_SPEECH_EO.get(pos, pos)

        return info

    def _format_explanation(self, analysis: Dict[str, Any]) -> str:
        """Format the grammatical analysis in Esperanto."""
        if not analysis.get('components'):
            return "Mi ne povis analizi la gramatikon de tiu frazo."

        lines = ["Gramatika analizo:"]

        for component in analysis['components']:
            role = component.get('role', '').capitalize()
            word = component.get('word', '?')

            parts = [f"- {role}: '{word}'"]

            if 'part_of_speech' in component:
                parts.append(f"vortspeco = {component['part_of_speech']}")

            if 'case' in component:
                parts.append(f"kazo = {component['case']}")

            if 'number' in component:
                parts.append(f"nombro = {component['number']}")

            if 'tense' in component:
                parts.append(f"tempo = {component['tense']}")

            if 'mood' in component:
                parts.append(f"modo = {component['mood']}")

            if 'modifiers' in component and component['modifiers']:
                modifiers_str = ', '.join(component['modifiers'])
                parts.append(f"priskriboj = [{modifiers_str}]")

            lines.append(' | '.join(parts))

        return '\n'.join(lines)


# Export
__all__ = ['GrammarExpert']
