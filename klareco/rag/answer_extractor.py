#!/usr/bin/env python3
"""
AST-Based Answer Extraction (Deterministic)

Extracts precise answers from retrieved documents by matching AST patterns.

This is the PRIMARY answer extraction method for Klareco's RAG system.
It uses deterministic pattern matching on Abstract Syntax Trees to extract
grammatically and semantically correct answers.

Architecture:
1. Parse question AST to determine question type (WHO, WHAT, WHERE, WHEN, HOW_MANY)
2. Parse document AST
3. Match patterns based on question type
4. Extract answer as complete vortgrupo (not just root)
5. Return structured answer with confidence and explanation

Question Type Detection:
- WHO (kiu): Extract person/agent (subject or object with animate semantics)
- WHAT (kio): Extract thing/concept (subject, object, or predicate)
- WHERE (kie): Extract location (aliaj with location semantics)
- WHEN (kiam): Extract time (aliaj with temporal semantics)
- HOW_MANY (kiom): Extract quantity (numeric modifier)
- WHICH (kiu + noun): Extract specific instance from category
- WHY (kial): Extract reason/cause (aliaj with causal semantics)
- HOW (kiel): Extract manner (aliaj with manner semantics)

Example:
    Query: "Kiu fondis Esperanton?"
    Document: "Zamenhof fondis Esperanton en 1887."

    Question type: WHO (kiu)
    Match pattern: subject of "fond"
    Extract: "Zamenhof" (complete vortgrupo)

    Answer: {
        'text': 'Zamenhof',
        'confidence': 0.95,
        'method': 'ast_pattern_match',
        'explanation': 'Subject of verb "fond" matching query pattern',
        'ast': {...}
    }
"""

from typing import Dict, Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


class ASTAnswerExtractor:
    """
    Deterministic answer extraction using AST pattern matching.

    This is the first-tier extraction method in the cascading fallback system.
    """

    # Question type mapping based on correlative suffix
    QUESTION_TYPES = {
        'u': 'WHO',      # kiu (who/which person)
        'o': 'WHAT',     # kio (what thing)
        'e': 'WHERE',    # kie (where location)
        'am': 'WHEN',    # kiam (when time)
        'om': 'HOW_MANY',# kiom (how many/much)
        'al': 'WHY',     # kial (why reason)
        'el': 'HOW',     # kiel (how manner)
        'a': 'WHICH',    # kia (which kind)
        'es': 'WHOSE',   # kies (whose possession)
    }

    # Ordinals to skip when extracting WHAT answers (predicate extraction)
    ORDINALS = {
        'unua', 'dua', 'tria', 'kvara', 'kvina', 'sesa', 'sepa', 'oka', 'naŭa', 'deka',
        'unue', 'due', 'trie', 'kvare', 'kvine', 'sese', 'sepe', 'oke', 'naŭe', 'deke',
        # Cardinal numbers (for HOW_MANY validation)
        'unu', 'du', 'tri', 'kvar', 'kvin', 'ses', 'sep', 'ok', 'naŭ', 'dek',
    }

    # Pronouns to reject for WHO questions
    PRONOUNS = {
        'mi', 'vi', 'li', 'ŝi', 'ĝi', 'ni', 'ili',
        'si', 'oni', 'mem',
    }

    def __init__(self):
        """Initialize answer extractor."""
        pass

    def extract_answer(
        self,
        query_ast: Dict,
        doc_ast: Dict,
        doc_text: str,
    ) -> Optional[Dict]:
        """
        Extract answer from document AST based on query pattern.

        Args:
            query_ast: Parsed query AST
            doc_ast: Parsed document AST
            doc_text: Original document text

        Returns:
            {
                'text': str,           # Answer text
                'confidence': float,   # [0-1] confidence score
                'method': str,         # 'ast_pattern_match'
                'explanation': str,    # Why this was extracted
                'ast': Dict,          # Full AST of answer
                'span': Tuple[int, int]  # Character offsets in doc_text (if available)
            }
            or None if no answer found
        """
        # Detect question type
        question_type = self._detect_question_type(query_ast)
        if not question_type:
            logger.debug("Could not detect question type")
            return None

        logger.debug(f"Question type: {question_type}")

        # Extract answer based on question type
        answer = None
        if question_type == 'WHO':
            answer = self._extract_who(query_ast, doc_ast, doc_text)
        elif question_type == 'WHAT':
            answer = self._extract_what(query_ast, doc_ast, doc_text)
        elif question_type == 'WHERE':
            answer = self._extract_where(query_ast, doc_ast, doc_text)
        elif question_type == 'WHEN':
            answer = self._extract_when(query_ast, doc_ast, doc_text)
        elif question_type == 'HOW_MANY':
            answer = self._extract_how_many(query_ast, doc_ast, doc_text)
        elif question_type == 'WHY':
            answer = self._extract_why(query_ast, doc_ast, doc_text)
        elif question_type == 'HOW':
            answer = self._extract_how(query_ast, doc_ast, doc_text)
        elif question_type == 'WHICH':
            answer = self._extract_which(query_ast, doc_ast, doc_text)
        elif question_type == 'WHOSE':
            answer = self._extract_whose(query_ast, doc_ast, doc_text)
        else:
            logger.warning(f"Unsupported question type: {question_type}")
            return None

        # Validate answer before returning
        if answer:
            answer_text = answer.get('text', '')
            answer_ast = answer.get('ast')

            if not self._validate_answer(question_type, answer_text, answer_ast):
                logger.debug(f"Answer validation failed for '{answer_text}'")
                return None

        return answer

    def _detect_question_type(self, query_ast: Dict) -> Optional[str]:
        """
        Detect question type from query AST.

        Looks for correlative (kiu, kio, kie, etc.) in subject, object, or aliaj.

        Args:
            query_ast: Parsed query AST

        Returns:
            Question type string (WHO, WHAT, WHERE, etc.) or None
        """
        # Check if it's marked as a question
        if query_ast.get('fraztipo') != 'demando':
            return None

        # Check subject for correlative
        subjekto = query_ast.get('subjekto')
        if subjekto:
            q_type = self._check_correlative(subjekto)
            if q_type:
                return q_type

        # Check object for correlative (e.g., "Kion X kreis?")
        objekto = query_ast.get('objekto')
        if objekto:
            q_type = self._check_correlative(objekto)
            if q_type:
                return q_type

        # Check aliaj (question words can appear in modifiers)
        for modifier in query_ast.get('aliaj', []):
            q_type = self._check_correlative(modifier)
            if q_type:
                return q_type

        return None

    def _check_correlative(self, node: Dict) -> Optional[str]:
        """
        Check if node contains correlative and return question type.

        Args:
            node: AST node (vortgrupo or vorto)

        Returns:
            Question type or None
        """
        # Handle vortgrupo - check kerno
        if node.get('tipo') == 'vortgrupo':
            kerno = node.get('kerno')
            if kerno:
                return self._check_correlative(kerno)

        # Handle vorto - check if it's a correlative
        if node.get('tipo') == 'vorto':
            if node.get('vortspeco') == 'korelativo':
                # Get correlative suffix (u, o, e, am, om, etc.)
                suffix = node.get('korelativo_sufikso', '')
                return self.QUESTION_TYPES.get(suffix)

        return None

    def _extract_who(
        self,
        query_ast: Dict,
        doc_ast: Dict,
        doc_text: str
    ) -> Optional[Dict]:
        """
        Extract WHO answer (person/agent).

        Strategy:
        1. If query verb matches doc verb → extract subject
        2. If subject is inanimate → extract object (passive construction)
        3. Look for person indicators: names, -ul suffix, -ist suffix

        Args:
            query_ast: Query AST
            doc_ast: Document AST
            doc_text: Document text

        Returns:
            Answer dict or None
        """
        query_verb = self._get_verb_root(query_ast)
        doc_verb = self._get_verb_root(doc_ast)

        # Check if verbs match (or are synonyms - future enhancement)
        if not query_verb or not doc_verb:
            return None

        verb_match = (query_verb == doc_verb)

        if not verb_match:
            # Try relaxed matching (check if roots share prefix)
            # This handles cases like "fond" vs "fondi"
            if query_verb[:4] == doc_verb[:4] and len(query_verb) >= 4:
                verb_match = True

        if not verb_match:
            return None

        # Extract subject as answer candidate
        subjekto = doc_ast.get('subjekto')
        if subjekto:
            answer_text = self._vortgrupo_to_text(subjekto)
            if answer_text:
                # Check if subject looks like a person
                is_person = self._is_person(subjekto)

                confidence = 0.9 if is_person else 0.7

                return {
                    'text': answer_text,
                    'confidence': confidence,
                    'method': 'ast_pattern_match',
                    'explanation': f'Subject of verb "{doc_verb}" matching query verb "{query_verb}"',
                    'ast': subjekto,
                }

        return None

    def _extract_what(
        self,
        query_ast: Dict,
        doc_ast: Dict,
        doc_text: str
    ) -> Optional[Dict]:
        """
        Extract WHAT answer (thing/concept).

        Strategy:
        1. If query has object → extract doc object
        2. If query asks "Kio estas X?" → extract predicate/definition (in aliaj)
        3. Otherwise → extract subject

        Args:
            query_ast: Query AST
            doc_ast: Document AST
            doc_text: Document text

        Returns:
            Answer dict or None
        """
        query_verb = self._get_verb_root(query_ast)
        doc_verb = self._get_verb_root(doc_ast)

        # Check for "estas" questions (definitions)
        if query_verb == 'est' and doc_verb == 'est':
            # Extract predicate - in Esperanto, predicate nominative is in aliaj
            # Strategy: Prefer substantives over adjectives, skip ordinals
            aliaj = doc_ast.get('aliaj', [])

            # Collect all candidates with priority
            candidates = []
            for modifier in aliaj:
                if modifier.get('tipo') == 'vorto':
                    vortspeco = modifier.get('vortspeco')
                    radiko = modifier.get('radiko', '').lower()

                    # Skip ordinals (unua, dua, etc.)
                    if radiko in self.ORDINALS:
                        continue

                    # Collect substantives (high priority)
                    if vortspeco == 'substantivo':
                        answer_text = self._vortgrupo_to_text(modifier)
                        if answer_text:
                            candidates.append({
                                'text': answer_text,
                                'ast': modifier,
                                'priority': 2,  # High priority
                                'confidence': 0.9,
                                'explanation': 'Substantive predicate after "estas"',
                            })

                    # Collect adjectives (low priority)
                    elif vortspeco == 'adjektivo':
                        answer_text = self._vortgrupo_to_text(modifier)
                        if answer_text:
                            candidates.append({
                                'text': answer_text,
                                'ast': modifier,
                                'priority': 1,  # Low priority
                                'confidence': 0.75,
                                'explanation': 'Adjective predicate after "estas"',
                            })

            # Return best candidate (prefer substantives)
            if candidates:
                best = max(candidates, key=lambda x: x['priority'])
                return {
                    'text': best['text'],
                    'confidence': best['confidence'],
                    'method': 'ast_pattern_match',
                    'explanation': best['explanation'],
                    'ast': best['ast'],
                }

            # Fallback: try object
            objekto = doc_ast.get('objekto')
            if objekto:
                answer_text = self._vortgrupo_to_text(objekto)
                if answer_text:
                    return {
                        'text': answer_text,
                        'confidence': 0.75,
                        'method': 'ast_pattern_match',
                        'explanation': 'Object after "estas"',
                        'ast': objekto,
                    }

        # Check if verbs match
        if query_verb and doc_verb and query_verb == doc_verb:
            # If query has object placeholder (kio) → extract doc object
            query_obj = query_ast.get('objekto')
            if query_obj and self._is_correlative(query_obj, 'kio'):
                objekto = doc_ast.get('objekto')
                if objekto:
                    answer_text = self._vortgrupo_to_text(objekto)
                    if answer_text:
                        return {
                            'text': answer_text,
                            'confidence': 0.9,
                            'method': 'ast_pattern_match',
                            'explanation': f'Object of verb "{doc_verb}"',
                            'ast': objekto,
                        }

            # Otherwise extract subject
            subjekto = doc_ast.get('subjekto')
            if subjekto:
                answer_text = self._vortgrupo_to_text(subjekto)
                if answer_text:
                    return {
                        'text': answer_text,
                        'confidence': 0.8,
                        'method': 'ast_pattern_match',
                        'explanation': f'Subject of verb "{doc_verb}"',
                        'ast': subjekto,
                    }

        return None

    def _extract_where(
        self,
        query_ast: Dict,
        doc_ast: Dict,
        doc_text: str
    ) -> Optional[Dict]:
        """
        Extract WHERE answer (location).

        Strategy:
        1. Look for location prepositions (en, sur, apud, etc.) followed by object
        2. Look for location suffixes (-ej = place for)
        3. Look for place names (proper nouns)

        Args:
            query_ast: Query AST
            doc_ast: Document AST
            doc_text: Document text

        Returns:
            Answer dict or None
        """
        # Location prepositions
        LOCATION_PREPS = {'en', 'sur', 'apud', 'ĉe', 'antaŭ', 'post', 'sub',
                          'super', 'inter', 'ekster', 'ĉirkaŭ', 'trans'}

        # Check aliaj for location modifiers
        # In parser output, preposition and object are separate consecutive items
        aliaj = doc_ast.get('aliaj', [])
        for i, modifier in enumerate(aliaj):
            if modifier.get('tipo') == 'vorto':
                # Check if it's a location preposition
                if modifier.get('vortspeco') == 'prepozicio':
                    radiko = modifier.get('radiko')
                    if radiko in LOCATION_PREPS:
                        # Get next item (the object of preposition)
                        if i + 1 < len(aliaj):
                            next_item = aliaj[i + 1]
                            answer_text = self._vortgrupo_to_text(next_item)
                            if answer_text:
                                return {
                                    'text': answer_text,
                                    'confidence': 0.95,
                                    'method': 'ast_pattern_match',
                                    'explanation': f'Location after preposition "{radiko}"',
                                    'ast': next_item,
                                }

        # Check for -ej suffix (place for)
        for key in ['subjekto', 'objekto']:
            node = doc_ast.get(key)
            if node and 'ej' in self._get_suffixes(node):
                answer_text = self._vortgrupo_to_text(node)
                if answer_text:
                    return {
                        'text': answer_text,
                        'confidence': 0.85,
                        'method': 'ast_pattern_match',
                        'explanation': 'Word with -ej suffix (place)',
                        'ast': node,
                    }

        return None

    def _extract_when(
        self,
        query_ast: Dict,
        doc_ast: Dict,
        doc_text: str
    ) -> Optional[Dict]:
        """
        Extract WHEN answer (time).

        Strategy:
        1. Look for time prepositions (en, dum, post, antaŭ)
        2. Look for year/date patterns (1887, januaro, etc.)
        3. Look for time adverbs (hieraŭ, hodiaŭ, morgaŭ)

        Args:
            query_ast: Query AST
            doc_ast: Document AST
            doc_text: Document text

        Returns:
            Answer dict or None
        """
        # Time prepositions
        TIME_PREPS = {'en', 'dum', 'post', 'antaŭ', 'ekde', 'ĝis'}

        # Check aliaj for time modifiers
        # Preposition and object are consecutive items in aliaj
        aliaj = doc_ast.get('aliaj', [])
        for i, modifier in enumerate(aliaj):
            if modifier.get('tipo') == 'vorto':
                # Check for time preposition
                if modifier.get('vortspeco') == 'prepozicio':
                    radiko = modifier.get('radiko')
                    if radiko in TIME_PREPS:
                        # Get next item (object of preposition)
                        if i + 1 < len(aliaj):
                            next_item = aliaj[i + 1]
                            answer_text = self._vortgrupo_to_text(next_item)
                            # Check if it looks like time
                            if answer_text and self._looks_like_time(answer_text):
                                return {
                                    'text': answer_text,
                                    'confidence': 0.95,
                                    'method': 'ast_pattern_match',
                                    'explanation': f'Time expression after "{radiko}"',
                                    'ast': next_item,
                                }

                # Time adverbs (hieraŭ, hodiaŭ, etc.)
                # Note: Parser may classify these as 'partiklo' or 'adverbo'
                vortspeco = modifier.get('vortspeco')
                if vortspeco in ['adverbo', 'partiklo']:
                    radiko = modifier.get('radiko', '')
                    if radiko in {'hieraŭ', 'hodiaŭ', 'morgaŭ', 'nun', 'tiam'}:
                        answer_text = self._vortgrupo_to_text(modifier)
                        if answer_text:
                            return {
                                'text': answer_text,
                                'confidence': 0.9,
                                'method': 'ast_pattern_match',
                                'explanation': 'Time adverb',
                                'ast': modifier,
                            }

        return None

    def _extract_how_many(
        self,
        query_ast: Dict,
        doc_ast: Dict,
        doc_text: str
    ) -> Optional[Dict]:
        """
        Extract HOW_MANY answer (quantity).

        Strategy:
        1. Look for numbers in document
        2. Look for quantity words (multe, malmulte, etc.)
        3. Extract numeric modifiers

        Args:
            query_ast: Query AST
            doc_ast: Document AST
            doc_text: Document text

        Returns:
            Answer dict or None
        """
        # Check priskriboj (modifiers) for numbers
        for key in ['subjekto', 'objekto']:
            node = doc_ast.get(key)
            if node and node.get('tipo') == 'vortgrupo':
                for priskribo in node.get('priskriboj', []):
                    if priskribo.get('tipo') == 'vorto':
                        radiko = priskribo.get('radiko', '')
                        # Check if it's a number
                        if radiko.isdigit() or self._is_number_word(radiko):
                            answer_text = self._vortgrupo_to_text(priskribo)
                            if answer_text:
                                return {
                                    'text': answer_text,
                                    'confidence': 0.95,
                                    'method': 'ast_pattern_match',
                                    'explanation': 'Numeric modifier',
                                    'ast': priskribo,
                                }

        # Check aliaj for standalone numbers
        for modifier in doc_ast.get('aliaj', []):
            if modifier.get('tipo') == 'vorto':
                radiko = modifier.get('radiko', '')
                if radiko.isdigit() or self._is_number_word(radiko):
                    answer_text = self._vortgrupo_to_text(modifier)
                    if answer_text:
                        return {
                            'text': answer_text,
                            'confidence': 0.9,
                            'method': 'ast_pattern_match',
                            'explanation': 'Number in sentence',
                            'ast': modifier,
                        }

        return None

    def _extract_why(
        self,
        query_ast: Dict,
        doc_ast: Dict,
        doc_text: str
    ) -> Optional[Dict]:
        """
        Extract WHY answer (reason/cause).

        Strategy:
        1. Look for causal prepositions (pro, ĉar)
        2. Look for purpose constructions (por + infinitive)
        3. Extract clause after causal marker

        Args:
            query_ast: Query AST
            doc_ast: Document AST
            doc_text: Document text

        Returns:
            Answer dict or None
        """
        # Causal markers
        CAUSAL_MARKERS = {'pro', 'ĉar', 'por', 'tial'}

        # Check aliaj for causal phrases
        # Preposition and object are consecutive items
        aliaj = doc_ast.get('aliaj', [])
        for i, modifier in enumerate(aliaj):
            if modifier.get('tipo') == 'vorto':
                if modifier.get('vortspeco') == 'prepozicio':
                    radiko = modifier.get('radiko')
                    if radiko in CAUSAL_MARKERS:
                        # Get next item (object of preposition)
                        if i + 1 < len(aliaj):
                            next_item = aliaj[i + 1]
                            answer_text = self._vortgrupo_to_text(next_item)
                            if answer_text:
                                return {
                                    'text': answer_text,
                                    'confidence': 0.85,
                                    'method': 'ast_pattern_match',
                                    'explanation': f'Reason/cause after "{radiko}"',
                                    'ast': next_item,
                                }

        return None

    def _extract_how(
        self,
        query_ast: Dict,
        doc_ast: Dict,
        doc_text: str
    ) -> Optional[Dict]:
        """
        Extract HOW answer (manner).

        Strategy:
        1. Look for manner adverbs (ending in -e)
        2. Look for manner prepositions (per, kun)
        3. Extract adverbial phrases

        Args:
            query_ast: Query AST
            doc_ast: Document AST
            doc_text: Document text

        Returns:
            Answer dict or None
        """
        # Manner prepositions
        MANNER_PREPS = {'per', 'kun', 'sen', 'laŭ'}

        # Check aliaj for manner modifiers
        # Preposition and object are consecutive items
        aliaj = doc_ast.get('aliaj', [])
        for i, modifier in enumerate(aliaj):
            if modifier.get('tipo') == 'vorto':
                # Prepositional phrases
                if modifier.get('vortspeco') == 'prepozicio':
                    radiko = modifier.get('radiko')
                    if radiko in MANNER_PREPS:
                        # Get next item (object of preposition)
                        if i + 1 < len(aliaj):
                            next_item = aliaj[i + 1]
                            answer_text = self._vortgrupo_to_text(next_item)
                            if answer_text:
                                return {
                                    'text': answer_text,
                                    'confidence': 0.85,
                                    'method': 'ast_pattern_match',
                                    'explanation': f'Manner expression with "{radiko}"',
                                    'ast': next_item,
                                }

                # Adverbs (ending in -e)
                if modifier.get('vortspeco') == 'adverbo':
                    answer_text = self._vortgrupo_to_text(modifier)
                    if answer_text:
                        return {
                            'text': answer_text,
                            'confidence': 0.8,
                            'method': 'ast_pattern_match',
                            'explanation': 'Manner adverb',
                            'ast': modifier,
                        }

        return None

    def _extract_which(
        self,
        query_ast: Dict,
        doc_ast: Dict,
        doc_text: str
    ) -> Optional[Dict]:
        """
        Extract WHICH answer (specific instance from category).

        Similar to WHO/WHAT but expects a specific selection.

        Args:
            query_ast: Query AST
            doc_ast: Document AST
            doc_text: Document text

        Returns:
            Answer dict or None
        """
        # For now, treat like WHO extraction
        return self._extract_who(query_ast, doc_ast, doc_text)

    def _extract_whose(
        self,
        query_ast: Dict,
        doc_ast: Dict,
        doc_text: str
    ) -> Optional[Dict]:
        """
        Extract WHOSE answer (possession).

        Strategy:
        1. Look for possessive constructions (de + possessor)
        2. Look for possessive adjectives (mia, via, lia, etc.)

        Args:
            query_ast: Query AST
            doc_ast: Document AST
            doc_text: Document text

        Returns:
            Answer dict or None
        """
        # Check for "de" prepositional phrases (possession)
        # Preposition and object are consecutive items
        aliaj = doc_ast.get('aliaj', [])
        for i, modifier in enumerate(aliaj):
            if modifier.get('tipo') == 'vorto':
                if modifier.get('vortspeco') == 'prepozicio':
                    if modifier.get('radiko') == 'de':
                        # Get next item (possessor)
                        if i + 1 < len(aliaj):
                            next_item = aliaj[i + 1]
                            answer_text = self._vortgrupo_to_text(next_item)
                            if answer_text:
                                return {
                                    'text': answer_text,
                                    'confidence': 0.9,
                                    'method': 'ast_pattern_match',
                                    'explanation': 'Possessor after "de"',
                                    'ast': next_item,
                                }

        return None

    # -------------------------------------------------------------------------
    # Answer Validation
    # -------------------------------------------------------------------------

    def _validate_answer(
        self,
        question_type: str,
        answer_text: str,
        answer_ast: Optional[Dict] = None
    ) -> bool:
        """
        Validate extracted answer matches expected answer type.

        This is a deterministic sanity check to reject clearly wrong extractions.

        Args:
            question_type: Question type (WHO, WHAT, WHERE, etc.)
            answer_text: Extracted answer text
            answer_ast: Optional AST of answer

        Returns:
            True if answer is valid, False if clearly wrong
        """
        answer_lower = answer_text.lower()

        # WHO questions should not return pronouns
        if question_type == 'WHO':
            # Check if answer is a pronoun
            if answer_lower in self.PRONOUNS:
                logger.debug(f"Rejecting pronoun '{answer_text}' for WHO question")
                return False

            # Check for generic words that aren't person names
            generic_non_persons = {'komparo', 'grupo', 'aro', 'afero', 'io'}
            if answer_lower in generic_non_persons:
                logger.debug(f"Rejecting generic non-person '{answer_text}' for WHO question")
                return False

        # HOW_MANY questions should contain numbers
        elif question_type == 'HOW_MANY':
            # Check if answer contains digits or number words
            has_digit = any(c.isdigit() for c in answer_text)
            is_number_word = self._is_number_word(answer_lower)

            if not (has_digit or is_number_word):
                logger.debug(f"Rejecting non-numeric '{answer_text}' for HOW_MANY question")
                return False

            # Reject index-style answers (all caps, single word)
            if answer_text.isupper() and len(answer_text.split()) == 1:
                logger.debug(f"Rejecting index entry '{answer_text}' for HOW_MANY question")
                return False

        # WHEN questions should look like time
        elif question_type == 'WHEN':
            if not self._looks_like_time(answer_text):
                logger.debug(f"Rejecting non-time '{answer_text}' for WHEN question")
                return False

        # WHAT questions should not be pronouns or ordinals
        elif question_type == 'WHAT':
            # Already handled ordinals in extraction, but double-check
            radiko = answer_ast.get('radiko', '').lower() if answer_ast else ''
            if radiko in self.ORDINALS:
                logger.debug(f"Rejecting ordinal '{answer_text}' for WHAT question")
                return False

            # Reject pronouns
            if answer_lower in self.PRONOUNS:
                logger.debug(f"Rejecting pronoun '{answer_text}' for WHAT question")
                return False

        return True

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _get_verb_root(self, ast: Dict) -> Optional[str]:
        """Extract verb root from AST."""
        verbo = ast.get('verbo')
        if verbo and verbo.get('tipo') == 'vorto':
            return verbo.get('radiko')
        return None

    def _vortgrupo_to_text(self, node: Dict) -> Optional[str]:
        """
        Convert vortgrupo AST node to text.

        Reconstructs the original text representation of a word group.

        Args:
            node: AST node (vortgrupo or vorto)

        Returns:
            Text string or None
        """
        if node.get('tipo') == 'vorto':
            return node.get('plena_vorto')

        if node.get('tipo') == 'vortgrupo':
            kerno = node.get('kerno')
            if kerno:
                # For now, just return the core word
                # TODO: Include priskriboj (modifiers)
                return self._vortgrupo_to_text(kerno)

        return None

    def _is_person(self, node: Dict) -> bool:
        """
        Check if node represents a person.

        Heuristics:
        - Has -ul suffix (person characterized by)
        - Has -ist suffix (professional)
        - Is a proper noun (starts with capital)
        - Is a correlative with 'u' (kiu)

        Args:
            node: AST node

        Returns:
            True if likely a person
        """
        suffixes = self._get_suffixes(node)
        if 'ul' in suffixes or 'ist' in suffixes or 'in' in suffixes:
            return True

        # Check if proper noun
        text = self._vortgrupo_to_text(node)
        if text and text[0].isupper():
            return True

        # Check if correlative (kiu)
        if node.get('tipo') == 'vorto':
            if node.get('korelativo_sufikso') == 'u':
                return True

        return False

    def _get_suffixes(self, node: Dict) -> List[str]:
        """Extract list of suffixes from node."""
        if node.get('tipo') == 'vortgrupo':
            kerno = node.get('kerno')
            if kerno:
                return self._get_suffixes(kerno)

        if node.get('tipo') == 'vorto':
            return node.get('sufiksoj', [])

        return []

    def _is_correlative(self, node: Dict, radiko: str) -> bool:
        """Check if node is a specific correlative (e.g., 'kio')."""
        if node.get('tipo') == 'vortgrupo':
            kerno = node.get('kerno')
            if kerno:
                return self._is_correlative(kerno, radiko)

        if node.get('tipo') == 'vorto':
            return (node.get('vortspeco') == 'korelativo' and
                    node.get('radiko') == radiko)

        return False

    def _looks_like_time(self, text: str) -> bool:
        """
        Heuristic check if text looks like a time expression.

        Args:
            text: Text string

        Returns:
            True if looks like time
        """
        # Check for year (4 digits)
        if text.isdigit() and len(text) == 4:
            year = int(text)
            if 1000 <= year <= 2100:
                return True

        # Check for month names
        months = {'januaro', 'februaro', 'marto', 'aprilo', 'majo', 'junio',
                  'julio', 'aŭgusto', 'septembro', 'oktobro', 'novembro', 'decembro'}
        if text.lower() in months:
            return True

        # Check for time words
        time_words = {'jaro', 'monato', 'semajno', 'tago', 'horo', 'minuto'}
        for word in time_words:
            if word in text.lower():
                return True

        return False

    def _is_number_word(self, radiko: str) -> bool:
        """
        Check if root is a number word.

        Args:
            radiko: Root string

        Returns:
            True if number word
        """
        number_words = {
            'unu', 'du', 'tri', 'kvar', 'kvin', 'ses', 'sep', 'ok', 'naŭ', 'dek',
            'cent', 'mil', 'milion', 'miliard',
            'multe', 'malmulte', 'kelke', 'sufiĉe'
        }
        return radiko.lower() in number_words
