"""
Question Type Classifier for AST-Aware Retrieval.

Analyzes question ASTs to determine:
- Question type (WHO, WHAT, WHERE, WHEN, HOW, BOOLEAN)
- Target entity type (PERSON, PLACE, TIME, THING, etc.)
- Query focus (what information is being sought)

This is a DETERMINISTIC classifier - no machine learning, just rule-based
analysis of AST structure and Esperanto question words.
"""

from typing import Dict, Optional, Tuple, List
from enum import Enum


class QuestionType(Enum):
    """Types of questions we can classify."""
    WHO = "who"           # Kiu? - seeks PERSON
    WHAT = "what"         # Kio? - seeks THING/DEFINITION
    WHERE = "where"       # Kie? - seeks PLACE
    WHEN = "when"         # Kiam? - seeks TIME
    HOW = "how"           # Kiel? - seeks METHOD/MANNER
    WHY = "why"           # Kial? - seeks REASON
    WHICH = "which"       # Kiu/Kiuj? (with noun) - seeks SPECIFIC
    HOW_MANY = "how_many" # Kiom? - seeks QUANTITY
    BOOLEAN = "boolean"   # Ĉu? - seeks YES/NO
    UNKNOWN = "unknown"   # Can't classify


class EntityType(Enum):
    """Types of entities/information being sought."""
    PERSON = "person"         # Human, name, profession
    PLACE = "place"           # Location, country, city
    TIME = "time"             # Date, year, period
    THING = "thing"           # Object, concept
    DEFINITION = "definition" # What is X?
    METHOD = "method"         # How to do X
    REASON = "reason"         # Why X happened
    QUANTITY = "quantity"     # Number, amount
    BOOLEAN = "boolean"       # Yes/no answer
    UNKNOWN = "unknown"


class QuestionClassifier:
    """
    Deterministic question classifier using AST analysis.

    Uses Esperanto correlatives (ki- words) and AST structure to identify:
    - What type of question is being asked
    - What type of information is being sought
    - Where to focus the search
    """

    # Esperanto question words (correlatives)
    QUESTION_WORDS = {
        # Who - person
        'kiu': (QuestionType.WHO, EntityType.PERSON),
        'kiuj': (QuestionType.WHO, EntityType.PERSON),
        'kiun': (QuestionType.WHO, EntityType.PERSON),
        'kiujn': (QuestionType.WHO, EntityType.PERSON),

        # What - thing/definition
        'kio': (QuestionType.WHAT, EntityType.THING),
        'kion': (QuestionType.WHAT, EntityType.THING),

        # Where - place
        'kie': (QuestionType.WHERE, EntityType.PLACE),
        'kien': (QuestionType.WHERE, EntityType.PLACE),

        # When - time
        'kiam': (QuestionType.WHEN, EntityType.TIME),

        # How - method/manner
        'kiel': (QuestionType.HOW, EntityType.METHOD),
        'kiela': (QuestionType.HOW, EntityType.METHOD),
        'kiele': (QuestionType.HOW, EntityType.METHOD),

        # Why - reason
        'kial': (QuestionType.WHY, EntityType.REASON),

        # How much/many - quantity
        'kiom': (QuestionType.HOW_MANY, EntityType.QUANTITY),

        # Boolean questions
        'ĉu': (QuestionType.BOOLEAN, EntityType.BOOLEAN),
    }

    def classify(self, query: str, query_ast: Dict) -> Dict:
        """
        Classify question type from query text and AST.

        Args:
            query: Raw question text
            query_ast: Parsed AST of the question

        Returns:
            {
                'question_type': QuestionType,
                'entity_type': EntityType,
                'focus': str,  # Main focus of query (SUBJ/VERB/OBJ)
                'question_word': str,  # The ki- word found
                'target_slots': [str],  # Which slots to match (SUBJ/VERB/OBJ)
            }
        """
        # Default classification
        result = {
            'question_type': QuestionType.UNKNOWN,
            'entity_type': EntityType.UNKNOWN,
            'focus': None,
            'question_word': None,
            'target_slots': ['SUBJ', 'VERB', 'OBJ'],  # Match all by default
        }

        # Check if it's a question
        if not self._is_question(query, query_ast):
            result['question_type'] = QuestionType.UNKNOWN
            return result

        # Find question word (ki- correlative)
        question_word = self._find_question_word(query_ast)

        if question_word:
            result['question_word'] = question_word

            # Lookup question type and entity type
            if question_word.lower() in self.QUESTION_WORDS:
                q_type, e_type = self.QUESTION_WORDS[question_word.lower()]
                result['question_type'] = q_type
                result['entity_type'] = e_type

            # Determine focus based on where question word appears
            result['focus'] = self._determine_focus(query_ast, question_word)

            # Refine entity type based on context
            result['entity_type'] = self._refine_entity_type(
                result['entity_type'],
                query_ast,
                question_word
            )

            # Determine which slots to prioritize in matching
            result['target_slots'] = self._determine_target_slots(
                result['question_type'],
                result['focus']
            )

        return result

    def _is_question(self, query: str, query_ast: Dict) -> bool:
        """Check if this is actually a question."""
        # Check 1: Ends with question mark
        if query.strip().endswith('?'):
            return True

        # Check 2: AST fraztipo is 'demando'
        if query_ast.get('fraztipo') == 'demando':
            return True

        # Check 3: Contains question word
        if self._find_question_word(query_ast):
            return True

        return False

    def _find_question_word(self, ast: Dict) -> Optional[str]:
        """
        Find the question word (ki- correlative) in the AST.

        Searches through all words in the sentence structure.
        """
        # Search in subject
        if ast.get('subjekto'):
            word = self._search_node_for_question_word(ast['subjekto'])
            if word:
                return word

        # Search in verb
        if ast.get('verbo'):
            word = self._search_node_for_question_word(ast['verbo'])
            if word:
                return word

        # Search in object
        if ast.get('objekto'):
            word = self._search_node_for_question_word(ast['objekto'])
            if word:
                return word

        # Search in modifiers
        if ast.get('aliaj'):
            for modifier in ast['aliaj']:
                word = self._search_node_for_question_word(modifier)
                if word:
                    return word

        return None

    def _search_node_for_question_word(self, node: Dict) -> Optional[str]:
        """Recursively search an AST node for question words."""
        if node.get('tipo') == 'vorto':
            # Check if this word is a question word
            # Use 'radiko' field from AST (parser uses this for word roots)
            text = node.get('radiko', '').lower()
            if text in self.QUESTION_WORDS or text.startswith('ki'):
                return text

        elif node.get('tipo') == 'vortgrupo':
            # Search in word group
            if node.get('kerno'):
                word = self._search_node_for_question_word(node['kerno'])
                if word:
                    return word

            if node.get('priskriboj'):
                for modifier in node['priskriboj']:
                    word = self._search_node_for_question_word(modifier)
                    if word:
                        return word

        return None

    def _determine_focus(self, ast: Dict, question_word: str) -> str:
        """
        Determine where the question focus is (SUBJ/VERB/OBJ).

        Returns which slot the question is asking about.
        """
        # Check where the question word appears
        if ast.get('subjekto'):
            if self._node_contains_word(ast['subjekto'], question_word):
                return 'SUBJ'

        if ast.get('objekto'):
            if self._node_contains_word(ast['objekto'], question_word):
                return 'OBJ'

        if ast.get('verbo'):
            if self._node_contains_word(ast['verbo'], question_word):
                return 'VERB'

        # Default: assume asking about object (most common)
        return 'OBJ'

    def _node_contains_word(self, node: Dict, word: str) -> bool:
        """Check if a node contains a specific word."""
        if node.get('tipo') == 'vorto':
            # Use 'radiko' field from AST (parser uses this for word roots)
            return node.get('radiko', '').lower() == word.lower()

        elif node.get('tipo') == 'vortgrupo':
            if node.get('kerno'):
                if self._node_contains_word(node['kerno'], word):
                    return True

            if node.get('priskriboj'):
                for modifier in node['priskriboj']:
                    if self._node_contains_word(modifier, word):
                        return True

        return False

    def _refine_entity_type(
        self,
        entity_type: EntityType,
        ast: Dict,
        question_word: str
    ) -> EntityType:
        """
        Refine entity type based on context clues in the AST.

        Examples:
        - "Kio estas Esperanto?" → DEFINITION (verb is "estas")
        - "Kiu fondis...?" → PERSON (past tense, agent action)
        - "Kiu urbo...?" → PLACE (followed by place noun)
        """
        # "Kio estas X?" pattern → DEFINITION
        if question_word.lower() in ['kio', 'kion']:
            verb = ast.get('verbo', {})
            if verb.get('tipo') == 'vorto':
                verb_root = verb.get('radiko', '').lower()
                if verb_root == 'est':  # "estas" = "is"
                    return EntityType.DEFINITION

        # "Kiu + noun" pattern → Check noun type
        if question_word.lower() in ['kiu', 'kiuj', 'kiun', 'kiujn']:
            # Look for noun after question word
            focus_node = None
            if ast.get('subjekto') and self._node_contains_word(ast['subjekto'], question_word):
                focus_node = ast['subjekto']
            elif ast.get('objekto') and self._node_contains_word(ast['objekto'], question_word):
                focus_node = ast['objekto']

            if focus_node:
                # Check for place nouns
                if self._contains_place_indicator(focus_node):
                    return EntityType.PLACE
                # Check for person nouns
                elif self._contains_person_indicator(focus_node):
                    return EntityType.PERSON

        return entity_type

    def _contains_place_indicator(self, node: Dict) -> bool:
        """Check if node contains place-related words."""
        place_words = {
            'urb', 'land', 'lok', 'reg', 'ŝtat', 'vil', 'insul',
            'mont', 'river', 'mar', 'ocean', 'kontinent'
        }

        return self._node_has_root_in_set(node, place_words)

    def _contains_person_indicator(self, node: Dict) -> bool:
        """Check if node contains person-related words."""
        person_words = {
            'hom', 'vir', 'virin', 'infan', 'person', 'famili',
            'prezident', 'reĝ', 'ministr', 'aŭtor'
        }

        return self._node_has_root_in_set(node, person_words)

    def _node_has_root_in_set(self, node: Dict, root_set: set) -> bool:
        """Check if any word in node has root in the given set."""
        if node.get('tipo') == 'vorto':
            root = node.get('radiko', '').lower()
            return root in root_set

        elif node.get('tipo') == 'vortgrupo':
            if node.get('kerno'):
                if self._node_has_root_in_set(node['kerno'], root_set):
                    return True

            if node.get('priskriboj'):
                for mod in node['priskriboj']:
                    if self._node_has_root_in_set(mod, root_set):
                        return True

        return False

    def _determine_target_slots(
        self,
        question_type: QuestionType,
        focus: str
    ) -> List[str]:
        """
        Determine which AST slots to prioritize when matching.

        Examples:
        - WHO question focused on SUBJ → prioritize matching SUBJ
        - WHAT question about OBJ → prioritize matching VERB and OBJ
        """
        if question_type == QuestionType.WHO:
            # WHO questions: focus on the subject slot
            if focus == 'SUBJ':
                return ['SUBJ', 'VERB']  # Match subject and action
            elif focus == 'OBJ':
                return ['OBJ', 'VERB']   # Match object and action
            else:
                return ['SUBJ', 'VERB', 'OBJ']

        elif question_type == QuestionType.WHAT:
            # WHAT questions: focus on object/definition
            if focus == 'OBJ':
                return ['VERB', 'OBJ']   # Match action and object
            elif focus == 'SUBJ':
                return ['SUBJ', 'VERB']  # Match subject and action
            else:
                return ['VERB', 'OBJ']   # Default: action and object

        elif question_type == QuestionType.WHERE:
            # WHERE questions: look for place modifiers and verb
            return ['VERB', 'OBJ']  # Places often appear as objects or modifiers

        elif question_type == QuestionType.WHEN:
            # WHEN questions: look for time modifiers and verb
            return ['VERB', 'SUBJ', 'OBJ']  # Time often in modifiers

        elif question_type == QuestionType.HOW:
            # HOW questions: focus on verb (method of action)
            return ['VERB', 'OBJ']

        elif question_type == QuestionType.WHY:
            # WHY questions: need full context
            return ['SUBJ', 'VERB', 'OBJ']

        elif question_type == QuestionType.BOOLEAN:
            # Boolean questions: match entire sentence
            return ['SUBJ', 'VERB', 'OBJ']

        else:
            # Unknown: match everything
            return ['SUBJ', 'VERB', 'OBJ']
