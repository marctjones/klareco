"""
Query Understanding for RAG System.

Identifies target entities and expected semantic roles in queries
to enable better document matching (Issue #577).

Example:
    Query: "Kiu kreis Esperanton?"
    - Target entity: "Esperanton" (the thing created)
    - Semantic role: PATIENT (thing being acted upon)
    - Expected doc role: OBJECT or PREP_ARG (de Esperanto)

Usage:
    analyzer = QueryAnalyzer()
    target = analyzer.identify_target(query_ast, question_type)
"""

from typing import Dict, List, Optional, Union
from enum import Enum

from klareco.rag.question_classifier import QuestionType as QType


class SemanticRole(Enum):
    """Semantic roles for entities in sentences."""
    AGENT = "agent"           # Doer of action (Zamenhof kreis X)
    PATIENT = "patient"       # Thing acted upon (X kreis Esperanton)
    TOPIC = "topic"           # Thing discussed (X parolas pri Esperanto)
    LOCATION = "location"     # Place (X naskiĝis en Varsovio)
    TIME = "time"             # Time/date (X okazis en 1887)
    UNKNOWN = "unknown"


class QueryTarget:
    """
    Target entity information for a query.

    Attributes:
        entity_root: Root form of target entity (e.g., "esperant")
        semantic_role: Expected semantic role (PATIENT, AGENT, etc.)
        expected_doc_roles: Grammatical roles in documents (objekto, subjekto, etc.)
    """
    def __init__(
        self,
        entity_root: Optional[str],
        semantic_role: SemanticRole,
        expected_doc_roles: List[str]
    ):
        self.entity_root = entity_root
        self.semantic_role = semantic_role
        self.expected_doc_roles = expected_doc_roles


class QueryAnalyzer:
    """Analyzes queries to identify target entities and expected roles."""

    # Verbs that create/found entities (patient = thing created)
    CREATION_VERBS = {'krei', 'fondi', 'fari', 'konstrui', 'starigi', 'establi'}

    # Verbs about communication/discussion (topic = thing discussed)
    COMMUNICATION_VERBS = {'paroli', 'skribi', 'diri', 'rakonti', 'klarigi', 'priskribi'}

    # Verbs about location/birth (location = place)
    LOCATION_VERBS = {'naskiĝi', 'loĝi', 'vivi', 'morti', 'okazi', 'esti'}

    def identify_target(
        self,
        query_ast: Dict,
        question_type: Union[str, QType]
    ) -> Optional[QueryTarget]:
        """
        Identify target entity and expected role in query.

        Args:
            query_ast: Parsed query AST
            question_type: Question type (WHO, WHAT, WHERE, WHEN, etc.) as string or QuestionType enum

        Returns:
            QueryTarget object or None if no clear target
        """
        # Convert to string for comparison
        if isinstance(question_type, QType):
            question_type = question_type.value.upper()
        else:
            question_type = str(question_type).upper()

        if question_type == 'WHO':
            # WHO questions ask about the AGENT (doer)
            # Target: the missing subject
            # Expected: SUBJECT or passive agent ("de X")
            return QueryTarget(
                entity_root=None,  # We're asking WHO
                semantic_role=SemanticRole.AGENT,
                expected_doc_roles=['subjekto', 'passive_agent']
            )

        elif question_type == 'WHAT':
            # WHAT questions can ask about different things
            # Check the verb to determine semantic role
            verb_root = self._get_verb_root(query_ast)
            objekto = query_ast.get('objekto')

            if not objekto:
                # No object in query (e.g., "Kio okazis?")
                return None

            entity_root = self._get_entity_root(objekto)

            if verb_root in self.CREATION_VERBS:
                # "Kiu kreis X?" → X is PATIENT (thing created)
                # Expected in doc: OBJECT or PREP_ARG
                return QueryTarget(
                    entity_root=entity_root,
                    semantic_role=SemanticRole.PATIENT,
                    expected_doc_roles=['objekto', 'prep_arg']
                )
            else:
                # Generic WHAT question
                return QueryTarget(
                    entity_root=entity_root,
                    semantic_role=SemanticRole.PATIENT,
                    expected_doc_roles=['objekto', 'subjekto']
                )

        elif question_type == 'WHERE':
            # WHERE questions ask about LOCATION
            # Target: the subject (thing being located)
            subjekto = query_ast.get('subjekto')
            if subjekto:
                entity_root = self._get_entity_root(subjekto)
                return QueryTarget(
                    entity_root=entity_root,
                    semantic_role=SemanticRole.LOCATION,
                    expected_doc_roles=['subjekto', 'objekto']
                )

        elif question_type == 'WHEN':
            # WHEN questions ask about TIME
            # Target: the event being timed (usually the verb's subject)
            subjekto = query_ast.get('subjekto')
            if subjekto:
                entity_root = self._get_entity_root(subjekto)
                return QueryTarget(
                    entity_root=entity_root,
                    semantic_role=SemanticRole.TIME,
                    expected_doc_roles=['subjekto', 'objekto']
                )

        return None

    def _get_verb_root(self, ast: Dict) -> Optional[str]:
        """Extract verb root from AST."""
        verbo = ast.get('verbo')
        if not verbo:
            return None

        if isinstance(verbo, dict):
            return verbo.get('radiko', '').lower()
        return None

    def _get_entity_root(self, node: Dict) -> Optional[str]:
        """
        Extract entity root from AST node.

        Handles both vorto and vortgrupo.
        """
        if not node:
            return None

        # Handle vortgrupo - get kerno
        if node.get('tipo') == 'vortgrupo':
            kerno = node.get('kerno', {})
            return kerno.get('radiko', '').lower()

        # Handle vorto directly
        elif node.get('tipo') == 'vorto':
            return node.get('radiko', '').lower()

        return None

    def check_role_match(
        self,
        query_target: QueryTarget,
        doc_ast: Dict
    ) -> float:
        """
        Check if document matches query target's expected roles.

        Args:
            query_target: Target entity from query
            doc_ast: Document AST

        Returns:
            Score multiplier (3.0 = strong match, 0.2 = mismatch)
        """
        if not query_target or not query_target.entity_root:
            return 1.0  # No target to match

        entity_root = query_target.entity_root
        expected_roles = query_target.expected_doc_roles

        # Check if entity appears in expected roles
        for role in expected_roles:
            if role == 'objekto':
                objekto = doc_ast.get('objekto')
                if self._entity_in_node(entity_root, objekto, check_compound=False):
                    return 3.0  # STRONG MATCH: entity in object role

            elif role == 'subjekto':
                subjekto = doc_ast.get('subjekto')
                if self._entity_in_node(entity_root, subjekto, check_compound=False):
                    return 3.0  # STRONG MATCH: entity in subject role

            elif role == 'prep_arg':
                # Check if entity appears after preposition "de"
                aliaj = doc_ast.get('aliaj', [])
                for i, word in enumerate(aliaj):
                    if word.get('radiko') == 'de' and i + 1 < len(aliaj):
                        next_word = aliaj[i + 1]
                        if self._entity_in_node(entity_root, next_word, check_compound=False):
                            return 3.0  # STRONG MATCH: entity after "de"

        # Check if entity appears as compound modifier (should penalize)
        for role in ['objekto', 'subjekto']:
            node = doc_ast.get(role)
            if self._entity_in_node(entity_root, node, check_compound=True, compound_only=True):
                return 0.2  # PENALTY: entity is just a modifier

        return 1.0  # Neutral (entity not found)

    def _entity_in_node(
        self,
        entity_root: str,
        node: Dict,
        check_compound: bool = False,
        compound_only: bool = False
    ) -> bool:
        """
        Check if entity root appears in AST node.

        Args:
            entity_root: Root to search for
            node: AST node (vorto or vortgrupo)
            check_compound: If True, also check kunmetajhoj
            compound_only: If True, only check kunmetajhoj (not main root)

        Returns:
            True if entity found
        """
        if not node:
            return False

        # Handle vortgrupo
        if node.get('tipo') == 'vortgrupo':
            kerno = node.get('kerno', {})
            return self._entity_in_node(entity_root, kerno, check_compound, compound_only)

        # Handle vorto
        if node.get('tipo') == 'vorto':
            # Check main root (unless compound_only)
            if not compound_only:
                node_root = node.get('radiko', '').lower()
                if node_root == entity_root:
                    return True

            # Check compound modifiers
            if check_compound:
                kunmetajhoj = node.get('kunmetajhoj', [])
                for kunmetajho in kunmetajhoj:
                    if isinstance(kunmetajho, dict):
                        modifier_root = kunmetajho.get('radiko', '').lower()
                        if modifier_root == entity_root:
                            return True

        return False
