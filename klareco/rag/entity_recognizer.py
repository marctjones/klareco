"""
Entity Recognizer for Esperanto AST-based extraction.

Extracts named entities from parsed Esperanto sentences using AST annotations:
- Proper names (parse_status: 'proper_name_unknown')
- Time expressions (temporal modifiers, dates)
- Place names (geographic entities)
- Quantities (numbers, measurements)

This is a DETERMINISTIC recognizer - uses AST structure and morphology,
not learned models.
"""

from typing import Dict, List, Set, Optional, Tuple
from enum import Enum
from dataclasses import dataclass


class EntityType(Enum):
    """Types of entities we can recognize."""
    PERSON = "person"           # Human names
    PLACE = "place"             # Locations, countries, cities
    ORGANIZATION = "organization"  # Companies, institutions
    TIME = "time"               # Dates, years, periods
    QUANTITY = "quantity"       # Numbers, measurements
    UNKNOWN = "unknown"


@dataclass
class Entity:
    """Recognized entity."""
    text: str                   # Original text
    entity_type: EntityType     # Type of entity
    root: str                   # Root word (if applicable)
    slot: str                   # Which slot it appeared in (SUBJ/VERB/OBJ/MODIFIER)
    confidence: float           # Confidence score [0.0, 1.0]


class EntityRecognizer:
    """
    Deterministic entity recognizer using AST annotations.

    Leverages Klareco's AST structure to identify entities based on:
    - Parse status (proper names marked as 'proper_name_unknown')
    - Morphology (capitalization, word structure)
    - Context (slot position, modifiers)
    """

    # Known place-related roots
    PLACE_ROOTS = {
        'urb', 'land', 'ŝtat', 'reg', 'vil', 'insul', 'mont', 'river',
        'mar', 'ocean', 'kontinent', 'lok', 'ej',
    }

    # Known person-related roots
    PERSON_ROOTS = {
        'hom', 'vir', 'virin', 'infan', 'person', 'famili',
        'prezident', 'reĝ', 'ministr', 'aŭtor', 'skribant',
    }

    # Known organization-related roots
    ORG_ROOTS = {
        'kompani', 'entrepren', 'organizaĵ', 'institu', 'universitat',
        'komputi', 'asoci', 'grup',
    }

    # Time-related roots
    TIME_ROOTS = {
        'jar', 'monat', 'tag', 'hor', 'minut', 'sekund', 'semajn',
        'period', 'epok', 'temp',
    }

    def __init__(self):
        """Initialize entity recognizer."""
        pass

    def recognize_entities(self, ast: Dict) -> List[Entity]:
        """
        Extract all entities from a parsed sentence.

        Args:
            ast: Parsed sentence AST

        Returns:
            List of recognized entities
        """
        entities = []

        # Extract from subject
        if ast.get('subjekto'):
            entities.extend(
                self._extract_from_node(ast['subjekto'], 'SUBJ')
            )

        # Extract from verb (rare, but possible for entity names)
        if ast.get('verbo'):
            entities.extend(
                self._extract_from_node(ast['verbo'], 'VERB')
            )

        # Extract from object
        if ast.get('objekto'):
            entities.extend(
                self._extract_from_node(ast['objekto'], 'OBJ')
            )

        # Extract from modifiers
        for modifier in ast.get('aliaj', []):
            entities.extend(
                self._extract_from_node(modifier, 'MODIFIER')
            )

        return entities

    def _extract_from_node(self, node: Dict, slot: str) -> List[Entity]:
        """Extract entities from an AST node."""
        entities = []

        if node.get('tipo') == 'vorto':
            # Check if this is a proper name
            entity = self._check_proper_name(node, slot)
            if entity:
                entities.append(entity)

            # Check for time expressions
            entity = self._check_time_expression(node, slot)
            if entity:
                entities.append(entity)

            # Check for quantities
            entity = self._check_quantity(node, slot)
            if entity:
                entities.append(entity)

        elif node.get('tipo') == 'vortgrupo':
            # Recursively extract from word group
            if node.get('kerno'):
                entities.extend(
                    self._extract_from_node(node['kerno'], slot)
                )

            for modifier in node.get('priskriboj', []):
                entities.extend(
                    self._extract_from_node(modifier, slot)
                )

        return entities

    def _check_proper_name(self, node: Dict, slot: str) -> Optional[Entity]:
        """
        Check if a word node is a proper name.

        Uses parse_status and capitalization clues.
        """
        # Check parse status
        parse_status = node.get('parse_status')

        if parse_status == 'proper_name_unknown':
            # Parser marked this as a proper name
            text = node.get('plena_vorto', '')
            root = node.get('radiko', '').lower()

            # Infer entity type from context
            entity_type = self._infer_proper_name_type(node, root)

            return Entity(
                text=text,
                entity_type=entity_type,
                root=root,
                slot=slot,
                confidence=0.9,  # High confidence - parser marked it
            )

        # Check for capitalized words (heuristic)
        text = node.get('plena_vorto', '')
        if text and text[0].isupper() and len(text) > 2:
            # Might be a proper name
            root = node.get('radiko', '').lower()
            entity_type = self._infer_proper_name_type(node, root)

            return Entity(
                text=text,
                entity_type=entity_type,
                root=root,
                slot=slot,
                confidence=0.6,  # Medium confidence - just capitalization
            )

        return None

    def _infer_proper_name_type(self, node: Dict, root: str) -> EntityType:
        """
        Infer the type of proper name from context.

        Uses morphological clues and known roots.
        """
        # Check for place indicators
        sufiksoj = node.get('sufiksoj', [])

        # -uj suffix indicates place (Germanujo, Francujo)
        if 'uj' in sufiksoj:
            return EntityType.PLACE

        # -an suffix with place roots indicates nationality/place
        if 'an' in sufiksoj:
            for place_root in self.PLACE_ROOTS:
                if place_root in root:
                    return EntityType.PLACE

        # Check root against known categories
        if root in self.PLACE_ROOTS:
            return EntityType.PLACE

        if root in self.PERSON_ROOTS:
            return EntityType.PERSON

        if root in self.ORG_ROOTS:
            return EntityType.ORGANIZATION

        # Default: if in subject or object slot, likely a person
        # (most proper names in Esperanto text are people)
        return EntityType.PERSON

    def _check_time_expression(self, node: Dict, slot: str) -> Optional[Entity]:
        """
        Check if a word node is a time expression.

        Looks for:
        - Numbers that could be years (1887, 1959, etc.)
        - Time-related roots
        """
        root = node.get('radiko', '').lower()

        # Check for time-related roots
        if root in self.TIME_ROOTS:
            text = node.get('plena_vorto', '')
            return Entity(
                text=text,
                entity_type=EntityType.TIME,
                root=root,
                slot=slot,
                confidence=0.8,
            )

        # Check for year-like numbers (4 digits starting with 1 or 2)
        vortspeco = node.get('vortspeco')
        if vortspeco == 'nombro':
            text = node.get('plena_vorto', '')
            # Simple heuristic: 4-digit numbers starting with 1 or 2 are likely years
            if len(text) == 4 and text[0] in ['1', '2'] and text.isdigit():
                return Entity(
                    text=text,
                    entity_type=EntityType.TIME,
                    root=root,
                    slot=slot,
                    confidence=0.7,
                )

        return None

    def _check_quantity(self, node: Dict, slot: str) -> Optional[Entity]:
        """
        Check if a word node is a quantity.

        Looks for numbers and measurements.
        """
        vortspeco = node.get('vortspeco')

        if vortspeco == 'nombro':
            text = node.get('plena_vorto', '')
            root = node.get('radiko', '').lower()

            # Skip year-like numbers (handled by time check)
            if len(text) == 4 and text[0] in ['1', '2'] and text.isdigit():
                return None

            return Entity(
                text=text,
                entity_type=EntityType.QUANTITY,
                root=root,
                slot=slot,
                confidence=0.9,
            )

        return None

    def filter_by_type(self, entities: List[Entity], entity_type: EntityType) -> List[Entity]:
        """Filter entities by type."""
        return [e for e in entities if e.entity_type == entity_type]

    def get_by_slot(self, entities: List[Entity], slot: str) -> List[Entity]:
        """Get entities from a specific slot."""
        return [e for e in entities if e.slot == slot]

    def has_entity_type(self, entities: List[Entity], entity_type: EntityType) -> bool:
        """Check if any entity of given type exists."""
        return any(e.entity_type == entity_type for e in entities)

    def get_entity_texts(self, entities: List[Entity]) -> Set[str]:
        """Get all entity text strings."""
        return {e.text for e in entities}

    def get_entity_roots(self, entities: List[Entity]) -> Set[str]:
        """Get all entity roots."""
        return {e.root for e in entities if e.root}
