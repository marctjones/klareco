"""
Entity Recognizer for Esperanto AST-based extraction.

Extracts named entities from parsed Esperanto sentences using AST annotations:
- Proper names (parse_status: 'proper_name_unknown')
- Time expressions (temporal modifiers, dates)
- Place names (geographic entities)
- Organizations (acronyms, institutional names)
- Works/publications

This is a DETERMINISTIC recognizer - uses AST structure, morphology,
gazetteers, and context clues - not learned models.
"""

import json
import logging
import re
from typing import Dict, List, Set, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


class EntityType(Enum):
    """Types of entities we can recognize."""
    PERSON = "person"           # Human names
    PLACE = "place"             # Locations, countries, cities
    ORGANIZATION = "organization"  # Companies, institutions
    TIME = "time"               # Dates, years, periods
    QUANTITY = "quantity"       # Numbers, measurements
    WORK = "work"               # Books, publications, artworks
    LANGUAGE = "language"       # Languages (Esperanto, English, etc.)
    UNKNOWN = "unknown"


@dataclass
class Entity:
    """Recognized entity."""
    text: str                   # Original text
    entity_type: EntityType     # Type of entity
    root: str                   # Root word (if applicable)
    slot: str                   # Which slot it appeared in (SUBJ/VERB/OBJ/MODIFIER)
    confidence: float           # Confidence score [0.0, 1.0]
    start_idx: int = -1         # Word position in sentence
    end_idx: int = -1           # End word position (for multi-word entities)


class EntityRecognizer:
    """
    Deterministic entity recognizer using AST annotations.

    Leverages Klareco's AST structure to identify entities based on:
    - Gazetteers (known persons, places, organizations, works)
    - Parse status (proper names marked as 'proper_name_unknown')
    - Morphology (capitalization, word structure, suffixes)
    - Context (slot position, prepositions, titles)
    """

    # Known place-related roots
    PLACE_ROOTS = {
        'urb', 'land', 'ŝtat', 'reg', 'vil', 'insul', 'mont', 'river',
        'mar', 'ocean', 'kontinent', 'lok', 'ej', 'ter', 'provinc',
        'region', 'lago', 'dezert', 'golf', 'voj',
    }

    # Known person-related roots
    PERSON_ROOTS = {
        'hom', 'vir', 'virin', 'infan', 'person', 'famili',
        'prezident', 'reĝ', 'ministr', 'aŭtor', 'skribant',
        'profesor', 'doktor', 'sinjor', 'majstr',
    }

    # Known organization-related roots
    ORG_ROOTS = {
        'kompani', 'entrepren', 'organizaĵ', 'institu', 'universitat',
        'komputi', 'asoci', 'grup', 'societ', 'akademi', 'parti',
        'klub', 'federaci', 'unuiĝ', 'lig', 'movad',
    }

    # Time-related roots
    TIME_ROOTS = {
        'jar', 'monat', 'tag', 'hor', 'minut', 'sekund', 'semajn',
        'period', 'epok', 'temp', 'dat', 'moment', 'nokt', 'maten',
        'vesper', 'jarcent', 'jardek',
    }

    # Work/publication roots
    WORK_ROOTS = {
        'libr', 'verk', 'roman', 'novel', 'poem', 'artikol',
        'gazet', 'revu', 'film', 'kant', 'oper', 'dram',
        'komedi', 'teatraĵ', 'manuskrip', 'publikaĵ',
    }

    # Language roots
    LANGUAGE_ROOTS = {
        'lingv', 'idiom', 'parolmanier', 'dialekt',
    }

    # Known language names (special entity type)
    KNOWN_LANGUAGES = {
        'esperanto', 'esperanton', 'esperanta', 'esperante',
        'ido', 'idon', 'volapuk', 'volapukon',
        'interlingua', 'interlinguan',
        'angla', 'anglan', 'anglo',
        'franca', 'francan', 'franco',
        'germana', 'germanan', 'germano',
        'hispana', 'hispanan', 'hispano',
        'rusa', 'rusan', 'ruso',
        'latina', 'latinan', 'latino',
        'pola', 'polan', 'polo',
    }

    # Person title indicators
    PERSON_TITLES = {
        'd-ro', 'doktoro', 'sinjoro', 's-ro', 'sinjorino', 's-ino',
        'profesoro', 'prof', 'majstro', 'reĝo', 'reĝino', 'princo',
        'princino', 'grafo', 'grafino', 'duko', 'dukino', 'papo',
        'episkopo', 'pastro', 'frato', 'fratino', 'prezidanto',
    }

    # Place prepositions
    PLACE_PREPOSITIONS = {'en', 'al', 'de', 'el', 'ĝis', 'tra', 'ĉe', 'apud'}

    # Place suffixes (countries, regions)
    PLACE_SUFFIXES = {'ujo', 'io', 'lando'}

    # Month names
    MONTHS = {
        'januaro', 'februaro', 'marto', 'aprilo', 'majo', 'junio',
        'julio', 'aŭgusto', 'septembro', 'oktobro', 'novembro', 'decembro',
    }

    def __init__(self, gazetteers_path: Optional[Path] = None):
        """
        Initialize entity recognizer.

        Args:
            gazetteers_path: Path to gazetteers directory
                            (default: data/gazetteers/)
        """
        if gazetteers_path is None:
            gazetteers_path = Path(__file__).parent.parent.parent / 'data' / 'gazetteers'

        self.gazetteers_path = Path(gazetteers_path)

        # Initialize gazetteers
        self.known_persons: Set[str] = set()
        self.known_places: Set[str] = set()
        self.known_orgs: Set[str] = set()
        self.known_works: Set[str] = set()

        self._load_gazetteers()

    def _load_gazetteers(self):
        """Load entity gazetteers from JSON files."""
        if not self.gazetteers_path.exists():
            logger.debug(f"Gazetteers directory not found at {self.gazetteers_path}")
            return

        # Load persons
        persons_file = self.gazetteers_path / 'persons.json'
        if persons_file.exists():
            with open(persons_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                persons = data.get('persons', data) if isinstance(data, dict) else data
                self.known_persons = {p.lower() for p in persons}
            logger.debug(f"Loaded {len(self.known_persons)} known persons")

        # Load places
        places_file = self.gazetteers_path / 'places.json'
        if places_file.exists():
            with open(places_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                places = data.get('places', data) if isinstance(data, dict) else data
                self.known_places = {p.lower() for p in places}
            logger.debug(f"Loaded {len(self.known_places)} known places")

        # Load organizations
        orgs_file = self.gazetteers_path / 'organizations.json'
        if orgs_file.exists():
            with open(orgs_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                orgs = data.get('organizations', data) if isinstance(data, dict) else data
                self.known_orgs = {o.lower() for o in orgs}
            logger.debug(f"Loaded {len(self.known_orgs)} known organizations")

        # Load works
        works_file = self.gazetteers_path / 'works.json'
        if works_file.exists():
            with open(works_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                works = data.get('works', data) if isinstance(data, dict) else data
                self.known_works = {w.lower() for w in works}
            logger.debug(f"Loaded {len(self.known_works)} known works")

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

    def _extract_from_node(
        self,
        node: Dict,
        slot: str,
        all_words: Optional[List[Dict]] = None,
        word_idx: int = -1
    ) -> List[Entity]:
        """Extract entities from an AST node."""
        entities = []

        if node.get('tipo') == 'vorto':
            # Check for time expressions FIRST (years shouldn't be names)
            entity = self._check_time_expression(node, slot)
            if entity:
                entities.append(entity)
            else:
                # Check if this is a proper name
                entity = self._check_proper_name(node, slot, all_words, word_idx)
                if entity:
                    entities.append(entity)

            # Check for quantities (skip if already time)
            if not entity or entity.entity_type != EntityType.TIME:
                entity = self._check_quantity(node, slot)
                if entity:
                    entities.append(entity)

        elif node.get('tipo') == 'vortgrupo':
            # Recursively extract from word group
            if node.get('kerno'):
                entities.extend(
                    self._extract_from_node(node['kerno'], slot, all_words, word_idx)
                )

            for modifier in node.get('priskriboj', []):
                entities.extend(
                    self._extract_from_node(modifier, slot, all_words, word_idx)
                )

        return entities

    def _check_proper_name(
        self,
        node: Dict,
        slot: str,
        all_words: Optional[List[Dict]] = None,
        word_idx: int = -1
    ) -> Optional[Entity]:
        """
        Check if a word node is a proper name.

        Uses parse_status and capitalization clues.
        """
        text = node.get('plena_vorto', '')
        text_lower = text.lower()
        root = node.get('radiko', '').lower()

        # Skip titles - they indicate the next word is a person, not themselves
        if text_lower in self.PERSON_TITLES or root in {'d', 'r'}:  # D-ro edge case
            return None

        # Check for known language names FIRST (before skipping common words)
        # This is important for Q&A about Esperanto itself
        if text_lower in self.KNOWN_LANGUAGES:
            return Entity(
                text=text,
                entity_type=EntityType.LANGUAGE,
                root=root or 'esperant',
                slot=slot,
                confidence=0.95,
            )

        # Skip function words that sometimes get capitalized
        # (but NOT content words like "Esperanto" which is handled above)
        function_words = {
            'la', 'kaj', 'de', 'en', 'al', 'por', 'kun', 'pri',
            'estas', 'estis', 'estos', 'estu',
        }
        if text_lower in function_words:
            return None

        # Check parse status
        parse_status = node.get('parse_status')
        category = node.get('category', '')

        if parse_status == 'proper_name_unknown' or 'proper' in category.lower():
            # Parser marked this as a proper name
            entity_type = self._infer_proper_name_type(
                node, root, slot, all_words, word_idx
            )

            return Entity(
                text=text,
                entity_type=entity_type,
                root=root,
                slot=slot,
                confidence=0.9,  # High confidence - parser marked it
            )

        # Check if in gazetteer
        if text_lower in self.known_persons:
            return Entity(
                text=text, entity_type=EntityType.PERSON,
                root=root, slot=slot, confidence=0.95
            )
        if text_lower in self.known_places:
            return Entity(
                text=text, entity_type=EntityType.PLACE,
                root=root, slot=slot, confidence=0.95
            )
        if text_lower in self.known_orgs:
            return Entity(
                text=text, entity_type=EntityType.ORGANIZATION,
                root=root, slot=slot, confidence=0.95
            )
        if text_lower in self.known_works:
            return Entity(
                text=text, entity_type=EntityType.WORK,
                root=root, slot=slot, confidence=0.90
            )

        # Check for capitalized words (heuristic)
        if text and text[0].isupper() and len(text) > 2:
            # Only treat as proper name if unknown to parser
            vortspeco = node.get('vortspeco', '')
            if vortspeco == 'nekonata' or parse_status == 'failed':
                entity_type = self._infer_proper_name_type(
                    node, root, slot, all_words, word_idx
                )

                return Entity(
                    text=text,
                    entity_type=entity_type,
                    root=root,
                    slot=slot,
                    confidence=0.6,  # Medium confidence - just capitalization
                )

        return None

    def _infer_proper_name_type(
        self,
        node: Dict,
        root: str,
        slot: str = '',
        all_words: Optional[List[Dict]] = None,
        word_idx: int = -1
    ) -> EntityType:
        """
        Infer the type of proper name from context.

        Uses gazetteers, morphological clues, and context.
        """
        text = node.get('plena_vorto', '')
        text_lower = text.lower()

        # 1. Check gazetteers first (highest confidence)
        if text_lower in self.known_persons:
            return EntityType.PERSON
        if text_lower in self.known_places:
            return EntityType.PLACE
        if text_lower in self.known_orgs:
            return EntityType.ORGANIZATION
        if text_lower in self.known_works:
            return EntityType.WORK

        # 2. Check morphological clues
        sufiksoj = node.get('sufiksoj', [])

        # -uj suffix indicates place (Germanujo, Francujo)
        if 'uj' in sufiksoj:
            return EntityType.PLACE

        # -io suffix for places (Italio, Germanio)
        if text_lower.endswith('io') and len(text) > 4:
            return EntityType.PLACE

        # -an suffix with place roots indicates nationality/place
        if 'an' in sufiksoj:
            for place_root in self.PLACE_ROOTS:
                if place_root in root:
                    return EntityType.PLACE

        # 3. Check context clues from surrounding words
        if all_words and word_idx >= 0:
            # Check preceding word for person titles
            if word_idx > 0:
                prec = all_words[word_idx - 1]
                prec_word = prec.get('plena_vorto', '').lower()
                prec_root = prec.get('radiko', '').lower()

                # Person titles
                if prec_word in self.PERSON_TITLES or prec_root in self.PERSON_TITLES:
                    return EntityType.PERSON

                # Place prepositions (en, al, de, etc.)
                if prec_root in self.PLACE_PREPOSITIONS:
                    return EntityType.PLACE

        # 4. Check root against known categories
        if root in self.PLACE_ROOTS:
            return EntityType.PLACE

        if root in self.PERSON_ROOTS:
            return EntityType.PERSON

        if root in self.ORG_ROOTS:
            return EntityType.ORGANIZATION

        if root in self.WORK_ROOTS:
            return EntityType.WORK

        # 5. Heuristics based on word form

        # All caps short word → likely organization acronym
        if text.isupper() and 2 <= len(text) <= 6:
            return EntityType.ORGANIZATION

        # Esperantized place name ending in -o (after locative prep)
        category = node.get('category', '')
        if category == 'proper_name_esperantized' and text.endswith('o'):
            return EntityType.PLACE

        # Default: if in subject or object slot, likely a person
        # (most proper names in Esperanto text are people)
        return EntityType.PERSON

    def _check_time_expression(self, node: Dict, slot: str) -> Optional[Entity]:
        """
        Check if a word node is a time expression.

        Looks for:
        - Numbers that could be years (1887, 1959, etc.)
        - Time-related roots
        - Month names
        """
        text = node.get('plena_vorto', '')
        text_lower = text.lower()
        root = node.get('radiko', '').lower()

        # Check for time-related roots
        if root in self.TIME_ROOTS:
            return Entity(
                text=text,
                entity_type=EntityType.TIME,
                root=root,
                slot=slot,
                confidence=0.8,
            )

        # Check for month names
        if text_lower in self.MONTHS:
            return Entity(
                text=text,
                entity_type=EntityType.TIME,
                root=root,
                slot=slot,
                confidence=0.9,
            )

        # Check for year-like numbers (4 digits starting with 1 or 2)
        # Accept both vortspeco=nombro and raw year strings
        if re.match(r'^[12]\d{3}$', text):
            return Entity(
                text=text,
                entity_type=EntityType.TIME,
                root=text,
                slot=slot,
                confidence=0.95,
            )

        # Check for date ordinals (e.g., "28-an" for 28th)
        if re.match(r'^\d{1,2}-an?$', text_lower):
            return Entity(
                text=text,
                entity_type=EntityType.TIME,
                root=root,
                slot=slot,
                confidence=0.80,
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
