"""
Entity Type Classification for RAG Disambiguation.

Classifies entities into semantic types using deterministic rules:
- LANGUAGE: Esperanto, la angla, la franca
- PERSON: Zamenhof, Schmidt, Tagore
- ORGANIZATION: UEA, Esperanto-klubo, ISAE
- PLACE: Berlino, Vaŝingtono, Parizo
- EVENT: Universala Kongreso, UK 1912
- UNKNOWN: Cannot determine type

100% deterministic - uses AST structure and vocabulary lists.
"""

from typing import Dict, Optional, Set, List
import logging

logger = logging.getLogger(__name__)


class EntityType:
    """Entity type constants."""
    LANGUAGE = 'LANGUAGE'
    PERSON = 'PERSON'
    ORGANIZATION = 'ORGANIZATION'
    PLACE = 'PLACE'
    EVENT = 'EVENT'
    UNKNOWN = 'UNKNOWN'


class EntityClassifier:
    """
    Classify entities into semantic types using deterministic rules.

    Uses AST structure (tipo, vortspeco, kunmetajhoj) and vocabulary lists.
    100% deterministic - zero learned parameters.
    """

    # Known languages by root
    LANGUAGES: Set[str] = {
        'esperant', 'angl', 'franc', 'german', 'rus', 'ital',
        'hispan', 'portugal', 'ĉin', 'japan', 'arab', 'hind',
        'latin', 'grek', 'hebr', 'pols', 'ĉeĥ', 'hungr',
        'turk', 'persian', 'korean', 'vietnm', 'nederland',
        'sved', 'dan', 'norv', 'suom', 'kroat', 'serb',
    }

    # Organization indicators (in compounds or alone)
    ORGANIZATION_MARKERS: Set[str] = {
        'klub', 'societ', 'asocia', 'ligo', 'federaci',
        'unuiĝ', 'komitat', 'komisi', 'organiz', 'institut',
        'grupp', 'mov', 'part', 'koalici', 'al', 'firm',
        'kompani', 'enterpren', 'fondaĵ', 'akademi',
    }

    # Place indicators
    PLACE_MARKERS: Set[str] = {
        'urb', 'vilaĝ', 'land', 'regi', 'ŝtat', 'provinc',
        'distrik', 'municip', 'kanton', 'gubernio', 'territori',
        'insul', 'kontinext', 'pen', 'region',
    }

    # Event indicators
    EVENT_MARKERS: Set[str] = {
        'kongres', 'kunsid', 'konferenc', 'simozi', 'seminari',
        'renkontiĝ', 'fest', 'celebr', 'ceremoniu', 'konkuris',
        'eksposici', 'foar', 'turnir', 'olimpik',
    }

    # Known people (common Esperanto figures)
    # This is a small set - most person detection relies on heuristics
    KNOWN_PEOPLE: Set[str] = {
        'zamenhof', 'waringhien', 'holebrink', 'auld', 'wennergren',
        'lapenna', 'tonkin', 'butler', 'piron', 'kalocsay',
        'boulton', 'corsetti', 'duc goninaz', 'fettes', 'régulo',
    }

    def __init__(self):
        """Initialize entity classifier."""
        logger.info("EntityClassifier initialized (deterministic, 0 params)")

    def classify(self, word_ast: Dict) -> str:
        """
        Classify an entity from its AST representation.

        Args:
            word_ast: AST node representing a word or word group

        Returns:
            Entity type: LANGUAGE | PERSON | ORGANIZATION | PLACE | EVENT | UNKNOWN
        """
        if not word_ast:
            return EntityType.UNKNOWN

        # Extract core word from vortgrupo
        if word_ast.get('tipo') == 'vortgrupo':
            kerno = word_ast.get('kerno', {})
        else:
            kerno = word_ast

        if not kerno:
            return EntityType.UNKNOWN

        root = kerno.get('radiko', '').lower()
        vortspeco = kerno.get('vortspeco', '')
        is_compound = kerno.get('estas_kunmetita', False)

        # 1. Check if it's a known language
        if root in self.LANGUAGES:
            # Even in compounds, language root usually means LANGUAGE
            # "Esperanto-klubo" is ORGANIZATION, but we check that below
            if not is_compound:
                return EntityType.LANGUAGE

        # 2. Check compounds for organization/place/event markers
        if is_compound:
            kunmetajhoj = kerno.get('kunmetajhoj', [])
            component_roots = [
                c.get('radiko', '').lower()
                for c in kunmetajhoj
                if isinstance(c, dict)
            ]

            # Check all components
            all_roots = component_roots + [root]

            # Organization: contains organization marker
            if any(r in self.ORGANIZATION_MARKERS for r in all_roots):
                return EntityType.ORGANIZATION

            # Place: contains place marker
            if any(r in self.PLACE_MARKERS for r in all_roots):
                return EntityType.PLACE

            # Event: contains event marker
            if any(r in self.EVENT_MARKERS for r in all_roots):
                return EntityType.EVENT

            # Language compound without other markers → could be ORGANIZATION
            # "Esperanto-movado" → ORGANIZATION
            if any(r in self.LANGUAGES for r in all_roots):
                return EntityType.ORGANIZATION

        # 3. Check standalone words
        if not is_compound:
            # Known organization marker as standalone
            if root in self.ORGANIZATION_MARKERS:
                return EntityType.ORGANIZATION

            # Known place marker as standalone
            if root in self.PLACE_MARKERS:
                return EntityType.PLACE

            # Known event marker as standalone
            if root in self.EVENT_MARKERS:
                return EntityType.EVENT

        # 4. Check if it's a known person
        if root in self.KNOWN_PEOPLE:
            return EntityType.PERSON

        # 5. Proper noun heuristics
        if vortspeco == 'propra_nomo':
            # Proper nouns are often PERSON or PLACE
            # Without more context, we can't determine which
            # For now, return UNKNOWN - let context decide
            return EntityType.UNKNOWN

        # 6. Default: unknown
        return EntityType.UNKNOWN

    def classify_from_text(self, root: str, is_proper_noun: bool = False) -> str:
        """
        Simplified classification from just root and proper noun flag.

        Args:
            root: Word root (lowercase)
            is_proper_noun: Whether the word is a proper noun

        Returns:
            Entity type
        """
        root = root.lower()

        # Check categories
        if root in self.LANGUAGES:
            return EntityType.LANGUAGE

        if root in self.ORGANIZATION_MARKERS:
            return EntityType.ORGANIZATION

        if root in self.PLACE_MARKERS:
            return EntityType.PLACE

        if root in self.EVENT_MARKERS:
            return EntityType.EVENT

        if root in self.KNOWN_PEOPLE:
            return EntityType.PERSON

        if is_proper_noun:
            return EntityType.UNKNOWN

        return EntityType.UNKNOWN

    def get_statistics(self) -> Dict[str, int]:
        """Get vocabulary statistics."""
        return {
            'languages': len(self.LANGUAGES),
            'organization_markers': len(self.ORGANIZATION_MARKERS),
            'place_markers': len(self.PLACE_MARKERS),
            'event_markers': len(self.EVENT_MARKERS),
            'known_people': len(self.KNOWN_PEOPLE),
        }
