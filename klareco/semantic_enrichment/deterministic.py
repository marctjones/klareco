"""
Deterministic Feature Extractor for AST Semantic Enrichment.

Extracts semantic features from AST using 100% deterministic rules:
- Correlative semantics (9 suffixes → entity types)
- Affix semantics (30+ suffixes/prefixes → semantic classes)
- Vortspeco mapping (grammar → Tier 1 categories)
- Capitalization + context heuristics

Returns confidence scores:
- 1.0 = Fully deterministic (e.g., correlatives, numbers)
- 0.95 = Strong deterministic signal (e.g., -ist suffix)
- 0.80 = Weak deterministic signal (e.g., capitalization alone)
- <0.70 = Ambiguous, needs learned model

Target coverage: ~70% of corpus entities fully determined without model.
"""

from typing import Dict, List, Optional, Set
import logging

from .taxonomy import (
    TopLevelCategory,
    EntityType,
    PersonType,
    LocationType,
    TimeType,
    ThingType,
)

logger = logging.getLogger(__name__)


class DeterministicFeatureExtractor:
    """
    Extract semantic features from AST using deterministic rules.

    Philosophy: Extract everything possible from Esperanto's systematic structure
    before invoking any learned model.

    Coverage estimates:
    - Tier 1: 100% deterministic (from vortspeco)
    - Tier 2: ~70% deterministic (from correlatives + affixes)
    - Tier 3: ~30% deterministic (strong affixes only)
    """

    # Correlative semantics (100% deterministic)
    # From Esperanto correlative table: 9 semantic suffixes
    CORRELATIVE_SEMANTICS: Dict[str, EntityType] = {
        'u': EntityType.PERSON,      # kiu, tiu = who/someone
        'o': EntityType.THING,       # kio, io = what/something
        'a': EntityType.QUALITY,     # kia = what kind
        'e': EntityType.LOCATION,    # kie = where
        'am': EntityType.TIME_POINT, # kiam = when
        'el': EntityType.MANNER,     # kiel = how
        'al': EntityType.REASON,     # kial = why
        'om': EntityType.QUANTITY,   # kiom = how much
        'es': EntityType.POSSESSIVE, # kies = whose
    }

    # Suffix semantics (95% deterministic)
    # From Zamenhof's systematic affix system
    SUFFIX_SEMANTICS: Dict[str, tuple] = {
        # Person indicators (95% confidence)
        'ist': (PersonType.PERSON_PROFESSION, 0.95),  # instruisto = teacher
        'ul': (PersonType.PERSON_ROLE, 0.95),         # saĝulo = wise person
        'in': (PersonType.PERSON_ROLE, 0.90),         # patrino = mother (derived)
        'an': (PersonType.PERSON_ROLE, 0.90),         # amerikano = American

        # Place indicators (90% confidence)
        'ej': (LocationType.PLACE_INSTITUTION, 0.90), # lernejo = school

        # Thing indicators (90% confidence)
        'il': (ThingType.THING_TOOL, 0.90),           # tranĉilo = knife
        'aĵ': (ThingType.THING_CONCRETE, 0.85),       # belajaĵo = beautiful thing
        'ar': (ThingType.THING_COLLECTION, 0.90),     # arbaro = forest

        # Abstract indicators (80% confidence)
        'ec': (EntityType.QUALITY, 0.80),             # beleco = beauty (abstract)
    }

    # Vortspeco → Tier 1 mapping (100% deterministic)
    VORTSPECO_TO_TIER1: Dict[str, TopLevelCategory] = {
        'substantivo': TopLevelCategory.ENTITY,
        'adjektivo': TopLevelCategory.ATTRIBUTE,
        'adverbo': TopLevelCategory.ATTRIBUTE,
        'verbo': TopLevelCategory.ACTION,
        'numero': TopLevelCategory.QUANTITY,
        'prepozicio': TopLevelCategory.RELATION,
        'konjunkcio': TopLevelCategory.RELATION,
        'partiklo': TopLevelCategory.ATTRIBUTE,
    }

    # Known time-related roots (for TIME detection)
    TIME_ROOTS: Set[str] = {
        'jar', 'monat', 'semajn', 'tag', 'hor', 'minut', 'sekond',
        'hodiaŭ', 'hieraŭ', 'morgaŭ', 'nun', 'antaŭ', 'post',
        'temp', 'dat', 'epok', 'period',
    }

    # Known place-related roots (for LOCATION detection)
    PLACE_ROOTS: Set[str] = {
        'urb', 'vilaĝ', 'land', 'ŝtat', 'regi', 'provinc',
        'lok', 'ej', 'hejm', 'dom',
    }

    # Known proper name indicators (Esperanto-specific)
    PROPER_NAME_SUFFIXES: Set[str] = {
        'land',  # Anglaland
        'io',    # Germanio, Hispanio (in some contexts)
    }

    def __init__(self):
        """Initialize deterministic feature extractor."""
        logger.info("DeterministicFeatureExtractor initialized (0 params)")

    def extract(
        self,
        word_ast: Dict,
        context_ast: Optional[Dict] = None
    ) -> Dict:
        """
        Extract deterministic semantic features from AST.

        Args:
            word_ast: AST node for word to classify
            context_ast: Optional surrounding context (±3 words, sentence structure)

        Returns:
            Dictionary with deterministic features and confidence scores:
            {
                'is_fully_determined': bool,  # True if no model needed
                'confidence': float,          # Overall confidence (0-1)
                'tier1_category': TopLevelCategory,
                'tier2_type': Optional[EntityType],
                'tier3_type': Optional[Enum],  # PersonType, LocationType, etc.
                'evidence': Dict,             # What features were detected
                'reasoning': str              # Human-readable explanation
            }
        """
        features = {
            'is_fully_determined': False,
            'confidence': 0.0,
            'tier1_category': None,
            'tier2_type': None,
            'tier3_type': None,
            'evidence': {},
            'reasoning': []
        }

        # STEP 1: Check correlatives (100% deterministic)
        if self._is_correlative(word_ast):
            return self._extract_from_correlative(word_ast)

        # STEP 2: Check numbers (100% deterministic)
        if self._is_number(word_ast):
            return self._extract_from_number(word_ast)

        # STEP 3: Extract from vortspeco (100% deterministic for Tier 1)
        tier1 = self._extract_tier1_from_vortspeco(word_ast)
        features['tier1_category'] = tier1
        features['evidence']['vortspeco'] = word_ast.get('vortspeco')
        features['reasoning'].append(f"Tier 1: {tier1.value} from vortspeco={word_ast.get('vortspeco')}")

        # STEP 4: Extract from affixes (95% deterministic for Tier 2/3)
        affix_result = self._extract_from_affixes(word_ast)
        if affix_result:
            features.update(affix_result)
            # If strong affix signal, might be fully determined
            if features['confidence'] >= 0.90:
                features['is_fully_determined'] = True
            return features

        # STEP 5: Extract from capitalization + context (80% confidence)
        cap_result = self._extract_from_capitalization(word_ast, context_ast)
        if cap_result:
            # Merge capitalization features
            for key, value in cap_result.items():
                if key == 'confidence':
                    # Take max confidence
                    features[key] = max(features[key], value)
                elif key == 'reasoning':
                    # Append reasoning
                    features[key].extend(value)
                elif key not in features or features[key] is None:
                    features[key] = value

        # STEP 6: Extract from root semantics (70% confidence)
        root_result = self._extract_from_root(word_ast)
        if root_result:
            for key, value in root_result.items():
                if key == 'confidence':
                    # Take max confidence
                    features[key] = max(features[key], value)
                elif key == 'reasoning':
                    # Append reasoning
                    features[key].extend(value)
                elif key not in features or features[key] is None:
                    features[key] = value

        # Final confidence check
        if features['confidence'] < 0.70:
            features['is_fully_determined'] = False
            features['reasoning'].append("Low confidence, needs learned model")

        return features

    def _is_correlative(self, word_ast: Dict) -> bool:
        """Check if word is a correlative (kiu, tio, etc.)."""
        return word_ast.get('vortspeco') == 'korelativo'

    def _extract_from_correlative(self, word_ast: Dict) -> Dict:
        """Extract features from correlative (100% deterministic)."""
        suffix = word_ast.get('korelativo_sufikso', '')
        tier2_type = self.CORRELATIVE_SEMANTICS.get(suffix)

        if tier2_type:
            return {
                'is_fully_determined': True,
                'confidence': 1.0,
                'tier1_category': TopLevelCategory.ENTITY if tier2_type in [
                    EntityType.PERSON, EntityType.THING, EntityType.LOCATION
                ] else TopLevelCategory.ATTRIBUTE,
                'tier2_type': tier2_type,
                'tier3_type': None,
                'evidence': {
                    'correlative_suffix': suffix,
                    'vortspeco': 'korelativo'
                },
                'reasoning': [f"Correlative -{suffix} → {tier2_type.value} (confidence=1.0)"]
            }

        # Unknown correlative suffix
        return {
            'is_fully_determined': False,
            'confidence': 0.5,
            'tier1_category': TopLevelCategory.ENTITY,
            'tier2_type': None,
            'tier3_type': None,
            'evidence': {'correlative_suffix': suffix},
            'reasoning': [f"Unknown correlative suffix: {suffix}"]
        }

    def _is_number(self, word_ast: Dict) -> bool:
        """Check if word is a number."""
        return word_ast.get('vortspeco') == 'numero'

    def _extract_from_number(self, word_ast: Dict) -> Dict:
        """Extract features from number (100% deterministic)."""
        return {
            'is_fully_determined': True,
            'confidence': 1.0,
            'tier1_category': TopLevelCategory.QUANTITY,
            'tier2_type': EntityType.NUMBER,
            'tier3_type': None,
            'evidence': {'vortspeco': 'numero'},
            'reasoning': ["Number → QUANTITY (confidence=1.0)"]
        }

    def _extract_tier1_from_vortspeco(self, word_ast: Dict) -> TopLevelCategory:
        """Extract Tier 1 category from vortspeco (100% deterministic)."""
        vortspeco = word_ast.get('vortspeco', 'nekonata')
        return self.VORTSPECO_TO_TIER1.get(vortspeco, TopLevelCategory.ENTITY)

    def _extract_from_affixes(self, word_ast: Dict) -> Optional[Dict]:
        """Extract features from suffixes/prefixes (95% confidence)."""
        sufiksoj = word_ast.get('sufiksoj', [])

        # Check each suffix
        for suffix in sufiksoj:
            if suffix in self.SUFFIX_SEMANTICS:
                tier3_type, confidence = self.SUFFIX_SEMANTICS[suffix]

                # Determine Tier 2 from Tier 3
                if isinstance(tier3_type, PersonType):
                    tier2_type = EntityType.PERSON
                elif isinstance(tier3_type, LocationType):
                    tier2_type = EntityType.LOCATION
                elif isinstance(tier3_type, ThingType):
                    tier2_type = EntityType.THING
                else:
                    tier2_type = tier3_type  # Already Tier 2 (like QUALITY)

                return {
                    'is_fully_determined': confidence >= 0.90,
                    'confidence': confidence,
                    'tier2_type': tier2_type,
                    'tier3_type': tier3_type if isinstance(tier3_type, (PersonType, LocationType, ThingType)) else None,
                    'evidence': {
                        'affix': suffix,
                        'affix_confidence': confidence
                    },
                    'reasoning': [f"Suffix -{suffix} → {tier3_type.value if hasattr(tier3_type, 'value') else tier2_type.value} (confidence={confidence})"]
                }

        return None

    def _extract_from_capitalization(
        self,
        word_ast: Dict,
        context_ast: Optional[Dict]
    ) -> Optional[Dict]:
        """Extract features from capitalization + context (80% confidence)."""
        text = word_ast.get('teksto', '')
        if not text or not text[0].isupper():
            return None

        # Check if sentence-initial (reduces confidence)
        is_sentence_initial = False
        if context_ast:
            position = context_ast.get('position', 'unknown')
            is_sentence_initial = (position == 'sentence_initial')

        # Capitalized + not sentence-initial → likely proper name
        if not is_sentence_initial:
            return {
                'confidence': 0.85,
                'tier3_type': PersonType.PERSON_NAME,  # Could be place too, but default to person
                'evidence': {
                    'capitalized': True,
                    'sentence_initial': False
                },
                'reasoning': [f"Capitalized '{text}' (not sentence-initial) → likely PERSON_NAME (confidence=0.85)"]
            }

        # Capitalized + sentence-initial → ambiguous
        return {
            'confidence': 0.50,
            'evidence': {
                'capitalized': True,
                'sentence_initial': True
            },
            'reasoning': [f"Capitalized '{text}' (sentence-initial) → ambiguous (confidence=0.50)"]
        }

    def _extract_from_root(self, word_ast: Dict) -> Optional[Dict]:
        """Extract features from root semantics (70% confidence)."""
        radiko = word_ast.get('radiko', '').lower()

        # Check time roots
        if radiko in self.TIME_ROOTS:
            return {
                'confidence': 0.70,
                'tier2_type': EntityType.TIME_POINT,
                'evidence': {'time_root': radiko},
                'reasoning': [f"Time root '{radiko}' → TIME_POINT (confidence=0.70)"]
            }

        # Check place roots
        if radiko in self.PLACE_ROOTS:
            return {
                'confidence': 0.70,
                'tier2_type': EntityType.LOCATION,
                'evidence': {'place_root': radiko},
                'reasoning': [f"Place root '{radiko}' → LOCATION (confidence=0.70)"]
            }

        return None

    def get_coverage_stats(self, corpus_asts: List[Dict]) -> Dict:
        """
        Analyze deterministic coverage on a corpus.

        Args:
            corpus_asts: List of word ASTs from corpus

        Returns:
            Coverage statistics:
            {
                'total_words': int,
                'fully_determined': int,    # confidence >= 0.90
                'partially_determined': int, # 0.70 <= confidence < 0.90
                'needs_model': int,          # confidence < 0.70
                'coverage_percent': float,   # fully_determined / total
                'tier1_coverage': float,
                'tier2_coverage': float,
                'tier3_coverage': float
            }
        """
        total = len(corpus_asts)
        fully_determined = 0
        partially_determined = 0
        needs_model = 0
        tier1_determined = 0
        tier2_determined = 0
        tier3_determined = 0

        for word_ast in corpus_asts:
            features = self.extract(word_ast)
            confidence = features['confidence']

            if confidence >= 0.90:
                fully_determined += 1
            elif confidence >= 0.70:
                partially_determined += 1
            else:
                needs_model += 1

            if features['tier1_category'] is not None:
                tier1_determined += 1
            if features['tier2_type'] is not None:
                tier2_determined += 1
            if features['tier3_type'] is not None:
                tier3_determined += 1

        return {
            'total_words': total,
            'fully_determined': fully_determined,
            'partially_determined': partially_determined,
            'needs_model': needs_model,
            'coverage_percent': (fully_determined / total * 100) if total > 0 else 0,
            'tier1_coverage': (tier1_determined / total * 100) if total > 0 else 0,
            'tier2_coverage': (tier2_determined / total * 100) if total > 0 else 0,
            'tier3_coverage': (tier3_determined / total * 100) if total > 0 else 0,
        }
