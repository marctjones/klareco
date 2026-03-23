"""
Deterministic Semantic Rules for Esperanto Affixes

This module provides 100% deterministic semantic transformations for
Esperanto affixes. These are GRAMMAR RULES, not learned patterns.

Philosophy (from CLAUDE.md):
    "Make grammar, morphology, and linguistic structure 100% programmatic"

Affixes are morphology → should be deterministic, not learned!

Usage:
    >>> features = get_affix_features(['ist'])
    >>> features
    {'animacy': 'animate', 'type': 'person', 'role': 'professional'}

    >>> semantics = compose_word_semantics('pom', ['ist'])
    >>> semantics['animacy']
    'animate'  # Because -ist makes a person
"""

from typing import Dict, List, Optional, Tuple


# ============================================================================
# AFFIX SEMANTIC RULES (100% Deterministic)
# ============================================================================

AFFIX_SEMANTICS = {
    # ========================================================================
    # AGENTIVE AFFIXES (Make Animate)
    # ========================================================================
    'ist': {
        'animacy': 'animate',
        'type': 'person',
        'role': 'professional',
        'description': 'One who practices/does X (profession)',
        'examples': ['bakisto (baker)', 'verkisto (writer)', 'instruisto (teacher)']
    },

    'ant': {
        'animacy': 'animate',
        'type': 'agent',
        'role': 'doer',
        'description': 'One who is doing X (active participle)',
        'examples': ['kuranto (runner)', 'kantanto (singer)', 'studanto (student)']
    },

    'ul': {
        'animacy': 'animate',
        'type': 'person',
        'role': 'characterized',
        'description': 'Person characterized by X',
        'examples': ['riĉulo (rich person)', 'belulo (beautiful person)', 'junulo (young person)']
    },

    'ind': {
        'animacy': 'animate',
        'type': 'agent',
        'role': 'worthy',
        'description': 'One who is worthy of X or should be X-ed',
        'examples': ['helpindulo (person needing help)', 'dankerinda (worthy of thanks)']
    },

    # ========================================================================
    # OBJECT/TOOL AFFIXES (Make Inanimate)
    # ========================================================================
    'il': {
        'animacy': 'inanimate',
        'type': 'tool',
        'function': 'instrument',
        'description': 'Tool/instrument for doing X',
        'examples': ['tranĉilo (knife)', 'skribilo (writing tool)', 'veturilo (vehicle)']
    },

    'aĵ': {
        'animacy': 'inanimate',  # Can be abstract when used that way
        'type': 'thing',
        'function': 'concrete',
        'description': 'Concrete thing characterized by X',
        'examples': ['manĝaĵo (food)', 'trinkaĵo (beverage)', 'konstruaĵo (building)']
    },

    # ========================================================================
    # SIZE/DEGREE MODIFIERS (Preserve Base Animacy)
    # ========================================================================
    'et': {
        'animacy': 'preserves',
        'type': 'diminutive',
        'function': 'small',
        'description': 'Small version of X',
        'examples': ['dometo (small house)', 'hundeto (puppy)', 'libreto (booklet)']
    },

    'eg': {
        'animacy': 'preserves',
        'type': 'augmentative',
        'function': 'large',
        'description': 'Large/intense version of X',
        'examples': ['domego (mansion)', 'hundego (huge dog)', 'pluvo (rain) → pluvego (downpour)']
    },

    # ========================================================================
    # PLACE AFFIXES (Make Location)
    # ========================================================================
    'ej': {
        'animacy': 'inanimate',
        'type': 'place',
        'function': 'location',
        'description': 'Place where X happens',
        'examples': ['lernejo (school)', 'kuirejo (kitchen)', 'laborejo (workplace)']
    },

    # ========================================================================
    # ABSTRACT AFFIXES (Make Abstract)
    # ========================================================================
    'ec': {
        'animacy': 'abstract',
        'type': 'quality',
        'function': 'essence',
        'description': 'Quality/essence of being X',
        'examples': ['beleco (beauty)', 'boneco (goodness)', 'homeco (humanity)']
    },

    'ad': {
        'animacy': 'abstract',
        'type': 'action',
        'function': 'continuous',
        'description': 'Continuous/repeated action of X-ing',
        'examples': ['parolado (speech/discourse)', 'kurado (running)', 'kantado (singing)']
    },

    'aĵ_abstract': {  # Special case: -aĵ can be abstract
        'animacy': 'abstract',
        'type': 'abstraction',
        'function': 'concept',
        'description': 'Abstract manifestation of X',
        'examples': ['miraĵo (miracle)', 'novaĵo (news)', 'okazaĵo (event)']
    },

    # ========================================================================
    # COLLECTIVE AFFIXES
    # ========================================================================
    'ar': {
        'animacy': 'preserves',  # Collective inherits from members
        'type': 'collective',
        'function': 'group',
        'description': 'Collection/group of X',
        'examples': ['homaro (humanity)', 'vortaro (dictionary)', 'arbaro (forest)']
    },

    # ========================================================================
    # RESULT/PRODUCT AFFIXES
    # ========================================================================
    'aĵ_result': {  # -aĵ for result/product
        'animacy': 'inanimate',
        'type': 'product',
        'function': 'result',
        'description': 'Result/product of X-ing',
        'examples': ['kuiraĵo (cooked food)', 'penturaĵo (painting)', 'verkaĵo (written work)']
    },

    'it': {
        'animacy': 'inanimate',
        'type': 'product',
        'function': 'result',
        'description': 'That which has been X-ed',
        'examples': ['senditaĵo (something sent)', 'elektito (chosen one)']
    },

    # ========================================================================
    # PASSIVE PARTICIPLE AFFIXES (Complex: depends on context)
    # ========================================================================
    'at': {
        'animacy': 'preserves',  # Passive participle present
        'type': 'passive',
        'function': 'ongoing',
        'description': 'Being X-ed (passive present)',
        'examples': ['amata (being loved)', 'vidata (being seen)']
    },

    'it_passive': {
        'animacy': 'preserves',  # Passive participle past
        'type': 'passive',
        'function': 'completed',
        'description': 'Having been X-ed (passive past)',
        'examples': ['amita (having been loved)', 'vidita (having been seen)']
    },

    # ========================================================================
    # CAPABILITY/PROPERTY AFFIXES
    # ========================================================================
    'ebl': {
        'animacy': 'abstract',  # Property, not entity
        'type': 'property',
        'function': 'capability',
        'description': 'Capable of being X-ed / possible to X',
        'examples': ['manĝebla (edible)', 'videbla (visible)', 'fareBla (doable)']
    },

    'em': {
        'animacy': 'abstract',  # Tendency/property
        'type': 'property',
        'function': 'tendency',
        'description': 'Tending/inclined to X',
        'examples': ['parolema (talkative)', 'helpema (helpful)', 'riskema (risky)']
    },

    # ========================================================================
    # RELATION/POSITION AFFIXES
    # ========================================================================
    'estr': {
        'animacy': 'animate',
        'type': 'person',
        'role': 'leader',
        'description': 'Leader/chief of X',
        'examples': ['estro (leader)', 'ŝipestro (ship captain)', 'urbestro (mayor)']
    },

    'id': {
        'animacy': 'animate',  # Offspring is animate
        'type': 'offspring',
        'function': 'descendant',
        'description': 'Offspring/descendant of X',
        'examples': ['hundido (puppy)', 'katido (kitten)', 'homido (human child)']
    },
}


# ============================================================================
# ANIMACY CATEGORIES
# ============================================================================

ANIMACY_CLASSES = ['animate', 'inanimate', 'abstract', 'preserves', 'unknown']
TYPE_CLASSES = [
    'person', 'agent', 'animal', 'tool', 'thing', 'place',
    'quality', 'action', 'product', 'collective', 'property',
    'offspring', 'diminutive', 'augmentative', 'abstraction', 'passive',
    'unknown'
]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_affix_features(affixes: List[str]) -> Dict[str, str]:
    """
    Get combined semantic features from a list of affixes.

    Affixes are applied left-to-right (earliest affix first).
    Later affixes override earlier ones.

    Args:
        affixes: List of affixes (e.g., ['mal', 'ist'])

    Returns:
        Dictionary with semantic features

    Example:
        >>> get_affix_features(['ist'])
        {'animacy': 'animate', 'type': 'person', 'role': 'professional'}
    """
    features = {
        'animacy': 'unknown',
        'type': 'unknown',
        'function': None,
        'role': None
    }

    for affix in affixes:
        if affix in AFFIX_SEMANTICS:
            affix_sem = AFFIX_SEMANTICS[affix]

            # Apply transformations
            if affix_sem.get('animacy') and affix_sem['animacy'] != 'preserves':
                features['animacy'] = affix_sem['animacy']

            if 'type' in affix_sem:
                features['type'] = affix_sem['type']

            if 'function' in affix_sem:
                features['function'] = affix_sem['function']

            if 'role' in affix_sem:
                features['role'] = affix_sem['role']

    return features


def compose_word_semantics(
    root: str,
    affixes: List[str],
    root_lexicon: Optional[Dict[str, Dict]] = None
) -> Dict[str, str]:
    """
    Compose word semantics from root + affixes.

    Strategy:
    1. Start with root semantics (from lexicon if available)
    2. Apply affix transformations in order
    3. Later affixes override earlier ones

    Args:
        root: Root word (e.g., 'pom')
        affixes: List of affixes (e.g., ['ist'])
        root_lexicon: Optional dictionary of root semantics

    Returns:
        Combined semantic features

    Example:
        >>> compose_word_semantics('pom', ['ist'])
        {'animacy': 'animate', 'type': 'person', ...}  # pomisto = apple-seller
    """
    # Start with root semantics (if available)
    if root_lexicon and root in root_lexicon:
        features = root_lexicon[root].copy()
    else:
        features = {
            'animacy': 'unknown',
            'type': 'unknown',
            'function': None,
            'role': None
        }

    # Apply affix transformations
    affix_features = get_affix_features(affixes)

    # Override with affix semantics
    for key in ['animacy', 'type', 'function', 'role']:
        if affix_features.get(key) and affix_features[key] not in ['preserves', None]:
            features[key] = affix_features[key]

    return features


def explain_word_semantics(
    root: str,
    affixes: List[str],
    root_lexicon: Optional[Dict[str, Dict]] = None
) -> Tuple[Dict, str]:
    """
    Get word semantics + human-readable explanation.

    Returns:
        (features_dict, explanation_string)

    Example:
        >>> features, explanation = explain_word_semantics('pom', ['ist'])
        >>> print(explanation)
        "pomisto: ROOT 'pom' (fruit) + AFFIX 'ist' (professional) → animate person"
    """
    features = compose_word_semantics(root, affixes, root_lexicon)

    # Build explanation
    parts = [f"ROOT '{root}'"]

    if root_lexicon and root in root_lexicon:
        root_sem = root_lexicon[root]
        parts.append(f"({root_sem.get('type', 'unknown')})")

    for affix in affixes:
        if affix in AFFIX_SEMANTICS:
            affix_sem = AFFIX_SEMANTICS[affix]
            desc = affix_sem.get('description', affix)
            parts.append(f"+ AFFIX '{affix}' ({desc})")

    result = f"→ {features['animacy']} {features['type']}"
    explanation = f"{root}{''.join(affixes)}: {' '.join(parts)} {result}"

    return features, explanation


# ============================================================================
# VALIDATION
# ============================================================================

def validate_affix_rules():
    """Validate that all affix rules are well-formed."""
    required_fields = {'animacy', 'type', 'description', 'examples'}

    for affix, rules in AFFIX_SEMANTICS.items():
        # Check required fields
        missing = required_fields - set(rules.keys())
        if missing:
            print(f"WARNING: Affix '{affix}' missing fields: {missing}")

        # Validate animacy value
        if rules['animacy'] not in ANIMACY_CLASSES:
            print(f"WARNING: Affix '{affix}' has invalid animacy: {rules['animacy']}")

        # Validate type value
        if rules['type'] not in TYPE_CLASSES:
            print(f"WARNING: Affix '{affix}' has invalid type: {rules['type']}")

    print(f"Validated {len(AFFIX_SEMANTICS)} affix rules")


if __name__ == '__main__':
    # Run validation
    validate_affix_rules()

    # Print examples
    print("\n" + "="*70)
    print("AFFIX SEMANTIC RULES - EXAMPLES")
    print("="*70)

    test_cases = [
        ('pom', ['ist']),   # pomisto = apple-seller
        ('bak', ['ist']),   # bakisto = baker
        ('tranĉ', ['il']),  # tranĉilo = knife
        ('lern', ['ej']),   # lernejo = school
        ('bel', ['ec']),    # beleco = beauty
        ('hom', ['ar']),    # homaro = humanity
        ('hund', ['et']),   # hundeto = puppy
    ]

    for root, affixes in test_cases:
        features, explanation = explain_word_semantics(root, affixes)
        print(f"\n{explanation}")
