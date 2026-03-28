"""
Root Semantic Lexicon for Esperanto

Hand-curated semantic features for common Esperanto roots.
These are LEXICAL features, not grammatical.

Coverage Strategy:
- Start with top 100 most frequent roots (~50% corpus coverage)
- Focus on unambiguous cases first
- Expand incrementally

Semantic Features:
- animacy: animate (living), inanimate (non-living), abstract (non-physical)
- type: person, animal, object, place, action, quality, etc.
- domain: Optional domain-specific info (bio, tech, social, etc.)

Philosophy:
- Roots are LEXICAL → need to be learned or looked up
- Affixes are GRAMMAR → 100% deterministic (in affix_semantics.py)
"""

# ============================================================================
# CORE ROOT LEXICON (Top ~100 roots by frequency)
# ============================================================================

ROOT_LEXICON = {
    # ========================================================================
    # HUMANS (animate, person)
    # ========================================================================
    'hom': {'animacy': 'animate', 'type': 'person', 'sentience': 'sentient'},
    'vir': {'animacy': 'animate', 'type': 'person', 'sentience': 'sentient', 'gender': 'male'},
    'virin': {'animacy': 'animate', 'type': 'person', 'sentience': 'sentient', 'gender': 'female'},
    'infant': {'animacy': 'animate', 'type': 'person', 'sentience': 'sentient', 'age': 'young'},
    'pli': {'animacy': 'animate', 'type': 'person', 'sentience': 'sentient'},
    'student': {'animacy': 'animate', 'type': 'person', 'sentience': 'sentient', 'role': 'learner'},
    'autor': {'animacy': 'animate', 'type': 'person', 'sentience': 'sentient', 'role': 'creator'},
    'pres': {'animacy': 'animate', 'type': 'person', 'sentience': 'sentient', 'role': 'leader'},
    'prezident': {'animacy': 'animate', 'type': 'person', 'sentience': 'sentient', 'role': 'leader'},

    # ========================================================================
    # ANIMALS (animate, animal)
    # ========================================================================
    'hund': {'animacy': 'animate', 'type': 'animal', 'sentience': 'sentient'},
    'kat': {'animacy': 'animate', 'type': 'animal', 'sentience': 'sentient'},
    'bird': {'animacy': 'animate', 'type': 'animal', 'sentience': 'sentient'},
    'fiŝ': {'animacy': 'animate', 'type': 'animal', 'sentience': 'sentient'},
    'insekt': {'animacy': 'animate', 'type': 'animal', 'sentience': 'sentient'},
    'best': {'animacy': 'animate', 'type': 'animal', 'sentience': 'sentient'},

    # ========================================================================
    # PHYSICAL OBJECTS (inanimate, object)
    # ========================================================================
    'tabl': {'animacy': 'inanimate', 'type': 'object', 'category': 'furniture'},
    'seĝ': {'animacy': 'inanimate', 'type': 'object', 'category': 'furniture'},
    'dom': {'animacy': 'inanimate', 'type': 'object', 'category': 'building'},
    'libr': {'animacy': 'inanimate', 'type': 'object', 'category': 'artifact'},
    'ŝton': {'animacy': 'inanimate', 'type': 'object', 'category': 'natural'},
    'akv': {'animacy': 'inanimate', 'type': 'substance', 'category': 'liquid'},
    'aer': {'animacy': 'inanimate', 'type': 'substance', 'category': 'gas'},
    'paper': {'animacy': 'inanimate', 'type': 'object', 'category': 'artifact'},
    'montr': {'animacy': 'inanimate', 'type': 'object', 'category': 'device'},
    'maŝin': {'animacy': 'inanimate', 'type': 'object', 'category': 'device'},
    'komputil': {'animacy': 'inanimate', 'type': 'object', 'category': 'device'},
    'telefon': {'animacy': 'inanimate', 'type': 'object', 'category': 'device'},

    # ========================================================================
    # FOOD (inanimate, consumable)
    # ========================================================================
    'pom': {'animacy': 'inanimate', 'type': 'food', 'category': 'fruit'},
    'pan': {'animacy': 'inanimate', 'type': 'food', 'category': 'baked'},
    'vian': {'animacy': 'inanimate', 'type': 'food', 'category': 'meat'},
    'frukt': {'animacy': 'inanimate', 'type': 'food', 'category': 'fruit'},

    # ========================================================================
    # PLACES (inanimate, location)
    # ========================================================================
    'urb': {'animacy': 'inanimate', 'type': 'place', 'category': 'settlement'},
    'land': {'animacy': 'inanimate', 'type': 'place', 'category': 'region'},
    'mond': {'animacy': 'inanimate', 'type': 'place', 'category': 'planet'},
    'ĉiel': {'animacy': 'inanimate', 'type': 'place', 'category': 'natural'},
    'mar': {'animacy': 'inanimate', 'type': 'place', 'category': 'natural'},
    'ter': {'animacy': 'inanimate', 'type': 'place', 'category': 'natural'},
    'reg': {'animacy': 'inanimate', 'type': 'place', 'category': 'region'},

    # ========================================================================
    # TIME/TEMPORAL (inanimate, temporal)
    # ========================================================================
    'jar': {'animacy': 'inanimate', 'type': 'temporal', 'category': 'duration'},
    'dat': {'animacy': 'inanimate', 'type': 'temporal', 'category': 'point'},
    'tag': {'animacy': 'inanimate', 'type': 'temporal', 'category': 'duration'},
    'temp': {'animacy': 'inanimate', 'type': 'temporal', 'category': 'abstract'},
    'monat': {'animacy': 'inanimate', 'type': 'temporal', 'category': 'duration'},

    # ========================================================================
    # ABSTRACT CONCEPTS (abstract)
    # ========================================================================
    'sci': {'animacy': 'abstract', 'type': 'knowledge'},
    'pens': {'animacy': 'abstract', 'type': 'mental'},
    'ide': {'animacy': 'abstract', 'type': 'mental'},
    'sonĝ': {'animacy': 'abstract', 'type': 'mental'},
    'am': {'animacy': 'abstract', 'type': 'emotion'},
    'sentiment': {'animacy': 'abstract', 'type': 'emotion'},
    'esper': {'animacy': 'abstract', 'type': 'emotion'},
    'tim': {'animacy': 'abstract', 'type': 'emotion'},
    'kor': {'animacy': 'inanimate', 'type': 'body-part'},  # Physical heart
    'kord': {'animacy': 'abstract', 'type': 'metaphorical'},  # Metaphorical heart (emotions)
    'lingv': {'animacy': 'abstract', 'type': 'system'},
    'esperant': {'animacy': 'abstract', 'type': 'language'},
    'form': {'animacy': 'abstract', 'type': 'structure'},
    'fort': {'animacy': 'abstract', 'type': 'quality'},
    'bel': {'animacy': 'abstract', 'type': 'quality'},
    'bon': {'animacy': 'abstract', 'type': 'quality'},
    'mal': {'animacy': 'abstract', 'type': 'quality'},
    'grav': {'animacy': 'abstract', 'type': 'quality'},
    'simpl': {'animacy': 'abstract', 'type': 'quality'},

    # ========================================================================
    # COMMON VERBS (abstract, action)
    # ========================================================================
    'est': {'animacy': 'abstract', 'type': 'copula', 'requires_subject': False},
    'hav': {'animacy': 'abstract', 'type': 'possession', 'requires_animate_agent': False},
    'far': {'animacy': 'abstract', 'type': 'action', 'requires_animate_agent': True},
    'dir': {'animacy': 'abstract', 'type': 'communication', 'requires_animate_agent': True},
    'parol': {'animacy': 'abstract', 'type': 'communication', 'requires_animate_agent': True},
    'vid': {'animacy': 'abstract', 'type': 'perception', 'requires_animate_agent': True, 'requires_sentient': True},
    'aŭd': {'animacy': 'abstract', 'type': 'perception', 'requires_animate_agent': True, 'requires_sentient': True},
    'sent': {'animacy': 'abstract', 'type': 'perception', 'requires_animate_agent': True, 'requires_sentient': True},
    'manĝ': {'animacy': 'abstract', 'type': 'consumption', 'requires_animate_agent': True, 'requires_physical_patient': True},
    'trink': {'animacy': 'abstract', 'type': 'consumption', 'requires_animate_agent': True, 'requires_liquid_patient': True},
    'lern': {'animacy': 'abstract', 'type': 'cognition', 'requires_animate_agent': True, 'requires_sentient': True},
    'sci_verb': {'animacy': 'abstract', 'type': 'cognition', 'requires_animate_agent': True, 'requires_sentient': True},
    'pens': {'animacy': 'abstract', 'type': 'cognition', 'requires_animate_agent': True, 'requires_sentient': True},
    'verk': {'animacy': 'abstract', 'type': 'creation', 'requires_animate_agent': True, 'requires_sentient': True},
    'kre': {'animacy': 'abstract', 'type': 'creation', 'requires_animate_agent': True},
    'konstru': {'animacy': 'abstract', 'type': 'creation', 'requires_animate_agent': True},
    'don': {'animacy': 'abstract', 'type': 'transfer', 'requires_animate_agent': True},
    'ricev': {'animacy': 'abstract', 'type': 'transfer', 'requires_animate_agent': True},
    'preten': {'animacy': 'abstract', 'type': 'action', 'requires_animate_agent': True},
    'plor': {'animacy': 'abstract', 'type': 'emotion_action', 'requires_animate_agent': True},
    'flug': {'animacy': 'abstract', 'type': 'motion', 'requires_capable_agent': True},
    'ir': {'animacy': 'abstract', 'type': 'motion', 'requires_animate_agent': True},
    'ven': {'animacy': 'abstract', 'type': 'motion', 'requires_animate_agent': True},
    'viv': {'animacy': 'abstract', 'type': 'biological', 'requires_animate_agent': True},
    'mort': {'animacy': 'abstract', 'type': 'biological', 'requires_animate_agent': True},

    # ========================================================================
    # SOCIAL/ORGANIZATIONAL (abstract or mixed)
    # ========================================================================
    'famili': {'animacy': 'collective', 'type': 'group'},
    'societ': {'animacy': 'abstract', 'type': 'organization'},
    'kompani': {'animacy': 'abstract', 'type': 'organization'},
    'ŝtat': {'animacy': 'abstract', 'type': 'organization'},
    'registr': {'animacy': 'abstract', 'type': 'organization'},

    # ========================================================================
    # QUANTITIES/NUMBERS (abstract)
    # ========================================================================
    'nombr': {'animacy': 'abstract', 'type': 'quantity'},
    'mil': {'animacy': 'abstract', 'type': 'quantity'},
    'cent': {'animacy': 'abstract', 'type': 'quantity'},
    'mult': {'animacy': 'abstract', 'type': 'quantity'},
    'pli': {'animacy': 'abstract', 'type': 'quantity'},

    # ========================================================================
    # EXPANDED LEXICON (High-frequency roots from top 200)
    # Auto-generated from corpus analysis, high-confidence (≥0.8)
    # Added: 2026-03-23 (50 new roots)
    # ========================================================================

    # Common verbs (high frequency)
    'pov': {'animacy': 'abstract', 'type': 'action'},
    'fond': {'animacy': 'abstract', 'type': 'action'},
    'komenc': {'animacy': 'abstract', 'type': 'action'},
    'kon': {'animacy': 'abstract', 'type': 'action'},
    'enhav': {'animacy': 'abstract', 'type': 'action'},
    'okaz': {'animacy': 'abstract', 'type': 'action'},
    'ating': {'animacy': 'abstract', 'type': 'action'},
    'kovr': {'animacy': 'abstract', 'type': 'action'},
    'prezent': {'animacy': 'abstract', 'type': 'action'},
    'trov': {'animacy': 'abstract', 'type': 'action'},
    'ebl': {'animacy': 'abstract', 'type': 'action'},
    'publik': {'animacy': 'abstract', 'type': 'action'},
    'konk': {'animacy': 'abstract', 'type': 'action'},
    'dev': {'animacy': 'abstract', 'type': 'action'},
    'aper': {'animacy': 'abstract', 'type': 'action'},
    'elekt': {'animacy': 'abstract', 'type': 'action'},
    'konsider': {'animacy': 'abstract', 'type': 'action'},
    'subskrib': {'animacy': 'abstract', 'type': 'action'},
    'sekv': {'animacy': 'abstract', 'type': 'action'},
    'daŭr': {'animacy': 'abstract', 'type': 'action'},
    'signif': {'animacy': 'abstract', 'type': 'action'},
    'serv': {'animacy': 'abstract', 'type': 'action'},
    'postul': {'animacy': 'abstract', 'type': 'action'},
    'konduk': {'animacy': 'abstract', 'type': 'action'},
    'ordin': {'animacy': 'abstract', 'type': 'action'},
    'eduk': {'animacy': 'abstract', 'type': 'action'},
    'decid': {'animacy': 'abstract', 'type': 'action'},
    'nask': {'animacy': 'abstract', 'type': 'biological'},
    'mort': {'animacy': 'abstract', 'type': 'biological'},
    'kontribu': {'animacy': 'abstract', 'type': 'action'},
    'spert': {'animacy': 'abstract', 'type': 'action'},
    'sufer': {'animacy': 'abstract', 'type': 'action'},
    'konsist': {'animacy': 'abstract', 'type': 'action'},
    'efektivig': {'animacy': 'abstract', 'type': 'action'},
    'kompren': {'animacy': 'abstract', 'type': 'cognition'},
    'partopreni': {'animacy': 'abstract', 'type': 'action'},
    'aplaŭd': {'animacy': 'abstract', 'type': 'action'},
    'form': {'animacy': 'abstract', 'type': 'structure'},
    'esper': {'animacy': 'abstract', 'type': 'action'},
    'apliki': {'animacy': 'abstract', 'type': 'action'},
    'ek': {'animacy': 'abstract', 'type': 'action'},
    'posedi': {'animacy': 'abstract', 'type': 'possession'},
    'alian': {'animacy': 'abstract', 'type': 'action'},
    'raport': {'animacy': 'abstract', 'type': 'communication'},
    'alport': {'animacy': 'abstract', 'type': 'action'},
    'komunik': {'animacy': 'abstract', 'type': 'communication'},
    'celebr': {'animacy': 'abstract', 'type': 'action'},
    'redakt': {'animacy': 'abstract', 'type': 'action'},
    'akompan': {'animacy': 'abstract', 'type': 'action'},
}


# ============================================================================
# SELECTIONAL RESTRICTIONS (Verb Constraints)
# ============================================================================

VERB_CONSTRAINTS = {
    # Actions requiring animate agents
    'animate_agent_verbs': [
        'manĝ', 'trink', 'lern', 'pens', 'verk', 'parol', 'dir',
        'kur', 'ir', 'ven', 'salut', 'rid', 'plor', 'kant',
        'labor', 'stud', 'instruk', 'help', 'am'
    ],

    # Actions requiring sentient agents (can think/perceive)
    'sentient_agent_verbs': [
        'vid', 'aŭd', 'sent', 'lern', 'pens', 'sci', 'kompreni',
        'verk', 'leg', 'kre', 'decid', 'eleg', 'vol'
    ],

    # Actions requiring physical objects (not abstract)
    'physical_patient_verbs': [
        'manĝ', 'trink', 'tranĉ', 'romp', 'konstru', 'port',
        'met', 'pren', 'jen', 'tir', 'puŝ'
    ],

    # Consumption verbs (special constraints)
    'consumption_verbs': {
        'manĝ': {'agent': 'animate', 'patient': 'physical_consumable'},
        'trink': {'agent': 'animate', 'patient': 'liquid'},
    },

    # Motion verbs (require capability)
    'motion_verbs': {
        'flug': {'agent': 'flying_capable'},  # birds, planes, etc
        'naĝ': {'agent': 'swimming_capable'},
    },
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_root_features(root: str) -> dict:
    """Get semantic features for a root."""
    return ROOT_LEXICON.get(root, {
        'animacy': 'unknown',
        'type': 'unknown'
    })


def is_animate(root: str) -> bool:
    """Check if root is animate."""
    return ROOT_LEXICON.get(root, {}).get('animacy') == 'animate'


def is_sentient(root: str) -> bool:
    """Check if root is sentient (can think/perceive)."""
    features = ROOT_LEXICON.get(root, {})
    return features.get('sentience') == 'sentient' or features.get('animacy') == 'animate'


def requires_animate_agent(verb: str) -> bool:
    """Check if verb requires animate agent."""
    return verb in VERB_CONSTRAINTS['animate_agent_verbs']


def requires_sentient_agent(verb: str) -> bool:
    """Check if verb requires sentient agent."""
    return verb in VERB_CONSTRAINTS['sentient_agent_verbs']


if __name__ == '__main__':
    print(f"Root lexicon contains {len(ROOT_LEXICON)} roots")
    print(f"\nExample lookups:")
    for root in ['hom', 'tabl', 'pens', 'manĝ']:
        features = get_root_features(root)
        print(f"  {root}: {features}")
