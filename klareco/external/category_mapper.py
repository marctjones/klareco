"""Map external semantic types to Klareco's 14 semantic categories.

Provides mapping logic for:
- ConceptNet relation types → Klareco categories
- Wikidata instance-of QIDs → Klareco categories
"""

from typing import Optional, Dict, List


class CategoryMapper:
    """Maps external semantic types to Klareco's 14 semantic categories.

    Klareco categories:
    - animate, person, body_part, edible, drinkable, abstract, concrete,
      place, time, readable, visual, sound, event, inanimate
    """

    # ConceptNet relation mappings
    # Maps target concepts from IsA/InstanceOf relations to our categories
    CONCEPTNET_MAPPINGS = {
        'person': [
            'person', 'human', 'human being', 'man', 'woman', 'child',
            'individual', 'someone', 'people', 'adult', 'professional'
        ],
        'animate': [
            'animal', 'living thing', 'organism', 'creature', 'being',
            'mammal', 'bird', 'fish', 'insect', 'reptile', 'plant'
        ],
        'edible': [
            'food', 'fruit', 'vegetable', 'meal', 'dish', 'meat',
            'snack', 'ingredient', 'grain', 'produce', 'edible'
        ],
        'drinkable': [
            'beverage', 'drink', 'liquid', 'fluid', 'juice', 'water',
            'alcoholic drink', 'tea', 'coffee'
        ],
        'place': [
            'location', 'place', 'building', 'city', 'town', 'country',
            'area', 'region', 'site', 'space', 'room', 'structure',
            'geographical feature', 'establishment', 'venue'
        ],
        'abstract': [
            'concept', 'idea', 'notion', 'quality', 'feeling', 'emotion',
            'state', 'condition', 'property', 'attribute', 'thought',
            'belief', 'theory', 'principle', 'relationship', 'abstraction'
        ],
        'concrete': [
            'object', 'thing', 'artifact', 'item', 'device', 'tool',
            'instrument', 'implement', 'article', 'product', 'material',
            'physical object', 'entity', 'piece'
        ],
        'time': [
            'time period', 'duration', 'moment', 'period', 'era', 'age',
            'time', 'point in time', 'interval', 'temporal unit', 'season'
        ],
        'event': [
            'event', 'occurrence', 'happening', 'activity', 'action',
            'process', 'phenomenon', 'incident', 'occasion', 'ceremony'
        ],
        'body_part': [
            'body part', 'organ', 'limb', 'anatomical structure',
            'part of body', 'anatomy'
        ],
        'readable': [
            'text', 'document', 'book', 'writing', 'publication',
            'written work', 'literature', 'manuscript'
        ],
        'visual': [
            'image', 'picture', 'visual representation', 'artwork',
            'visual object', 'display'
        ],
        'sound': [
            'sound', 'noise', 'audio', 'music', 'acoustic signal',
            'auditory sensation'
        ]
    }

    # Wikidata P31 (instance-of) mappings
    # Maps Wikidata QIDs to our categories
    WIKIDATA_MAPPINGS = {
        'person': [
            'Q5',       # human
            'Q215627',  # person
        ],
        'place': [
            'Q515',     # city
            'Q486972',  # human settlement
            'Q3957',    # town
            'Q532',     # village
            'Q41176',   # building
            'Q27096213', # geographic entity
            'Q17334923', # location
        ],
        'edible': [
            'Q2095',    # food
            'Q3314483', # dish (food)
            'Q3314483', # meal
        ],
        'drinkable': [
            'Q40050',   # beverage
            'Q8492',    # drink
        ],
        'animate': [
            'Q729',     # animal
            'Q16521',   # taxon (biological classification)
        ],
        'time': [
            'Q186081',  # time interval
            'Q1190554', # time period
        ],
        'event': [
            'Q1190554', # occurrence
            'Q1656682', # event
        ],
        'abstract': [
            'Q6881511', # concept
            'Q9081',    # idea
        ],
        'concrete': [
            'Q488383',  # object
            'Q223557',  # physical object
        ],
        'readable': [
            'Q7725634', # literary work
            'Q571',     # book
            'Q49848',   # document
        ],
        'visual': [
            'Q3305213', # visual artwork
            'Q478798',  # image
        ],
        'body_part': [
            'Q4936952', # anatomical structure
            'Q4167410', # organ
        ]
    }

    def __init__(self):
        """Initialize the category mapper with normalized lookup tables."""
        # Create lowercase lookup for ConceptNet
        self._cn_lookup = {}
        for category, terms in self.CONCEPTNET_MAPPINGS.items():
            for term in terms:
                self._cn_lookup[term.lower()] = category

        # Create lookup for Wikidata QIDs
        self._wd_lookup = {}
        for category, qids in self.WIKIDATA_MAPPINGS.items():
            for qid in qids:
                self._wd_lookup[qid] = category

    def map_conceptnet(self, target_label: str) -> Optional[str]:
        """Map ConceptNet target label to Klareco category.

        Args:
            target_label: Target label from ConceptNet relation (e.g., "living thing")

        Returns:
            Klareco category name, or None if no mapping found
        """
        if not target_label:
            return None

        # Try exact match first
        target_lower = target_label.lower().strip()
        if target_lower in self._cn_lookup:
            return self._cn_lookup[target_lower]

        # Try fuzzy matching (check if any keyword appears in the target)
        for term, category in self._cn_lookup.items():
            if term in target_lower or target_lower in term:
                return category

        return None

    def map_wikidata(self, instance_of_qid: str) -> Optional[str]:
        """Map Wikidata instance-of QID to Klareco category.

        Args:
            instance_of_qid: Wikidata QID (e.g., "Q5" for human)

        Returns:
            Klareco category name, or None if no mapping found
        """
        if not instance_of_qid:
            return None

        # Normalize QID format (remove URL prefix if present)
        qid = instance_of_qid.split('/')[-1] if '/' in instance_of_qid else instance_of_qid

        return self._wd_lookup.get(qid)

    def map_conceptnet_relations(self, relations: List[Dict]) -> Optional[str]:
        """Map a list of ConceptNet relations to best matching category.

        Args:
            relations: List of relation dicts with 'relation', 'start', 'end' keys

        Returns:
            Best matching Klareco category, or None if no match
        """
        # Try IsA and InstanceOf relations first (most reliable)
        for rel in relations:
            if rel.get('relation') in ['IsA', 'InstanceOf']:
                # Try mapping both start and end labels
                for label in [rel.get('end'), rel.get('start')]:
                    if label:
                        category = self.map_conceptnet(label)
                        if category:
                            return category

        # Fallback: try other relations
        for rel in relations:
            for label in [rel.get('end'), rel.get('start')]:
                if label:
                    category = self.map_conceptnet(label)
                    if category:
                        return category

        return None

    def get_category_stats(self) -> Dict[str, int]:
        """Get statistics about mapping coverage.

        Returns:
            Dict with category names and their mapping counts
        """
        cn_stats = {}
        for category, terms in self.CONCEPTNET_MAPPINGS.items():
            cn_stats[f'conceptnet_{category}'] = len(terms)

        wd_stats = {}
        for category, qids in self.WIKIDATA_MAPPINGS.items():
            wd_stats[f'wikidata_{category}'] = len(qids)

        return {**cn_stats, **wd_stats}
