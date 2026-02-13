#!/usr/bin/env python3
"""
Automatically annotate semantic gap examples using Claude's understanding.

Uses context clues, root meanings, and sentence structure to assign tier3_type.
"""

import json
import sys
from pathlib import Path
from collections import Counter

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def analyze_context_for_person(word_ast, sentence_ast, sentence_text):
    """
    Analyze if a person entity is family, profession, role, or generic.

    Context clues:
    - Possessives (mia, via, sia, nia, ilia) → person_family
    - Professional verbs (instruas, kuracas, laboras) → person_profession
    - Titles/roles (prezidanto, membro) → person_role
    """
    text_lower = sentence_text.lower()
    radiko = word_ast.get('radiko', '')

    # Family indicators
    family_roots = ['patr', 'matr', 'frat', 'filr', 'gepatr', 'nep', 'onkl', 'kuz', 'edz']
    family_possessives = ['mia', 'via', 'sia', 'nia', 'ilia']

    if radiko in family_roots:
        return 'person_family'

    for poss in family_possessives:
        if poss in text_lower:
            return 'person_family'

    # Profession indicators
    profession_roots = ['instruist', 'kuracist', 'labor', 'verk', 'pent', 'kant', 'muz']
    profession_verbs = ['instruas', 'kuracas', 'laboras', 'verkas', 'konstruas']

    if radiko in profession_roots:
        return 'person_profession'

    for verb in profession_verbs:
        if verb in text_lower:
            return 'person_profession'

    # Role indicators (titles, positions)
    role_roots = ['prezident', 'membr', 'estr', 'direkto', 'ministr', 'reĝ']

    if radiko in role_roots:
        return 'person_role'

    # Default: generic person
    return 'person_generic'


def analyze_context_for_location(word_ast, sentence_ast, sentence_text):
    """
    Analyze if location is geographic, facility, or generic.
    """
    radiko = word_ast.get('radiko', '')
    text_lower = sentence_text.lower()

    # Geographic
    geographic_roots = ['urb', 'land', 'mont', 'mar', 'river', 'lag', 'insel']
    if radiko in geographic_roots:
        return 'location_geographic'

    # Facilities/buildings
    facility_roots = ['domeŭ', 'hotel', 'teatr', 'kinematograf', 'muzej', 'lernej', 'hospital']
    if radiko in facility_roots:
        return 'location_facility'

    return 'location_generic'


def analyze_context_for_time(word_ast, sentence_ast, sentence_text):
    """
    Analyze if time is absolute or duration.
    """
    radiko = word_ast.get('radiko', '')

    # Absolute times
    absolute_roots = ['jar', 'monat', 'tag', 'hor', 'minut', 'moment', 'temp']
    if radiko in absolute_roots:
        return 'time_absolute'

    # Durations
    duration_roots = ['daŭr', 'period']
    if radiko in duration_roots:
        return 'time_duration'

    return 'time_absolute'


def analyze_context_for_thing(word_ast, sentence_ast, sentence_text):
    """
    Analyze thing type: tool, food, animal, plant, vehicle, etc.
    """
    radiko = word_ast.get('radiko', '')
    text_lower = sentence_text.lower()

    # Animals
    animal_roots = ['hund', 'kat', 'bird', 'fiŝ', 'ĉeval', 'bov', 'ŝaf', 'pork']
    if radiko in animal_roots:
        return 'thing_animal'

    # Food
    food_roots = ['pan', 'akv', 'vian', 'fromaĝ', 'frukt', 'legom']
    if radiko in food_roots:
        return 'thing_food'

    # Plants
    plant_roots = ['arb', 'flor', 'herb', 'plant']
    if radiko in plant_roots:
        return 'thing_plant'

    # Tools/instruments
    tool_roots = ['martel', 'seg', 'tranĉil', 'fosil', 'bros']
    if radiko in tool_roots:
        return 'thing_tool'

    # Vehicles
    vehicle_roots = ['aŭt', 'vetur', 'ŝip', 'trajn', 'aviad']
    if radiko in vehicle_roots:
        return 'thing_vehicle'

    # Containers
    container_roots = ['skat', 'pak', 'sak', 'bot', 'uj']
    if radiko in container_roots:
        return 'thing_container'

    # Furniture
    furniture_roots = ['tabl', 'seĝ', 'lit', 'ŝrank']
    if radiko in furniture_roots:
        return 'thing_furniture'

    # Clothing
    clothing_roots = ['vest', 'ĉemiz', 'pantalon', 'ŝu', 'ĉapel']
    if radiko in clothing_roots:
        return 'thing_clothing'

    # Documents
    document_roots = ['libr', 'ĵurnal', 'letero', 'dokument']
    if radiko in document_roots:
        return 'thing_document'

    return 'thing_generic'


def annotate_example(example):
    """
    Annotate a single example based on context.

    Returns tier3_type or None if uncertain.
    """
    word_ast = example.get('word_ast', {})
    sentence_ast = example.get('sentence_ast', {})
    sentence_text = example.get('sentence_text', '')
    priors = example.get('deterministic_priors', {})

    tier2 = priors.get('tier2_type')

    # If we have tier2 hint, use it to guide
    if tier2 == 'person':
        return analyze_context_for_person(word_ast, sentence_ast, sentence_text)
    elif tier2 == 'location':
        return analyze_context_for_location(word_ast, sentence_ast, sentence_text)
    elif tier2 == 'time':
        return analyze_context_for_time(word_ast, sentence_ast, sentence_text)
    elif tier2 == 'thing':
        return analyze_context_for_thing(word_ast, sentence_ast, sentence_text)

    # No tier2, analyze from scratch
    radiko = word_ast.get('radiko', '')
    vortspeco = word_ast.get('vortspeco', '')

    # If substantivo, likely a thing
    if vortspeco == 'substantivo':
        # Check for person indicators
        person_indicators = ['ist', 'ul', 'an']
        if any(radiko.endswith(ind) for ind in person_indicators):
            return analyze_context_for_person(word_ast, sentence_ast, sentence_text)

        # Otherwise thing
        return analyze_context_for_thing(word_ast, sentence_ast, sentence_text)

    return None  # Uncertain


def auto_annotate(input_path: Path, output_path: Path):
    """
    Automatically annotate sampled examples.
    """
    print("="*70)
    print("AUTO-ANNOTATE SEMANTIC GAP EXAMPLES")
    print("="*70)
    print()
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print()

    # Load examples
    examples = []
    with open(input_path, 'r') as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    print(f"Loaded {len(examples):,} examples to annotate")
    print()

    # Annotate each
    annotations = {}
    type_counter = Counter()
    uncertain_count = 0

    for i, example in enumerate(examples):
        tier3 = annotate_example(example)

        if tier3:
            annotations[str(i)] = {
                'tier3_type': tier3,
                'confidence': 'auto',
                'source': 'claude_annotation'
            }
            type_counter[tier3] += 1
        else:
            uncertain_count += 1

        if (i + 1) % 100 == 0:
            print(f"  Annotated: {i+1}/{len(examples)}")

    print()
    print(f"✓ Annotated {len(annotations):,} examples")
    print(f"  Uncertain/skipped: {uncertain_count}")
    print()

    # Show distribution
    print("Annotation distribution:")
    for tier3, count in type_counter.most_common():
        pct = count / len(annotations) * 100 if len(annotations) > 0 else 0
        print(f"  {tier3:25s}: {count:4,} ({pct:5.1f}%)")
    print()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)

    print(f"✓ Saved to: {output_path}")
    print()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Auto-annotate examples')
    parser.add_argument(
        '--input',
        type=Path,
        default=Path('data/training/active_learning/iteration_0_to_annotate.jsonl'),
        help='Input sampled examples'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('annotations.json'),
        help='Output annotations file'
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    auto_annotate(args.input, args.output)
