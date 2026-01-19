#!/usr/bin/env python3
"""
Generate M1 Training Data with Semantic Violations

Uses expanded semantic categories to create selectional preference violations.

Strategy:
1. Extract positive (plausible) triples from corpus
2. Generate semantic violations:
   - Verb selectional constraints (e.g., manĝ requires edible object)
   - Type mismatches (e.g., abstract subject for physical verb)
   - Animate/inanimate violations
3. Mix with some random corruption (hybrid approach)

Output: data/training/m1_semantic_violations/
"""

import argparse
import json
import logging
import random
import sys
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Selectional constraints for common verbs
VERB_CONSTRAINTS = {
    # Eating/drinking verbs require edible/drinkable objects
    'manĝ': {'subject': ['animate', 'person'], 'object': ['edible']},
    'trink': {'subject': ['animate', 'person'], 'object': ['drinkable']},
    'glut': {'subject': ['animate', 'person'], 'object': ['edible']},

    # Cognitive verbs require animate subjects
    'pens': {'subject': ['animate', 'person'], 'object': ['abstract', 'concrete']},
    'sci': {'subject': ['animate', 'person'], 'object': ['abstract', 'concrete']},
    'kompr': {'subject': ['animate', 'person'], 'object': ['abstract', 'concrete']},
    'kred': {'subject': ['animate', 'person'], 'object': ['abstract']},
    'memor': {'subject': ['animate', 'person'], 'object': ['abstract', 'concrete']},
    'sag': {'subject': ['animate', 'person'], 'object': ['abstract', 'concrete']},

    # Communication verbs require animate subjects
    'dir': {'subject': ['animate', 'person'], 'object': ['abstract']},
    'parol': {'subject': ['animate', 'person'], 'object': ['abstract']},
    'demand': {'subject': ['animate', 'person'], 'object': ['abstract']},
    'respond': {'subject': ['animate', 'person'], 'object': ['abstract']},

    # Reading verbs require readable objects
    'leg': {'subject': ['animate', 'person'], 'object': ['readable', 'abstract']},
    'stud': {'subject': ['animate', 'person'], 'object': ['readable', 'abstract']},

    # Perception verbs
    'vid': {'subject': ['animate', 'person'], 'object': ['concrete', 'person', 'visual']},
    'aŭd': {'subject': ['animate', 'person'], 'object': ['sound', 'concrete']},
    'sent': {'subject': ['animate', 'person'], 'object': ['abstract', 'concrete']},
    'rigard': {'subject': ['animate', 'person'], 'object': ['concrete', 'person', 'visual']},

    # Physical action verbs
    'pren': {'subject': ['animate', 'person'], 'object': ['concrete']},
    'met': {'subject': ['animate', 'person'], 'object': ['concrete']},
    'port': {'subject': ['animate', 'person'], 'object': ['concrete']},
}


def load_semantic_categories(path: Path) -> Dict[str, List[str]]:
    """Load expanded semantic categories."""
    with open(path) as f:
        categories = json.load(f)

    # Build reverse mapping
    word_to_categories = defaultdict(set)
    for category, words in categories.items():
        for word in words:
            word_to_categories[word].add(category)

    logger.info(f"Loaded {len(categories)} semantic categories")
    logger.info(f"Total categorized words: {sum(len(w) for w in categories.values())}")

    return categories, dict(word_to_categories)


def extract_svo_triple(ast: Dict) -> Optional[Tuple[str, str, str]]:
    """Extract (subject_root, verb_root, object_root) from AST."""
    if not all(k in ast for k in ['subjekto', 'verbo', 'objekto']):
        return None

    try:
        subjekto = ast['subjekto']
        verbo = ast['verbo']
        objekto = ast['objekto']

        subj_root = subjekto.get('kerno', {}).get('radiko') if isinstance(subjekto, dict) else None
        verb_root = verbo.get('radiko') if isinstance(verbo, dict) else None
        obj_root = objekto.get('kerno', {}).get('radiko') if isinstance(objekto, dict) else None

        if subj_root and verb_root and obj_root:
            return (subj_root.lower(), verb_root.lower(), obj_root.lower())
    except (AttributeError, KeyError):
        pass

    return None


def generate_semantic_violation(
    triple: Dict,
    categories: Dict[str, List[str]],
    word_to_cat: Dict[str, Set[str]]
) -> Optional[Dict]:
    """
    Generate semantic violation by replacing word with incompatible category.

    Returns violation dict or None if can't generate good violation.
    """
    subj = triple['subject_root']
    verb = triple['verb_root']
    obj = triple['object_root']

    # Check if verb has selectional constraints
    if verb in VERB_CONSTRAINTS:
        constraints = VERB_CONSTRAINTS[verb]

        # Try object violation (most common)
        if 'object' in constraints:
            required_cats = constraints['object']
            obj_cats = word_to_cat.get(obj, set())

            # If object matches requirement, corrupt it with incompatible category
            if any(cat in required_cats for cat in obj_cats):
                # Find incompatible categories
                incompatible = [
                    cat for cat in categories.keys()
                    if cat not in required_cats and cat not in ['inanimate', 'concrete']
                ]

                if incompatible:
                    violation_cat = random.choice(incompatible)
                    if categories[violation_cat]:
                        new_obj = random.choice(categories[violation_cat])
                        return {
                            'subject_root': subj,
                            'verb_root': verb,
                            'object_root': new_obj,
                            'label': 0.0,
                            'corruption': f'object_semantic_{violation_cat}',
                            'source': triple['source'],
                            'original_text': triple.get('original_text', '')
                        }

        # Try subject violation
        if 'subject' in constraints:
            required_cats = constraints['subject']
            subj_cats = word_to_cat.get(subj, set())

            if any(cat in required_cats for cat in subj_cats):
                # Corrupt with inanimate if requires animate
                if 'animate' in required_cats or 'person' in required_cats:
                    incompatible_cats = ['concrete', 'abstract', 'inanimate']
                    for cat in incompatible_cats:
                        if categories.get(cat):
                            new_subj = random.choice(categories[cat])
                            return {
                                'subject_root': new_subj,
                                'verb_root': verb,
                                'object_root': obj,
                                'label': 0.0,
                                'corruption': f'subject_semantic_{cat}',
                                'source': triple['source'],
                                'original_text': triple.get('original_text', '')
                            }

    return None


def main():
    parser = argparse.ArgumentParser(description='Generate M1 training data with semantic violations')
    parser.add_argument('--corpus', type=str,
                        default='data/corpus/unified_corpus.jsonl',
                        help='Parsed corpus path')
    parser.add_argument('--output-dir', type=str,
                        default='data/training/m1_semantic_violations',
                        help='Output directory')
    parser.add_argument('--semantic-categories', type=str,
                        default='data/vocabularies/semantic_categories_merged.json',
                        help='Semantic categories file (merged from manual + external sources)')
    parser.add_argument('--max-triples', type=int, default=50000,
                        help='Maximum positive triples to extract')
    parser.add_argument('--semantic-ratio', type=float, default=0.7,
                        help='Ratio of semantic violations (rest are random)')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("M1 Semantic Violations Data Generation")
    logger.info("=" * 70)
    logger.info(f"Corpus: {args.corpus}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Semantic ratio: {args.semantic_ratio:.1%}")
    logger.info("")

    # Load semantic categories
    categories, word_to_cat = load_semantic_categories(Path(args.semantic_categories))
    logger.info("")

    # Extract positive triples from corpus
    logger.info("Extracting positive triples from corpus...")
    positive_triples = []

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        logger.error(f"Corpus not found: {corpus_path}")
        return 1

    with open(corpus_path) as f:
        for i, line in enumerate(f):
            if i >= args.max_triples * 2:  # Read more to get enough valid ones
                break

            if i > 0 and i % 10000 == 0:
                logger.info(f"  Processed {i:,} sentences, extracted {len(positive_triples):,} triples")

            try:
                data = json.loads(line)
                if 'ast' in data:
                    triple = extract_svo_triple(data['ast'])
                    if triple:
                        subj, verb, obj = triple
                        positive_triples.append({
                            'subject_root': subj,
                            'verb_root': verb,
                            'object_root': obj,
                            'label': 1.0,
                            'corruption': None,
                            'source': data.get('metadata', {}).get('source', 'corpus'),
                            'original_text': data.get('sentence', '')
                        })

                        if len(positive_triples) >= args.max_triples:
                            break
            except json.JSONDecodeError:
                continue

    logger.info(f"✓ Extracted {len(positive_triples):,} positive triples")
    logger.info("")

    # Generate negative samples
    logger.info("Generating negative samples...")
    negatives = []
    semantic_violations = 0
    random_corruptions = 0

    for i, pos_triple in enumerate(positive_triples):
        if i > 0 and i % 5000 == 0:
            logger.info(f"  Generated {len(negatives):,} negatives ({semantic_violations} semantic, {random_corruptions} random)")

        # Try semantic violation first
        if random.random() < args.semantic_ratio:
            violation = generate_semantic_violation(pos_triple, categories, word_to_cat)
            if violation:
                negatives.append(violation)
                semantic_violations += 1
                continue

        # Fallback to random corruption
        corruption_type = random.choice(['subject', 'object', 'verb'])
        if corruption_type == 'subject':
            all_nouns = [w for words in categories.values() for w in words]
            if all_nouns:
                new_subj = random.choice(all_nouns)
                negatives.append({
                    'subject_root': new_subj,
                    'verb_root': pos_triple['verb_root'],
                    'object_root': pos_triple['object_root'],
                    'label': 0.0,
                    'corruption': 'subject_random',
                    'source': pos_triple['source'],
                    'original_text': pos_triple['original_text']
                })
                random_corruptions += 1
        elif corruption_type == 'object':
            all_nouns = [w for words in categories.values() for w in words]
            if all_nouns:
                new_obj = random.choice(all_nouns)
                negatives.append({
                    'subject_root': pos_triple['subject_root'],
                    'verb_root': pos_triple['verb_root'],
                    'object_root': new_obj,
                    'label': 0.0,
                    'corruption': 'object_random',
                    'source': pos_triple['source'],
                    'original_text': pos_triple['original_text']
                })
                random_corruptions += 1

    logger.info(f"✓ Generated {len(negatives):,} negative samples")
    logger.info(f"  Semantic violations: {semantic_violations:,} ({semantic_violations/len(negatives)*100:.1f}%)")
    logger.info(f"  Random corruptions: {random_corruptions:,} ({random_corruptions/len(negatives)*100:.1f}%)")
    logger.info("")

    # Combine and split
    all_data = positive_triples + negatives
    random.shuffle(all_data)

    # Split 80/10/10
    train_size = int(0.8 * len(all_data))
    val_size = int(0.1 * len(all_data))

    train_data = all_data[:train_size]
    val_data = all_data[train_size:train_size + val_size]
    test_data = all_data[train_size + val_size:]

    # Save splits
    for split_name, split_data in [('train', train_data), ('val', val_data), ('test', test_data)]:
        output_path = output_dir / f'{split_name}.jsonl'
        with open(output_path, 'w') as f:
            for item in split_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        logger.info(f"✓ Saved {split_name}: {len(split_data):,} examples -> {output_path}")

    # Save metadata
    metadata = {
        'total_examples': len(all_data),
        'train_examples': len(train_data),
        'val_examples': len(val_data),
        'test_examples': len(test_data),
        'positive_examples': len(positive_triples),
        'negative_examples': len(negatives),
        'semantic_violations': semantic_violations,
        'random_corruptions': random_corruptions,
        'semantic_ratio': args.semantic_ratio
    }

    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info("")
    logger.info("=" * 70)
    logger.info("✓ Data generation complete!")
    logger.info(f"Total: {len(all_data):,} examples")
    logger.info(f"Train: {len(train_data):,} | Val: {len(val_data):,} | Test: {len(test_data):,}")
    logger.info("=" * 70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
