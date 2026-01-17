#!/usr/bin/env python3
"""
Prepare M1 (Selectional Preference) Training Data

Extracts subject-verb-object triples from the parsed corpus and generates
training data for the M1 selectional preference model.

M1 learns compatibility between roots in grammatical roles:
- Subject-Verb compatibility: Can X be the subject of Y?
- Verb-Object compatibility: Can Y take Z as object?
- Triple plausibility: Is (X, Y, Z) plausible?

Training data format:
    {
        "subject_root": "hund",
        "verb_root": "manĝ",
        "object_root": "viand",
        "label": 1.0,  # 1.0 = plausible (from corpus), 0.0 = implausible (corrupted)
        "corruption": null,  # or "subject", "verb", "object"
        "source": "wikipedia",
        "original_text": "La hundo manĝas viandon."
    }

Negative sampling strategies:
1. Corrupt subject: Replace with random noun
2. Corrupt object: Replace with random noun
3. Corrupt verb: Replace with random verb

Usage:
    python scripts/prepare_m1_training_data.py
    python scripts/prepare_m1_training_data.py --output data/training/m1_triples.jsonl
    python scripts/prepare_m1_training_data.py --negatives-per-positive 2
    python scripts/prepare_m1_training_data.py --max-triples 100000
"""

import argparse
import json
import logging
import random
import sys
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


def extract_svo_triple(ast: Dict) -> Optional[Tuple[str, str, str]]:
    """
    Extract (subject_root, verb_root, object_root) from AST.

    Returns:
        (subj, verb, obj) tuple if all three are present, else None
    """
    # Check structure exists
    if not all(k in ast for k in ['subjekto', 'verbo', 'objekto']):
        return None

    subjekto = ast['subjekto']
    verbo = ast['verbo']
    objekto = ast['objekto']

    # Extract roots
    try:
        # Subject and object have 'kerno' structure
        subj_root = subjekto.get('kerno', {}).get('radiko') if isinstance(subjekto, dict) else None
        verb_root = verbo.get('radiko') if isinstance(verbo, dict) else None
        obj_root = objekto.get('kerno', {}).get('radiko') if isinstance(objekto, dict) else None

        if subj_root and verb_root and obj_root:
            return (subj_root.lower(), verb_root.lower(), obj_root.lower())
    except (AttributeError, KeyError, TypeError):
        pass

    return None


def load_corpus_triples(
    corpus_path: Path,
    max_triples: Optional[int] = None,
    min_parse_rate: float = 0.7
) -> Tuple[List[Dict], Dict[str, Set[str]]]:
    """
    Load positive triples from corpus.

    Returns:
        (triples, vocabularies) where:
        - triples: List of dicts with subject_root, verb_root, object_root, etc.
        - vocabularies: Dict with 'nouns', 'verbs' sets
    """
    logger.info(f"Loading triples from {corpus_path}")

    triples = []
    nouns = set()
    verbs = set()

    triple_counts = Counter()  # Count frequency of each triple

    with open(corpus_path) as f:
        for i, line in enumerate(f):
            if i % 100000 == 0:
                logger.info(f"  Processed {i:,} sentences, found {len(triples):,} triples")

            if max_triples and len(triples) >= max_triples:
                break

            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Skip low parse rate
            if entry.get('parse_rate', 0) < min_parse_rate:
                continue

            # Extract triple
            ast = entry.get('ast')
            if not ast:
                continue

            triple = extract_svo_triple(ast)
            if not triple:
                continue

            subj, verb, obj = triple

            # Track vocabularies
            nouns.add(subj)
            nouns.add(obj)
            verbs.add(verb)

            # Count triple frequency
            triple_counts[triple] += 1

            # Store triple
            triples.append({
                'subject_root': subj,
                'verb_root': verb,
                'object_root': obj,
                'label': 1.0,
                'corruption': None,
                'source': entry.get('source', 'unknown'),
                'original_text': entry.get('text', ''),
                'frequency': 1  # Will be updated below
            })

    # Update frequencies
    for triple_dict in triples:
        key = (triple_dict['subject_root'], triple_dict['verb_root'], triple_dict['object_root'])
        triple_dict['frequency'] = triple_counts[key]

    logger.info(f"Loaded {len(triples):,} positive triples")
    logger.info(f"  Unique nouns: {len(nouns):,}")
    logger.info(f"  Unique verbs: {len(verbs):,}")

    vocabularies = {
        'nouns': nouns,
        'verbs': verbs
    }

    return triples, vocabularies


def generate_negative_samples(
    positive_triples: List[Dict],
    vocabularies: Dict[str, Set[str]],
    negatives_per_positive: int = 1
) -> List[Dict]:
    """
    Generate negative samples by corrupting positive triples.

    Corruption strategies:
    1. Corrupt subject: (X, verb, obj) → (X', verb, obj) where X' is random noun
    2. Corrupt object: (subj, verb, Y) → (subj, verb, Y') where Y' is random noun
    3. Corrupt verb: (subj, Z, obj) → (subj, Z', obj) where Z' is random verb

    Returns:
        List of negative triple dicts
    """
    logger.info(f"Generating {negatives_per_positive} negative samples per positive")

    nouns = list(vocabularies['nouns'])
    verbs = list(vocabularies['verbs'])

    negatives = []
    corruption_types = ['subject', 'object', 'verb']

    total_to_generate = len(positive_triples) * negatives_per_positive

    for idx, pos_triple in enumerate(positive_triples):
        # Progress updates every 100k triples
        if idx > 0 and idx % 100000 == 0:
            logger.info(f"  Generated {len(negatives):,} / {total_to_generate:,} negatives ({100*len(negatives)/total_to_generate:.1f}%)")

        subj = pos_triple['subject_root']
        verb = pos_triple['verb_root']
        obj = pos_triple['object_root']

        for _ in range(negatives_per_positive):
            # Choose corruption type
            corruption = random.choice(corruption_types)

            if corruption == 'subject':
                # Replace subject with random noun (avoid same)
                candidates = [n for n in nouns if n != subj]
                corrupted_subj = random.choice(candidates) if candidates else subj
                neg_triple = {
                    'subject_root': corrupted_subj,
                    'verb_root': verb,
                    'object_root': obj,
                    'label': 0.0,
                    'corruption': 'subject',
                    'source': pos_triple['source'],
                    'original_text': pos_triple['original_text']
                }

            elif corruption == 'object':
                # Replace object with random noun (avoid same)
                candidates = [n for n in nouns if n != obj]
                corrupted_obj = random.choice(candidates) if candidates else obj
                neg_triple = {
                    'subject_root': subj,
                    'verb_root': verb,
                    'object_root': corrupted_obj,
                    'label': 0.0,
                    'corruption': 'object',
                    'source': pos_triple['source'],
                    'original_text': pos_triple['original_text']
                }

            else:  # corruption == 'verb'
                # Replace verb with random verb (avoid same)
                candidates = [v for v in verbs if v != verb]
                corrupted_verb = random.choice(candidates) if candidates else verb
                neg_triple = {
                    'subject_root': subj,
                    'verb_root': corrupted_verb,
                    'object_root': obj,
                    'label': 0.0,
                    'corruption': 'verb',
                    'source': pos_triple['source'],
                    'original_text': pos_triple['original_text']
                }

            negatives.append(neg_triple)

    logger.info(f"Generated {len(negatives):,} negative samples")

    return negatives


def split_data(
    data: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split data into train/val/test sets."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    random.seed(seed)
    random.shuffle(data)

    n = len(data)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train = data[:train_end]
    val = data[train_end:val_end]
    test = data[val_end:]

    logger.info(f"Split: train={len(train):,}, val={len(val):,}, test={len(test):,}")

    return train, val, test


def save_splits(
    train: List[Dict],
    val: List[Dict],
    test: List[Dict],
    output_dir: Path
):
    """Save train/val/test splits to JSONL files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save each split
    for split_name, split_data in [('train', train), ('val', val), ('test', test)]:
        output_path = output_dir / f"{split_name}.jsonl"

        with open(output_path, 'w') as f:
            for item in split_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        logger.info(f"Saved {len(split_data):,} examples to {output_path}")

    # Save vocabulary
    vocab_path = output_dir / "vocabulary.json"

    # Collect unique roots from train set only
    nouns = set()
    verbs = set()
    for item in train:
        nouns.add(item['subject_root'])
        nouns.add(item['object_root'])
        verbs.add(item['verb_root'])

    vocab = {
        'nouns': sorted(list(nouns)),
        'verbs': sorted(list(verbs)),
        'num_nouns': len(nouns),
        'num_verbs': len(verbs)
    }

    with open(vocab_path, 'w') as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved vocabulary to {vocab_path}")
    logger.info(f"  Nouns: {len(nouns):,}")
    logger.info(f"  Verbs: {len(verbs):,}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare M1 selectional preference training data"
    )
    parser.add_argument(
        '--corpus',
        type=Path,
        default=Path('data/enhanced_corpus/corpus_with_metadata.jsonl'),
        help='Path to parsed corpus (default: data/enhanced_corpus/corpus_with_metadata.jsonl)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/training/m1_selectional'),
        help='Output directory for training data (default: data/training/m1_selectional)'
    )
    parser.add_argument(
        '--max-triples',
        type=int,
        default=None,
        help='Maximum number of positive triples to extract (default: all)'
    )
    parser.add_argument(
        '--negatives-per-positive',
        type=int,
        default=1,
        help='Number of negative samples per positive (default: 1)'
    )
    parser.add_argument(
        '--min-parse-rate',
        type=float,
        default=0.7,
        help='Minimum parse rate to include sentence (default: 0.7)'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help='Train set ratio (default: 0.8)'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.1,
        help='Validation set ratio (default: 0.1)'
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.1,
        help='Test set ratio (default: 0.1)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )

    args = parser.parse_args()

    # Validate corpus exists
    if not args.corpus.exists():
        logger.error(f"Corpus not found: {args.corpus}")
        return 1

    # Load positive triples
    positive_triples, vocabularies = load_corpus_triples(
        args.corpus,
        max_triples=args.max_triples,
        min_parse_rate=args.min_parse_rate
    )

    if not positive_triples:
        logger.error("No valid triples found in corpus")
        return 1

    # Generate negative samples
    negative_triples = generate_negative_samples(
        positive_triples,
        vocabularies,
        negatives_per_positive=args.negatives_per_positive
    )

    # Combine and shuffle
    all_data = positive_triples + negative_triples
    logger.info(f"Total examples: {len(all_data):,} ({len(positive_triples):,} positive, {len(negative_triples):,} negative)")

    # Split into train/val/test
    train, val, test = split_data(
        all_data,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )

    # Save splits
    save_splits(train, val, test, args.output_dir)

    # Save metadata
    metadata = {
        'corpus_path': str(args.corpus),
        'total_examples': len(all_data),
        'positive_examples': len(positive_triples),
        'negative_examples': len(negative_triples),
        'negatives_per_positive': args.negatives_per_positive,
        'min_parse_rate': args.min_parse_rate,
        'train_size': len(train),
        'val_size': len(val),
        'test_size': len(test),
        'unique_nouns': len(vocabularies['nouns']),
        'unique_verbs': len(vocabularies['verbs']),
        'seed': args.seed
    }

    metadata_path = args.output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved metadata to {metadata_path}")
    logger.info("Done!")

    return 0


if __name__ == '__main__':
    sys.exit(main())
