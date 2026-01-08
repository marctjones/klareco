#!/usr/bin/env python3
"""
Prepare Semantic Training Data for Contrastive Learning.

Generates triplet training data from SemanticRelationDB:
- Anchor: a root word
- Positive: a synonym of the anchor
- Negative: a random unrelated root

This enables training embeddings that capture semantic similarity,
not just distributional co-occurrence.

Usage:
    python scripts/prepare_semantic_training_data.py
    python scripts/prepare_semantic_training_data.py --output data/training/semantic_triplets.jsonl
    python scripts/prepare_semantic_training_data.py --negatives-per-pair 5
"""

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Set, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


def load_semantic_relations(relations_path: Path) -> Dict:
    """Load semantic relations from JSON file."""
    with open(relations_path) as f:
        return json.load(f)


def load_root_vocabulary(vocab_path: Path) -> Set[str]:
    """Load root vocabulary to filter valid roots."""
    if not vocab_path.exists():
        logger.warning(f"Vocabulary not found: {vocab_path}")
        return set()

    with open(vocab_path) as f:
        vocab_data = json.load(f)

    # Handle different vocab formats
    if isinstance(vocab_data, dict):
        return set(vocab_data.keys())
    elif isinstance(vocab_data, list):
        return set(vocab_data)
    return set()


def extract_synonym_pairs(relations: Dict, valid_roots: Set[str]) -> List[Tuple[str, str]]:
    """Extract all synonym pairs from relations."""
    pairs = []
    synonyms = relations.get('synonyms', {})

    for root, syn_list in synonyms.items():
        root = root.lower()
        if valid_roots and root not in valid_roots:
            continue

        for syn in syn_list:
            syn = syn.lower()
            if valid_roots and syn not in valid_roots:
                continue
            if root != syn:
                pairs.append((root, syn))

    logger.info(f"Extracted {len(pairs)} synonym pairs")
    return pairs


def extract_hypernym_pairs(relations: Dict, valid_roots: Set[str]) -> List[Tuple[str, str]]:
    """Extract hypernym pairs (word, more_general_word)."""
    pairs = []
    hypernyms = relations.get('hypernyms', {})

    for root, hyper_list in hypernyms.items():
        root = root.lower()
        if valid_roots and root not in valid_roots:
            continue

        for hyper in hyper_list:
            hyper = hyper.lower()
            if valid_roots and hyper not in valid_roots:
                continue
            if root != hyper:
                pairs.append((root, hyper))

    logger.info(f"Extracted {len(pairs)} hypernym pairs")
    return pairs


def extract_hyponym_pairs(relations: Dict, valid_roots: Set[str]) -> List[Tuple[str, str]]:
    """Extract hyponym pairs (word, more_specific_word)."""
    pairs = []
    hyponyms = relations.get('hyponyms', {})

    for root, hypo_list in hyponyms.items():
        root = root.lower()
        if valid_roots and root not in valid_roots:
            continue

        for hypo in hypo_list:
            hypo = hypo.lower()
            if valid_roots and hypo not in valid_roots:
                continue
            if root != hypo:
                pairs.append((root, hypo))

    logger.info(f"Extracted {len(pairs)} hyponym pairs")
    return pairs


def build_related_set(pairs: List[Tuple[str, str]]) -> Dict[str, Set[str]]:
    """Build mapping of each root to all its related roots."""
    related = defaultdict(set)
    for a, b in pairs:
        related[a].add(b)
        related[b].add(a)
    return related


def generate_triplets(
    pairs: List[Tuple[str, str]],
    all_roots: List[str],
    related: Dict[str, Set[str]],
    negatives_per_pair: int = 3,
) -> List[Dict]:
    """
    Generate triplets: (anchor, positive, negative).

    For each synonym pair, we generate multiple triplets with different negatives.
    Negatives are sampled from roots that are NOT related to the anchor.
    """
    triplets = []

    for anchor, positive in pairs:
        # Find valid negatives (not related to anchor)
        related_to_anchor = related.get(anchor, set()) | {anchor, positive}
        valid_negatives = [r for r in all_roots if r not in related_to_anchor]

        if not valid_negatives:
            continue

        # Sample negatives
        n_neg = min(negatives_per_pair, len(valid_negatives))
        negatives = random.sample(valid_negatives, n_neg)

        for negative in negatives:
            triplets.append({
                'anchor': anchor,
                'positive': positive,
                'negative': negative,
            })

    return triplets


def save_triplets(triplets: List[Dict], output_path: Path):
    """Save triplets to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for triplet in triplets:
            f.write(json.dumps(triplet, ensure_ascii=False) + '\n')

    logger.info(f"Saved {len(triplets)} triplets to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare semantic training data")
    parser.add_argument(
        "--relations",
        type=Path,
        default=PROJECT_ROOT / "data/raw/eo/dictionaries/revo/revo_semantic_relations.json",
        help="Path to semantic relations JSON",
    )
    parser.add_argument(
        "--vocab",
        type=Path,
        default=PROJECT_ROOT / "data/vocabularies/root_vocab.json",
        help="Path to root vocabulary (for filtering)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data/training/semantic_triplets.jsonl",
        help="Output path for triplet training data",
    )
    parser.add_argument(
        "--negatives-per-pair",
        type=int,
        default=3,
        help="Number of negative samples per positive pair",
    )
    parser.add_argument(
        "--include-hypernyms",
        action="store_true",
        help="Include hypernym/hyponym pairs as positives (with lower weight)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()
    random.seed(args.seed)

    # Check input files
    if not args.relations.exists():
        logger.error(f"Relations file not found: {args.relations}")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Preparing Semantic Training Data")
    logger.info("=" * 60)

    # Load data
    logger.info(f"Loading relations from {args.relations}")
    relations = load_semantic_relations(args.relations)

    logger.info(f"Loading vocabulary from {args.vocab}")
    valid_roots = load_root_vocabulary(args.vocab)
    logger.info(f"Vocabulary size: {len(valid_roots)} roots")

    # Extract pairs
    synonym_pairs = extract_synonym_pairs(relations, valid_roots)

    all_pairs = list(synonym_pairs)

    if args.include_hypernyms:
        hypernym_pairs = extract_hypernym_pairs(relations, valid_roots)
        hyponym_pairs = extract_hyponym_pairs(relations, valid_roots)
        all_pairs.extend(hypernym_pairs)
        all_pairs.extend(hyponym_pairs)

    logger.info(f"Total positive pairs: {len(all_pairs)}")

    # Build related set for negative sampling
    related = build_related_set(all_pairs)

    # Get all unique roots
    all_roots = list(set(r for pair in all_pairs for r in pair))
    logger.info(f"Unique roots in training data: {len(all_roots)}")

    # Generate triplets
    logger.info(f"Generating triplets with {args.negatives_per_pair} negatives per pair...")
    triplets = generate_triplets(
        all_pairs,
        all_roots,
        related,
        negatives_per_pair=args.negatives_per_pair,
    )

    # Shuffle triplets
    random.shuffle(triplets)

    # Save
    save_triplets(triplets, args.output)

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    logger.info(f"Synonym pairs: {len(synonym_pairs)}")
    if args.include_hypernyms:
        logger.info(f"Hypernym pairs: {len(hypernym_pairs)}")
        logger.info(f"Hyponym pairs: {len(hyponym_pairs)}")
    logger.info(f"Total triplets: {len(triplets)}")
    logger.info(f"Output: {args.output}")

    # Show sample triplets
    logger.info("")
    logger.info("Sample triplets:")
    for t in triplets[:5]:
        logger.info(f"  {t['anchor']} ~ {t['positive']} ≠ {t['negative']}")


if __name__ == "__main__":
    main()
