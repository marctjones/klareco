#!/usr/bin/env python3
"""
Generate training pairs for Tree-LSTM from properly segmented corpus.

Strategy:
- Positive pairs: Sentences from same source within sliding window (contextually similar)
- Negative pairs: Random sentences from different sources (dissimilar)

This creates a much larger and higher quality training set than the original 5,495 pairs.

Usage:
    python scripts/generate_training_pairs.py --corpus data/corpus_sentences.jsonl --output data/training_pairs_v2 --positive-pairs 10000 --negative-pairs 50000
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict


def load_corpus_by_source(corpus_path: Path) -> Dict[str, List[Dict]]:
    """
    Load corpus grouped by source.

    Returns:
        Dict mapping source_id -> list of sentence entries
    """
    by_source = defaultdict(list)

    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            source = entry['source']
            by_source[source].append(entry)

    return dict(by_source)


def generate_positive_pairs(
    by_source: Dict[str, List[Dict]],
    num_pairs: int,
    window_size: int = 5
) -> List[Tuple[str, str, int]]:
    """
    Generate positive pairs (similar sentences).

    Strategy: Pick sentences from same source within a sliding window.
    Sentences near each other in the text are likely contextually similar.

    Args:
        by_source: Corpus grouped by source
        num_pairs: Number of positive pairs to generate
        window_size: Max distance between sentences (default: 5)

    Returns:
        List of (sentence1, sentence2, label=1) tuples
    """
    pairs = []

    for _ in range(num_pairs):
        # Pick a random source
        source = random.choice(list(by_source.keys()))
        sentences = by_source[source]

        if len(sentences) < 2:
            continue

        # Pick a random position
        idx1 = random.randint(0, len(sentences) - 1)

        # Pick another sentence within window
        min_idx = max(0, idx1 - window_size)
        max_idx = min(len(sentences) - 1, idx1 + window_size)

        # Ensure we don't pick the same sentence
        if min_idx == max_idx:
            continue

        idx2 = random.randint(min_idx, max_idx)
        while idx2 == idx1:
            idx2 = random.randint(min_idx, max_idx)

        sent1 = sentences[idx1]['sentence']
        sent2 = sentences[idx2]['sentence']

        pairs.append((sent1, sent2, 1))

    return pairs


def generate_negative_pairs(
    by_source: Dict[str, List[Dict]],
    num_pairs: int
) -> List[Tuple[str, str, int]]:
    """
    Generate negative pairs (dissimilar sentences).

    Strategy: Pick random sentences from different sources.

    Args:
        by_source: Corpus grouped by source
        num_pairs: Number of negative pairs to generate

    Returns:
        List of (sentence1, sentence2, label=0) tuples
    """
    pairs = []
    sources = list(by_source.keys())

    if len(sources) < 2:
        raise ValueError("Need at least 2 sources for negative pairs")

    for _ in range(num_pairs):
        # Pick two different sources
        source1, source2 = random.sample(sources, 2)

        # Pick random sentence from each
        sent1 = random.choice(by_source[source1])['sentence']
        sent2 = random.choice(by_source[source2])['sentence']

        pairs.append((sent1, sent2, 0))

    return pairs


def write_pairs(pairs: List[Tuple[str, str, int]], output_dir: Path):
    """
    Write training pairs to files.

    Args:
        pairs: List of (sent1, sent2, label) tuples
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Separate by label
    positive_pairs = [(s1, s2) for s1, s2, label in pairs if label == 1]
    negative_pairs = [(s1, s2) for s1, s2, label in pairs if label == 0]

    # Write positive pairs
    positive_path = output_dir / "positive_pairs.txt"
    with open(positive_path, 'w', encoding='utf-8') as f:
        for sent1, sent2 in positive_pairs:
            f.write(f"{sent1}\t{sent2}\n")

    # Write negative pairs
    negative_path = output_dir / "negative_pairs.txt"
    with open(negative_path, 'w', encoding='utf-8') as f:
        for sent1, sent2 in negative_pairs:
            f.write(f"{sent1}\t{sent2}\n")

    print(f"✅ Wrote {len(positive_pairs):,} positive pairs to {positive_path}")
    print(f"✅ Wrote {len(negative_pairs):,} negative pairs to {negative_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate training pairs for Tree-LSTM")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/corpus_sentences.jsonl"),
        help="Corpus JSONL file (properly segmented)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/training_pairs_v2"),
        help="Output directory for training pairs"
    )
    parser.add_argument(
        "--positive-pairs",
        type=int,
        default=10000,
        help="Number of positive pairs (default: 10000)"
    )
    parser.add_argument(
        "--negative-pairs",
        type=int,
        default=50000,
        help="Number of negative pairs (default: 50000)"
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=5,
        help="Max sentence distance for positive pairs (default: 5)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )

    args = parser.parse_args()

    # Set seed for reproducibility
    random.seed(args.seed)

    print("=" * 70)
    print("GENERATING TRAINING PAIRS FOR TREE-LSTM")
    print("=" * 70)
    print()

    # Load corpus
    print(f"Loading corpus from {args.corpus}...")
    by_source = load_corpus_by_source(args.corpus)

    total_sentences = sum(len(sents) for sents in by_source.values())
    print(f"✅ Loaded {total_sentences:,} sentences from {len(by_source)} sources")
    print()

    for source, sents in by_source.items():
        print(f"  {source:30s} {len(sents):6,} sentences")
    print()

    # Generate positive pairs
    print(f"Generating {args.positive_pairs:,} positive pairs (window={args.window_size})...")
    positive_pairs = generate_positive_pairs(by_source, args.positive_pairs, args.window_size)
    print(f"✅ Generated {len(positive_pairs):,} positive pairs")
    print()

    # Generate negative pairs
    print(f"Generating {args.negative_pairs:,} negative pairs...")
    negative_pairs = generate_negative_pairs(by_source, args.negative_pairs)
    print(f"✅ Generated {len(negative_pairs):,} negative pairs")
    print()

    # Combine and shuffle
    all_pairs = positive_pairs + negative_pairs
    random.shuffle(all_pairs)

    # Write to files
    print(f"Writing pairs to {args.output}...")
    write_pairs(all_pairs, args.output)
    print()

    # Statistics
    print("=" * 70)
    print("STATISTICS")
    print("=" * 70)
    print(f"Positive pairs: {len(positive_pairs):,}")
    print(f"Negative pairs: {len(negative_pairs):,}")
    print(f"Total pairs:    {len(all_pairs):,}")
    print(f"Class ratio:    {len(negative_pairs)/len(positive_pairs):.1f}:1 (negative:positive)")
    print()
    print("✅ Done!")
    print()


if __name__ == "__main__":
    main()
