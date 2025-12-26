#!/usr/bin/env python3
"""
Prepare training data for Tree-LSTM contrastive learning.

Creates positive and negative pairs from the AST corpus:
- Positive pairs: ASTs with similar semantics (shared vocabulary, similar structure)
- Negative pairs: ASTs with different semantics

Usage:
    python scripts/prepare_training_data.py --corpus data/ast_corpus --output data/training_pairs --num-pairs 10000
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Set
from collections import defaultdict
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.logging_config import setup_logging


def load_ast_corpus(corpus_dir: Path, max_asts: int = None) -> List[Dict]:
    """
    Load AST corpus from JSONL files.

    Args:
        corpus_dir: Directory containing *_asts.jsonl files
        max_asts: Maximum ASTs to load (None = all)

    Returns:
        List of AST dictionaries with metadata
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading AST corpus from {corpus_dir}...")

    asts = []

    jsonl_files = sorted(corpus_dir.glob('*_asts.jsonl'))
    if not jsonl_files:
        raise FileNotFoundError(f"No *_asts.jsonl files found in {corpus_dir}")

    logger.info(f"Found {len(jsonl_files)} JSONL files")

    for jsonl_file in jsonl_files:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    asts.append(data)

                    if max_asts and len(asts) >= max_asts:
                        logger.info(f"Reached max_asts limit ({max_asts})")
                        return asts

                except Exception as e:
                    logger.warning(f"Skipping invalid line: {e}")

        if len(asts) % 10000 == 0:
            logger.info(f"  Loaded {len(asts):,} ASTs so far")

    logger.info(f"Total ASTs loaded: {len(asts):,}")
    return asts


def extract_vocabulary(ast: Dict) -> Set[str]:
    """
    Extract vocabulary (roots) from an AST.

    Args:
        ast: AST dictionary

    Returns:
        Set of root words
    """
    vocab = set()

    def traverse(node):
        """Recursively extract roots from AST nodes."""
        if isinstance(node, dict):
            # Extract root
            if 'radiko' in node and node['radiko']:
                vocab.add(node['radiko'])

            # Traverse nested structures
            if node.get('tipo') == 'frazo':
                # Sentence node
                if 'subjekto' in node:
                    traverse(node['subjekto'])
                if 'verbo' in node:
                    traverse(node['verbo'])
                if 'objekto' in node:
                    traverse(node['objekto'])
                for alia in node.get('aliaj', []):
                    traverse(alia)

            elif node.get('tipo') == 'vortgrupo':
                # Word group (noun phrase)
                if 'kerno' in node:
                    traverse(node['kerno'])
                for priskribo in node.get('priskriboj', []):
                    traverse(priskribo)

    traverse(ast)
    return vocab


def compute_similarity(ast1: Dict, ast2: Dict) -> float:
    """
    Compute semantic similarity between two ASTs using vocabulary overlap.

    Args:
        ast1, ast2: AST dictionaries

    Returns:
        Similarity score (0-1)
    """
    vocab1 = extract_vocabulary(ast1['ast'])
    vocab2 = extract_vocabulary(ast2['ast'])

    if not vocab1 or not vocab2:
        return 0.0

    # Jaccard similarity
    intersection = len(vocab1 & vocab2)
    union = len(vocab1 | vocab2)

    return intersection / union if union > 0 else 0.0


def create_positive_pairs(
    asts: List[Dict],
    num_pairs: int,
    min_similarity: float = 0.3,
    logger=None
) -> List[Tuple[Dict, Dict, float]]:
    """
    Create positive pairs (similar ASTs).

    Args:
        asts: List of AST dictionaries
        num_pairs: Number of positive pairs to create
        min_similarity: Minimum similarity threshold

    Returns:
        List of (ast1, ast2, similarity) tuples
    """
    logger.info(f"Creating {num_pairs} positive pairs (min_similarity={min_similarity})...")

    pairs = []
    attempts = 0
    max_attempts = num_pairs * 100  # Prevent infinite loops

    while len(pairs) < num_pairs and attempts < max_attempts:
        attempts += 1

        # Sample two random ASTs
        ast1, ast2 = random.sample(asts, 2)

        # Compute similarity
        similarity = compute_similarity(ast1, ast2)

        # Accept if similar enough
        if similarity >= min_similarity:
            pairs.append((ast1, ast2, similarity))

        # Progress update
        if attempts % 10000 == 0:
            logger.info(f"  Attempts: {attempts:,}, Pairs found: {len(pairs):,}")

    if len(pairs) < num_pairs:
        logger.warning(f"Only found {len(pairs)} positive pairs after {attempts} attempts")
    else:
        logger.info(f"Created {len(pairs)} positive pairs")

    return pairs


def create_negative_pairs(
    asts: List[Dict],
    num_pairs: int,
    max_similarity: float = 0.1,
    logger=None
) -> List[Tuple[Dict, Dict, float]]:
    """
    Create negative pairs (dissimilar ASTs).

    Args:
        asts: List of AST dictionaries
        num_pairs: Number of negative pairs to create
        max_similarity: Maximum similarity threshold

    Returns:
        List of (ast1, ast2, similarity) tuples
    """
    logger.info(f"Creating {num_pairs} negative pairs (max_similarity={max_similarity})...")

    pairs = []
    attempts = 0
    max_attempts = num_pairs * 100

    while len(pairs) < num_pairs and attempts < max_attempts:
        attempts += 1

        # Sample two random ASTs
        ast1, ast2 = random.sample(asts, 2)

        # Compute similarity
        similarity = compute_similarity(ast1, ast2)

        # Accept if dissimilar enough
        if similarity <= max_similarity:
            pairs.append((ast1, ast2, similarity))

        # Progress update
        if attempts % 10000 == 0:
            logger.info(f"  Attempts: {attempts:,}, Pairs found: {len(pairs):,}")

    if len(pairs) < num_pairs:
        logger.warning(f"Only found {len(pairs)} negative pairs after {attempts} attempts")
    else:
        logger.info(f"Created {len(pairs)} negative pairs")

    return pairs


def save_training_pairs(
    output_dir: Path,
    positive_pairs: List[Tuple[Dict, Dict, float]],
    negative_pairs: List[Tuple[Dict, Dict, float]],
    logger=None
):
    """
    Save training pairs to disk.

    Args:
        output_dir: Output directory
        positive_pairs: List of positive (ast1, ast2, similarity) tuples
        negative_pairs: List of negative (ast1, ast2, similarity) tuples
    """
    logger.info(f"Saving training pairs to {output_dir}...")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save positive pairs
    positive_file = output_dir / 'positive_pairs.jsonl'
    with open(positive_file, 'w', encoding='utf-8') as f:
        for ast1, ast2, similarity in positive_pairs:
            json.dump({
                'ast1': ast1,
                'ast2': ast2,
                'similarity': similarity,
                'label': 1  # Positive
            }, f, ensure_ascii=False)
            f.write('\n')
    logger.info(f"  Saved {len(positive_pairs)} positive pairs: {positive_file}")

    # Save negative pairs
    negative_file = output_dir / 'negative_pairs.jsonl'
    with open(negative_file, 'w', encoding='utf-8') as f:
        for ast1, ast2, similarity in negative_pairs:
            json.dump({
                'ast1': ast1,
                'ast2': ast2,
                'similarity': similarity,
                'label': 0  # Negative
            }, f, ensure_ascii=False)
            f.write('\n')
    logger.info(f"  Saved {len(negative_pairs)} negative pairs: {negative_file}")

    # Save metadata
    metadata = {
        'num_positive_pairs': len(positive_pairs),
        'num_negative_pairs': len(negative_pairs),
        'total_pairs': len(positive_pairs) + len(negative_pairs),
        'positive_similarity': {
            'mean': sum(s for _, _, s in positive_pairs) / len(positive_pairs) if positive_pairs else 0,
            'min': min(s for _, _, s in positive_pairs) if positive_pairs else 0,
            'max': max(s for _, _, s in positive_pairs) if positive_pairs else 0,
        },
        'negative_similarity': {
            'mean': sum(s for _, _, s in negative_pairs) / len(negative_pairs) if negative_pairs else 0,
            'min': min(s for _, _, s in negative_pairs) if negative_pairs else 0,
            'max': max(s for _, _, s in negative_pairs) if negative_pairs else 0,
        }
    }

    metadata_file = output_dir / 'metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"  Saved metadata: {metadata_file}")

    logger.info("✅ Training pairs saved successfully")


def main():
    """Prepare training data for contrastive learning."""
    parser = argparse.ArgumentParser(description='Prepare training data for Tree-LSTM')
    parser.add_argument('--corpus', type=str, default='data/ast_corpus',
                        help='AST corpus directory')
    parser.add_argument('--output', type=str, default='data/training_pairs',
                        help='Output directory for training pairs')
    parser.add_argument('--num-pairs', type=int, default=10000,
                        help='Number of pairs to create (total = 2x this)')
    parser.add_argument('--max-asts', type=int, default=100000,
                        help='Maximum ASTs to load from corpus')
    parser.add_argument('--min-positive-similarity', type=float, default=0.3,
                        help='Minimum similarity for positive pairs')
    parser.add_argument('--max-negative-similarity', type=float, default=0.1,
                        help='Maximum similarity for negative pairs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    args = parser.parse_args()

    # Setup logging
    setup_logging(level=logging.DEBUG if args.debug else logging.INFO)
    logger = logging.getLogger(__name__)

    # Set random seed
    random.seed(args.seed)

    logger.info("="*70)
    logger.info("TRAINING DATA PREPARATION - PHASE 3")
    logger.info("="*70)
    logger.info(f"Corpus: {args.corpus}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Pairs per class: {args.num_pairs}")
    logger.info(f"Max ASTs: {args.max_asts}")
    logger.info(f"Min positive similarity: {args.min_positive_similarity}")
    logger.info(f"Max negative similarity: {args.max_negative_similarity}")
    logger.info(f"Random seed: {args.seed}")
    logger.info("")

    try:
        # Step 1: Load AST corpus
        corpus_dir = Path(args.corpus)
        asts = load_ast_corpus(corpus_dir, args.max_asts)
        logger.info("")

        # Step 2: Create positive pairs
        positive_pairs = create_positive_pairs(
            asts,
            args.num_pairs,
            args.min_positive_similarity,
            logger
        )
        logger.info("")

        # Step 3: Create negative pairs
        negative_pairs = create_negative_pairs(
            asts,
            args.num_pairs,
            args.max_negative_similarity,
            logger
        )
        logger.info("")

        # Step 4: Save training pairs
        output_dir = Path(args.output)
        save_training_pairs(output_dir, positive_pairs, negative_pairs, logger)
        logger.info("")

        # Summary
        logger.info("="*70)
        logger.info("TRAINING DATA PREPARATION COMPLETE")
        logger.info("="*70)
        logger.info(f"Positive pairs: {len(positive_pairs):,}")
        logger.info(f"Negative pairs: {len(negative_pairs):,}")
        logger.info(f"Total pairs: {len(positive_pairs) + len(negative_pairs):,}")
        logger.info(f"Output: {output_dir}")
        logger.info("")
        logger.info("✅ Training data ready for Tree-LSTM!")

        return 0

    except Exception as e:
        logger.error(f"ERROR: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
