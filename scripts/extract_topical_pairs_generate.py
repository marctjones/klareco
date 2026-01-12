#!/usr/bin/env python3
"""
Smart generation of topical training pairs with coverage guarantees.

Strategy:
1. Build vocabulary from full corpus (Pass 1)
2. Generate pairs with PER-ROOT coverage targets (Pass 2)
   - Target: 300 pairs per root
   - Sample pairs during generation (not after)
   - Stop sampling for roots that hit target
3. Generate balanced negative samples

This produces high-quality, balanced training data in a single pass.
No post-processing needed.
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def setup_file_logging(log_path: Path):
    """Add file handler for logging."""
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)


# Function words to exclude
FUNCTION_WORDS = {
    'kaj', 'aŭ', 'sed', 'nek', 'do', 'tamen', 'ĉar', 'ke', 'se',
    'al', 'de', 'en', 'el', 'kun', 'per', 'por', 'pri', 'sen', 'sur', 'sub', 'ĉe', 'tra', 'ĉirkaŭ',
    'mi', 'vi', 'li', 'ŝi', 'ĝi', 'ni', 'ili', 'si', 'oni',
    'kiu', 'kio', 'kia', 'kie', 'kiel', 'kiam', 'kiom', 'kial',
    'tiu', 'tio', 'tia', 'tie', 'tiel', 'tiam', 'tiom', 'tial',
    'ĉiu', 'ĉio', 'ĉia', 'ĉie', 'ĉiel', 'ĉiam', 'ĉiom', 'ĉial',
    'neniu', 'nenio', 'nenia', 'nenie', 'neniel', 'neniam', 'neniom', 'nenial',
    'iu', 'io', 'ia', 'ie', 'iel', 'iam', 'iom', 'ial',
    'est', 'far', 'hav', 'pov', 'dev', 'vol', 'deb',
    'la', 'ne', 'tre', 'nur', 'ankaŭ', 'eĉ', 'ja', 'jen', 'jes', 'plej', 'pli', 'tro',
}


def extract_roots_from_ast(ast_node: dict, roots: List[str]):
    """Recursively extract roots from AST."""
    if not isinstance(ast_node, dict):
        return

    node_type = ast_node.get('tipo')

    if node_type == 'vorto':
        root = ast_node.get('radiko')
        parse_status = ast_node.get('parse_status')
        if root and parse_status == 'success' and root not in FUNCTION_WORDS:
            roots.append(root)
    elif node_type == 'frazo':
        for key in ['subjekto', 'verbo', 'objekto']:
            if ast_node.get(key):
                extract_roots_from_ast(ast_node[key], roots)
        for child in ast_node.get('aliaj', []):
            extract_roots_from_ast(child, roots)
    elif node_type == 'vortgrupo':
        if ast_node.get('kerno'):
            extract_roots_from_ast(ast_node['kerno'], roots)
        for desc in ast_node.get('priskriboj', []):
            extract_roots_from_ast(desc, roots)


def generate_skipgram_pairs(roots: List[str], window_size: int = 5) -> List[Tuple[str, str, float]]:
    """Generate skip-gram pairs with distance-based targets."""
    pairs = []
    for i, center_root in enumerate(roots):
        start = max(0, i - window_size)
        end = min(len(roots), i + window_size + 1)
        for j in range(start, end):
            if i == j:
                continue
            context_root = roots[j]
            distance = abs(i - j)
            target = max(0.3, 0.8 - (distance - 1) * 0.1)
            pairs.append((center_root, context_root, target))
    return pairs


def build_vocabulary_from_corpus(
    corpus_path: Path,
    min_frequency: int = 5
) -> Tuple[Dict[str, int], Dict[int, str], Counter]:
    """Build vocabulary from full corpus."""
    logger.info("Building vocabulary from full corpus...")

    root_freq = Counter()
    total_sentences = 0
    total_roots_extracted = 0

    with open(corpus_path) as f:
        for i, line in enumerate(f):
            if i % 100000 == 0 and i > 0:
                logger.info(f"  Processed {i:,} sentences, {len(root_freq):,} unique roots")

            try:
                entry = json.loads(line)
                ast = entry.get('ast')
                if not ast:
                    continue

                roots = []
                extract_roots_from_ast(ast, roots)

                if roots:
                    root_freq.update(roots)
                    total_roots_extracted += len(roots)
                    total_sentences += 1

            except json.JSONDecodeError:
                continue

    logger.info(f"Processed {total_sentences:,} sentences")
    logger.info(f"Extracted {total_roots_extracted:,} total roots")
    logger.info(f"Found {len(root_freq):,} unique roots before filtering")

    # Filter by frequency
    filtered_roots = {root for root, freq in root_freq.items() if freq >= min_frequency}
    logger.info(f"After min_frequency={min_frequency} filtering: {len(filtered_roots):,} roots")

    # Build vocabulary
    root_to_idx = {root: idx for idx, root in enumerate(sorted(filtered_roots))}
    idx_to_root = {idx: root for root, idx in root_to_idx.items()}

    return root_to_idx, idx_to_root, root_freq


def generate_pairs_with_coverage(
    corpus_path: Path,
    root_to_idx: Dict[str, int],
    output_path: Path,
    target_pairs_per_root: int = 300,
    window_size: int = 5,
    min_root_frequency: int = 50,
    checkpoint_path: Path = None,
    checkpoint_interval: int = 500000,
) -> int:
    """
    Generate pairs with coverage guarantees.

    Strategy:
    - Track how many pairs each root has generated
    - Sample pairs with probability inversely proportional to current coverage
    - Stop when target coverage reached

    This ensures balanced representation without generating excessive data.
    """
    logger.info(f"Generating pairs with coverage target: {target_pairs_per_root} per root...")

    idx_to_root = {idx: root for root, idx in root_to_idx.items()}
    root_pair_count = defaultdict(int)  # Track pairs per root
    positive_pair_set = set()
    total_positive = 0
    start_sentence = 0
    sentences_since_checkpoint = 0

    # Try to load checkpoint
    if checkpoint_path and checkpoint_path.exists():
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)
        start_sentence = checkpoint['sentences_processed']
        root_pair_count = defaultdict(int, checkpoint['root_pair_count'])
        positive_pair_set = set(tuple(p) for p in checkpoint['positive_pairs'])
        total_positive = checkpoint['total_positive']
        logger.info(f"Resuming from sentence {start_sentence:,}, {total_positive:,} pairs so far")

    # Compute sampling probabilities
    # Roots with fewer pairs get higher sampling probability
    def get_sampling_prob(root: str) -> float:
        count = root_pair_count[root]
        if count >= target_pairs_per_root:
            return 0.0  # Stop sampling
        elif count < target_pairs_per_root // 2:
            return 1.0  # Always sample if below half target
        else:
            # Gradually reduce probability as we approach target
            return 1.0 - (count - target_pairs_per_root // 2) / (target_pairs_per_root // 2)

    # Open output in append mode if resuming
    write_mode = 'a' if start_sentence > 0 else 'w'

    with open(output_path, write_mode) as output_file:
        with open(corpus_path) as f:
            # Skip to checkpoint position
            for _ in range(start_sentence):
                next(f)

            for i, line in enumerate(f, start=start_sentence):
                if i % 100000 == 0 and i > 0:
                    logger.info(f"  Processed {i:,} sentences, {total_positive:,} pairs generated")

                try:
                    entry = json.loads(line)
                    ast = entry.get('ast')
                    if not ast:
                        continue

                    # Extract roots
                    roots = []
                    extract_roots_from_ast(ast, roots)
                    roots = [r for r in roots if r in root_to_idx]

                    if len(roots) < 2:
                        continue

                    # Generate skip-gram pairs
                    sentence_pairs = generate_skipgram_pairs(roots, window_size=window_size)

                    for r1, r2, target in sentence_pairs:
                        # Check if we should sample this pair
                        prob1 = get_sampling_prob(r1)
                        prob2 = get_sampling_prob(r2)

                        # Use minimum probability (conservative)
                        sampling_prob = min(prob1, prob2)

                        if sampling_prob == 0.0:
                            continue  # Both roots at capacity

                        # Sample with probability
                        if random.random() > sampling_prob:
                            continue

                        idx1, idx2 = root_to_idx[r1], root_to_idx[r2]
                        pair_key = (min(idx1, idx2), max(idx1, idx2))

                        # Skip duplicates
                        if pair_key in positive_pair_set:
                            continue

                        # Write to disk immediately
                        weight = 2.0 + 3.0 * target
                        pair_data = {
                            'idx1': pair_key[0],
                            'idx2': pair_key[1],
                            'target_similarity': target,
                            'weight': weight,
                            'root1': idx_to_root[pair_key[0]],
                            'root2': idx_to_root[pair_key[1]]
                        }
                        output_file.write(json.dumps(pair_data) + '\n')

                        # Update counters
                        positive_pair_set.add(pair_key)
                        root_pair_count[r1] += 1
                        root_pair_count[r2] += 1
                        total_positive += 1

                sentences_since_checkpoint += 1

                # Checkpoint periodically
                if checkpoint_path and sentences_since_checkpoint >= checkpoint_interval:
                    output_file.flush()
                    save_checkpoint_positive(
                        checkpoint_path,
                        i + 1,  # sentences_processed
                        dict(root_pair_count),
                        list(positive_pair_set),
                        total_positive
                    )
                    sentences_since_checkpoint = 0

                except json.JSONDecodeError:
                    continue

    logger.info(f"Generated {total_positive:,} positive pairs")

    # Coverage statistics
    counts = list(root_pair_count.values())
    if counts:
        counts.sort()
        logger.info(f"Coverage statistics:")
        logger.info(f"  Roots with pairs: {len(root_pair_count):,}")
        logger.info(f"  Min pairs per root: {counts[0]:,}")
        logger.info(f"  Median pairs per root: {counts[len(counts)//2]:,}")
        logger.info(f"  Max pairs per root: {counts[-1]:,}")
        logger.info(f"  Avg pairs per root: {sum(counts) / len(counts):.1f}")

    return total_positive, positive_pair_set, root_pair_count


def save_checkpoint_positive(
    checkpoint_path: Path,
    sentences_processed: int,
    root_pair_count: Dict,
    positive_pairs: List,
    total_positive: int
):
    """Save checkpoint for positive pair generation."""
    temp_path = checkpoint_path.with_suffix('.tmp')
    checkpoint = {
        'sentences_processed': sentences_processed,
        'root_pair_count': root_pair_count,
        'positive_pairs': list(positive_pairs),
        'total_positive': total_positive
    }
    try:
        with open(temp_path, 'w') as f:
            json.dump(checkpoint, f)
        temp_path.rename(checkpoint_path)
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")
        if temp_path.exists():
            temp_path.unlink()


def generate_smart_negatives(
    output_path: Path,
    root_to_idx: Dict[str, int],
    root_freq: Counter,
    positive_pair_set: Set,
    root_pair_count: Dict,
    target_positive: int,
    negative_ratio: float = 2.0,
    target_pairs_per_root: int = 300
):
    """
    Generate balanced negative samples.

    Strategy:
    - Sample negatives to maintain per-root coverage balance
    - Mix of rare-rare, rare-common, common-common pairs
    - Frequency-weighted sampling to match positive distribution

    Optimized version: Pre-computes weight lists for fast sampling
    """
    logger.info(f"Generating smart negative samples (ratio={negative_ratio}:1)...")

    idx_to_root = {idx: root for root, idx in root_to_idx.items()}
    target_negatives = int(target_positive * negative_ratio)

    # Compute frequency-based sampling weights (PRE-COMPUTE ONCE!)
    total_freq = sum(root_freq.values())
    roots_list = list(root_to_idx.keys())
    weights_list = [root_freq.get(r, 0) / total_freq for r in roots_list]

    logger.info(f"  Target negatives: {target_negatives:,}")
    logger.info(f"  Sampling from {len(roots_list):,} roots")

    negative_count = 0
    attempts = 0
    max_attempts = target_negatives * 10

    # Batch size for sampling
    batch_size = 10000

    with open(output_path, 'a') as output_file:
        while negative_count < target_negatives and attempts < max_attempts:
            # Sample batch of root pairs
            remaining = min(batch_size, target_negatives - negative_count)

            # Sample pairs in batch (MUCH FASTER!)
            r1_batch = random.choices(roots_list, weights=weights_list, k=remaining)
            r2_batch = random.choices(roots_list, weights=weights_list, k=remaining)

            for r1, r2 in zip(r1_batch, r2_batch):
                attempts += 1

                if r1 == r2:
                    continue

                idx1, idx2 = root_to_idx[r1], root_to_idx[r2]
                pair_key = (min(idx1, idx2), max(idx1, idx2))

                if pair_key in positive_pair_set:
                    continue

                # Check coverage balance
                count1 = root_pair_count.get(r1, 0)
                count2 = root_pair_count.get(r2, 0)

                # Skip if either root is over-represented
                if count1 > target_pairs_per_root * 1.5 or count2 > target_pairs_per_root * 1.5:
                    continue

                # Write negative pair
                pair_data = {
                    'idx1': pair_key[0],
                    'idx2': pair_key[1],
                    'target_similarity': 0.0,
                    'weight': 1.0,
                    'root1': idx_to_root[pair_key[0]],
                    'root2': idx_to_root[pair_key[1]]
                }
                output_file.write(json.dumps(pair_data) + '\n')

                positive_pair_set.add(pair_key)
                root_pair_count[r1] += 1
                root_pair_count[r2] += 1
                negative_count += 1

                if negative_count >= target_negatives:
                    break

            if negative_count % 100000 == 0 and negative_count > 0:
                logger.info(f"  Generated {negative_count:,} negative samples ({negative_count/target_negatives*100:.1f}%)")
                output_file.flush()  # Flush to disk periodically

    logger.info(f"Generated {negative_count:,} negative samples")
    logger.info(f"Ratio: 1:{negative_count/target_positive:.1f}")
    logger.info(f"Total attempts: {attempts:,}")

    return negative_count


def main():
    parser = argparse.ArgumentParser(description='Smart generation of topical pairs')
    parser.add_argument('--corpus', type=Path,
                        default=Path('data/corpus/unified_corpus.jsonl'),
                        help='Unified corpus path')
    parser.add_argument('--output', type=Path,
                        default=Path('data/training/topical_pairs_smart.jsonl'),
                        help='Output training pairs file')
    parser.add_argument('--vocab-output', type=Path,
                        default=Path('data/vocabularies/topical_vocab.json'),
                        help='Output vocabulary file')
    parser.add_argument('--log-dir', type=Path,
                        default=Path('logs/training'),
                        help='Log directory')
    parser.add_argument('--target-per-root', type=int, default=300,
                        help='Target pairs per root')
    parser.add_argument('--window-size', type=int, default=5,
                        help='Skip-gram window size')
    parser.add_argument('--negative-ratio', type=float, default=2.0,
                        help='Negative:positive sample ratio')
    parser.add_argument('--min-frequency', type=int, default=5,
                        help='Minimum root frequency')
    parser.add_argument('--min-root-freq', type=int, default=50,
                        help='Minimum root frequency for pair generation')

    args = parser.parse_args()

    # Setup
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.vocab_output.parent.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    log_path = args.log_dir / f'generate_topical_smart_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    setup_file_logging(log_path)

    logger.info("=" * 60)
    logger.info("Smart Topical Pair Generation")
    logger.info("=" * 60)
    logger.info(f"Strategy: Coverage-based sampling")
    logger.info(f"Target per root: {args.target_per_root}")
    logger.info(f"Window size: {args.window_size}")
    logger.info(f"Negative ratio: {args.negative_ratio}:1")
    logger.info(f"Min frequency: {args.min_frequency}")

    if not args.corpus.exists():
        logger.error(f"Corpus not found: {args.corpus}")
        return 1

    # Build vocabulary
    root_to_idx, idx_to_root, root_freq = build_vocabulary_from_corpus(
        args.corpus,
        min_frequency=args.min_frequency
    )

    # Save vocabulary
    vocab_data = {
        'root_to_idx': root_to_idx,
        'idx_to_root': idx_to_root,
        'metadata': {
            'vocab_size': len(root_to_idx),
            'min_frequency': args.min_frequency,
            'created': datetime.now().isoformat()
        }
    }

    logger.info(f"Saving vocabulary to {args.vocab_output}")
    with open(args.vocab_output, 'w') as f:
        json.dump(vocab_data, f, indent=2, ensure_ascii=False)

    # Generate positive pairs with coverage guarantees
    checkpoint_path = args.output.with_suffix('.checkpoint.json')
    total_positive, positive_pair_set, root_pair_count = generate_pairs_with_coverage(
        args.corpus,
        root_to_idx,
        args.output,
        target_pairs_per_root=args.target_per_root,
        window_size=args.window_size,
        min_root_frequency=args.min_root_freq,
        checkpoint_path=checkpoint_path,
        checkpoint_interval=500000
    )

    # Generate smart negative samples
    total_negative = generate_smart_negatives(
        args.output,
        root_to_idx,
        root_freq,
        positive_pair_set,
        root_pair_count,
        total_positive,
        negative_ratio=args.negative_ratio,
        target_pairs_per_root=args.target_per_root
    )

    # Final statistics
    total_pairs = total_positive + total_negative
    output_size = args.output.stat().st_size / (1024**3)

    logger.info(f"\n{'=' * 60}")
    logger.info("Complete!")
    logger.info(f"{'=' * 60}")
    logger.info(f"Vocabulary: {len(root_to_idx):,} roots")
    logger.info(f"Positive pairs: {total_positive:,}")
    logger.info(f"Negative pairs: {total_negative:,}")
    logger.info(f"Total pairs: {total_pairs:,}")
    logger.info(f"Output size: {output_size:.2f} GB")
    logger.info(f"Output: {args.output}")
    logger.info(f"Log: {log_path}")

    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        logger.info("Checkpoint removed (complete)")

    return 0


if __name__ == '__main__':
    sys.exit(main())
