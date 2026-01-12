#!/usr/bin/env python3
"""
Prepare topical training pairs from corpus for dual embeddings.

This script extracts skip-gram training pairs from the unified corpus to train
the topical embedding component of DualRootEmbeddings.

Unlike linguistic embeddings (trained on semantic relations), topical embeddings
learn from corpus-level word co-occurrence patterns to capture contextual and
topical similarity.

Key differences from linguistic training:
- Linguistic: "hundo" similar to "kato" (both animals) via ReVo relations
- Topical: "hundo" similar to "mangxo" (dogs eat food) via corpus co-occurrence

Approach:
1. Read unified corpus (4.3M sentences)
2. Extract roots from AST parse results
3. Generate skip-gram pairs (window=5)
4. Filter function words (no semantic content)
5. Create negative samples (5:1 ratio)
6. Output training pairs with graded targets

Output format:
- Positive pairs: (root1, root2, target_sim, weight)
- Negative pairs: (root1, root2, 0.0, weight)
- Target similarity based on distance in window

Task #70: Topical training data preparation
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Set, Dict
from collections import defaultdict, Counter

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


# Function words to exclude (same as train_root_embeddings.py)
# These don't carry semantic meaning and cause embedding collapse
FUNCTION_WORDS = {
    # Conjunctions
    'kaj', 'aŭ', 'sed', 'nek', 'do', 'tamen', 'ĉar', 'ke', 'se',
    # Prepositions
    'al', 'de', 'en', 'el', 'kun', 'per', 'por', 'pri', 'sen', 'sur', 'sub', 'ĉe', 'tra', 'ĉirkaŭ',
    # Pronouns/correlatives
    'mi', 'vi', 'li', 'ŝi', 'ĝi', 'ni', 'ili', 'si', 'oni',
    'kiu', 'kio', 'kia', 'kie', 'kiel', 'kiam', 'kiom', 'kial',
    'tiu', 'tio', 'tia', 'tie', 'tiel', 'tiam', 'tiom', 'tial',
    'ĉiu', 'ĉio', 'ĉia', 'ĉie', 'ĉiel', 'ĉiam', 'ĉiom', 'ĉial',
    'neniu', 'nenio', 'nenia', 'nenie', 'neniel', 'neniam', 'neniom', 'nenial',
    'iu', 'io', 'ia', 'ie', 'iel', 'iam', 'iom', 'ial',
    # Common verbs/copula
    'est', 'far', 'hav', 'pov', 'dev', 'vol', 'deb',
    # Articles/particles
    'la', 'ne', 'tre', 'nur', 'ankaŭ', 'eĉ', 'ja', 'jen', 'jes', 'plej', 'pli', 'tro',
}


def extract_roots_from_ast(ast_node: dict, roots: List[str]):
    """
    Recursively extract roots from AST while preserving sentence order.

    Args:
        ast_node: AST node dictionary
        roots: List to append roots to (preserves order)
    """
    if not isinstance(ast_node, dict):
        return

    node_type = ast_node.get('tipo')

    # Handle word nodes
    if node_type == 'vorto':
        root = ast_node.get('radiko')
        parse_status = ast_node.get('parse_status')

        # Only include successfully parsed roots (not proper names or failed parses)
        if root and parse_status == 'success' and root not in FUNCTION_WORDS:
            roots.append(root)

    # Recursively process children based on AST structure
    elif node_type == 'frazo':
        # Process in sentence order: subject, verb, object, other
        for key in ['subjekto', 'verbo', 'objekto']:
            if ast_node.get(key):
                extract_roots_from_ast(ast_node[key], roots)

        for child in ast_node.get('aliaj', []):
            extract_roots_from_ast(child, roots)

    elif node_type == 'vortgrupo':
        # Process kernel then modifiers
        if ast_node.get('kerno'):
            extract_roots_from_ast(ast_node['kerno'], roots)
        for desc in ast_node.get('priskriboj', []):
            extract_roots_from_ast(desc, roots)


def generate_skipgram_pairs(
    roots: List[str],
    window_size: int = 5,
    use_distance_grading: bool = True
) -> List[Tuple[str, str, float]]:
    """
    Generate skip-gram training pairs from a sequence of roots.

    Args:
        roots: Ordered list of roots from a sentence
        window_size: Context window size (default: 5)
        use_distance_grading: If True, closer words get higher similarity targets

    Returns:
        List of (root1, root2, target_similarity) tuples
    """
    pairs = []

    for i, center_root in enumerate(roots):
        # Get context words within window
        start = max(0, i - window_size)
        end = min(len(roots), i + window_size + 1)

        for j in range(start, end):
            if i == j:
                continue  # Skip self

            context_root = roots[j]
            distance = abs(i - j)

            # Grade similarity by distance
            if use_distance_grading:
                # Closer words = higher similarity
                # Distance 1: 0.7, Distance 2: 0.6, ... Distance 5: 0.3
                target = max(0.3, 0.8 - (distance - 1) * 0.1)
            else:
                # Uniform target for all pairs in window
                target = 0.6

            pairs.append((center_root, context_root, target))

    return pairs


def build_vocabulary_from_corpus(
    corpus_path: Path,
    min_frequency: int = 5,
    max_sentences: int = None
) -> Tuple[Dict[str, int], Dict[int, str], Counter]:
    """
    Build root vocabulary from corpus with frequency filtering.

    Args:
        corpus_path: Path to unified_corpus.jsonl
        min_frequency: Minimum root frequency to include in vocabulary
        max_sentences: Maximum sentences to process (for testing)

    Returns:
        (root_to_idx, idx_to_root, root_freq)
    """
    logger.info("Building vocabulary from corpus...")

    root_freq = Counter()
    total_sentences = 0
    total_roots_extracted = 0

    with open(corpus_path) as f:
        for i, line in enumerate(f):
            if max_sentences and i >= max_sentences:
                break

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
                logger.warning(f"Failed to parse line {i}")
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


def generate_training_pairs(
    corpus_path: Path,
    root_to_idx: Dict[str, int],
    output_path: Path,
    window_size: int = 5,
    negative_ratio: int = 5,
    max_sentences: int = None,
    checkpoint_path: Path = None,
    checkpoint_interval: int = 500000
) -> Tuple[int, int]:
    """
    Generate training pairs from corpus with STREAMING to disk.

    MEMORY EFFICIENT: Writes pairs to disk immediately instead of accumulating in memory.

    Args:
        corpus_path: Path to unified_corpus.jsonl
        root_to_idx: Root vocabulary
        output_path: Path to write pairs (streaming)
        window_size: Skip-gram window size
        negative_ratio: Ratio of negative to positive samples
        max_sentences: Maximum sentences to process
        checkpoint_path: Path to save/load checkpoint
        checkpoint_interval: Save checkpoint every N sentences

    Returns:
        (positive_count, sentences_processed)
    """
    logger.info("Generating training pairs (STREAMING mode - memory efficient)...")

    # Try to load checkpoint
    start_sentence = 0
    positive_pair_set = set()
    positive_count = 0
    idx_to_root = {idx: root for root, idx in root_to_idx.items()}

    if checkpoint_path and checkpoint_path.exists():
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)
        start_sentence = checkpoint['sentences_processed']
        positive_pair_set = set(tuple(p) for p in checkpoint['positive_pairs'])
        positive_count = checkpoint['positive_count']
        logger.info(f"Resuming from sentence {start_sentence:,}")

    total_sentences = start_sentence
    sentences_since_checkpoint = 0

    # Open output file for streaming writes (append mode if resuming)
    write_mode = 'a' if start_sentence > 0 else 'w'
    output_file = open(output_path, write_mode)

    try:
        with open(corpus_path) as f:
            # Skip to checkpoint position
            for _ in range(start_sentence):
                next(f)

            for i, line in enumerate(f, start=start_sentence):
                if max_sentences and i >= max_sentences:
                    break

                if i % 100000 == 0 and i > 0:
                    logger.info(f"  Processed {i:,} sentences, {positive_count:,} pairs generated")

                try:
                    entry = json.loads(line)
                    ast = entry.get('ast')

                    if not ast:
                        continue

                    # Extract roots
                    roots = []
                    extract_roots_from_ast(ast, roots)

                    # Filter to vocabulary
                    roots = [r for r in roots if r in root_to_idx]

                    if len(roots) < 2:
                        continue

                    # Generate skip-gram pairs for this sentence
                    sentence_pairs = generate_skipgram_pairs(roots, window_size=window_size)

                    for r1, r2, target in sentence_pairs:
                        idx1, idx2 = root_to_idx[r1], root_to_idx[r2]
                        pair_key = (min(idx1, idx2), max(idx1, idx2))

                        # Weight based on target similarity
                        weight = 2.0 + 3.0 * target  # Range: 2.0-5.0

                        # STREAM TO DISK immediately (memory efficient!)
                        pair_data = {
                            'idx1': pair_key[0],
                            'idx2': pair_key[1],
                            'target_similarity': target,
                            'weight': weight,
                            'root1': idx_to_root[pair_key[0]],
                            'root2': idx_to_root[pair_key[1]]
                        }
                        output_file.write(json.dumps(pair_data) + '\n')

                        positive_pair_set.add(pair_key)
                        positive_count += 1

                    total_sentences += 1
                    sentences_since_checkpoint += 1

                    # Checkpoint periodically
                    if checkpoint_path and sentences_since_checkpoint >= checkpoint_interval:
                        output_file.flush()  # Flush to disk
                        save_checkpoint(
                            checkpoint_path,
                            list(positive_pair_set),
                            positive_count,
                            total_sentences
                        )
                        sentences_since_checkpoint = 0

                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse line {i}")
                    continue

    finally:
        output_file.close()

    logger.info(f"Generated {positive_count:,} positive pairs from {total_sentences:,} sentences")

    # Generate negative samples (also streaming to disk)
    logger.info(f"Generating negative samples (ratio={negative_ratio}:1) - STREAMING...")
    content_indices = list(root_to_idx.values())
    target_negatives = positive_count * negative_ratio

    negative_count = 0
    attempts = 0
    max_attempts = target_negatives * 10

    # Reopen output file in append mode for negatives
    with open(output_path, 'a') as output_file:
        while negative_count < target_negatives and attempts < max_attempts:
            attempts += 1
            idx1, idx2 = random.sample(content_indices, 2)
            pair_key = (min(idx1, idx2), max(idx1, idx2))

            if pair_key not in positive_pair_set:
                # STREAM negative pairs to disk
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
                negative_count += 1

                if negative_count % 1000000 == 0:
                    logger.info(f"  Generated {negative_count:,} negative samples")
                    output_file.flush()

    logger.info(f"Generated {negative_count:,} negative samples")
    total_pairs = positive_count + negative_count
    logger.info(f"Total training pairs: {total_pairs:,}")
    logger.info(f"Positive:Negative ratio = {positive_count}:{negative_count} (1:{negative_count/positive_count:.1f})")

    # Final checkpoint (just metadata, no pairs list!)
    if checkpoint_path:
        save_checkpoint_final(
            checkpoint_path,
            list(positive_pair_set),
            positive_count,
            total_sentences
        )

    return positive_count, total_sentences


def save_checkpoint(
    checkpoint_path: Path,
    positive_pairs: List[Tuple[int, int]],
    positive_count: int,
    sentences_processed: int
):
    """Save checkpoint atomically (metadata only, no pairs list)."""
    temp_path = checkpoint_path.with_suffix('.tmp')

    checkpoint = {
        'positive_pairs': list(positive_pairs),
        'positive_count': positive_count,
        'sentences_processed': sentences_processed
    }

    try:
        with open(temp_path, 'w') as f:
            json.dump(checkpoint, f)
        temp_path.rename(checkpoint_path)
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")
        if temp_path.exists():
            temp_path.unlink()


def save_checkpoint_final(
    checkpoint_path: Path,
    positive_pairs: List[Tuple[int, int]],
    positive_count: int,
    sentences_processed: int
):
    """Save final checkpoint."""
    save_checkpoint(checkpoint_path, positive_pairs, positive_count, sentences_processed)


def main():
    parser = argparse.ArgumentParser(description='Prepare topical training pairs from corpus')
    parser.add_argument('--corpus', type=Path,
                        default=Path('data/corpus/unified_corpus.jsonl'),
                        help='Unified corpus path')
    parser.add_argument('--output', type=Path,
                        default=Path('data/training/topical_pairs.jsonl'),
                        help='Output training pairs file')
    parser.add_argument('--vocab-output', type=Path,
                        default=Path('data/vocabularies/topical_vocab.json'),
                        help='Output vocabulary file')
    parser.add_argument('--log-dir', type=Path,
                        default=Path('logs/training'),
                        help='Log directory')
    parser.add_argument('--window-size', type=int, default=5,
                        help='Skip-gram window size')
    parser.add_argument('--negative-ratio', type=int, default=5,
                        help='Negative:positive sample ratio')
    parser.add_argument('--min-frequency', type=int, default=5,
                        help='Minimum root frequency to include')
    parser.add_argument('--max-sentences', type=int, default=None,
                        help='Maximum sentences to process (for testing)')
    parser.add_argument('--checkpoint-interval', type=int, default=500000,
                        help='Save checkpoint every N sentences')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint if exists')
    parser.add_argument('--fresh', action='store_true',
                        help='Start fresh, ignore checkpoint')

    args = parser.parse_args()

    # Setup
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.vocab_output.parent.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    log_path = args.log_dir / f'prepare_topical_pairs_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    setup_file_logging(log_path)

    logger.info("=" * 60)
    logger.info("Topical Training Pair Generation")
    logger.info("=" * 60)
    logger.info(f"Corpus: {args.corpus}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Window size: {args.window_size}")
    logger.info(f"Negative ratio: {args.negative_ratio}:1")
    logger.info(f"Min frequency: {args.min_frequency}")

    if not args.corpus.exists():
        logger.error(f"Corpus not found: {args.corpus}")
        return 1

    # Build vocabulary
    root_to_idx, idx_to_root, root_freq = build_vocabulary_from_corpus(
        args.corpus,
        min_frequency=args.min_frequency,
        max_sentences=args.max_sentences
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

    # Generate training pairs (STREAMING - writes directly to output file)
    checkpoint_path = args.output.with_suffix('.checkpoint.json')
    if args.fresh and checkpoint_path.exists():
        logger.info("Removing old checkpoint (--fresh mode)")
        checkpoint_path.unlink()

    if args.fresh and args.output.exists():
        logger.info("Removing old output file (--fresh mode)")
        args.output.unlink()

    positive_count, sentences_processed = generate_training_pairs(
        args.corpus,
        root_to_idx,
        args.output,  # Pairs are STREAMED to this file
        window_size=args.window_size,
        negative_ratio=args.negative_ratio,
        max_sentences=args.max_sentences,
        checkpoint_path=checkpoint_path if args.resume else None,
        checkpoint_interval=args.checkpoint_interval
    )

    # Count total pairs in output file
    total_pairs = 0
    with open(args.output) as f:
        for _ in f:
            total_pairs += 1

    logger.info(f"\nComplete!")
    logger.info(f"Vocabulary: {len(root_to_idx):,} roots")
    logger.info(f"Training pairs: {total_pairs:,} (positive: {positive_count:,})")
    logger.info(f"Sentences processed: {sentences_processed:,}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Log: {log_path}")

    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        logger.info("Checkpoint removed (complete)")

    return 0


if __name__ == '__main__':
    sys.exit(main())
