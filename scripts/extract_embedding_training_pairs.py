#!/usr/bin/env python3
"""
Extract Root Embedding Training Pairs with Cross-Sentence Context

VERSION: v2.1
COMPATIBLE WITH: v2.1 database schema (AST-native)
DEPENDENCIES: Parser (for content root identification)
STAGE: Data

Description:
    Extracts training pairs for skip-gram root embeddings with paragraph-aware
    cross-sentence context. Filters function words, includes adjacent sentence
    context with reduced weight.

Pipeline Position:
    v2.1 DB → [THIS SCRIPT] → training_pairs.jsonl → train_root_embeddings.py

Usage:
    python scripts/extract_embedding_training_pairs.py \
        --db-path data/indexes/kuzu_v2.1 \
        --output data/training/root_embedding_pairs.jsonl \
        --window-size 5 \
        --cross-sentence-weight 0.5 \
        --min-frequency 5

Inputs:
    - Kuzu v2.1 database with AST-native schema
    - Frazoteksto, Paragrafo, Vorto nodes
    - SEKVA_FRAZOTEKSTO relationships

Outputs:
    - training_pairs.jsonl: (target_root, context_root, weight) tuples
    - Format: {"target": "hund", "context": "kat", "weight": 1.0}
    - Cross-sentence pairs have weight=0.5

Quality Checks:
    - Function words filtered (artikolo, prepozicio, konjunkcio, pronomo)
    - Only content roots included (substantivo, verbo, adjektivo, adverbo)
    - Minimum frequency threshold (default: 5 occurrences)
    - Paragraph boundaries respected (no cross-paragraph context)

Last Updated: 2026-03-09
Author: Claude + Marc
Related Issues: Phase 1 Root Embeddings
See Also: docs/ROOT_EMBEDDINGS_DESIGN.md, docs/CROSS_SENTENCE_CONTEXT_EMBEDDINGS.md
"""

import argparse
import json
import logging
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Set, Optional
import random
import numpy as np
import kuzu

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Function word categories to EXCLUDE from embeddings
FUNCTION_WORD_CATEGORIES = {
    'artikolo',      # la
    'prepozicio',    # de, al, en, sur, sub, etc.
    'konjunkcio',    # kaj, aŭ, sed, ĉar, etc.
    'pronomo',       # mi, vi, li, ŝi, ĝi, etc. (most)
}

# Content word categories to INCLUDE in embeddings
CONTENT_WORD_CATEGORIES = {
    'substantivo',   # dog, cat, tree
    'verbo',         # run, eat, sleep
    'adjektivo',     # big, small, red
    'adverbo',       # quickly, slowly, well
}


def extract_paragraph_sentences(db: kuzu.Database) -> List[Dict]:
    """
    Extract all sentences grouped by paragraph with sequential ordering.

    Returns:
        List of paragraphs, each containing:
        {
            'paragrafo_id': int,
            'sentences': [
                {'id': int, 'teksto': str, 'frazo_ordo': int, 'roots': [...]},
                ...
            ]
        }
    """
    logger.info("Extracting sentences grouped by paragraph...")

    conn = kuzu.Connection(db)

    # Query: Get all sentences with their paragraph and ordering
    query = """
    MATCH (f:Frazoteksto)-[:EN_PARAGRAFO]->(p:Paragrafo)
    RETURN
        p.id AS paragrafo_id,
        f.id AS frazoteksto_id,
        f.teksto AS teksto,
        f.frazo_ordo AS frazo_ordo
    ORDER BY p.id, f.frazo_ordo
    """

    result = conn.execute(query)

    paragraphs = defaultdict(list)

    while result.has_next():
        row = result.get_next()
        paragrafo_id = row[0]
        frazoteksto_id = row[1]
        teksto = row[2]
        frazo_ordo = row[3]

        paragraphs[paragrafo_id].append({
            'id': frazoteksto_id,
            'teksto': teksto,
            'frazo_ordo': frazo_ordo,
            'roots': []  # Will be populated with content roots
        })

    # Convert to list and sort by paragraph ID
    paragraph_list = [
        {
            'paragrafo_id': pid,
            'sentences': sorted(sents, key=lambda s: s['frazo_ordo'])
        }
        for pid, sents in sorted(paragraphs.items())
    ]

    logger.info(f"Extracted {len(paragraph_list)} paragraphs with {sum(len(p['sentences']) for p in paragraph_list)} sentences")

    return paragraph_list


def extract_content_roots_from_sentence(db: kuzu.Database, frazoteksto_id: int) -> List[str]:
    """
    Extract content roots from a sentence (excluding function words).

    Returns:
        List of content root strings (substantivo, verbo, adjektivo, adverbo only)
    """
    conn = kuzu.Connection(db)

    query = """
    MATCH (f:Frazoteksto)-[:FRAZOTEKSTO_HAVAS_AST]->(ast:AST)-[:AST_HAVAS_FRAZON]->(frazo:Frazo)
    WHERE f.id = $frazoteksto_id
    MATCH (frazo)-[:HAVAS_VERBON]->(v:Vorto)
    WHERE v.radiko IS NOT NULL
      AND v.radiko <> ''
      AND v.vortspeco IN ['substantivo', 'verbo', 'adjektivo', 'adverbo']
    RETURN DISTINCT v.radiko AS radiko
    UNION
    MATCH (f:Frazoteksto)-[:FRAZOTEKSTO_HAVAS_AST]->(ast:AST)-[:AST_HAVAS_FRAZON]->(frazo:Frazo)
    WHERE f.id = $frazoteksto_id
    MATCH (frazo)-[:HAVAS_SUBJEKTON_VORTO]->(v:Vorto)
    WHERE v.radiko IS NOT NULL AND v.radiko <> ''
      AND v.vortspeco IN ['substantivo', 'verbo', 'adjektivo', 'adverbo']
    RETURN DISTINCT v.radiko AS radiko
    UNION
    MATCH (f:Frazoteksto)-[:FRAZOTEKSTO_HAVAS_AST]->(ast:AST)-[:AST_HAVAS_FRAZON]->(frazo:Frazo)
    WHERE f.id = $frazoteksto_id
    MATCH (frazo)-[:HAVAS_OBJEKTON_VORTO]->(v:Vorto)
    WHERE v.radiko IS NOT NULL AND v.radiko <> ''
      AND v.vortspeco IN ['substantivo', 'verbo', 'adjektivo', 'adverbo']
    RETURN DISTINCT v.radiko AS radiko
    UNION
    MATCH (f:Frazoteksto)-[:FRAZOTEKSTO_HAVAS_AST]->(ast:AST)-[:AST_HAVAS_FRAZON]->(frazo:Frazo)
    WHERE f.id = $frazoteksto_id
    MATCH (frazo)-[:HAVAS_SUBJEKTON_VORTGRUPO|HAVAS_OBJEKTON_VORTGRUPO]->(vg:Vortgrupo)
    MATCH (vg)-[:HAVAS_KERNON|HAVAS_PRISKRIBON]->(v:Vorto)
    WHERE v.radiko IS NOT NULL AND v.radiko <> ''
      AND v.vortspeco IN ['substantivo', 'verbo', 'adjektivo', 'adverbo']
    RETURN DISTINCT v.radiko AS radiko
    """

    result = conn.execute(query, parameters={'frazoteksto_id': frazoteksto_id})

    roots = []
    while result.has_next():
        row = result.get_next()
        radiko = row[0]
        if radiko:
            roots.append(radiko)

    return roots


def populate_content_roots(db: kuzu.Database, paragraphs: List[Dict]) -> None:
    """
    Populate content roots for all sentences using chunked batch queries with checkpointing.

    Processes in chunks for:
    - Memory efficiency (don't load all 5M sentences at once)
    - Progress updates (every chunk ~1-2 minutes)
    - Restartability (checkpoint after each chunk)
    """
    import time
    import tempfile
    from pathlib import Path

    logger.info("Extracting content roots for all sentences (chunked batch mode)...")

    conn = kuzu.Connection(db)
    total_sentences = sum(len(p['sentences']) for p in paragraphs)

    # Create sentence ID lookup
    sentence_map = {}
    sentence_ids = []
    for paragraph in paragraphs:
        for sentence in paragraph['sentences']:
            sentence_map[sentence['id']] = sentence
            sentence['roots'] = []
            sentence_ids.append(sentence['id'])

    sentence_ids.sort()

    # Checkpoint file
    checkpoint_file = Path(tempfile.gettempdir()) / 'klareco_root_extraction_checkpoint.json'

    # Load checkpoint if exists
    start_chunk = 0
    if checkpoint_file.exists():
        try:
            import json
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                start_chunk = checkpoint.get('last_completed_chunk', 0) + 1
                logger.info(f"Resuming from chunk {start_chunk} (checkpoint found)")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}, starting fresh")
            start_chunk = 0

    # Process in chunks
    CHUNK_SIZE = 100000
    total_chunks = (len(sentence_ids) + CHUNK_SIZE - 1) // CHUNK_SIZE

    # 4 separate queries (faster than combined OPTIONAL MATCH)
    query_templates = [
        # Verbs
        '''
        MATCH (f:Frazoteksto)-[:FRAZOTEKSTO_HAVAS_AST]->(ast:AST)-[:AST_HAVAS_FRAZON]->(frazo:Frazo)
        MATCH (frazo)-[:HAVAS_VERBON]->(v:Vorto)
        WHERE f.id >= $start_id AND f.id < $end_id
          AND v.radiko IS NOT NULL AND v.radiko <> ''
          AND v.vortspeco IN ['substantivo', 'verbo', 'adjektivo', 'adverbo']
        RETURN f.id, v.radiko
        ''',
        # Subject words
        '''
        MATCH (f:Frazoteksto)-[:FRAZOTEKSTO_HAVAS_AST]->(ast:AST)-[:AST_HAVAS_FRAZON]->(frazo:Frazo)
        MATCH (frazo)-[:HAVAS_SUBJEKTON_VORTO]->(v:Vorto)
        WHERE f.id >= $start_id AND f.id < $end_id
          AND v.radiko IS NOT NULL AND v.radiko <> ''
          AND v.vortspeco IN ['substantivo', 'verbo', 'adjektivo', 'adverbo']
        RETURN f.id, v.radiko
        ''',
        # Object words
        '''
        MATCH (f:Frazoteksto)-[:FRAZOTEKSTO_HAVAS_AST]->(ast:AST)-[:AST_HAVAS_FRAZON]->(frazo:Frazo)
        MATCH (frazo)-[:HAVAS_OBJEKTON_VORTO]->(v:Vorto)
        WHERE f.id >= $start_id AND f.id < $end_id
          AND v.radiko IS NOT NULL AND v.radiko <> ''
          AND v.vortspeco IN ['substantivo', 'verbo', 'adjektivo', 'adverbo']
        RETURN f.id, v.radiko
        ''',
        # Word group words
        '''
        MATCH (f:Frazoteksto)-[:FRAZOTEKSTO_HAVAS_AST]->(ast:AST)-[:AST_HAVAS_FRAZON]->(frazo:Frazo)
        MATCH (frazo)-[:HAVAS_SUBJEKTON_VORTGRUPO|HAVAS_OBJEKTON_VORTGRUPO]->(vg:Vortgrupo)
        MATCH (vg)-[:HAVAS_KERNON|HAVAS_PRISKRIBON]->(v:Vorto)
        WHERE f.id >= $start_id AND f.id < $end_id
          AND v.radiko IS NOT NULL AND v.radiko <> ''
          AND v.vortspeco IN ['substantivo', 'verbo', 'adjektivo', 'adverbo']
        RETURN f.id, v.radiko
        '''
    ]

    total_roots = 0

    for chunk_idx in range(start_chunk, total_chunks):
        chunk_start_time = time.time()

        start_idx = chunk_idx * CHUNK_SIZE
        end_idx = min(start_idx + CHUNK_SIZE, len(sentence_ids))

        start_id = sentence_ids[start_idx]
        end_id = sentence_ids[end_idx - 1] + 1 if end_idx < len(sentence_ids) else sentence_ids[-1] + 1000000

        logger.info(f"Processing chunk {chunk_idx + 1}/{total_chunks} (sentences {start_idx:,}-{end_idx:,}, IDs {start_id}-{end_id})")

        # Execute all 4 queries for this chunk
        chunk_roots = 0
        for query_template in query_templates:
            result = conn.execute(query_template, parameters={'start_id': start_id, 'end_id': end_id})

            while result.has_next():
                row = result.get_next()
                frazoteksto_id = row[0]
                radiko = row[1]

                if frazoteksto_id in sentence_map:
                    sentence_map[frazoteksto_id]['roots'].append(radiko)
                    chunk_roots += 1

        total_roots += chunk_roots
        chunk_elapsed = time.time() - chunk_start_time

        # Progress update
        sentences_processed = end_idx
        pct_complete = 100 * sentences_processed / total_sentences
        avg_time_per_chunk = chunk_elapsed
        remaining_chunks = total_chunks - (chunk_idx + 1)
        eta_minutes = (remaining_chunks * avg_time_per_chunk) / 60

        logger.info(f"  Chunk complete: {chunk_roots:,} roots in {chunk_elapsed:.1f}s | Total: {sentences_processed:,}/{total_sentences:,} ({pct_complete:.1f}%) | ETA: {eta_minutes:.1f} min")

        # Save checkpoint
        try:
            import json
            with open(checkpoint_file, 'w') as f:
                json.dump({'last_completed_chunk': chunk_idx, 'total_roots': total_roots}, f)
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")

    # Clean up checkpoint
    if checkpoint_file.exists():
        checkpoint_file.unlink()

    logger.info(f"Content roots extracted: {total_roots:,} total roots from {total_sentences:,} sentences")


def compute_subsampling_probability(
    root_frequency: int,
    total_roots: int,
    threshold: float = 1e-3
) -> float:
    """
    Compute probability of keeping a word based on its frequency.

    Frequent words are subsampled to reduce training time and improve quality.
    Formula from original word2vec paper (Mikolov et al., 2013).

    Args:
        root_frequency: Number of times this root appears
        total_roots: Total number of root occurrences
        threshold: Subsampling threshold (default: 1e-3)

    Returns:
        Probability of keeping this word (0.0 to 1.0)
    """
    if threshold == 0:
        return 1.0  # Subsampling disabled

    freq = root_frequency / total_roots

    # Original word2vec formula
    keep_prob = (np.sqrt(freq / threshold) + 1) * (threshold / freq)

    return min(1.0, keep_prob)


def generate_training_pairs(
    paragraphs: List[Dict],
    window_size: int = 5,
    cross_sentence_weight: float = 0.5,
    subsample_threshold: float = 1e-3
) -> List[Tuple[str, str, float]]:
    """
    Generate (target, context, weight) training pairs with cross-sentence context.

    Within-sentence pairs: weight = 1.0
    Adjacent-sentence pairs: weight = cross_sentence_weight (default 0.5)

    Includes subsampling of frequent words for faster training and better quality.

    Args:
        paragraphs: List of paragraphs with sentences and roots
        window_size: Context window size (default 5)
        cross_sentence_weight: Weight for adjacent sentence pairs (default 0.5)
        subsample_threshold: Threshold for subsampling (default 1e-3, 0 to disable)

    Returns:
        List of (target_root, context_root, weight) tuples
    """
    logger.info("Generating training pairs with cross-sentence context...")

    # Compute root frequencies for subsampling
    root_counts = Counter()
    for paragraph in paragraphs:
        for sentence in paragraph['sentences']:
            for root in sentence['roots']:
                root_counts[root] += 1

    total_roots = sum(root_counts.values())

    # Compute subsampling probabilities
    subsample_probs = {}
    if subsample_threshold > 0:
        for root, count in root_counts.items():
            subsample_probs[root] = compute_subsampling_probability(
                count, total_roots, subsample_threshold
            )

        # Log subsampling statistics
        subsampled_roots = sum(1 for p in subsample_probs.values() if p < 1.0)
        logger.info(f"Subsampling enabled: {subsampled_roots}/{len(subsample_probs)} roots will be subsampled")

        # Show examples of heavily subsampled roots
        heavily_subsampled = [(root, prob) for root, prob in subsample_probs.items() if prob < 0.5]
        if heavily_subsampled:
            heavily_subsampled.sort(key=lambda x: x[1])
            logger.info(f"Most heavily subsampled roots (keep prob < 0.5): {heavily_subsampled[:5]}")
    else:
        logger.info("Subsampling disabled")
        subsample_probs = {root: 1.0 for root in root_counts.keys()}

    pairs = []

    for paragraph in paragraphs:
        sentences = paragraph['sentences']

        # Process each sentence
        for sent_idx, sentence in enumerate(sentences):
            target_roots = sentence['roots']

            if not target_roots:
                continue

            # Within-sentence context (weight = 1.0)
            for i, target in enumerate(target_roots):
                # Subsample check for target word
                if random.random() > subsample_probs.get(target, 1.0):
                    continue  # Skip this target word

                # Look at context window around target
                start = max(0, i - window_size)
                end = min(len(target_roots), i + window_size + 1)

                for j in range(start, end):
                    if i != j:  # Don't pair with self
                        context = target_roots[j]
                        # Subsample check for context word
                        if random.random() > subsample_probs.get(context, 1.0):
                            continue
                        pairs.append((target, context, 1.0))

            # Cross-sentence context (weight = cross_sentence_weight)
            # Only look at adjacent sentences (prev and next)

            # Previous sentence
            if sent_idx > 0:
                prev_sentence = sentences[sent_idx - 1]
                prev_roots = prev_sentence['roots']

                for target in target_roots:
                    if random.random() > subsample_probs.get(target, 1.0):
                        continue
                    # Take last window_size roots from previous sentence
                    for context in prev_roots[-window_size:]:
                        if random.random() > subsample_probs.get(context, 1.0):
                            continue
                        pairs.append((target, context, cross_sentence_weight))

            # Next sentence
            if sent_idx < len(sentences) - 1:
                next_sentence = sentences[sent_idx + 1]
                next_roots = next_sentence['roots']

                for target in target_roots:
                    if random.random() > subsample_probs.get(target, 1.0):
                        continue
                    # Take first window_size roots from next sentence
                    for context in next_roots[:window_size]:
                        if random.random() > subsample_probs.get(context, 1.0):
                            continue
                        pairs.append((target, context, cross_sentence_weight))

    logger.info(f"Generated {len(pairs)} training pairs")

    return pairs


def filter_by_frequency(
    pairs: List[Tuple[str, str, float]],
    min_frequency: int = 5,
    target_vocabulary: Optional[Path] = None
) -> Tuple[List[Tuple[str, str, float]], List[str]]:
    """
    Filter training pairs by frequency or target vocabulary.

    Args:
        pairs: List of (target, context, weight) tuples
        min_frequency: Minimum occurrences for auto-generated vocabulary
        target_vocabulary: Optional path to JSON file with target vocabulary list

    Returns:
        (filtered_pairs, vocabulary)
    """
    # Determine vocabulary (target vocabulary or auto-generated)
    if target_vocabulary:
        logger.info(f"Using target vocabulary from {target_vocabulary}")
        with open(target_vocabulary, 'r', encoding='utf-8') as f:
            vocabulary = set(json.load(f))
        logger.info(f"Target vocabulary size: {len(vocabulary)} roots")

        # Filter pairs immediately (no need to count frequencies)
        filtered_pairs = [
            (target, context, weight)
            for target, context, weight in pairs
            if target in vocabulary and context in vocabulary
        ]

        logger.info(f"Filtered pairs: {len(filtered_pairs)} / {len(pairs)} ({100*len(filtered_pairs)/len(pairs):.1f}%)")

        # Build vocabulary list (only roots that appear in filtered pairs)
        roots_in_pairs = set()
        for target, context, weight in filtered_pairs:
            roots_in_pairs.add(target)
            roots_in_pairs.add(context)
        vocabulary = sorted(roots_in_pairs)
        logger.info(f"Roots from target vocabulary present in pairs: {len(vocabulary)}")

    else:
        logger.info(f"Auto-generating vocabulary with min_frequency={min_frequency}...")

        # Count root frequencies
        root_counts = Counter()
        for target, context, weight in pairs:
            root_counts[target] += 1
            root_counts[context] += 1

        # Build vocabulary of frequent roots
        vocabulary = {root for root, count in root_counts.items() if count >= min_frequency}

        logger.info(f"Vocabulary size: {len(vocabulary)} roots (>= {min_frequency} occurrences)")

        # Filter pairs
        filtered_pairs = [
            (target, context, weight)
            for target, context, weight in pairs
            if target in vocabulary and context in vocabulary
        ]

        logger.info(f"Filtered pairs: {len(filtered_pairs)} / {len(pairs)} ({100*len(filtered_pairs)/len(pairs):.1f}%)")

        vocabulary = sorted(vocabulary)

    return filtered_pairs, vocabulary


def save_training_pairs(
    pairs: List[Tuple[str, str, float]],
    output_path: Path
) -> None:
    """
    Save training pairs to JSONL file.
    """
    logger.info(f"Saving training pairs to {output_path}...")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for target, context, weight in pairs:
            record = {
                'target': target,
                'context': context,
                'weight': weight
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    logger.info(f"Saved {len(pairs)} training pairs")


def save_vocabulary(vocabulary: Set[str], output_path: Path) -> None:
    """
    Save vocabulary to JSON file.
    """
    logger.info(f"Saving vocabulary to {output_path}...")

    vocab_list = sorted(vocabulary)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(vocab_list, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved vocabulary with {len(vocab_list)} roots")


def compute_statistics(
    pairs: List[Tuple[str, str, float]],
    vocabulary: Set[str]
) -> Dict:
    """
    Compute training data statistics.
    """
    within_sentence_pairs = sum(1 for _, _, w in pairs if w == 1.0)
    cross_sentence_pairs = sum(1 for _, _, w in pairs if w < 1.0)

    total_weighted_pairs = sum(w for _, _, w in pairs)

    root_pair_counts = Counter()
    for target, context, weight in pairs:
        root_pair_counts[target] += weight

    avg_pairs_per_root = total_weighted_pairs / len(vocabulary) if vocabulary else 0

    stats = {
        'total_pairs': len(pairs),
        'within_sentence_pairs': within_sentence_pairs,
        'cross_sentence_pairs': cross_sentence_pairs,
        'total_weighted_pairs': total_weighted_pairs,
        'vocabulary_size': len(vocabulary),
        'avg_pairs_per_root': avg_pairs_per_root,
        'min_pairs_per_root': min(root_pair_counts.values()) if root_pair_counts else 0,
        'max_pairs_per_root': max(root_pair_counts.values()) if root_pair_counts else 0,
    }

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Extract root embedding training pairs with cross-sentence context'
    )
    parser.add_argument(
        '--db-path',
        type=Path,
        required=True,
        help='Path to Kuzu v2.1 database directory'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output path for training pairs JSONL file'
    )
    parser.add_argument(
        '--vocab-output',
        type=Path,
        help='Output path for vocabulary JSON file (default: same dir as output with _vocab.json suffix)'
    )
    parser.add_argument(
        '--window-size',
        type=int,
        default=5,
        help='Context window size (default: 5)'
    )
    parser.add_argument(
        '--cross-sentence-weight',
        type=float,
        default=0.5,
        help='Weight for cross-sentence context pairs (default: 0.5)'
    )
    parser.add_argument(
        '--min-frequency',
        type=int,
        default=5,
        help='Minimum root frequency to include in vocabulary (default: 5)'
    )
    parser.add_argument(
        '--subsample-threshold',
        type=float,
        default=1e-3,
        help='Subsampling threshold for frequent words (default: 1e-3, 0 to disable)'
    )
    parser.add_argument(
        '--target-vocabulary',
        type=Path,
        help='Optional: Target vocabulary file (JSON list of roots). If provided, only pairs with roots in this vocabulary will be extracted. If not provided, vocabulary is auto-generated from corpus with min-frequency filtering.'
    )

    args = parser.parse_args()

    # Set default vocabulary output path
    if args.vocab_output is None:
        args.vocab_output = args.output.parent / (args.output.stem + '_vocab.json')

    logger.info("=" * 80)
    logger.info("Root Embedding Training Pair Extraction")
    logger.info("=" * 80)
    logger.info(f"Database: {args.db_path}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Vocabulary: {args.vocab_output}")
    logger.info(f"Window size: {args.window_size}")
    logger.info(f"Cross-sentence weight: {args.cross_sentence_weight}")
    logger.info(f"Min frequency: {args.min_frequency}")
    logger.info("=" * 80)

    # Open database
    logger.info("Opening Kuzu database...")
    db = kuzu.Database(str(args.db_path))

    # Extract sentences grouped by paragraph
    paragraphs = extract_paragraph_sentences(db)

    # Populate content roots for all sentences
    populate_content_roots(db, paragraphs)

    # Generate training pairs with cross-sentence context and subsampling
    pairs = generate_training_pairs(
        paragraphs,
        window_size=args.window_size,
        cross_sentence_weight=args.cross_sentence_weight,
        subsample_threshold=args.subsample_threshold
    )

    # Filter by frequency or target vocabulary
    filtered_pairs, vocabulary = filter_by_frequency(
        pairs,
        args.min_frequency,
        args.target_vocabulary
    )

    # Compute statistics
    stats = compute_statistics(filtered_pairs, vocabulary)

    logger.info("=" * 80)
    logger.info("Training Data Statistics")
    logger.info("=" * 80)
    logger.info(f"Total pairs: {stats['total_pairs']:,}")
    logger.info(f"Within-sentence pairs: {stats['within_sentence_pairs']:,}")
    logger.info(f"Cross-sentence pairs: {stats['cross_sentence_pairs']:,}")
    logger.info(f"Total weighted pairs: {stats['total_weighted_pairs']:,.1f}")
    logger.info(f"Vocabulary size: {stats['vocabulary_size']:,} roots")
    logger.info(f"Avg pairs per root: {stats['avg_pairs_per_root']:,.1f}")
    logger.info(f"Min pairs per root: {stats['min_pairs_per_root']:,.0f}")
    logger.info(f"Max pairs per root: {stats['max_pairs_per_root']:,.0f}")
    logger.info("=" * 80)

    # Save outputs
    save_training_pairs(filtered_pairs, args.output)
    save_vocabulary(vocabulary, args.vocab_output)

    # Save statistics
    stats_path = args.output.parent / (args.output.stem + '_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved statistics to {stats_path}")

    logger.info("=" * 80)
    logger.info("DONE")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
