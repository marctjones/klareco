#!/usr/bin/env python3
"""
Extract Root Embedding Training Pairs with AST-Aware Semantic Pairing (OPTIMIZED)

VERSION: v3.0 (AST-aware, batched)
COMPATIBLE WITH: v2.1 database schema (AST-native)
DEPENDENCIES: Parser (for AST structure)
STAGE: Data

Description:
    OPTIMIZED version with batch processing and combined queries.

    PERFORMANCE IMPROVEMENTS:
    - Batch queries: Process 1000 sentences per query (vs 1)
    - Combined queries: Single query for all pair types per batch
    - Expected: 6-12x faster (~5-10 minutes vs ~1 hour)

Pipeline Position:
    v2.1 DB → [THIS SCRIPT] → training_pairs.jsonl → train_root_embeddings.py

Usage:
    python scripts/extract_embedding_training_pairs_ast_aware_optimized.py \
        --db-path data/indexes/kuzu_v2.1 \
        --output data/training/root_embedding_pairs.jsonl \
        --target-vocabulary data/vocabularies/production_semantic_roots_15k.json \
        --cross-sentence-weight 0.5 \
        --subsample-threshold 1e-3 \
        --batch-size 1000

Last Updated: 2026-03-10
Author: Claude + Marc
See Also: docs/ROOT_EMBEDDINGS_DESIGN.md (updated 2026-03-10)
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


def extract_all_semantic_pairs_batch(
    db: kuzu.Database,
    sentence_ids: List[int],
    subsample_probs: Dict[str, float]
) -> Dict[int, List[Tuple[str, str, float]]]:
    """
    Extract ALL semantic pairs for a batch of sentences in optimized queries.

    OPTIMIZATION: Process 1000 sentences per query instead of 1.

    Returns all three pair types:
    1. Modifier-head relationships (weight 1.0)
    2. Semantic arguments (weight 0.8)
    3. Semantic heads for cross-sentence (returned separately for efficiency)

    Args:
        db: Kuzu database
        sentence_ids: Batch of sentence IDs (typically 1000)
        subsample_probs: Subsampling probabilities

    Returns:
        Dictionary mapping sentence_id → list of (target, context, weight) pairs
    """
    conn = kuzu.Connection(db)
    pairs_by_sentence = defaultdict(list)

    if not sentence_ids:
        return pairs_by_sentence

    # ============================================================================
    # QUERY 1: Modifier-Head Relationships (adjective-noun, adverb-verb)
    # ============================================================================

    query_modifier_head = """
    MATCH (f:Frazoteksto)-[:FRAZOTEKSTO_HAVAS_AST]->(ast:AST)-[:AST_HAVAS_FRAZON]->(frazo:Frazo)
    WHERE f.id IN $sentence_ids

    // Adjective-noun pairs from vortgrupo
    MATCH (frazo)-[:HAVAS_SUBJEKTON_VORTGRUPO|HAVAS_OBJEKTON_VORTGRUPO]->(vg:Vortgrupo)
    MATCH (vg)-[:HAVAS_KERNON]->(head:Vorto)
    MATCH (vg)-[:HAVAS_PRISKRIBON]->(modifier:Vorto)
    WHERE head.radiko IS NOT NULL AND head.radiko <> ''
      AND modifier.radiko IS NOT NULL AND modifier.radiko <> ''
      AND head.vortspeco = 'substantivo'
      AND modifier.vortspeco = 'adjektivo'

    RETURN f.id AS sentence_id, modifier.radiko AS modifier, head.radiko AS head

    UNION ALL

    // Adverb-verb pairs
    MATCH (f:Frazoteksto)-[:FRAZOTEKSTO_HAVAS_AST]->(ast:AST)-[:AST_HAVAS_FRAZON]->(frazo:Frazo)
    WHERE f.id IN $sentence_ids

    MATCH (frazo)-[:HAVAS_VERBON]->(verb:Vorto)
    MATCH (frazo)-[:HAVAS_ALIAJN]->(modifier:Vorto)
    WHERE verb.radiko IS NOT NULL AND verb.radiko <> ''
      AND modifier.radiko IS NOT NULL AND modifier.radiko <> ''
      AND verb.vortspeco = 'verbo'
      AND modifier.vortspeco = 'adverbo'

    RETURN f.id AS sentence_id, modifier.radiko AS modifier, verb.radiko AS head
    """

    try:
        result = conn.execute(query_modifier_head, parameters={'sentence_ids': sentence_ids})

        while result.has_next():
            row = result.get_next()
            sentence_id = row[0]
            modifier = row[1]
            head = row[2]

            # Apply subsampling
            if random.random() <= subsample_probs.get(modifier, 1.0) and \
               random.random() <= subsample_probs.get(head, 1.0):
                # Bidirectional pairs
                pairs_by_sentence[sentence_id].append((modifier, head, 1.0))
                pairs_by_sentence[sentence_id].append((head, modifier, 1.0))

    except Exception as e:
        logger.warning(f"Failed to extract modifier-head pairs for batch: {e}")

    # ============================================================================
    # QUERY 2: Semantic Arguments (subject-object pairs)
    # ============================================================================

    query_semantic_args = """
    MATCH (f:Frazoteksto)-[:FRAZOTEKSTO_HAVAS_AST]->(ast:AST)-[:AST_HAVAS_FRAZON]->(frazo:Frazo)
    WHERE f.id IN $sentence_ids

    // Get subject head
    OPTIONAL MATCH (frazo)-[:HAVAS_SUBJEKTON_VORTO]->(subj:Vorto)
    WHERE subj.radiko IS NOT NULL AND subj.radiko <> ''
      AND subj.vortspeco IN ['substantivo', 'adjektivo']

    OPTIONAL MATCH (frazo)-[:HAVAS_SUBJEKTON_VORTGRUPO]->(subj_vg:Vortgrupo)
    OPTIONAL MATCH (subj_vg)-[:HAVAS_KERNON]->(subj_head:Vorto)
    WHERE subj_head.radiko IS NOT NULL AND subj_head.radiko <> ''
      AND subj_head.vortspeco IN ['substantivo', 'adjektivo']

    // Get object head
    OPTIONAL MATCH (frazo)-[:HAVAS_OBJEKTON_VORTO]->(obj:Vorto)
    WHERE obj.radiko IS NOT NULL AND obj.radiko <> ''
      AND obj.vortspeco IN ['substantivo', 'adjektivo']

    OPTIONAL MATCH (frazo)-[:HAVAS_OBJEKTON_VORTGRUPO]->(obj_vg:Vortgrupo)
    OPTIONAL MATCH (obj_vg)-[:HAVAS_KERNON]->(obj_head:Vorto)
    WHERE obj_head.radiko IS NOT NULL AND obj_head.radiko <> ''
      AND obj_head.vortspeco IN ['substantivo', 'adjektivo']

    WITH f.id AS sentence_id,
         COALESCE(subj.radiko, subj_head.radiko) AS subj_root,
         COALESCE(obj.radiko, obj_head.radiko) AS obj_root

    WHERE subj_root IS NOT NULL AND obj_root IS NOT NULL
      AND subj_root <> obj_root

    RETURN sentence_id, subj_root, obj_root
    """

    try:
        result = conn.execute(query_semantic_args, parameters={'sentence_ids': sentence_ids})

        while result.has_next():
            row = result.get_next()
            sentence_id = row[0]
            subj_root = row[1]
            obj_root = row[2]

            # Apply subsampling
            if random.random() <= subsample_probs.get(subj_root, 1.0) and \
               random.random() <= subsample_probs.get(obj_root, 1.0):
                # Bidirectional pairs
                pairs_by_sentence[sentence_id].append((subj_root, obj_root, 0.8))
                pairs_by_sentence[sentence_id].append((obj_root, subj_root, 0.8))

    except Exception as e:
        logger.warning(f"Failed to extract semantic arguments for batch: {e}")

    return pairs_by_sentence


def extract_semantic_heads_batch(
    db: kuzu.Database,
    sentence_ids: List[int]
) -> Dict[int, List[str]]:
    """
    Extract semantic heads (subject/object nouns) for batch of sentences.

    Used for cross-sentence pairing. Batched for efficiency.

    Args:
        db: Kuzu database
        sentence_ids: Batch of sentence IDs

    Returns:
        Dictionary mapping sentence_id → list of semantic heads
    """
    conn = kuzu.Connection(db)
    heads_by_sentence = defaultdict(list)

    if not sentence_ids:
        return heads_by_sentence

    query = """
    MATCH (f:Frazoteksto)-[:FRAZOTEKSTO_HAVAS_AST]->(ast:AST)-[:AST_HAVAS_FRAZON]->(frazo:Frazo)
    WHERE f.id IN $sentence_ids

    // Get subject head
    OPTIONAL MATCH (frazo)-[:HAVAS_SUBJEKTON_VORTO]->(subj:Vorto)
    WHERE subj.radiko IS NOT NULL AND subj.radiko <> ''
      AND subj.vortspeco = 'substantivo'

    OPTIONAL MATCH (frazo)-[:HAVAS_SUBJEKTON_VORTGRUPO]->(subj_vg:Vortgrupo)
    OPTIONAL MATCH (subj_vg)-[:HAVAS_KERNON]->(subj_head:Vorto)
    WHERE subj_head.radiko IS NOT NULL AND subj_head.radiko <> ''
      AND subj_head.vortspeco = 'substantivo'

    // Get object head
    OPTIONAL MATCH (frazo)-[:HAVAS_OBJEKTON_VORTO]->(obj:Vorto)
    WHERE obj.radiko IS NOT NULL AND obj.radiko <> ''
      AND obj.vortspeco = 'substantivo'

    OPTIONAL MATCH (frazo)-[:HAVAS_OBJEKTON_VORTGRUPO]->(obj_vg:Vortgrupo)
    OPTIONAL MATCH (obj_vg)-[:HAVAS_KERNON]->(obj_head:Vorto)
    WHERE obj_head.radiko IS NOT NULL AND obj_head.radiko <> ''
      AND obj_head.vortspeco = 'substantivo'

    RETURN f.id AS sentence_id,
           COALESCE(subj.radiko, subj_head.radiko) AS subj_root,
           COALESCE(obj.radiko, obj_head.radiko) AS obj_root
    """

    try:
        result = conn.execute(query, parameters={'sentence_ids': sentence_ids})

        while result.has_next():
            row = result.get_next()
            sentence_id = row[0]
            subj_root = row[1]
            obj_root = row[2]

            if subj_root:
                heads_by_sentence[sentence_id].append(subj_root)
            if obj_root:
                heads_by_sentence[sentence_id].append(obj_root)

    except Exception as e:
        logger.warning(f"Failed to extract semantic heads for batch: {e}")

    return heads_by_sentence


def compute_subsampling_probability(count: int, total_count: int, threshold: float) -> float:
    """Compute Mikolov et al. 2013 subsampling probability."""
    if count == 0 or total_count == 0:
        return 1.0

    frequency = count / total_count

    if frequency > threshold:
        prob = (np.sqrt(threshold / frequency) + threshold / frequency)
        return min(1.0, prob)
    else:
        return 1.0


def extract_paragraph_sentences(db: kuzu.Database) -> List[Dict]:
    """Extract all sentences grouped by paragraph with sequential ordering."""
    logger.info("Extracting sentences grouped by paragraph...")

    conn = kuzu.Connection(db)

    query = """
    MATCH (f:Frazoteksto)-[:EN_PARAGRAFO]->(p:Paragrafo)
    RETURN
        p.id AS paragrafo_id,
        f.id AS frazoteksto_id,
        f.frazo_ordo AS frazo_ordo
    ORDER BY p.id, f.frazo_ordo
    """

    result = conn.execute(query)

    paragraphs = defaultdict(list)

    while result.has_next():
        row = result.get_next()
        paragrafo_id = row[0]
        frazoteksto_id = row[1]
        frazo_ordo = row[2]

        paragraphs[paragrafo_id].append({
            'id': frazoteksto_id,
            'frazo_ordo': frazo_ordo
        })

    # Convert to list and sort
    paragraph_list = [
        {
            'paragrafo_id': pid,
            'sentences': sorted(sents, key=lambda s: s['frazo_ordo'])
        }
        for pid, sents in sorted(paragraphs.items())
    ]

    logger.info(f"Extracted {len(paragraph_list)} paragraphs with {sum(len(p['sentences']) for p in paragraph_list)} sentences")

    return paragraph_list


def compute_root_frequencies(db: kuzu.Database) -> Dict[str, int]:
    """Compute frequency of each content root across entire corpus."""
    logger.info("Computing root frequencies across corpus...")

    conn = kuzu.Connection(db)

    query = """
    MATCH (f:Frazoteksto)-[:FRAZOTEKSTO_HAVAS_AST]->(ast:AST)-[:AST_HAVAS_FRAZON]->(frazo:Frazo)

    MATCH (frazo)-[:HAVAS_VERBON]->(v:Vorto)
    WHERE v.radiko IS NOT NULL AND v.radiko <> ''
      AND v.vortspeco IN ['substantivo', 'verbo', 'adjektivo', 'adverbo']
    RETURN v.radiko AS radiko

    UNION ALL

    MATCH (frazo)-[:HAVAS_SUBJEKTON_VORTO|HAVAS_OBJEKTON_VORTO|HAVAS_ALIAJN]->(v:Vorto)
    WHERE v.radiko IS NOT NULL AND v.radiko <> ''
      AND v.vortspeco IN ['substantivo', 'verbo', 'adjektivo', 'adverbo']
    RETURN v.radiko AS radiko

    UNION ALL

    MATCH (frazo)-[:HAVAS_SUBJEKTON_VORTGRUPO|HAVAS_OBJEKTON_VORTGRUPO]->(vg:Vortgrupo)
    MATCH (vg)-[:HAVAS_KERNON|HAVAS_PRISKRIBON]->(v:Vorto)
    WHERE v.radiko IS NOT NULL AND v.radiko <> ''
      AND v.vortspeco IN ['substantivo', 'verbo', 'adjektivo', 'adverbo']
    RETURN v.radiko AS radiko
    """

    result = conn.execute(query)

    root_counts = Counter()

    while result.has_next():
        row = result.get_next()
        radiko = row[0]
        if radiko:
            root_counts[radiko] += 1

    logger.info(f"Found {len(root_counts)} unique content roots")

    return root_counts


def generate_training_pairs_optimized(
    db: kuzu.Database,
    paragraphs: List[Dict],
    subsample_threshold: float = 1e-3,
    cross_sentence_weight: float = 0.5,
    batch_size: int = 1000
) -> List[Tuple[str, str, float]]:
    """
    Generate AST-aware semantic training pairs with BATCH PROCESSING.

    OPTIMIZATION: Process sentences in batches of 1000 instead of one-by-one.
    Expected speedup: 6-12x faster (5-10 minutes vs 1 hour).

    Args:
        db: Kuzu database
        paragraphs: List of paragraphs with sentence IDs
        subsample_threshold: Mikolov subsampling threshold
        cross_sentence_weight: Weight for cross-sentence pairs
        batch_size: Number of sentences to process per batch (default: 1000)

    Returns:
        List of (target, context, weight) tuples
    """
    logger.info(f"Generating AST-aware semantic training pairs (OPTIMIZED, batch_size={batch_size})...")

    # Compute root frequencies for subsampling
    root_counts = compute_root_frequencies(db)
    total_roots = sum(root_counts.values())

    # Compute subsampling probabilities
    subsample_probs = {}
    if subsample_threshold > 0:
        for root, count in root_counts.items():
            subsample_probs[root] = compute_subsampling_probability(
                count, total_roots, subsample_threshold
            )

        subsampled_roots = sum(1 for p in subsample_probs.values() if p < 1.0)
        logger.info(f"Subsampling enabled: {subsampled_roots}/{len(subsample_probs)} roots will be subsampled")

        heavily_subsampled = [(root, prob) for root, prob in subsample_probs.items() if prob < 0.5]
        if heavily_subsampled:
            heavily_subsampled.sort(key=lambda x: x[1])
            logger.info(f"Most heavily subsampled roots (keep prob < 0.5): {heavily_subsampled[:5]}")
    else:
        logger.info("Subsampling disabled")
        subsample_probs = {root: 1.0 for root in root_counts.keys()}

    pairs = []
    total_sentences = sum(len(p['sentences']) for p in paragraphs)
    processed = 0

    # Process paragraphs in batches
    paragraph_count = 0
    for paragraph in paragraphs:
        sentences = paragraph['sentences']
        sentence_ids = [s['id'] for s in sentences]

        # Process sentences in batches
        for batch_start in range(0, len(sentence_ids), batch_size):
            batch_end = min(batch_start + batch_size, len(sentence_ids))
            batch_ids = sentence_ids[batch_start:batch_end]

            # Log progress every 100 batches (100K sentences)
            if paragraph_count % 100 == 0 and batch_start == 0:
                logger.info(f"Processing paragraph {paragraph_count}/{len(paragraphs)}, {processed} sentences processed so far")

            # Extract within-sentence pairs (batched!)
            pairs_by_sentence = extract_all_semantic_pairs_batch(
                db, batch_ids, subsample_probs
            )

            for sentence_id in batch_ids:
                pairs.extend(pairs_by_sentence.get(sentence_id, []))

            # Extract semantic heads for cross-sentence pairing (batched!)
            heads_by_sentence = extract_semantic_heads_batch(db, batch_ids)

            # Generate cross-sentence pairs (within paragraph only)
            for idx in range(batch_start, batch_end):
                sentence = sentences[idx]
                sentence_id = sentence['id']
                current_heads = heads_by_sentence.get(sentence_id, [])

                if not current_heads:
                    continue

                # Previous sentence (if exists in same paragraph)
                if idx > 0:
                    prev_sentence = sentences[idx - 1]
                    prev_heads = heads_by_sentence.get(prev_sentence['id'], [])

                    for curr_head in current_heads:
                        if random.random() > subsample_probs.get(curr_head, 1.0):
                            continue
                        for prev_head in prev_heads:
                            if random.random() > subsample_probs.get(prev_head, 1.0):
                                continue
                            pairs.append((curr_head, prev_head, cross_sentence_weight))

                # Next sentence (if exists in same paragraph)
                if idx < len(sentences) - 1:
                    next_sentence = sentences[idx + 1]
                    next_heads = heads_by_sentence.get(next_sentence['id'], [])

                    for curr_head in current_heads:
                        if random.random() > subsample_probs.get(curr_head, 1.0):
                            continue
                        for next_head in next_heads:
                            if random.random() > subsample_probs.get(next_head, 1.0):
                                continue
                            pairs.append((curr_head, next_head, cross_sentence_weight))

            processed += len(batch_ids)
            if processed % 10000 == 0:
                logger.info(f"Processed {processed}/{total_sentences} sentences, {len(pairs)} pairs so far")

        paragraph_count += 1

    logger.info(f"Generated {len(pairs)} total training pairs")

    # Log pair type distribution
    weight_distribution = Counter(weight for _, _, weight in pairs)
    logger.info(f"Pair weight distribution: {dict(weight_distribution)}")

    return pairs


def filter_by_vocabulary(
    pairs: List[Tuple[str, str, float]],
    target_vocabulary: Path
) -> Tuple[List[Tuple[str, str, float]], List[str]]:
    """Filter training pairs by target vocabulary."""
    logger.info(f"Filtering pairs by target vocabulary: {target_vocabulary}")

    with open(target_vocabulary, 'r') as f:
        vocab_data = json.load(f)
        if isinstance(vocab_data, dict):
            vocabulary_set = set(vocab_data.keys())
        else:
            vocabulary_set = set(vocab_data)

    logger.info(f"Target vocabulary size: {len(vocabulary_set)} roots")

    filtered_pairs = [
        (target, context, weight)
        for target, context, weight in pairs
        if target in vocabulary_set and context in vocabulary_set
    ]

    logger.info(f"Filtered pairs: {len(pairs)} → {len(filtered_pairs)} ({len(filtered_pairs)/len(pairs)*100:.1f}% kept)")

    vocabulary = sorted(vocabulary_set)

    return filtered_pairs, vocabulary


def save_training_data(
    pairs: List[Tuple[str, str, float]],
    vocabulary: List[str],
    output_path: Path,
    vocab_output_path: Path
) -> None:
    """Save training pairs and vocabulary to disk."""
    logger.info(f"Saving training pairs to {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for target, context, weight in pairs:
            json.dump({
                'target': target,
                'context': context,
                'weight': weight
            }, f)
            f.write('\n')

    logger.info(f"Saving vocabulary to {vocab_output_path}")
    vocab_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(vocab_output_path, 'w') as f:
        json.dump(vocabulary, f, ensure_ascii=False, indent=2)

    # Save statistics
    stats_path = output_path.parent / (output_path.stem + '_stats.json')
    logger.info(f"Saving statistics to {stats_path}")

    root_pair_counts = Counter()
    weight_distribution = Counter()
    for target, context, weight in pairs:
        root_pair_counts[target] += 1
        root_pair_counts[context] += 1
        weight_distribution[weight] += 1

    avg_pairs_per_root = sum(root_pair_counts.values()) / len(root_pair_counts) if root_pair_counts else 0

    stats = {
        'total_pairs': len(pairs),
        'vocabulary_size': len(vocabulary),
        'unique_roots_in_pairs': len(root_pair_counts),
        'avg_pairs_per_root': avg_pairs_per_root,
        'weight_distribution': {str(k): v for k, v in sorted(weight_distribution.items())},
        'most_frequent_roots': root_pair_counts.most_common(20)
    }

    with open(stats_path, 'w') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved {len(pairs)} pairs with {len(vocabulary)} vocabulary")
    logger.info(f"Average pairs per root: {avg_pairs_per_root:.1f}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract AST-aware semantic training pairs (OPTIMIZED with batching)"
    )
    parser.add_argument('--db-path', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--vocab-output', type=Path)
    parser.add_argument('--target-vocabulary', type=Path)
    parser.add_argument('--subsample-threshold', type=float, default=1e-3)
    parser.add_argument('--cross-sentence-weight', type=float, default=0.5)
    parser.add_argument('--batch-size', type=int, default=1000,
                       help='Number of sentences to process per batch (default: 1000)')

    args = parser.parse_args()

    if args.vocab_output is None:
        args.vocab_output = args.output.parent / (args.output.stem + '_vocab.json')

    logger.info(f"Opening database: {args.db_path}")
    db = kuzu.Database(str(args.db_path))

    paragraphs = extract_paragraph_sentences(db)

    pairs = generate_training_pairs_optimized(
        db, paragraphs,
        subsample_threshold=args.subsample_threshold,
        cross_sentence_weight=args.cross_sentence_weight,
        batch_size=args.batch_size
    )

    if args.target_vocabulary:
        pairs, vocabulary = filter_by_vocabulary(pairs, args.target_vocabulary)
    else:
        vocabulary = sorted(set(root for pair in pairs for root in pair[:2]))
        logger.info(f"Created vocabulary with {len(vocabulary)} roots from pairs")

    save_training_data(pairs, vocabulary, args.output, args.vocab_output)

    logger.info("Extraction complete!")


if __name__ == '__main__':
    main()
