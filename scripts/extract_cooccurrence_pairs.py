#!/usr/bin/env python3
"""
Extract co-occurrence pairs from GOLD sources.

Adds distributional semantics to AST-aware training by capturing
which roots tend to appear together in the same sentence.

Usage:
    python scripts/extract_cooccurrence_pairs.py \
        --db-path data/indexes/v2.1_kuzu_index_full \
        --vocabulary data/vocabularies/fundamento_revo_tier12.json \
        --output data/training/fundamento_cooccurrence/pairs.jsonl \
        --window-size 10
"""

import argparse
import json
import kuzu
from pathlib import Path
from typing import List, Tuple, Set, Dict
from collections import Counter, defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_vocabulary(vocab_path: Path) -> Set[str]:
    """Load target vocabulary (Tier 1-2 roots)."""
    with open(vocab_path) as f:
        vocab_data = json.load(f)

    # Vocab format: {root: {tier, source, frequency}, ...}
    roots = set(vocab_data.keys())

    logger.info(f"Loaded vocabulary: {len(roots):,} roots")
    return roots


def extract_sentence_roots(
    db: kuzu.Database,
    gold_sources: List[int],
    vocabulary: Set[str]
) -> List[List[str]]:
    """
    Extract all sentences from GOLD sources as lists of roots.

    Uses simplified approach: query vocabulary in batches to avoid buffer limits.

    Returns:
        List of sentences, where each sentence is a list of roots
    """
    logger.info("Extracting sentences from GOLD sources...")

    conn = kuzu.Connection(db)

    # Get all sentences with their AST-parsed words
    # Process in batches by gold source to avoid buffer overflow
    all_sentence_roots = defaultdict(list)

    for source_id in gold_sources:
        logger.info(f"  Processing source {source_id}...")

        # Get subject heads (through Vortgrupo)
        query_subj = """
        MATCH (f:Frazoteksto)-[:EN_PARAGRAFO]->(p:Paragrafo)-[:EN_SEKCIO]->(s:Sekcio)-[:EN_DOKUMENTO]->(d:Dokumento)-[:EN_FONTARO]->(fo:Fontaro)
        WHERE fo.id = $source_id
        MATCH (f)-[:FRAZOTEKSTO_HAVAS_AST]->(ast:AST)-[:AST_HAVAS_FRAZON]->(frazo:Frazo)
        MATCH (frazo)-[:HAVAS_SUBJEKTON_VORTGRUPO]->(vg:Vortgrupo)
        MATCH (vg)-[:HAVAS_KERNON]->(v:Vorto)
        WHERE v.radiko IS NOT NULL AND v.radiko <> ''
        RETURN f.id AS frazo_id, v.radiko AS root
        """

        result = conn.execute(query_subj, {'source_id': source_id})

        while result.has_next():
            row = result.get_next()
            frazo_id = row[0]
            root = row[1].lower().strip()

            if root in vocabulary:
                all_sentence_roots[frazo_id].append(root)

        # Get verbs (direct Vorto)
        query_verbs = """
        MATCH (f:Frazoteksto)-[:EN_PARAGRAFO]->(p:Paragrafo)-[:EN_SEKCIO]->(s:Sekcio)-[:EN_DOKUMENTO]->(d:Dokumento)-[:EN_FONTARO]->(fo:Fontaro)
        WHERE fo.id = $source_id
        MATCH (f)-[:FRAZOTEKSTO_HAVAS_AST]->(ast:AST)-[:AST_HAVAS_FRAZON]->(frazo:Frazo)
        MATCH (frazo)-[:HAVAS_VERBON]->(v:Vorto)
        WHERE v.radiko IS NOT NULL AND v.radiko <> ''
        RETURN f.id AS frazo_id, v.radiko AS root
        """

        result = conn.execute(query_verbs, {'source_id': source_id})

        while result.has_next():
            row = result.get_next()
            frazo_id = row[0]
            root = row[1].lower().strip()

            if root in vocabulary:
                all_sentence_roots[frazo_id].append(root)

        # Get object heads (through Vortgrupo)
        query_obj = """
        MATCH (f:Frazoteksto)-[:EN_PARAGRAFO]->(p:Paragrafo)-[:EN_SEKCIO]->(s:Sekcio)-[:EN_DOKUMENTO]->(d:Dokumento)-[:EN_FONTARO]->(fo:Fontaro)
        WHERE fo.id = $source_id
        MATCH (f)-[:FRAZOTEKSTO_HAVAS_AST]->(ast:AST)-[:AST_HAVAS_FRAZON]->(frazo:Frazo)
        MATCH (frazo)-[:HAVAS_OBJEKTON_VORTGRUPO]->(vg:Vortgrupo)
        MATCH (vg)-[:HAVAS_KERNON]->(v:Vorto)
        WHERE v.radiko IS NOT NULL AND v.radiko <> ''
        RETURN f.id AS frazo_id, v.radiko AS root
        """

        result = conn.execute(query_obj, {'source_id': source_id})

        while result.has_next():
            row = result.get_next()
            frazo_id = row[0]
            root = row[1].lower().strip()

            if root in vocabulary:
                all_sentence_roots[frazo_id].append(root)

        # Get modifiers (adjectives, adverbs in Vortgrupo)
        query_mods = """
        MATCH (f:Frazoteksto)-[:EN_PARAGRAFO]->(p:Paragrafo)-[:EN_SEKCIO]->(s:Sekcio)-[:EN_DOKUMENTO]->(d:Dokumento)-[:EN_FONTARO]->(fo:Fontaro)
        WHERE fo.id = $source_id
        MATCH (f)-[:FRAZOTEKSTO_HAVAS_AST]->(ast:AST)-[:AST_HAVAS_FRAZON]->(frazo:Frazo)
        MATCH (frazo)-[:HAVAS_SUBJEKTON_VORTGRUPO|HAVAS_OBJEKTON_VORTGRUPO]->(vg:Vortgrupo)
        MATCH (vg)-[:HAVAS_PRISKRIBON]->(v:Vorto)
        WHERE v.radiko IS NOT NULL AND v.radiko <> ''
        RETURN f.id AS frazo_id, v.radiko AS root
        """

        result = conn.execute(query_mods, {'source_id': source_id})

        while result.has_next():
            row = result.get_next()
            frazo_id = row[0]
            root = row[1].lower().strip()

            if root in vocabulary:
                all_sentence_roots[frazo_id].append(root)

    logger.info(f"Found {len(all_sentence_roots):,} sentences")

    # Convert to list of sentences (filter for 2+ roots)
    sentences = [
        roots for roots in all_sentence_roots.values()
        if len(roots) >= 2
    ]

    logger.info(f"Extracted {len(sentences):,} sentences with 2+ roots")
    return sentences


def compute_cooccurrence_pairs(
    sentences: List[List[str]],
    window_size: int = 10
) -> List[Tuple[str, str, float]]:
    """
    Compute co-occurrence pairs from sentences using a sliding window.

    Args:
        sentences: List of sentences (each is list of roots)
        window_size: Maximum distance between roots to consider co-occurrence

    Returns:
        List of (target, context, weight) tuples
    """
    logger.info(f"Computing co-occurrence with window size {window_size}...")

    # Count co-occurrences
    cooccurrence_counts = defaultdict(int)

    for sentence in sentences:
        n = len(sentence)

        for i, target in enumerate(sentence):
            # Look at context within window
            start = max(0, i - window_size)
            end = min(n, i + window_size + 1)

            for j in range(start, end):
                if i == j:
                    continue

                context = sentence[j]

                # Create ordered pair (alphabetically sorted for consistency)
                pair = tuple(sorted([target, context]))
                cooccurrence_counts[pair] += 1

    logger.info(f"Found {len(cooccurrence_counts):,} unique co-occurrence pairs")

    # Convert to weighted pairs
    # Weight based on frequency (log scale to avoid dominating rare AST pairs)
    import math

    pairs = []
    for (root1, root2), count in cooccurrence_counts.items():
        # Weight: log(count + 1) / 10 to keep in range [0, 1]
        # Typical counts: 1-100, so log gives ~0-4.6, divide by 10 gives 0-0.46
        weight = min(1.0, math.log(count + 1) / 10)

        # Add both directions for symmetry
        pairs.append((root1, root2, weight))
        pairs.append((root2, root1, weight))

    logger.info(f"Generated {len(pairs):,} co-occurrence pairs")

    return pairs


def main():
    parser = argparse.ArgumentParser(description="Extract co-occurrence pairs from GOLD sources")
    parser.add_argument('--db-path', type=Path, required=True,
                        help='Path to Kuzu database')
    parser.add_argument('--vocabulary', type=Path, required=True,
                        help='Path to vocabulary JSON (Tier 1-2)')
    parser.add_argument('--output', type=Path, required=True,
                        help='Output path for pairs.jsonl')
    parser.add_argument('--window-size', type=int, default=10,
                        help='Co-occurrence window size (default: 10)')

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("Co-occurrence Pair Extraction")
    logger.info("=" * 80)
    logger.info(f"Database: {args.db_path}")
    logger.info(f"Vocabulary: {args.vocabulary}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Window size: {args.window_size}")

    # Load vocabulary
    vocabulary = load_vocabulary(args.vocabulary)

    # GOLD sources (same as AST extraction)
    gold_sources = [1, 2, 3, 4, 5, 6]

    # Connect to database
    logger.info(f"\nConnecting to database: {args.db_path}")
    db = kuzu.Database(str(args.db_path))

    # Extract sentences
    sentences = extract_sentence_roots(db, gold_sources, vocabulary)

    # Compute co-occurrence pairs
    pairs = compute_cooccurrence_pairs(sentences, args.window_size)

    # Get unique roots
    unique_roots = set()
    for target, context, _ in pairs:
        unique_roots.add(target)
        unique_roots.add(context)

    # Save pairs
    args.output.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"\nSaving pairs to {args.output}...")
    with open(args.output, 'w') as f:
        for target, context, weight in pairs:
            f.write(json.dumps({
                'target': target,
                'context': context,
                'weight': weight
            }) + '\n')

    # Save vocabulary
    vocab_output = args.output.parent / 'vocab.json'
    logger.info(f"Saving vocabulary to {vocab_output}...")
    with open(vocab_output, 'w') as f:
        json.dump(sorted(unique_roots), f, indent=2)

    # Save statistics
    stats_output = args.output.parent / 'stats.json'
    logger.info(f"Saving statistics to {stats_output}...")

    stats = {
        'total_pairs': len(pairs),
        'unique_roots': len(unique_roots),
        'vocabulary_size': len(vocabulary),
        'coverage': len(unique_roots) / len(vocabulary),
        'sentences_processed': len(sentences),
        'window_size': args.window_size,
        'gold_sources': gold_sources
    }

    with open(stats_output, 'w') as f:
        json.dump(stats, f, indent=2)

    logger.info("\n" + "=" * 80)
    logger.info("Co-occurrence Extraction Complete")
    logger.info("=" * 80)
    logger.info(f"Total pairs: {len(pairs):,}")
    logger.info(f"Unique roots: {len(unique_roots):,}")
    logger.info(f"Coverage: {stats['coverage']:.1%}")
    logger.info(f"Output: {args.output}")


if __name__ == '__main__':
    main()
