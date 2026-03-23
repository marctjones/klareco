#!/usr/bin/env python3
"""
Extract AST-Aware Training Pairs from GOLD Sources (Fundamento-Focused)

VERSION: v3.0-fundamento
COMPATIBLE WITH: v2.1 database schema (AST-native)
DEPENDENCIES: Kuzu v2.1, Tier 1-2 vocabulary
STAGE: Data

Description:
    Extracts AST-aware training pairs from GOLD sources only (Tier 0).
    Uses Fundamento + ReVo Tier 1-2 vocabulary (~9,800 roots).

    Key improvements:
    - AST-aware semantic pairing (not bag-of-words)
    - GOLD sources only (authoritative)
    - Optimized queries (split OPTIONAL MATCH to avoid bottleneck)
    - Fundamento-focused vocabulary

Pipeline Position:
    v2.1 DB (GOLD sources) → [THIS SCRIPT] → fundamento_ast_pairs.jsonl → train

Usage:
    python scripts/extract_fundamento_ast_pairs.py \
        --db-path data/indexes/v2.1_kuzu_index_full \
        --vocabulary data/vocabularies/fundamento_revo_tier12.json \
        --output data/training/fundamento_ast_pairs/pairs.jsonl

Outputs:
    - pairs.jsonl: {"target": "hund", "context": "bel", "weight": 1.0}
    - vocab.json: List of roots used
    - stats.json: Extraction statistics

Last Updated: 2026-03-22
Author: Claude + Marc
See Also: docs/ROOT_EMBEDDINGS_DESIGN.md
"""

import argparse
import json
import logging
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Set
import random
import kuzu

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# GOLD sources (Tier 0: authoritative)
GOLD_FONTARO_IDS = [1, 2, 3, 4, 5, 6]  # lingvaj_respondoj, pmeg, alice, andersen, ekzercaro, krestomatio


def extract_modifier_head_pairs(
    db: kuzu.Database,
    gold_sources: List[int],
    vocabulary: Set[str]
) -> List[Tuple[str, str, float]]:
    """
    Extract modifier-head pairs from AST structure.

    Examples:
    - bela hundo → (bel, hund, 1.0)
    - rapide kuri → (rapid, kur, 1.0)

    OPTIMIZATION: Uses direct MATCH (not OPTIONAL) for speed.
    """
    logger.info("Extracting modifier-head pairs (AST-aware)...")

    conn = kuzu.Connection(db)
    pairs = []

    # Query 1: Adjective → Noun (in vortgrupo)
    query_adj_noun = """
    MATCH (f:Frazoteksto)-[:EN_PARAGRAFO]->(p:Paragrafo)-[:EN_SEKCIO]->(s:Sekcio)-[:EN_DOKUMENTO]->(d:Dokumento)-[:EN_FONTARO]->(fo:Fontaro)
    WHERE fo.id IN $gold_sources
    MATCH (f)-[:FRAZOTEKSTO_HAVAS_AST]->(ast:AST)-[:AST_HAVAS_FRAZON]->(frazo:Frazo)
    MATCH (frazo)-[:HAVAS_SUBJEKTON_VORTGRUPO|HAVAS_OBJEKTON_VORTGRUPO]->(vg:Vortgrupo)
    MATCH (vg)-[:HAVAS_KERNON]->(head:Vorto)
    MATCH (vg)-[:HAVAS_PRISKRIBON]->(modifier:Vorto)
    WHERE head.radiko IS NOT NULL AND head.radiko <> ''
      AND modifier.radiko IS NOT NULL AND modifier.radiko <> ''
      AND head.vortspeco = 'substantivo'
      AND modifier.vortspeco = 'adjektivo'
    RETURN modifier.radiko AS modifier, head.radiko AS head
    """

    result = conn.execute(query_adj_noun, parameters={'gold_sources': gold_sources})

    count = 0
    while result.has_next():
        row = result.get_next()
        modifier = row[0]
        head = row[1]

        # Filter to vocabulary
        if modifier in vocabulary and head in vocabulary:
            # Bidirectional pairs (both directions informative)
            pairs.append((modifier, head, 1.0))
            pairs.append((head, modifier, 1.0))
            count += 1

    logger.info(f"  Adjective→Noun: {count:,} pairs")

    # Query 2: Adverb → Verb
    query_adv_verb = """
    MATCH (f:Frazoteksto)-[:EN_PARAGRAFO]->(p:Paragrafo)-[:EN_SEKCIO]->(s:Sekcio)-[:EN_DOKUMENTO]->(d:Dokumento)-[:EN_FONTARO]->(fo:Fontaro)
    WHERE fo.id IN $gold_sources
    MATCH (f)-[:FRAZOTEKSTO_HAVAS_AST]->(ast:AST)-[:AST_HAVAS_FRAZON]->(frazo:Frazo)
    MATCH (frazo)-[:HAVAS_VERBON]->(verb:Vorto)
    MATCH (frazo)-[:HAVAS_ALIAJN]->(modifier:Vorto)
    WHERE verb.radiko IS NOT NULL AND verb.radiko <> ''
      AND modifier.radiko IS NOT NULL AND modifier.radiko <> ''
      AND verb.vortspeco = 'verbo'
      AND modifier.vortspeco = 'adverbo'
    RETURN modifier.radiko AS modifier, verb.radiko AS verb
    """

    result = conn.execute(query_adv_verb, parameters={'gold_sources': gold_sources})

    count = 0
    while result.has_next():
        row = result.get_next()
        modifier = row[0]
        verb = row[1]

        if modifier in vocabulary and verb in vocabulary:
            pairs.append((modifier, verb, 1.0))
            pairs.append((verb, modifier, 1.0))
            count += 1

    logger.info(f"  Adverb→Verb: {count:,} pairs")

    return pairs


def extract_semantic_argument_pairs(
    db: kuzu.Database,
    gold_sources: List[int],
    vocabulary: Set[str]
) -> List[Tuple[str, str, float]]:
    """
    Extract subject-object semantic pairs from AST.

    Examples:
    - La hundo manĝas katon → (hund, kat, 0.8)

    OPTIMIZATION: Split into two queries to avoid slow OPTIONAL MATCH cascade.
    """
    logger.info("Extracting semantic argument pairs (subject-object)...")

    conn = kuzu.Connection(db)
    pairs = []

    # Query 1: Direct VORTO subjects and objects
    query_direct = """
    MATCH (f:Frazoteksto)-[:EN_PARAGRAFO]->(p:Paragrafo)-[:EN_SEKCIO]->(s:Sekcio)-[:EN_DOKUMENTO]->(d:Dokumento)-[:EN_FONTARO]->(fo:Fontaro)
    WHERE fo.id IN $gold_sources
    MATCH (f)-[:FRAZOTEKSTO_HAVAS_AST]->(ast:AST)-[:AST_HAVAS_FRAZON]->(frazo:Frazo)
    MATCH (frazo)-[:HAVAS_SUBJEKTON_VORTO]->(subj:Vorto)
    MATCH (frazo)-[:HAVAS_OBJEKTON_VORTO]->(obj:Vorto)
    WHERE subj.radiko IS NOT NULL AND subj.radiko <> ''
      AND obj.radiko IS NOT NULL AND obj.radiko <> ''
      AND subj.vortspeco IN ['substantivo', 'adjektivo']
      AND obj.vortspeco IN ['substantivo', 'adjektivo']
    RETURN subj.radiko AS subj, obj.radiko AS obj
    """

    result = conn.execute(query_direct, parameters={'gold_sources': gold_sources})

    count = 0
    while result.has_next():
        row = result.get_next()
        subj = row[0]
        obj = row[1]

        if subj in vocabulary and obj in vocabulary and subj != obj:
            # Semantic argument pairs (lower weight than modifiers)
            pairs.append((subj, obj, 0.8))
            pairs.append((obj, subj, 0.8))
            count += 1

    logger.info(f"  Direct VORTO: {count:,} pairs")

    # Query 2: VORTGRUPO subjects and objects (extract heads)
    query_vortgrupo = """
    MATCH (f:Frazoteksto)-[:EN_PARAGRAFO]->(p:Paragrafo)-[:EN_SEKCIO]->(s:Sekcio)-[:EN_DOKUMENTO]->(d:Dokumento)-[:EN_FONTARO]->(fo:Fontaro)
    WHERE fo.id IN $gold_sources
    MATCH (f)-[:FRAZOTEKSTO_HAVAS_AST]->(ast:AST)-[:AST_HAVAS_FRAZON]->(frazo:Frazo)
    MATCH (frazo)-[:HAVAS_SUBJEKTON_VORTGRUPO]->(subj_vg:Vortgrupo)
    MATCH (subj_vg)-[:HAVAS_KERNON]->(subj_head:Vorto)
    MATCH (frazo)-[:HAVAS_OBJEKTON_VORTGRUPO]->(obj_vg:Vortgrupo)
    MATCH (obj_vg)-[:HAVAS_KERNON]->(obj_head:Vorto)
    WHERE subj_head.radiko IS NOT NULL AND subj_head.radiko <> ''
      AND obj_head.radiko IS NOT NULL AND obj_head.radiko <> ''
      AND subj_head.vortspeco IN ['substantivo', 'adjektivo']
      AND obj_head.vortspeco IN ['substantivo', 'adjektivo']
    RETURN subj_head.radiko AS subj, obj_head.radiko AS obj
    """

    result = conn.execute(query_vortgrupo, parameters={'gold_sources': gold_sources})

    count = 0
    while result.has_next():
        row = result.get_next()
        subj = row[0]
        obj = row[1]

        if subj in vocabulary and obj in vocabulary and subj != obj:
            pairs.append((subj, obj, 0.8))
            pairs.append((obj, subj, 0.8))
            count += 1

    logger.info(f"  VORTGRUPO heads: {count:,} pairs")

    return pairs


def generate_antonym_pairs(vocabulary: Set[str]) -> List[Tuple[str, str, float]]:
    """
    Generate systematic antonym pairs from mal- prefix.

    Examples:
    - bon ↔ malbon → (bon, malbon, -0.7)
    - long ↔ mallong → (long, mallong, -0.7)
    """
    logger.info("Generating systematic antonym pairs (mal- prefix)...")

    pairs = []

    for root in vocabulary:
        if not root.startswith('mal'):
            continue

        # Extract positive root
        positive_root = root[3:]  # Remove 'mal-'

        # Skip if too short or positive not in vocabulary
        if len(positive_root) < 2:
            continue
        if positive_root not in vocabulary:
            continue

        # Add antonym pair with NEGATIVE similarity
        pairs.append((root, positive_root, -0.7))
        pairs.append((positive_root, root, -0.7))

    logger.info(f"  Generated {len(pairs)//2:,} antonym pairs")

    return pairs


def main():
    parser = argparse.ArgumentParser(
        description='Extract AST-aware training pairs from GOLD sources (Fundamento-focused)'
    )
    parser.add_argument('--db-path', type=Path, required=True,
                        help='Path to Kuzu database')
    parser.add_argument('--vocabulary', type=Path, required=True,
                        help='Tier 1-2 vocabulary JSON')
    parser.add_argument('--output', type=Path, required=True,
                        help='Output JSONL file')
    parser.add_argument('--add-antonyms', action='store_true', default=True,
                        help='Add systematic mal- antonym pairs')

    args = parser.parse_args()

    # Load vocabulary
    logger.info(f"Loading vocabulary from {args.vocabulary}...")
    with open(args.vocabulary) as f:
        vocab_data = json.load(f)

    if isinstance(vocab_data, dict):
        vocabulary = set(vocab_data.keys())
    else:
        vocabulary = set(vocab_data)

    logger.info(f"Loaded {len(vocabulary):,} roots (Tier 1-2: Fundamento + ReVo)")

    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Open database
    logger.info(f"Opening Kuzu database: {args.db_path}...")
    db = kuzu.Database(str(args.db_path))

    # Extract pairs
    all_pairs = []

    # 1. Modifier-head pairs (AST-aware)
    modifier_pairs = extract_modifier_head_pairs(db, GOLD_FONTARO_IDS, vocabulary)
    all_pairs.extend(modifier_pairs)

    # 2. Semantic argument pairs (subject-object)
    semantic_pairs = extract_semantic_argument_pairs(db, GOLD_FONTARO_IDS, vocabulary)
    all_pairs.extend(semantic_pairs)

    # 3. Antonym pairs (systematic mal- negation)
    if args.add_antonyms:
        antonym_pairs = generate_antonym_pairs(vocabulary)
        all_pairs.extend(antonym_pairs)

    # Deduplicate pairs (keep highest weight)
    logger.info("Deduplicating pairs...")
    pair_weights = {}
    for target, context, weight in all_pairs:
        key = (target, context)
        if key not in pair_weights or abs(weight) > abs(pair_weights[key]):
            pair_weights[key] = weight

    final_pairs = [(t, c, w) for (t, c), w in pair_weights.items()]

    logger.info(f"Total unique pairs: {len(final_pairs):,}")

    # Write pairs
    logger.info(f"Writing pairs to {args.output}...")
    with open(args.output, 'w') as f:
        for target, context, weight in final_pairs:
            f.write(json.dumps({
                'target': target,
                'context': context,
                'weight': weight
            }, ensure_ascii=False) + '\n')

    # Write vocabulary
    vocab_output = args.output.parent / 'vocab.json'
    used_roots = set()
    for target, context, _ in final_pairs:
        used_roots.add(target)
        used_roots.add(context)

    with open(vocab_output, 'w') as f:
        json.dump(sorted(used_roots), f, ensure_ascii=False, indent=2)

    # Write statistics
    stats_output = args.output.parent / 'stats.json'

    weight_counts = Counter(w for _, _, w in final_pairs)

    stats = {
        'total_pairs': len(final_pairs),
        'unique_roots': len(used_roots),
        'vocabulary_size': len(vocabulary),
        'coverage': len(used_roots) / len(vocabulary),
        'pair_type_counts': {
            'modifier_head': weight_counts.get(1.0, 0),
            'semantic_args': weight_counts.get(0.8, 0),
            'antonyms': weight_counts.get(-0.7, 0)
        },
        'gold_sources': GOLD_FONTARO_IDS
    }

    with open(stats_output, 'w') as f:
        json.dump(stats, f, indent=2)

    logger.info("=" * 80)
    logger.info("Extraction Complete!")
    logger.info("=" * 80)
    logger.info(f"Output: {args.output}")
    logger.info(f"Vocabulary: {vocab_output}")
    logger.info(f"Statistics: {stats_output}")
    logger.info(f"")
    logger.info(f"Total pairs: {len(final_pairs):,}")
    logger.info(f"Unique roots: {len(used_roots):,} / {len(vocabulary):,} ({len(used_roots)/len(vocabulary)*100:.1f}%)")
    logger.info(f"")
    logger.info("Pair breakdown:")
    logger.info(f"  Modifier-head (1.0):  {weight_counts.get(1.0, 0):,}")
    logger.info(f"  Semantic args (0.8):  {weight_counts.get(0.8, 0):,}")
    logger.info(f"  Antonyms (-0.7):      {weight_counts.get(-0.7, 0):,}")


if __name__ == '__main__':
    main()
