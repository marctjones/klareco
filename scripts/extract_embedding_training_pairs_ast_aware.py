#!/usr/bin/env python3
"""
Extract Root Embedding Training Pairs with AST-Aware Semantic Pairing

VERSION: v3.0 (AST-aware)
COMPATIBLE WITH: v2.1 database schema (AST-native)
DEPENDENCIES: Parser (for AST structure)
STAGE: Data

Description:
    Extracts training pairs for skip-gram root embeddings using AST structure
    to identify SEMANTIC relationships (not positional co-occurrence).

    KEY DIFFERENCE FROM v2.1:
    - OLD: Positional 5-word window → learns grammar + semantics mixed
    - NEW: AST-aware semantic pairing → learns PURE semantics (no grammar)

Pipeline Position:
    v2.1 DB → [THIS SCRIPT] → training_pairs.jsonl → train_root_embeddings.py

Usage:
    python scripts/extract_embedding_training_pairs_ast_aware.py \
        --db-path data/indexes/kuzu_v2.1 \
        --output data/training/root_embedding_pairs.jsonl \
        --target-vocabulary data/vocabularies/production_semantic_roots_15k.json \
        --cross-sentence-weight 0.5 \
        --subsample-threshold 1e-3

Inputs:
    - Kuzu v2.1 database with AST-native schema
    - Target vocabulary (clean semantic roots only)

Outputs:
    - training_pairs.jsonl: (target_root, context_root, weight) tuples
    - Format: {"target": "hund", "context": "kat", "weight": 1.0}
    - Weights: 1.0 (modifier-head), 0.9 (compound), 0.8 (semantic args), 0.5 (discourse)

Quality Checks:
    - Function words excluded (handled by AST deterministically)
    - Only semantic relationships (NOT grammatical like subject-verb)
    - Modifier-head pairs (adjective-noun, adverb-verb)
    - Semantic arguments (subject-object: both participants in event)
    - Cross-sentence discourse (topical coherence)

Last Updated: 2026-03-10
Author: Claude + Marc
Related Issues: Phase 1 Root Embeddings Design Update
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


# Content word categories to INCLUDE in embeddings
CONTENT_WORD_CATEGORIES = {
    'substantivo',   # dog, cat, tree
    'verbo',         # run, eat, sleep
    'adjektivo',     # big, small, red
    'adverbo',       # quickly, slowly, well
}


def extract_semantic_pairs_from_sentence(
    db: kuzu.Database,
    frazoteksto_id: int,
    subsample_probs: Dict[str, float]
) -> List[Tuple[str, str, float]]:
    """
    Extract semantic pairs from a sentence using AST structure.

    Creates pairs based on SEMANTIC relationships, not positional proximity:
    1. Modifier-head relationships (adjektivo-substantivo, adverbo-verbo)
    2. Semantic arguments (subjekto-objekto: both participants in event)
    3. Compound root components (fiŝhundo = fish + dog)

    Args:
        db: Kuzu database connection
        frazoteksto_id: Sentence ID
        subsample_probs: Subsampling probabilities for frequent roots

    Returns:
        List of (target, context, weight) tuples with semantic relationships
    """
    conn = kuzu.Connection(db)
    pairs = []

    # ============================================================================
    # 1. MODIFIER-HEAD RELATIONSHIPS (adjective-noun, adverb-verb)
    #    These indicate semantic similarity - modifiers describe properties
    # ============================================================================

    # Query: Get adjective-noun pairs from vortgrupo structures
    query_adj_noun = """
    MATCH (f:Frazoteksto)-[:FRAZOTEKSTO_HAVAS_AST]->(ast:AST)-[:AST_HAVAS_FRAZON]->(frazo:Frazo)
    WHERE f.id = $frazoteksto_id

    // Subject phrase modifiers
    MATCH (frazo)-[:HAVAS_SUBJEKTON_VORTGRUPO]->(vg:Vortgrupo)
    MATCH (vg)-[:HAVAS_KERNON]->(head:Vorto)
    MATCH (vg)-[:HAVAS_PRISKRIBON]->(modifier:Vorto)
    WHERE head.radiko IS NOT NULL AND head.radiko <> ''
      AND modifier.radiko IS NOT NULL AND modifier.radiko <> ''
      AND head.vortspeco = 'substantivo'
      AND modifier.vortspeco = 'adjektivo'
    RETURN modifier.radiko AS modifier_root, head.radiko AS head_root

    UNION

    // Object phrase modifiers
    MATCH (frazo)-[:HAVAS_OBJEKTON_VORTGRUPO]->(vg:Vortgrupo)
    MATCH (vg)-[:HAVAS_KERNON]->(head:Vorto)
    MATCH (vg)-[:HAVAS_PRISKRIBON]->(modifier:Vorto)
    WHERE head.radiko IS NOT NULL AND head.radiko <> ''
      AND modifier.radiko IS NOT NULL AND modifier.radiko <> ''
      AND head.vortspeco = 'substantivo'
      AND modifier.vortspeco = 'adjektivo'
    RETURN modifier.radiko AS modifier_root, head.radiko AS head_root
    """

    try:
        result = conn.execute(query_adj_noun, parameters={'frazoteksto_id': frazoteksto_id})

        while result.has_next():
            row = result.get_next()
            modifier_root = row[0]
            head_root = row[1]

            # Apply subsampling
            if random.random() <= subsample_probs.get(modifier_root, 1.0) and \
               random.random() <= subsample_probs.get(head_root, 1.0):
                # Strong semantic link: adjective describes noun property
                pairs.append((modifier_root, head_root, 1.0))
                pairs.append((head_root, modifier_root, 1.0))  # Bidirectional

    except Exception as e:
        logger.warning(f"Failed to extract adjective-noun pairs for sentence {frazoteksto_id}: {e}")

    # Query: Get adverb-verb pairs
    query_adv_verb = """
    MATCH (f:Frazoteksto)-[:FRAZOTEKSTO_HAVAS_AST]->(ast:AST)-[:AST_HAVAS_FRAZON]->(frazo:Frazo)
    WHERE f.id = $frazoteksto_id

    MATCH (frazo)-[:HAVAS_VERBON]->(verb:Vorto)
    MATCH (frazo)-[:HAVAS_ALIAJN]->(modifier:Vorto)
    WHERE verb.radiko IS NOT NULL AND verb.radiko <> ''
      AND modifier.radiko IS NOT NULL AND modifier.radiko <> ''
      AND verb.vortspeco = 'verbo'
      AND modifier.vortspeco = 'adverbo'
    RETURN modifier.radiko AS modifier_root, verb.radiko AS verb_root
    """

    try:
        result = conn.execute(query_adv_verb, parameters={'frazoteksto_id': frazoteksto_id})

        while result.has_next():
            row = result.get_next()
            modifier_root = row[0]
            verb_root = row[1]

            # Apply subsampling
            if random.random() <= subsample_probs.get(modifier_root, 1.0) and \
               random.random() <= subsample_probs.get(verb_root, 1.0):
                # Strong semantic link: adverb describes verb manner
                pairs.append((modifier_root, verb_root, 1.0))
                pairs.append((verb_root, modifier_root, 1.0))  # Bidirectional

    except Exception as e:
        logger.warning(f"Failed to extract adverb-verb pairs for sentence {frazoteksto_id}: {e}")

    # ============================================================================
    # 2. SEMANTIC ARGUMENTS (subject-object pairs: both participants in event)
    #    Subject and object often semantically related (both animals, both people, etc.)
    #    We DON'T pair them with the verb (that's grammar, AST already knows!)
    # ============================================================================

    query_semantic_args = """
    MATCH (f:Frazoteksto)-[:FRAZOTEKSTO_HAVAS_AST]->(ast:AST)-[:AST_HAVAS_FRAZON]->(frazo:Frazo)
    WHERE f.id = $frazoteksto_id

    // Get subject head root
    OPTIONAL MATCH (frazo)-[:HAVAS_SUBJEKTON_VORTO]->(subj:Vorto)
    WHERE subj.radiko IS NOT NULL AND subj.radiko <> ''
      AND subj.vortspeco IN ['substantivo', 'adjektivo']

    OPTIONAL MATCH (frazo)-[:HAVAS_SUBJEKTON_VORTGRUPO]->(subj_vg:Vortgrupo)
    OPTIONAL MATCH (subj_vg)-[:HAVAS_KERNON]->(subj_head:Vorto)
    WHERE subj_head.radiko IS NOT NULL AND subj_head.radiko <> ''
      AND subj_head.vortspeco IN ['substantivo', 'adjektivo']

    // Get object head root
    OPTIONAL MATCH (frazo)-[:HAVAS_OBJEKTON_VORTO]->(obj:Vorto)
    WHERE obj.radiko IS NOT NULL AND obj.radiko <> ''
      AND obj.vortspeco IN ['substantivo', 'adjektivo']

    OPTIONAL MATCH (frazo)-[:HAVAS_OBJEKTON_VORTGRUPO]->(obj_vg:Vortgrupo)
    OPTIONAL MATCH (obj_vg)-[:HAVAS_KERNON]->(obj_head:Vorto)
    WHERE obj_head.radiko IS NOT NULL AND obj_head.radiko <> ''
      AND obj_head.vortspeco IN ['substantivo', 'adjektivo']

    WITH
        COALESCE(subj.radiko, subj_head.radiko) AS subj_root,
        COALESCE(obj.radiko, obj_head.radiko) AS obj_root

    WHERE subj_root IS NOT NULL AND obj_root IS NOT NULL
      AND subj_root <> obj_root

    RETURN subj_root, obj_root
    """

    try:
        result = conn.execute(query_semantic_args, parameters={'frazoteksto_id': frazoteksto_id})

        while result.has_next():
            row = result.get_next()
            subj_root = row[0]
            obj_root = row[1]

            # Apply subsampling
            if random.random() <= subsample_probs.get(subj_root, 1.0) and \
               random.random() <= subsample_probs.get(obj_root, 1.0):
                # Moderate semantic link: both participants in same event
                # (Often same semantic category: both animals, both people, etc.)
                pairs.append((subj_root, obj_root, 0.8))
                pairs.append((obj_root, subj_root, 0.8))  # Bidirectional

    except Exception as e:
        logger.warning(f"Failed to extract semantic arguments for sentence {frazoteksto_id}: {e}")

    # ============================================================================
    # 3. COMPOUND ROOT COMPONENTS (fiŝhundo = fiŝ + hund, both semantically related)
    #    NOT IMPLEMENTED YET - Would need compound word detection in schema
    # ============================================================================
    # TODO: Add compound root extraction if schema supports it

    return pairs


def extract_semantic_heads_from_sentence(
    db: kuzu.Database,
    frazoteksto_id: int
) -> List[str]:
    """
    Extract semantic heads (main entities) from a sentence using AST structure.

    Gets subject and object heads (not verbs, not modifiers) - the main entities
    that the sentence is ABOUT. These are what we pair across sentences for
    discourse continuity.

    Args:
        db: Kuzu database connection
        frazoteksto_id: Sentence ID

    Returns:
        List of semantic head roots (substantivo only)
    """
    conn = kuzu.Connection(db)

    query = """
    MATCH (f:Frazoteksto)-[:FRAZOTEKSTO_HAVAS_AST]->(ast:AST)-[:AST_HAVAS_FRAZON]->(frazo:Frazo)
    WHERE f.id = $frazoteksto_id

    // Get subject head (the main noun being talked about)
    OPTIONAL MATCH (frazo)-[:HAVAS_SUBJEKTON_VORTO]->(subj:Vorto)
    WHERE subj.radiko IS NOT NULL AND subj.radiko <> ''
      AND subj.vortspeco = 'substantivo'

    OPTIONAL MATCH (frazo)-[:HAVAS_SUBJEKTON_VORTGRUPO]->(subj_vg:Vortgrupo)
    OPTIONAL MATCH (subj_vg)-[:HAVAS_KERNON]->(subj_head:Vorto)
    WHERE subj_head.radiko IS NOT NULL AND subj_head.radiko <> ''
      AND subj_head.vortspeco = 'substantivo'

    // Get object head (the main noun being acted upon)
    OPTIONAL MATCH (frazo)-[:HAVAS_OBJEKTON_VORTO]->(obj:Vorto)
    WHERE obj.radiko IS NOT NULL AND obj.radiko <> ''
      AND obj.vortspeco = 'substantivo'

    OPTIONAL MATCH (frazo)-[:HAVAS_OBJEKTON_VORTGRUPO]->(obj_vg:Vortgrupo)
    OPTIONAL MATCH (obj_vg)-[:HAVAS_KERNON]->(obj_head:Vorto)
    WHERE obj_head.radiko IS NOT NULL AND obj_head.radiko <> ''
      AND obj_head.vortspeco = 'substantivo'

    WITH
        COALESCE(subj.radiko, subj_head.radiko) AS subj_root,
        COALESCE(obj.radiko, obj_head.radiko) AS obj_root

    RETURN subj_root, obj_root
    """

    try:
        result = conn.execute(query, parameters={'frazoteksto_id': frazoteksto_id})

        heads = []
        if result.has_next():
            row = result.get_next()
            subj_root = row[0]
            obj_root = row[1]

            if subj_root:
                heads.append(subj_root)
            if obj_root:
                heads.append(obj_root)

        return heads

    except Exception as e:
        logger.warning(f"Failed to extract semantic heads from sentence {frazoteksto_id}: {e}")
        return []


def extract_cross_sentence_pairs(
    db: kuzu.Database,
    current_sentence_id: int,
    prev_sentence_id: Optional[int],
    next_sentence_id: Optional[int],
    subsample_probs: Dict[str, float],
    cross_sentence_weight: float
) -> List[Tuple[str, str, float]]:
    """
    Extract AST-aware cross-sentence discourse pairs (topical coherence).

    Uses AST structure to identify semantic heads (subject/object nouns) from
    both sentences, then pairs them. This captures discourse continuity:
    - Same entity mentioned across sentences
    - Related entities in adjacent sentences
    - Topic continuation

    Weight is lower (0.5) than within-sentence semantic relationships because
    the connection is weaker (discourse vs. grammatical structure).

    NOTE: Only pairs within same paragraph (paragraph boundary = hard stop).

    Args:
        db: Kuzu database connection
        current_sentence_id: Current sentence ID
        prev_sentence_id: Previous sentence ID (if exists)
        next_sentence_id: Next sentence ID (if exists)
        subsample_probs: Subsampling probabilities
        cross_sentence_weight: Weight for cross-sentence pairs (default: 0.5)

    Returns:
        List of (target, context, weight) tuples for discourse relationships

    Example:
        Sentence 1: "Zamenhof kreis Esperanton"
                    AST heads: [Zamenhof, Esperanton]

        Sentence 2: "La lingvo estas facila"
                    AST heads: [lingvo]

        Cross-sentence pairs:
        - (Zamenhof, lingvo) - creator and creation
        - (Esperanton, lingvo) - same entity (proper noun, common noun)
    """
    conn = kuzu.Connection(db)
    pairs = []

    # Get semantic heads (subject/object nouns only) from current sentence
    current_heads = extract_semantic_heads_from_sentence(db, current_sentence_id)

    if not current_heads:
        return pairs

    # Process previous sentence
    if prev_sentence_id is not None:
        prev_heads = extract_semantic_heads_from_sentence(db, prev_sentence_id)

        # Pair semantic heads across sentences
        for curr_head in current_heads:
            if random.random() > subsample_probs.get(curr_head, 1.0):
                continue
            for prev_head in prev_heads:
                if random.random() > subsample_probs.get(prev_head, 1.0):
                    continue
                # Bidirectional pairing for discourse continuity
                pairs.append((curr_head, prev_head, cross_sentence_weight))

    # Process next sentence
    if next_sentence_id is not None:
        next_heads = extract_semantic_heads_from_sentence(db, next_sentence_id)

        # Pair semantic heads across sentences
        for curr_head in current_heads:
            if random.random() > subsample_probs.get(curr_head, 1.0):
                continue
            for next_head in next_heads:
                if random.random() > subsample_probs.get(next_head, 1.0):
                    continue
                # Bidirectional pairing for discourse continuity
                pairs.append((curr_head, next_head, cross_sentence_weight))

    return pairs


def compute_subsampling_probability(count: int, total_count: int, threshold: float) -> float:
    """
    Compute Mikolov et al. 2013 subsampling probability.

    Frequent words (like "est", "hav") are downsampled to prevent them
    from dominating the training signal.

    Args:
        count: Frequency of this root
        total_count: Total root count across corpus
        threshold: Subsampling threshold (1e-3 to 1e-5 typical)

    Returns:
        Probability of keeping this root (0.0 to 1.0)
    """
    if count == 0 or total_count == 0:
        return 1.0

    frequency = count / total_count

    # Mikolov formula: sqrt(t/f) + t/f
    # where t = threshold, f = frequency
    if frequency > threshold:
        prob = (np.sqrt(threshold / frequency) + threshold / frequency)
        return min(1.0, prob)
    else:
        return 1.0


def extract_paragraph_sentences(db: kuzu.Database) -> List[Dict]:
    """
    Extract all sentences grouped by paragraph with sequential ordering.

    Returns:
        List of paragraphs, each containing sentence IDs in order
    """
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
    """
    Compute frequency of each content root across entire corpus.

    Returns:
        Dictionary mapping root → frequency count
    """
    logger.info("Computing root frequencies across corpus...")

    conn = kuzu.Connection(db)

    query = """
    MATCH (f:Frazoteksto)-[:FRAZOTEKSTO_HAVAS_AST]->(ast:AST)-[:AST_HAVAS_FRAZON]->(frazo:Frazo)

    // Get all content roots from all structural positions
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


def generate_training_pairs(
    db: kuzu.Database,
    paragraphs: List[Dict],
    subsample_threshold: float = 1e-3,
    cross_sentence_weight: float = 0.5
) -> List[Tuple[str, str, float]]:
    """
    Generate AST-aware semantic training pairs.

    Uses AST structure to identify semantic relationships:
    1. Modifier-head (adjective-noun, adverb-verb) - weight 1.0
    2. Semantic arguments (subject-object) - weight 0.8
    3. Cross-sentence discourse - weight 0.5

    Args:
        db: Kuzu database
        paragraphs: List of paragraphs with sentence IDs
        subsample_threshold: Mikolov subsampling threshold (1e-3 typical)
        cross_sentence_weight: Weight for cross-sentence pairs (0.5 typical)

    Returns:
        List of (target, context, weight) tuples
    """
    logger.info("Generating AST-aware semantic training pairs...")

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
    total_sentences = sum(len(p['sentences']) for p in paragraphs)
    processed = 0

    # Process each paragraph
    for paragraph in paragraphs:
        sentences = paragraph['sentences']

        # Process each sentence
        for sent_idx, sentence in enumerate(sentences):
            sentence_id = sentence['id']

            # 1. Extract within-sentence semantic pairs (AST-aware)
            sentence_pairs = extract_semantic_pairs_from_sentence(
                db, sentence_id, subsample_probs
            )
            pairs.extend(sentence_pairs)

            # 2. Extract cross-sentence discourse pairs
            prev_id = sentences[sent_idx - 1]['id'] if sent_idx > 0 else None
            next_id = sentences[sent_idx + 1]['id'] if sent_idx < len(sentences) - 1 else None

            discourse_pairs = extract_cross_sentence_pairs(
                db, sentence_id, prev_id, next_id,
                subsample_probs, cross_sentence_weight
            )
            pairs.extend(discourse_pairs)

            processed += 1
            if processed % 10000 == 0:
                logger.info(f"Processed {processed}/{total_sentences} sentences, {len(pairs)} pairs so far")

    logger.info(f"Generated {len(pairs)} total training pairs")

    # Log pair type distribution
    weight_distribution = Counter(weight for _, _, weight in pairs)
    logger.info(f"Pair weight distribution: {dict(weight_distribution)}")

    return pairs


def filter_by_vocabulary(
    pairs: List[Tuple[str, str, float]],
    target_vocabulary: Path
) -> Tuple[List[Tuple[str, str, float]], List[str]]:
    """
    Filter training pairs by target vocabulary (clean semantic roots only).

    Args:
        pairs: Raw training pairs
        target_vocabulary: Path to production vocabulary JSON

    Returns:
        Tuple of (filtered_pairs, vocabulary_list)
    """
    logger.info(f"Filtering pairs by target vocabulary: {target_vocabulary}")

    # Load target vocabulary
    with open(target_vocabulary, 'r') as f:
        vocab_data = json.load(f)
        if isinstance(vocab_data, dict):
            vocabulary_set = set(vocab_data.keys())
        else:
            vocabulary_set = set(vocab_data)

    logger.info(f"Target vocabulary size: {len(vocabulary_set)} roots")

    # Filter pairs
    filtered_pairs = [
        (target, context, weight)
        for target, context, weight in pairs
        if target in vocabulary_set and context in vocabulary_set
    ]

    logger.info(f"Filtered pairs: {len(pairs)} → {len(filtered_pairs)} ({len(filtered_pairs)/len(pairs)*100:.1f}% kept)")

    # Create vocabulary list (sorted)
    vocabulary = sorted(vocabulary_set)

    return filtered_pairs, vocabulary


def save_training_data(
    pairs: List[Tuple[str, str, float]],
    vocabulary: List[str],
    output_path: Path,
    vocab_output_path: Path
) -> None:
    """
    Save training pairs and vocabulary to disk.

    Args:
        pairs: Training pairs (target, context, weight)
        vocabulary: Sorted list of unique roots
        output_path: Output path for pairs JSONL
        vocab_output_path: Output path for vocabulary JSON
    """
    logger.info(f"Saving training pairs to {output_path}")

    # Save pairs as JSONL
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for target, context, weight in pairs:
            json.dump({
                'target': target,
                'context': context,
                'weight': weight
            }, f)
            f.write('\n')

    # Save vocabulary
    logger.info(f"Saving vocabulary to {vocab_output_path}")
    vocab_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(vocab_output_path, 'w') as f:
        json.dump(vocabulary, f, ensure_ascii=False, indent=2)

    # Save statistics
    stats_path = output_path.parent / (output_path.stem + '_stats.json')
    logger.info(f"Saving statistics to {stats_path}")

    # Compute statistics
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
        description="Extract AST-aware semantic training pairs for root embeddings"
    )
    parser.add_argument(
        '--db-path',
        type=Path,
        required=True,
        help='Path to Kuzu v2.1 database'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output path for training pairs JSONL'
    )
    parser.add_argument(
        '--vocab-output',
        type=Path,
        help='Output path for vocabulary JSON (default: {output_dir}/vocab.json)'
    )
    parser.add_argument(
        '--target-vocabulary',
        type=Path,
        help='Target vocabulary (clean semantic roots only, e.g., production_semantic_roots_15k.json)'
    )
    parser.add_argument(
        '--subsample-threshold',
        type=float,
        default=1e-3,
        help='Mikolov subsampling threshold (default: 1e-3)'
    )
    parser.add_argument(
        '--cross-sentence-weight',
        type=float,
        default=0.5,
        help='Weight for cross-sentence discourse pairs (default: 0.5)'
    )

    args = parser.parse_args()

    # Set default vocab output path
    if args.vocab_output is None:
        args.vocab_output = args.output.parent / (args.output.stem + '_vocab.json')

    # Open database
    logger.info(f"Opening database: {args.db_path}")
    db = kuzu.Database(str(args.db_path))

    # Extract paragraph structure
    paragraphs = extract_paragraph_sentences(db)

    # Generate AST-aware semantic pairs
    pairs = generate_training_pairs(
        db, paragraphs,
        subsample_threshold=args.subsample_threshold,
        cross_sentence_weight=args.cross_sentence_weight
    )

    # Filter by target vocabulary (if provided)
    if args.target_vocabulary:
        pairs, vocabulary = filter_by_vocabulary(pairs, args.target_vocabulary)
    else:
        # Create vocabulary from pairs
        vocabulary = sorted(set(root for pair in pairs for root in pair[:2]))
        logger.info(f"Created vocabulary with {len(vocabulary)} roots from pairs")

    # Save training data
    save_training_data(pairs, vocabulary, args.output, args.vocab_output)

    logger.info("Extraction complete!")
    logger.info(f"Training pairs: {args.output}")
    logger.info(f"Vocabulary: {args.vocab_output}")
    logger.info(f"Statistics: {args.output.parent / (args.output.stem + '_stats.json')}")


if __name__ == '__main__':
    main()
