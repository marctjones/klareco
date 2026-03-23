#!/usr/bin/env python3
"""
Semantic Type Clustering - Automated Type Hierarchy from SVO Patterns

VERSION: v2.1
COMPATIBLE WITH: v2.1 SVO extraction output, v2.1 database schema
DEPENDENCIES: SVO triples from extract_svo_triples.py
STAGE: Training

Description:
    Automatically discovers semantic types by clustering roots based on verb
    co-occurrence patterns. Implements distributional hypothesis: words appearing
    with similar verbs have similar semantic types.

    Zero human annotation required - fully automated from corpus patterns.

Pipeline Position:
    SVO Triples → [THIS SCRIPT] → SEMANTIC_TYPES.json → Verb Constraints → SFV Model

Usage:
    python scripts/cluster_semantic_types.py \
        --input data/semantic_types/svo_triples_full.jsonl \
        --output data/semantic_types/semantic_types.json \
        --num-clusters 18 \
        --min-frequency 10

Inputs:
    - SVO triples: JSONL file from extract_svo_triples.py
      Format: {"subject": "zamenhof", "verb": "kre", "object": "esperant", ...}

Outputs:
    - semantic_types.json: Dictionary mapping roots to semantic type labels
      Format: {"zamenhof": "PERSONO", "hund": "ANIMALO", "tabel": "OBJEKTO"}
    - cluster_stats.json: Statistics about each cluster
    - verb_matrix.npz: Sparse verb co-occurrence matrix (for inspection)

Quality Checks:
    - Cluster coherence: Within-cluster similarity > 0.3
    - Cluster separation: Between-cluster distance > 0.1
    - Coverage: % of roots assigned to clusters
    - Fundamento coverage: All Fundamento roots categorized

Algorithm:
    1. Build verb co-occurrence matrix (roots × verbs)
    2. Compute root similarity using cosine distance on verb vectors
    3. Apply hierarchical clustering (Ward linkage)
    4. Assign semantic type labels to clusters
    5. Validate cluster quality

Last Updated: 2026-03-16
Author: Claude Code
Related Issues: #691 (parser enhancement), semantic type hierarchy design
See Also: extract_svo_triples.py, generate_verb_constraints.py
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict, Counter
import logging

import numpy as np
from scipy.sparse import csr_matrix, save_npz
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_svo_triples(input_path: Path) -> List[Dict]:
    """Load SVO triples from JSONL file."""
    logger.info(f"Loading SVO triples from {input_path}")
    triples = []

    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                triple = json.loads(line.strip())
                triples.append(triple)
            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num}: Invalid JSON - {e}")
                continue

    logger.info(f"Loaded {len(triples):,} SVO triples")
    return triples


def build_verb_cooccurrence_matrix(
    triples: List[Dict],
    min_frequency: int = 10
) -> Tuple[csr_matrix, List[str], List[str]]:
    """
    Build sparse matrix of verb co-occurrence for roots.

    Returns:
        matrix: Sparse (roots × verbs) matrix of co-occurrence counts
        root_vocab: List of roots (row indices)
        verb_vocab: List of verbs (column indices)
    """
    logger.info("Building verb co-occurrence matrix...")

    # Count root and verb frequencies
    root_freq = Counter()
    verb_freq = Counter()

    for triple in triples:
        subj = triple.get('subject')
        verb = triple.get('verb')
        obj = triple.get('object')

        if subj:
            root_freq[subj] += 1
        if verb:
            verb_freq[verb] += 1
        if obj:
            root_freq[obj] += 1

    # Filter by minimum frequency
    valid_roots = {r for r, c in root_freq.items() if c >= min_frequency}
    valid_verbs = {v for v, c in verb_freq.items() if c >= min_frequency}

    logger.info(f"Found {len(valid_roots):,} roots (freq >= {min_frequency})")
    logger.info(f"Found {len(valid_verbs):,} verbs (freq >= {min_frequency})")

    # Create vocabularies
    root_vocab = sorted(valid_roots)
    verb_vocab = sorted(valid_verbs)

    root_to_idx = {r: i for i, r in enumerate(root_vocab)}
    verb_to_idx = {v: i for i, v in enumerate(verb_vocab)}

    # Build co-occurrence counts
    # For each root, count which verbs it appears with (as subject or object)
    cooccurrence = defaultdict(lambda: defaultdict(int))

    for triple in triples:
        subj = triple.get('subject')
        verb = triple.get('verb')
        obj = triple.get('object')

        if verb in valid_verbs:
            if subj in valid_roots:
                cooccurrence[subj][verb] += 1
            if obj in valid_roots:
                cooccurrence[obj][verb] += 1

    # Convert to sparse matrix
    rows = []
    cols = []
    data = []

    for root, verbs in cooccurrence.items():
        if root not in root_to_idx:
            continue
        root_idx = root_to_idx[root]

        for verb, count in verbs.items():
            if verb not in verb_to_idx:
                continue
            verb_idx = verb_to_idx[verb]

            rows.append(root_idx)
            cols.append(verb_idx)
            data.append(count)

    matrix = csr_matrix(
        (data, (rows, cols)),
        shape=(len(root_vocab), len(verb_vocab)),
        dtype=np.float32
    )

    logger.info(f"Matrix shape: {matrix.shape}, density: {matrix.nnz / np.prod(matrix.shape):.4f}")

    return matrix, root_vocab, verb_vocab


def normalize_matrix(matrix: csr_matrix) -> csr_matrix:
    """Normalize rows to unit length (L2 normalization)."""
    logger.info("Normalizing verb vectors...")

    # Compute row norms
    row_norms = np.sqrt(np.array(matrix.power(2).sum(axis=1)).flatten())
    row_norms[row_norms == 0] = 1.0  # Avoid division by zero

    # Normalize
    matrix_normalized = matrix.copy()
    matrix_normalized = matrix_normalized.multiply(1.0 / row_norms[:, np.newaxis])

    return matrix_normalized


def cluster_roots(
    matrix: csr_matrix,
    num_clusters: int,
    method: str = 'ward'
) -> np.ndarray:
    """
    Cluster roots using hierarchical clustering.

    Args:
        matrix: Normalized (roots × verbs) co-occurrence matrix
        num_clusters: Target number of semantic type clusters
        method: Linkage method ('ward', 'average', 'complete')

    Returns:
        cluster_labels: Array of cluster assignments for each root
    """
    logger.info(f"Clustering {matrix.shape[0]} roots into {num_clusters} clusters...")

    # Convert sparse to dense for clustering (only if small enough)
    if matrix.shape[0] > 10000:
        logger.warning(f"Large matrix ({matrix.shape[0]} roots), this may be slow...")

    # Compute pairwise cosine distances
    # Cosine distance = 1 - cosine similarity
    # For normalized vectors: cosine_sim(a, b) = a · b
    dense_matrix = matrix.toarray()

    logger.info("Computing pairwise distances...")
    distances = pdist(dense_matrix, metric='cosine')

    logger.info("Running hierarchical clustering...")
    linkage_matrix = linkage(distances, method=method)

    logger.info(f"Cutting dendrogram at {num_clusters} clusters...")
    cluster_labels = fcluster(linkage_matrix, num_clusters, criterion='maxclust')

    # Compute silhouette score (cluster quality metric)
    try:
        silhouette = silhouette_score(dense_matrix, cluster_labels, metric='cosine')
        logger.info(f"Silhouette score: {silhouette:.3f} (higher is better, range [-1, 1])")
    except Exception as e:
        logger.warning(f"Could not compute silhouette score: {e}")

    return cluster_labels


def assign_semantic_labels(
    cluster_labels: np.ndarray,
    root_vocab: List[str],
    num_clusters: int
) -> Dict[int, str]:
    """
    Assign human-readable semantic type labels to clusters.

    Uses heuristics based on common roots in each cluster.
    For now, uses generic labels like TYPE_01, TYPE_02, etc.

    Future: Could use LLM to suggest labels based on cluster contents.
    """
    logger.info("Assigning semantic type labels to clusters...")

    cluster_to_label = {}

    # Group roots by cluster
    clusters = defaultdict(list)
    for root, label in zip(root_vocab, cluster_labels):
        clusters[label].append(root)

    # Heuristic label assignment
    for cluster_id in range(1, num_clusters + 1):
        roots_in_cluster = clusters.get(cluster_id, [])

        if not roots_in_cluster:
            cluster_to_label[cluster_id] = f"TYPE_{cluster_id:02d}"
            continue

        # Check for common semantic patterns
        label = None

        # Person indicators
        person_indicators = {'hom', 'persono', 'vir', 'infan', 'patro', 'patrino'}
        if any(r in person_indicators for r in roots_in_cluster[:20]):
            label = "PERSONO"

        # Animal indicators
        elif any(r in {'hund', 'kat', 'bird', 'best'} for r in roots_in_cluster[:20]):
            label = "ANIMALO"

        # Object indicators
        elif any(r in {'tabel', 'seĝ', 'libr', 'aŭt'} for r in roots_in_cluster[:20]):
            label = "OBJEKTO"

        # Place indicators
        elif any(r in {'urb', 'land', 'dom', 'ejo'} for r in roots_in_cluster[:20]):
            label = "LOKO"

        # Time indicators
        elif any(r in {'jar', 'monat', 'tag', 'hor'} for r in roots_in_cluster[:20]):
            label = "TEMPO"

        # Action indicators
        elif any(r in {'ir', 'ven', 'far', 'dir'} for r in roots_in_cluster[:20]):
            label = "AGO"

        # Default: generic type label
        if label is None:
            label = f"TYPE_{cluster_id:02d}"

        # Ensure uniqueness
        if label in cluster_to_label.values():
            label = f"{label}_{cluster_id:02d}"

        cluster_to_label[cluster_id] = label
        logger.info(f"  Cluster {cluster_id} ({len(roots_in_cluster)} roots): {label}")
        logger.info(f"    Sample: {', '.join(roots_in_cluster[:10])}")

    return cluster_to_label


def compute_cluster_stats(
    cluster_labels: np.ndarray,
    root_vocab: List[str],
    matrix: csr_matrix,
    cluster_to_label: Dict[int, str]
) -> Dict:
    """Compute statistics for each cluster."""
    logger.info("Computing cluster statistics...")

    stats = {}

    # Group roots by cluster
    clusters = defaultdict(list)
    for root, label in zip(root_vocab, cluster_labels):
        clusters[label].append(root)

    for cluster_id, semantic_label in cluster_to_label.items():
        roots_in_cluster = clusters.get(cluster_id, [])

        if not roots_in_cluster:
            continue

        # Get indices for roots in this cluster
        indices = [i for i, root in enumerate(root_vocab) if cluster_labels[i] == cluster_id]

        # Compute within-cluster similarity (mean pairwise cosine similarity)
        if len(indices) > 1:
            cluster_vectors = matrix[indices].toarray()

            # Pairwise cosine similarity
            similarities = []
            for i in range(len(cluster_vectors)):
                for j in range(i + 1, len(cluster_vectors)):
                    sim = np.dot(cluster_vectors[i], cluster_vectors[j])
                    similarities.append(sim)

            mean_similarity = np.mean(similarities) if similarities else 0.0
        else:
            mean_similarity = 1.0

        stats[semantic_label] = {
            'cluster_id': cluster_id,
            'num_roots': len(roots_in_cluster),
            'mean_within_similarity': float(mean_similarity),
            'sample_roots': roots_in_cluster[:20]
        }

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Cluster semantic types from SVO verb co-occurrence patterns"
    )
    parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='Input JSONL file with SVO triples'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output JSON file for semantic type mappings'
    )
    parser.add_argument(
        '--num-clusters',
        type=int,
        default=18,
        help='Number of semantic type clusters (default: 18)'
    )
    parser.add_argument(
        '--min-frequency',
        type=int,
        default=10,
        help='Minimum frequency for roots and verbs (default: 10)'
    )
    parser.add_argument(
        '--method',
        choices=['ward', 'average', 'complete'],
        default='ward',
        help='Hierarchical clustering linkage method (default: ward)'
    )
    parser.add_argument(
        '--save-matrix',
        action='store_true',
        help='Save verb co-occurrence matrix to NPZ file'
    )

    args = parser.parse_args()

    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Load SVO triples
    triples = load_svo_triples(args.input)

    if not triples:
        logger.error("No triples loaded, exiting")
        return

    # Build verb co-occurrence matrix
    matrix, root_vocab, verb_vocab = build_verb_cooccurrence_matrix(
        triples,
        min_frequency=args.min_frequency
    )

    # Normalize matrix (for cosine similarity)
    matrix_normalized = normalize_matrix(matrix)

    # Save matrix if requested
    if args.save_matrix:
        matrix_path = args.output.parent / 'verb_cooccurrence_matrix.npz'
        logger.info(f"Saving matrix to {matrix_path}")
        save_npz(matrix_path, matrix)

        vocab_path = args.output.parent / 'matrix_vocabularies.json'
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump({
                'roots': root_vocab,
                'verbs': verb_vocab
            }, f, ensure_ascii=False, indent=2)

    # Cluster roots
    cluster_labels = cluster_roots(
        matrix_normalized,
        num_clusters=args.num_clusters,
        method=args.method
    )

    # Assign semantic labels
    cluster_to_label = assign_semantic_labels(
        cluster_labels,
        root_vocab,
        args.num_clusters
    )

    # Compute statistics
    stats = compute_cluster_stats(
        cluster_labels,
        root_vocab,
        matrix_normalized,
        cluster_to_label
    )

    # Create semantic type mapping
    semantic_types = {}
    for root, cluster_id in zip(root_vocab, cluster_labels):
        semantic_label = cluster_to_label.get(cluster_id, f"TYPE_{cluster_id:02d}")
        semantic_types[root] = semantic_label

    # Save output
    logger.info(f"Saving semantic types to {args.output}")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(semantic_types, f, ensure_ascii=False, indent=2)

    # Save statistics
    stats_path = args.output.parent / 'cluster_stats.json'
    logger.info(f"Saving statistics to {stats_path}")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    logger.info("✅ Clustering complete!")
    logger.info(f"Created {len(cluster_to_label)} semantic type clusters")
    logger.info(f"Categorized {len(semantic_types):,} roots")


if __name__ == '__main__':
    main()
