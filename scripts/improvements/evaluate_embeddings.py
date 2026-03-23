#!/usr/bin/env python3
"""
Evaluate root embedding quality.

Tests:
1. Synonym accuracy (similar roots should have high similarity)
2. Antonym detection (mal- pairs should have negative similarity)
3. Embedding collapse check (mean similarity should be low)
4. Cluster separation (semantic clusters should be distinct)

Usage:
    python scripts/improvements/evaluate_embeddings.py \
        --model models/root_embedder/frozen_v1.0.pt \
        --vocab data/vocabularies/tier_filtered_roots.json
"""

import argparse
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model_and_vocab(model_path: Path):
    """Load frozen model and extract embeddings."""
    checkpoint = torch.load(model_path, map_location='cpu')

    embeddings = checkpoint['model_state_dict']['embeddings.weight']
    root_to_idx = checkpoint['root_to_idx']
    idx_to_root = checkpoint['idx_to_root']

    logger.info(f"Loaded model: {model_path}")
    logger.info(f"  Vocabulary: {len(root_to_idx):,} roots")
    logger.info(f"  Embedding dim: {embeddings.shape[1]}")
    logger.info(f"  Frozen: {checkpoint.get('frozen', False)}")

    return embeddings, root_to_idx, idx_to_root


def compute_similarity(embeddings: torch.Tensor, idx1: int, idx2: int) -> float:
    """Compute cosine similarity between two embeddings."""
    emb1 = F.normalize(embeddings[idx1:idx1+1], dim=-1)
    emb2 = F.normalize(embeddings[idx2:idx2+1], dim=-1)
    return (emb1 * emb2).sum().item()


def test_antonyms(embeddings: torch.Tensor, root_to_idx: Dict[str, int]) -> Dict:
    """
    Test mal- antonym pairs.

    Esperanto systematically forms antonyms with mal- prefix:
    - bon (good) ↔ malbon (bad)
    - long (long) ↔ mallong (short)
    - varm (warm) ↔ malvarm (cold)

    Good embeddings should give NEGATIVE similarity for antonyms.
    """
    logger.info("\n=== Testing Antonym Detection ===")

    antonym_sims = []
    examples = []

    for root in root_to_idx:
        if not root.startswith('mal'):
            continue

        positive_root = root[3:]
        if len(positive_root) < 2 or positive_root not in root_to_idx:
            continue

        idx1 = root_to_idx[root]
        idx2 = root_to_idx[positive_root]

        sim = compute_similarity(embeddings, idx1, idx2)
        antonym_sims.append(sim)

        if len(examples) < 10:
            examples.append((positive_root, root, sim))

    if not antonym_sims:
        logger.warning("No antonym pairs found!")
        return {'mean_sim': 0.0, 'negative_rate': 0.0, 'count': 0}

    mean_sim = sum(antonym_sims) / len(antonym_sims)
    negative_rate = sum(1 for s in antonym_sims if s < 0) / len(antonym_sims)

    logger.info(f"Found {len(antonym_sims)} antonym pairs")
    logger.info(f"Mean similarity: {mean_sim:.4f}")
    logger.info(f"Negative rate: {negative_rate:.2%}")

    logger.info("\nExamples:")
    for pos, neg, sim in examples:
        logger.info(f"  {pos:15} ↔ {neg:15} : {sim:+.3f}")

    result = {
        'mean_sim': mean_sim,
        'negative_rate': negative_rate,
        'count': len(antonym_sims),
        'examples': examples
    }

    # Scoring
    if negative_rate > 0.8 and mean_sim < -0.3:
        logger.info("✓ EXCELLENT antonym detection!")
    elif negative_rate > 0.6 and mean_sim < 0.0:
        logger.info("✓ GOOD antonym detection")
    elif negative_rate > 0.4:
        logger.info("⚠ MODERATE antonym detection")
    else:
        logger.info("✗ POOR antonym detection")

    return result


def test_embedding_collapse(embeddings: torch.Tensor) -> Dict:
    """
    Check for embedding collapse.

    In collapsed embeddings, all vectors become similar (high mean similarity).
    Good embeddings should have low mean pairwise similarity.
    """
    logger.info("\n=== Testing Embedding Collapse ===")

    # Sample 1000 random pairs
    n = min(1000, embeddings.shape[0] // 2)
    indices1 = torch.randint(0, embeddings.shape[0], (n,))
    indices2 = torch.randint(0, embeddings.shape[0], (n,))

    # Compute similarities
    emb1 = F.normalize(embeddings[indices1], dim=-1)
    emb2 = F.normalize(embeddings[indices2], dim=-1)
    sims = (emb1 * emb2).sum(dim=-1).tolist()

    mean_sim = sum(sims) / len(sims)
    std_sim = (sum((s - mean_sim) ** 2 for s in sims) / len(sims)) ** 0.5

    logger.info(f"Random pair similarities (n={n}):")
    logger.info(f"  Mean: {mean_sim:.4f}")
    logger.info(f"  Std:  {std_sim:.4f}")

    result = {
        'mean_random_sim': mean_sim,
        'std_random_sim': std_sim,
        'sample_size': n
    }

    # Scoring
    if mean_sim < 0.1:
        logger.info("✓ EXCELLENT separation (no collapse)")
    elif mean_sim < 0.3:
        logger.info("✓ GOOD separation")
    elif mean_sim < 0.5:
        logger.info("⚠ MODERATE separation (some collapse)")
    else:
        logger.info("✗ POOR separation (collapsed embeddings!)")

    return result


def test_semantic_clusters(
    embeddings: torch.Tensor,
    root_to_idx: Dict[str, int],
    idx_to_root: Dict[int, str]
) -> Dict:
    """
    Test if semantically related roots cluster together.

    Check predefined semantic categories:
    - Animals: hund, kat, bird, fiŝ, ...
    - Colors: ruĝ, blu, verd, flav, ...
    - Emotions: ĝoj, trist, kolor, timo, ...
    """
    logger.info("\n=== Testing Semantic Clustering ===")

    # Define test clusters
    test_clusters = {
        'animals': ['hund', 'kat', 'bird', 'fiŝ', 'bov', 'ĉeval', 'pork'],
        'colors': ['ruĝ', 'blu', 'verd', 'flav', 'blank', 'nigr', 'brun'],
        'emotions': ['ĝoj', 'trist', 'kolor', 'timo', 'am'],
        'actions': ['ir', 'ven', 'far', 'don', 'pren', 'vid'],
    }

    results = {}

    for cluster_name, roots in test_clusters.items():
        # Filter roots that exist in vocabulary
        valid_roots = [r for r in roots if r in root_to_idx]

        if len(valid_roots) < 3:
            logger.warning(f"  {cluster_name}: too few roots in vocabulary")
            continue

        # Compute within-cluster similarity
        sims = []
        for i, r1 in enumerate(valid_roots):
            for r2 in valid_roots[i+1:]:
                idx1, idx2 = root_to_idx[r1], root_to_idx[r2]
                sim = compute_similarity(embeddings, idx1, idx2)
                sims.append(sim)

        if not sims:
            continue

        mean_sim = sum(sims) / len(sims)
        results[cluster_name] = {
            'mean_sim': mean_sim,
            'count': len(valid_roots),
            'pairs': len(sims)
        }

        logger.info(f"  {cluster_name:12}: {mean_sim:.3f} (n={len(valid_roots)} roots, {len(sims)} pairs)")

    overall_mean = sum(r['mean_sim'] for r in results.values()) / len(results) if results else 0.0

    logger.info(f"\nOverall cluster coherence: {overall_mean:.3f}")

    if overall_mean > 0.4:
        logger.info("✓ EXCELLENT semantic clustering")
    elif overall_mean > 0.3:
        logger.info("✓ GOOD semantic clustering")
    elif overall_mean > 0.2:
        logger.info("⚠ MODERATE semantic clustering")
    else:
        logger.info("✗ POOR semantic clustering")

    return {
        'clusters': results,
        'overall_mean': overall_mean
    }


def test_nearest_neighbors(
    embeddings: torch.Tensor,
    root_to_idx: Dict[str, int],
    idx_to_root: Dict[int, str],
    test_roots: List[str] = ['hund', 'am', 'bon', 'ruĝ']
) -> Dict:
    """
    Find nearest neighbors for test roots.

    This is a qualitative check - nearest neighbors should make sense.
    """
    logger.info("\n=== Testing Nearest Neighbors ===")

    results = {}

    for root in test_roots:
        if root not in root_to_idx:
            logger.warning(f"  {root}: not in vocabulary")
            continue

        idx = root_to_idx[root]
        emb = F.normalize(embeddings[idx:idx+1], dim=-1)

        # Compute similarity to all other roots
        all_embs = F.normalize(embeddings, dim=-1)
        sims = (emb * all_embs).sum(dim=-1)

        # Get top 10 (excluding self)
        top_indices = sims.argsort(descending=True)[1:11].tolist()
        neighbors = [(idx_to_root[str(i)], sims[i].item()) for i in top_indices]

        results[root] = neighbors

        logger.info(f"\n  {root}:")
        for neighbor, sim in neighbors[:5]:
            logger.info(f"    {neighbor:15} : {sim:.3f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate root embeddings")
    parser.add_argument('--model', type=Path, required=True,
                        help='Path to frozen model')
    parser.add_argument('--output', type=Path,
                        help='Path to save evaluation results (JSON)')

    args = parser.parse_args()

    # Load model
    embeddings, root_to_idx, idx_to_root = load_model_and_vocab(args.model)

    # Run all tests
    results = {
        'model_path': str(args.model),
        'vocab_size': len(root_to_idx),
        'embedding_dim': embeddings.shape[1],
    }

    # Test 1: Antonyms
    results['antonyms'] = test_antonyms(embeddings, root_to_idx)

    # Test 2: Embedding collapse
    results['collapse'] = test_embedding_collapse(embeddings)

    # Test 3: Semantic clusters
    results['clusters'] = test_semantic_clusters(embeddings, root_to_idx, idx_to_root)

    # Test 4: Nearest neighbors (qualitative)
    results['nearest_neighbors'] = test_nearest_neighbors(embeddings, root_to_idx, idx_to_root)

    # Overall score
    logger.info("\n" + "=" * 60)
    logger.info("OVERALL EVALUATION")
    logger.info("=" * 60)

    scores = []

    # Antonym score
    if results['antonyms']['negative_rate'] > 0.8:
        scores.append(100)
    elif results['antonyms']['negative_rate'] > 0.6:
        scores.append(80)
    elif results['antonyms']['negative_rate'] > 0.4:
        scores.append(60)
    else:
        scores.append(40)

    # Collapse score
    if results['collapse']['mean_random_sim'] < 0.1:
        scores.append(100)
    elif results['collapse']['mean_random_sim'] < 0.3:
        scores.append(80)
    elif results['collapse']['mean_random_sim'] < 0.5:
        scores.append(60)
    else:
        scores.append(40)

    # Clustering score
    cluster_mean = results['clusters']['overall_mean']
    if cluster_mean > 0.4:
        scores.append(100)
    elif cluster_mean > 0.3:
        scores.append(80)
    elif cluster_mean > 0.2:
        scores.append(60)
    else:
        scores.append(40)

    overall_score = sum(scores) / len(scores)
    results['overall_score'] = overall_score

    logger.info(f"Overall Score: {overall_score:.1f}/100")

    if overall_score >= 90:
        logger.info("✓ EXCELLENT embeddings - ready for production!")
    elif overall_score >= 75:
        logger.info("✓ GOOD embeddings - acceptable quality")
    elif overall_score >= 60:
        logger.info("⚠ MODERATE embeddings - consider improvements")
    else:
        logger.info("✗ POOR embeddings - needs redesign")

    # Save results
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            # Convert torch tensors to lists for JSON serialization
            json_results = results.copy()
            json_results['nearest_neighbors'] = {
                k: [(n, float(s)) for n, s in v]
                for k, v in results['nearest_neighbors'].items()
            }
            json.dump(json_results, f, indent=2)
        logger.info(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
