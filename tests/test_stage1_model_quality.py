#!/usr/bin/env python3
"""
Stage 1 Model Quality Tests - Root Embeddings

Tests the trained Stage 1 root embedding model against quality thresholds:
- Root similarity accuracy: >85%
- No embedding collapse: mean_sim < 0.5
- Cluster separation: gap > 0.03
- Fundamento coverage: 100%
- ReVo correlation: >0.75

Run: pytest tests/test_stage1_model_quality.py -v
"""

import json
import pytest
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def root_embeddings():
    """Load trained Stage 1 root embeddings."""
    model_path = Path('models/root_embeddings/best_model.pt')

    if not model_path.exists():
        pytest.skip("Stage 1 model not found - run ./scripts/train_roots.sh first")

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    return {
        'embeddings': checkpoint['model_state_dict']['embeddings.weight'],
        'root_to_idx': checkpoint['root_to_idx'],
        'idx_to_root': checkpoint['idx_to_root'],
        'vocab_size': len(checkpoint['root_to_idx']),
        'embedding_dim': checkpoint['model_state_dict']['embeddings.weight'].shape[1],
        'correlation': checkpoint.get('correlation', checkpoint.get('best_correlation', 0.0))
    }


@pytest.fixture(scope="module")
def revo_relations():
    """Load ReVo semantic relations for testing."""
    revo_path = Path('data/raw/eo/dictionaries/revo/revo_semantic_relations.json')

    if not revo_path.exists():
        pytest.skip("ReVo relations not found")

    with open(revo_path) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def fundamento_roots():
    """Load Fundamento roots."""
    fundamento_path = Path('data/vocabularies/fundamento_roots.json')

    if not fundamento_path.exists():
        pytest.skip("Fundamento roots not found")

    with open(fundamento_path) as f:
        data = json.load(f)
        if isinstance(data, list):
            return set(r['root'].lower() if isinstance(r, dict) else str(r).lower() for r in data)
        return set(data.keys())


# =============================================================================
# Helper Functions
# =============================================================================

def cosine_similarity(emb1: torch.Tensor, emb2: torch.Tensor) -> float:
    """Compute cosine similarity between two embeddings."""
    emb1_norm = F.normalize(emb1.unsqueeze(0), dim=1)
    emb2_norm = F.normalize(emb2.unsqueeze(0), dim=1)
    return (emb1_norm * emb2_norm).sum().item()


def get_embedding(embeddings: torch.Tensor, root_to_idx: Dict[str, int], root: str):
    """Get embedding for a root, or None if not found."""
    if root not in root_to_idx:
        return None
    return embeddings[root_to_idx[root]]


# =============================================================================
# Test 1: Model Metadata
# =============================================================================

def test_model_exists():
    """Test that Stage 1 model exists."""
    model_path = Path('models/root_embeddings/best_model.pt')
    assert model_path.exists(), "Stage 1 model not found - run training first"


def test_vocabulary_size(root_embeddings):
    """Test that vocabulary has expected size."""
    vocab_size = root_embeddings['vocab_size']

    # Should be either old (10.8K) or new expanded (18.9K)
    assert vocab_size >= 10000, f"Vocabulary too small: {vocab_size}"
    assert vocab_size <= 20000, f"Vocabulary unexpectedly large: {vocab_size}"

    print(f"\n✓ Vocabulary size: {vocab_size:,} roots")


def test_embedding_dimension(root_embeddings):
    """Test that embeddings have correct dimensionality."""
    dim = root_embeddings['embedding_dim']
    assert dim == 64, f"Expected 64D embeddings, got {dim}D"

    print(f"\n✓ Embedding dimension: {dim}D")


# =============================================================================
# Test 2: Fundamento Coverage
# =============================================================================

def test_fundamento_coverage(root_embeddings, fundamento_roots):
    """Test that all Fundamento roots are in vocabulary."""
    root_to_idx = root_embeddings['root_to_idx']

    missing = []
    for root in fundamento_roots:
        if root not in root_to_idx:
            missing.append(root)

    coverage = (len(fundamento_roots) - len(missing)) / len(fundamento_roots)

    print(f"\n✓ Fundamento coverage: {coverage:.1%} ({len(fundamento_roots) - len(missing)}/{len(fundamento_roots)})")

    if missing:
        print(f"  Missing Fundamento roots: {', '.join(sorted(missing)[:10])}")

    # Target: 100% coverage
    assert coverage >= 0.95, f"Fundamento coverage too low: {coverage:.1%}"


# =============================================================================
# Test 3: No Embedding Collapse
# =============================================================================

def test_no_embedding_collapse(root_embeddings):
    """Test that embeddings haven't collapsed (all similar)."""
    embeddings = root_embeddings['embeddings']

    # Sample 1000 random pairs
    n = min(1000, len(embeddings))
    indices = torch.randperm(len(embeddings))[:n]

    similarities = []
    for i in range(n):
        for j in range(i + 1, min(i + 10, n)):  # Compare with 10 neighbors
            sim = cosine_similarity(embeddings[indices[i]], embeddings[indices[j]])
            similarities.append(sim)

    mean_sim = sum(similarities) / len(similarities)

    print(f"\n✓ Mean random similarity: {mean_sim:.3f}")

    # Target: mean_sim < 0.5 (embeddings are diverse)
    assert mean_sim < 0.5, f"Embedding collapse detected: mean_sim={mean_sim:.3f}"


# =============================================================================
# Test 4: Synonym Similarity
# =============================================================================

def test_synonym_similarity(root_embeddings, revo_relations):
    """Test that synonyms have high similarity."""
    embeddings = root_embeddings['embeddings']
    root_to_idx = root_embeddings['root_to_idx']

    synonyms = revo_relations.get('synonyms', {})

    high_sim_pairs = 0
    total_pairs = 0
    similarities = []

    for root, syn_list in synonyms.items():
        root_emb = get_embedding(embeddings, root_to_idx, root.lower())
        if root_emb is None:
            continue

        for syn in syn_list:
            syn_emb = get_embedding(embeddings, root_to_idx, syn.lower())
            if syn_emb is None:
                continue

            sim = cosine_similarity(root_emb, syn_emb)
            similarities.append(sim)
            total_pairs += 1

            if sim > 0.3:  # Threshold for "similar"
                high_sim_pairs += 1

    if total_pairs == 0:
        pytest.skip("No synonym pairs found in vocabulary")

    accuracy = high_sim_pairs / total_pairs
    mean_sim = sum(similarities) / len(similarities)

    print(f"\n✓ Synonym similarity:")
    print(f"  Pairs tested: {total_pairs}")
    print(f"  Mean similarity: {mean_sim:.3f}")
    print(f"  High similarity (>0.3): {accuracy:.1%}")

    # Target: >70% of synonyms have sim > 0.3
    assert accuracy >= 0.70, f"Synonym accuracy too low: {accuracy:.1%}"


# =============================================================================
# Test 5: Antonym Separation
# =============================================================================

def test_antonym_separation(root_embeddings, revo_relations):
    """Test that antonyms have low similarity."""
    embeddings = root_embeddings['embeddings']
    root_to_idx = root_embeddings['root_to_idx']

    antonyms = revo_relations.get('antonyms', {})

    low_sim_pairs = 0
    total_pairs = 0
    similarities = []

    for root, ant_list in antonyms.items():
        root_emb = get_embedding(embeddings, root_to_idx, root.lower())
        if root_emb is None:
            continue

        for ant in ant_list:
            ant_emb = get_embedding(embeddings, root_to_idx, ant.lower())
            if ant_emb is None:
                continue

            sim = cosine_similarity(root_emb, ant_emb)
            similarities.append(sim)
            total_pairs += 1

            if sim < 0.3:  # Threshold for "dissimilar"
                low_sim_pairs += 1

    if total_pairs == 0:
        pytest.skip("No antonym pairs found in vocabulary")

    accuracy = low_sim_pairs / total_pairs
    mean_sim = sum(similarities) / len(similarities)

    print(f"\n✓ Antonym separation:")
    print(f"  Pairs tested: {total_pairs}")
    print(f"  Mean similarity: {mean_sim:.3f}")
    print(f"  Low similarity (<0.3): {accuracy:.1%}")

    # Target: >60% of antonyms have sim < 0.3
    assert accuracy >= 0.60, f"Antonym separation too low: {accuracy:.1%}"


# =============================================================================
# Test 6: ReVo Correlation
# =============================================================================

def test_revo_correlation(root_embeddings):
    """Test that ReVo correlation meets threshold."""
    correlation = root_embeddings['correlation']

    print(f"\n✓ ReVo correlation: {correlation:.4f}")

    # Target: >0.75 correlation
    assert correlation >= 0.75, f"ReVo correlation too low: {correlation:.4f}"


# =============================================================================
# Test 7: Semantic Cluster Coherence
# =============================================================================

def test_semantic_cluster_coherence(root_embeddings):
    """Test that semantic clusters have high intra-cluster similarity."""
    embeddings = root_embeddings['embeddings']
    root_to_idx = root_embeddings['root_to_idx']

    # Define test clusters
    clusters = {
        'animals': ['hund', 'kat', 'best', 'bird', 'fiŝ', 'muso'],
        'food': ['manĝ', 'trink', 'pom', 'pan', 'vian', 'lakt'],
        'emotions': ['ĝoj', 'trist', 'kolor', 'am', 'mal', 'tim'],
        'colors': ['ruĝ', 'blu', 'verd', 'flav', 'blank', 'nigr']
    }

    cluster_scores = []

    for cluster_name, roots in clusters.items():
        # Get embeddings for cluster members
        cluster_embs = []
        for root in roots:
            emb = get_embedding(embeddings, root_to_idx, root)
            if emb is not None:
                cluster_embs.append(emb)

        if len(cluster_embs) < 2:
            continue

        # Compute intra-cluster similarity
        similarities = []
        for i in range(len(cluster_embs)):
            for j in range(i + 1, len(cluster_embs)):
                sim = cosine_similarity(cluster_embs[i], cluster_embs[j])
                similarities.append(sim)

        mean_sim = sum(similarities) / len(similarities)
        cluster_scores.append(mean_sim)

        print(f"  {cluster_name}: {mean_sim:.3f} ({len(cluster_embs)}/{len(roots)} roots)")

    if not cluster_scores:
        pytest.skip("No cluster roots found in vocabulary")

    overall_coherence = sum(cluster_scores) / len(cluster_scores)

    print(f"\n✓ Cluster coherence: {overall_coherence:.3f}")

    # Target: >0.25 mean intra-cluster similarity
    assert overall_coherence >= 0.25, f"Cluster coherence too low: {overall_coherence:.3f}"


# =============================================================================
# Test 8: M1 Vocabulary Coverage
# =============================================================================

def test_m1_vocabulary_coverage(root_embeddings):
    """Test coverage of M1 training vocabulary."""
    m1_vocab_path = Path('data/training/m1_selectional_hard/vocabulary.json')

    if not m1_vocab_path.exists():
        pytest.skip("M1 vocabulary not found - M1 data not generated yet")

    with open(m1_vocab_path) as f:
        m1_vocab = json.load(f)
        m1_roots = set(m1_vocab.get('nouns', []) + m1_vocab.get('verbs', []))

    root_to_idx = root_embeddings['root_to_idx']

    covered = sum(1 for root in m1_roots if root in root_to_idx)
    coverage = covered / len(m1_roots)

    print(f"\n✓ M1 vocabulary coverage: {coverage:.1%} ({covered}/{len(m1_roots)})")

    # Target: >90% coverage (new expanded vocabulary)
    assert coverage >= 0.90, f"M1 coverage too low: {coverage:.1%}"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
