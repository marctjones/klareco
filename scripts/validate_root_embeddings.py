#!/usr/bin/env python3
"""
Validate root embeddings quality.

Tests:
1. Semantic similarity - similar roots should be close
2. Anti-patterns - opposite roots should be far apart
3. Cluster coherence - semantic groups should cluster
4. Fundamento coverage - all essential roots embedded
5. No embedding collapse - embeddings should be diverse
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import argparse


def load_model(model_path: Path) -> Tuple[torch.nn.Embedding, Dict[str, int], Dict[int, str]]:
    """Load trained embedding model and vocabularies."""
    checkpoint = torch.load(model_path, map_location='cpu')

    embedding = torch.nn.Embedding(
        checkpoint['vocab_size'],
        checkpoint['embedding_dim']
    )
    embedding.load_state_dict({'weight': checkpoint['embedding_weights']})

    root_to_idx = checkpoint['root_to_idx']
    idx_to_root = {v: k for k, v in root_to_idx.items()}

    return embedding, root_to_idx, idx_to_root


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine similarity between two vectors."""
    return (a @ b) / (torch.norm(a) * torch.norm(b))


def test_semantic_similarity(embedding, root_to_idx):
    """Test if semantically similar roots have high similarity."""

    # Define test pairs: (root1, root2, expected_similarity_range)
    similar_pairs = [
        ("hund", "kat", (0.3, 1.0)),  # dog, cat (both animals)
        ("bel", "bon", (0.3, 1.0)),   # beautiful, good (positive qualities)
        ("pens", "sci", (0.3, 1.0)),  # think, know (cognitive)
        ("leg", "skrib", (0.3, 1.0)), # read, write (literacy)
        ("patr", "parol", (0.2, 1.0)), # father, speak (may be weaker)
        ("dom", "lok", (0.3, 1.0)),   # house, place (locations)
        ("infan", "jun", (0.3, 1.0)), # child, young (youth)
    ]

    results = []
    for r1, r2, (min_sim, max_sim) in similar_pairs:
        if r1 not in root_to_idx or r2 not in root_to_idx:
            results.append((r1, r2, None, "SKIP", "Root not in vocab"))
            continue

        emb1 = embedding(torch.tensor(root_to_idx[r1]))
        emb2 = embedding(torch.tensor(root_to_idx[r2]))
        sim = cosine_similarity(emb1, emb2).item()

        status = "PASS" if min_sim <= sim <= max_sim else "FAIL"
        results.append((r1, r2, sim, status, f"Expected [{min_sim:.2f}, {max_sim:.2f}]"))

    return results


def test_anti_patterns(embedding, root_to_idx):
    """Test if opposite/unrelated roots have low similarity."""

    # Define test pairs: (root1, root2, max_expected_similarity)
    dissimilar_pairs = [
        ("bon", "malbon", 0.5),  # good, bad (if malbon exists)
        ("hund", "libr", 0.4),   # dog, book (unrelated)
        ("bel", "danĝer", 0.4),  # beautiful, danger (unrelated)
        ("am", "hav", 0.5),      # love, have (different semantic fields)
    ]

    results = []
    for r1, r2, max_sim in dissimilar_pairs:
        if r1 not in root_to_idx or r2 not in root_to_idx:
            results.append((r1, r2, None, "SKIP", "Root not in vocab"))
            continue

        emb1 = embedding(torch.tensor(root_to_idx[r1]))
        emb2 = embedding(torch.tensor(root_to_idx[r2]))
        sim = cosine_similarity(emb1, emb2).item()

        status = "PASS" if sim <= max_sim else "FAIL"
        results.append((r1, r2, sim, status, f"Expected <= {max_sim:.2f}"))

    return results


def test_cluster_coherence(embedding, root_to_idx):
    """Test if semantic groups cluster together."""

    # Define semantic groups
    groups = {
        "animals": ["hund", "kat", "bird"],
        "family": ["patr", "frat", "fil", "infan"],
        "cognitive": ["pens", "sci", "lern", "stud"],
        "communication": ["parol", "dir", "skrib", "leg"],
        "positive_qualities": ["bel", "bon", "feliĉ"],
    }

    results = []
    for group_name, roots in groups.items():
        # Filter roots that exist
        valid_roots = [r for r in roots if r in root_to_idx]
        if len(valid_roots) < 2:
            results.append((group_name, None, "SKIP", f"Only {len(valid_roots)} roots in vocab"))
            continue

        # Compute pairwise similarities within group
        embeddings = [embedding(torch.tensor(root_to_idx[r])) for r in valid_roots]
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                sim = cosine_similarity(embeddings[i], embeddings[j]).item()
                similarities.append(sim)

        avg_sim = np.mean(similarities)
        min_sim = np.min(similarities)

        # Expect average within-group similarity > 0.3
        status = "PASS" if avg_sim > 0.3 else "FAIL"
        results.append((group_name, avg_sim, status, f"min={min_sim:.3f}, roots={valid_roots}"))

    return results


def test_fundamento_coverage(embedding, root_to_idx, fundamento_path: Path):
    """Test if all Fundamento roots are embedded."""

    with open(fundamento_path) as f:
        data = json.load(f)
        fundamento_roots = data.get('roots', data)

    missing = []
    present = []

    for root in fundamento_roots.keys():
        if root in root_to_idx:
            present.append(root)
        else:
            missing.append(root)

    coverage = len(present) / len(fundamento_roots)
    status = "PASS" if coverage == 1.0 else "WARN" if coverage > 0.9 else "FAIL"

    return {
        "total": len(fundamento_roots),
        "present": len(present),
        "missing": missing,
        "coverage": coverage,
        "status": status
    }


def test_embedding_collapse(embedding, root_to_idx):
    """Test if embeddings are diverse (not collapsed to similar vectors)."""

    # Sample 100 random roots
    all_roots = list(root_to_idx.keys())
    sample_size = min(100, len(all_roots))
    sample_roots = np.random.choice(all_roots, sample_size, replace=False)

    # Compute pairwise similarities
    embeddings = [embedding(torch.tensor(root_to_idx[r])) for r in sample_roots]
    similarities = []
    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            sim = cosine_similarity(embeddings[i], embeddings[j]).item()
            similarities.append(sim)

    mean_sim = np.mean(similarities)
    std_sim = np.std(similarities)

    # Embeddings collapsed if mean similarity > 0.5
    status = "PASS" if mean_sim < 0.5 else "WARN" if mean_sim < 0.7 else "FAIL"

    return {
        "sample_size": sample_size,
        "mean_similarity": mean_sim,
        "std_similarity": std_sim,
        "status": status,
        "note": "Mean similarity should be < 0.5 for diverse embeddings"
    }


def print_results(results, title):
    """Pretty print test results."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")

    if isinstance(results, dict):
        for key, value in results.items():
            print(f"  {key}: {value}")
    else:
        for result in results:
            if len(result) == 5:
                r1, r2, sim, status, note = result
                if sim is None:
                    print(f"  [{status}] {r1} <-> {r2}: {note}")
                else:
                    print(f"  [{status}] {r1} <-> {r2}: {sim:.3f} ({note})")
            elif len(result) == 4:
                name, value, status, note = result
                if value is None:
                    print(f"  [{status}] {name}: {note}")
                else:
                    print(f"  [{status}] {name}: {value:.3f} ({note})")


def main():
    parser = argparse.ArgumentParser(description="Validate root embeddings")
    parser.add_argument('--model', type=Path,
                       default=Path('models/root_embeddings/best_model.pt'),
                       help='Path to trained model')
    parser.add_argument('--fundamento', type=Path,
                       default=Path('data/vocabularies/fundamento_roots.json'),
                       help='Path to Fundamento roots')

    args = parser.parse_args()

    print("Loading model...")
    embedding, root_to_idx, idx_to_root = load_model(args.model)
    print(f"Loaded {len(root_to_idx)} root embeddings (dim={embedding.embedding_dim})")

    # Run tests
    print("\n" + "="*60)
    print("ROOT EMBEDDING VALIDATION")
    print("="*60)

    # Test 1: Semantic similarity
    results = test_semantic_similarity(embedding, root_to_idx)
    print_results(results, "Test 1: Semantic Similarity")
    pass_count = sum(1 for r in results if r[3] == "PASS")
    print(f"\nPassed: {pass_count}/{len(results)}")

    # Test 2: Anti-patterns
    results = test_anti_patterns(embedding, root_to_idx)
    print_results(results, "Test 2: Anti-Patterns (Dissimilarity)")
    pass_count = sum(1 for r in results if r[3] == "PASS")
    print(f"\nPassed: {pass_count}/{len(results)}")

    # Test 3: Cluster coherence
    results = test_cluster_coherence(embedding, root_to_idx)
    print_results(results, "Test 3: Cluster Coherence")
    pass_count = sum(1 for r in results if r[3] == "PASS")
    print(f"\nPassed: {pass_count}/{len(results)}")

    # Test 4: Fundamento coverage
    if args.fundamento.exists():
        results = test_fundamento_coverage(embedding, root_to_idx, args.fundamento)
        print_results(results, "Test 4: Fundamento Coverage")
    else:
        print(f"\nSkipping Test 4: Fundamento file not found: {args.fundamento}")

    # Test 5: Embedding collapse
    results = test_embedding_collapse(embedding, root_to_idx)
    print_results(results, "Test 5: Embedding Collapse Check")

    print("\n" + "="*60)
    print("VALIDATION COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
