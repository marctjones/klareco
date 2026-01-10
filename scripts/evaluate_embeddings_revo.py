#!/usr/bin/env python3
"""
Evaluate embedding quality using ReVo semantic relations.

Uses 1,943 synonym pairs and 173 antonym pairs from ReVo (Reta Vortaro)
to measure how well current compositional embeddings capture semantic similarity.

Tests:
1. Synonym Similarity - Mean cosine similarity between synonym pairs
2. Synonym vs Random Gap - Difference between synonym and random word similarities
3. Antonym Discrimination - Whether antonyms have lower similarity than synonyms

Usage:
    python scripts/evaluate_embeddings_revo.py
    python scripts/evaluate_embeddings_revo.py --verbose
    python scripts/evaluate_embeddings_revo.py --output results.json
"""

import argparse
import json
import sys
import time
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field, asdict

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class EvaluationResults:
    """Results from embedding evaluation."""
    # Synonym test
    synonym_mean_similarity: float = 0.0
    synonym_std_similarity: float = 0.0
    synonym_pairs_tested: int = 0
    synonym_pairs_high_quality: int = 0  # sim > 0.7
    synonym_pairs_medium_quality: int = 0  # 0.5 < sim <= 0.7
    synonym_pairs_low_quality: int = 0  # sim <= 0.5

    # Baseline test
    random_mean_similarity: float = 0.0
    synonym_random_gap: float = 0.0

    # Antonym test
    antonym_mean_similarity: float = 0.0
    antonym_pairs_tested: int = 0
    synonym_antonym_gap: float = 0.0

    # Coverage
    roots_in_vocab: int = 0
    roots_missing: int = 0
    missing_roots: List[str] = field(default_factory=list)

    # Timing
    evaluation_time_seconds: float = 0.0


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def load_revo_relations(revo_path: Path) -> Dict:
    """Load ReVo semantic relations."""
    with open(revo_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_embeddings(model_type: str = 'hybrid'):
    """
    Load the embedding model.

    Args:
        model_type: 'hybrid' (both linguistic + topical),
                   'linguistic' (only linguistic),
                   'topical' (only topical)

    Returns:
        Embedding model with get_root_embedding(root) method
    """
    linguistic_path = PROJECT_ROOT / "models" / "root_embeddings" / "best_model.pt"
    topical_path = PROJECT_ROOT / "models" / "topical_embeddings" / "best_model.pt"

    if model_type == 'hybrid':
        # Load hybrid model (combines both)
        from klareco.embeddings.hybrid_embeddings import HybridEmbeddings

        if not linguistic_path.exists():
            raise FileNotFoundError(f"Linguistic model not found: {linguistic_path}")
        if not topical_path.exists():
            raise FileNotFoundError(f"Topical model not found: {topical_path}")

        return HybridEmbeddings.from_checkpoints(
            linguistic_checkpoint=linguistic_path,
            topical_checkpoint=topical_path,
            pad_missing=True,
            default_mode='hybrid'
        )

    elif model_type == 'linguistic':
        # Load only linguistic embeddings
        from klareco.embeddings.linguistic_embeddings import LinguisticEmbeddings

        if not linguistic_path.exists():
            raise FileNotFoundError(f"Linguistic model not found: {linguistic_path}")

        return LinguisticEmbeddings.from_checkpoint(linguistic_path)

    elif model_type == 'topical':
        # Load only topical embeddings
        from klareco.embeddings.topical_embeddings import TopicalEmbeddings

        if not topical_path.exists():
            raise FileNotFoundError(f"Topical model not found: {topical_path}")

        return TopicalEmbeddings.from_checkpoint(topical_path)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def get_root_embedding(embedder, root: str) -> Optional[np.ndarray]:
    """Get embedding for a root word."""
    try:
        # Try to get root embedding directly
        if hasattr(embedder, 'get_root_embedding'):
            emb = embedder.get_root_embedding(root)
            if emb is not None:
                return emb.detach().cpu().numpy() if hasattr(emb, 'detach') else np.array(emb)

        # Fall back to embedding the root as a word
        if hasattr(embedder, 'embed'):
            # Add 'o' ending to make it a noun
            emb = embedder.embed(root + 'o')
            if emb is not None:
                return emb.detach().cpu().numpy() if hasattr(emb, 'detach') else np.array(emb)

        return None
    except Exception:
        return None


def evaluate_synonyms(
    embedder,
    synonym_pairs: List[Tuple[str, str]],
    verbose: bool = False,
) -> Tuple[List[float], List[str]]:
    """Evaluate synonym similarity."""
    similarities = []
    missing_roots = set()

    for root1, root2 in synonym_pairs:
        emb1 = get_root_embedding(embedder, root1)
        emb2 = get_root_embedding(embedder, root2)

        if emb1 is None:
            missing_roots.add(root1)
            continue
        if emb2 is None:
            missing_roots.add(root2)
            continue

        sim = cosine_similarity(emb1, emb2)
        similarities.append(sim)

        if verbose and sim < 0.3:
            print(f"  Low similarity: {root1} <-> {root2} = {sim:.3f}")

    return similarities, list(missing_roots)


def evaluate_random_baseline(
    embedder,
    synonym_pairs: List[Tuple[str, str]],
    all_roots: List[str],
    num_samples: int = 100,
) -> List[float]:
    """Compute similarity between random word pairs as baseline."""
    random_sims = []

    # Get embeddings for a sample of roots
    root_embeddings = {}
    for root in random.sample(all_roots, min(500, len(all_roots))):
        emb = get_root_embedding(embedder, root)
        if emb is not None:
            root_embeddings[root] = emb

    if len(root_embeddings) < 20:
        return []

    roots_list = list(root_embeddings.keys())

    for _ in range(num_samples):
        r1, r2 = random.sample(roots_list, 2)
        sim = cosine_similarity(root_embeddings[r1], root_embeddings[r2])
        random_sims.append(sim)

    return random_sims


def evaluate_antonyms(
    embedder,
    antonym_pairs: List[Tuple[str, str]],
    verbose: bool = False,
) -> List[float]:
    """Evaluate antonym similarity."""
    similarities = []

    for root1, root2 in antonym_pairs:
        emb1 = get_root_embedding(embedder, root1)
        emb2 = get_root_embedding(embedder, root2)

        if emb1 is None or emb2 is None:
            continue

        sim = cosine_similarity(emb1, emb2)
        similarities.append(sim)

        if verbose and sim > 0.7:
            print(f"  High antonym similarity: {root1} <-> {root2} = {sim:.3f}")

    return similarities


def run_evaluation(
    revo_path: Path,
    model_type: str = 'hybrid',
    verbose: bool = False,
) -> EvaluationResults:
    """Run full embedding evaluation."""
    start_time = time.time()
    results = EvaluationResults()

    print("Loading ReVo semantic relations...")
    revo_data = load_revo_relations(revo_path)

    synonym_pairs = [tuple(p) for p in revo_data['relations']['synonym']]
    antonym_pairs = [tuple(p) for p in revo_data['relations']['antonym']]

    print(f"  Synonym pairs: {len(synonym_pairs)}")
    print(f"  Antonym pairs: {len(antonym_pairs)}")

    print("\nLoading embedding model...")
    embedder = load_embeddings(model_type)
    print(f"  Model loaded: {model_type}")

    # Collect all roots for random baseline
    all_roots = set()
    for r1, r2 in synonym_pairs:
        all_roots.add(r1)
        all_roots.add(r2)
    for r1, r2 in antonym_pairs:
        all_roots.add(r1)
        all_roots.add(r2)
    all_roots = list(all_roots)

    # Test 1: Synonym Similarity
    print("\n" + "=" * 60)
    print("TEST 1: Synonym Similarity")
    print("=" * 60)

    syn_sims, missing = evaluate_synonyms(embedder, synonym_pairs, verbose)

    if syn_sims:
        results.synonym_mean_similarity = float(np.mean(syn_sims))
        results.synonym_std_similarity = float(np.std(syn_sims))
        results.synonym_pairs_tested = len(syn_sims)
        results.synonym_pairs_high_quality = sum(1 for s in syn_sims if s > 0.7)
        results.synonym_pairs_medium_quality = sum(1 for s in syn_sims if 0.5 < s <= 0.7)
        results.synonym_pairs_low_quality = sum(1 for s in syn_sims if s <= 0.5)

        print(f"\n  Pairs tested: {results.synonym_pairs_tested}")
        print(f"  Mean similarity: {results.synonym_mean_similarity:.3f}")
        print(f"  Std deviation: {results.synonym_std_similarity:.3f}")
        print(f"  High quality (>0.7): {results.synonym_pairs_high_quality} ({results.synonym_pairs_high_quality/len(syn_sims)*100:.1f}%)")
        print(f"  Medium quality (0.5-0.7): {results.synonym_pairs_medium_quality} ({results.synonym_pairs_medium_quality/len(syn_sims)*100:.1f}%)")
        print(f"  Low quality (<0.5): {results.synonym_pairs_low_quality} ({results.synonym_pairs_low_quality/len(syn_sims)*100:.1f}%)")

    results.missing_roots = missing[:20]  # Keep first 20
    results.roots_missing = len(missing)
    results.roots_in_vocab = len(all_roots) - len(missing)

    print(f"\n  Roots in vocab: {results.roots_in_vocab}")
    print(f"  Roots missing: {results.roots_missing}")
    if missing and verbose:
        print(f"  Sample missing: {missing[:10]}")

    # Test 2: Random Baseline
    print("\n" + "=" * 60)
    print("TEST 2: Synonym vs Random Baseline")
    print("=" * 60)

    random_sims = evaluate_random_baseline(embedder, synonym_pairs, all_roots)

    if random_sims:
        results.random_mean_similarity = float(np.mean(random_sims))
        results.synonym_random_gap = results.synonym_mean_similarity - results.random_mean_similarity

        print(f"\n  Random pairs mean: {results.random_mean_similarity:.3f}")
        print(f"  Synonym mean: {results.synonym_mean_similarity:.3f}")
        print(f"  Gap: {results.synonym_random_gap:.3f}")

        if results.synonym_random_gap > 0.2:
            print("  Status: GOOD (gap > 0.2)")
        elif results.synonym_random_gap > 0.1:
            print("  Status: MODERATE (0.1 < gap < 0.2)")
        else:
            print("  Status: POOR (gap < 0.1)")

    # Test 3: Antonym Discrimination
    print("\n" + "=" * 60)
    print("TEST 3: Antonym Discrimination")
    print("=" * 60)

    ant_sims = evaluate_antonyms(embedder, antonym_pairs, verbose)

    if ant_sims:
        results.antonym_mean_similarity = float(np.mean(ant_sims))
        results.antonym_pairs_tested = len(ant_sims)
        results.synonym_antonym_gap = results.synonym_mean_similarity - results.antonym_mean_similarity

        print(f"\n  Pairs tested: {results.antonym_pairs_tested}")
        print(f"  Antonym mean similarity: {results.antonym_mean_similarity:.3f}")
        print(f"  Synonym mean similarity: {results.synonym_mean_similarity:.3f}")
        print(f"  Gap (syn - ant): {results.synonym_antonym_gap:.3f}")

        if results.synonym_antonym_gap > 0.1:
            print("  Status: GOOD (synonyms more similar than antonyms)")
        elif results.synonym_antonym_gap > 0:
            print("  Status: WEAK (small difference)")
        else:
            print("  Status: FAIL (antonyms more similar than synonyms!)")

    results.evaluation_time_seconds = time.time() - start_time

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n  Synonym similarity: {results.synonym_mean_similarity:.3f} (target: >0.7)")
    print(f"  Synonym-random gap: {results.synonym_random_gap:.3f} (target: >0.2)")
    print(f"  Synonym-antonym gap: {results.synonym_antonym_gap:.3f} (target: >0.1)")
    print(f"  Vocabulary coverage: {results.roots_in_vocab}/{results.roots_in_vocab + results.roots_missing} ({results.roots_in_vocab/(results.roots_in_vocab + results.roots_missing)*100:.1f}%)")
    print(f"\n  Evaluation time: {results.evaluation_time_seconds:.1f}s")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate embeddings using ReVo semantic relations'
    )
    parser.add_argument('--revo', type=Path,
                        default=PROJECT_ROOT / 'data' / 'raw' / 'eo' / 'dictionaries' / 'revo' / 'revo_semantic_relations.json',
                        help='Path to ReVo relations JSON')
    parser.add_argument('--model', type=str, default='hybrid',
                        choices=['hybrid', 'linguistic', 'topical'],
                        help='Embedding model type (default: hybrid)')
    parser.add_argument('--output', type=Path,
                        help='Path to save results JSON')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed output')

    args = parser.parse_args()

    if not args.revo.exists():
        print(f"ERROR: ReVo file not found: {args.revo}")
        sys.exit(1)

    print("=" * 60)
    print("EMBEDDING EVALUATION WITH REVO SEMANTIC RELATIONS")
    print("=" * 60)

    results = run_evaluation(args.revo, args.model, args.verbose)

    # Save results
    if args.output:
        output_path = args.output
    else:
        output_dir = PROJECT_ROOT / 'benchmark_results' / 'embeddings'
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f'revo_evaluation_{time.strftime("%Y%m%d_%H%M%S")}.json'

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(asdict(results), f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
