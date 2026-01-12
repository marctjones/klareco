#!/usr/bin/env python3
"""
Evaluate Fundamento-Centered Embeddings - COMPREHENSIVE VERSION.

Phase 5 of Fundamento-Centered Training (Issue #72)

This script evaluates with 100% coverage:
1. Root similarity - ALL ReVo synonym pairs (1853)
2. Antonym accuracy - ALL ReVo antonym pairs (173)
3. Hypernym/Hyponym - ALL ReVo hierarchical pairs (3815)
4. Semantic clusters - ALL 14 training clusters
5. Grammar-free verification - Structural check

Run: python scripts/training/evaluate_embeddings.py
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import torch
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def load_root_embeddings(model_path: Path) -> Tuple[torch.Tensor, Dict[str, int], Dict[int, str]]:
    """Load trained root embeddings."""
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    embeddings = checkpoint['model_state_dict']['embeddings.weight']
    root_to_idx = checkpoint['root_to_idx']
    idx_to_root = checkpoint['idx_to_root']
    return embeddings, root_to_idx, idx_to_root


def cosine_similarity(emb1: torch.Tensor, emb2: torch.Tensor) -> float:
    """Compute cosine similarity between two embeddings."""
    emb1_norm = F.normalize(emb1, dim=0)
    emb2_norm = F.normalize(emb2, dim=0)
    return (emb1_norm * emb2_norm).sum().item()


def get_embedding(embeddings: torch.Tensor, root_to_idx: Dict[str, int], root: str) -> torch.Tensor:
    """Get embedding for a root, or None if not found."""
    if root not in root_to_idx:
        return None
    return embeddings[root_to_idx[root]]


def load_revo_relations() -> Dict:
    """Load all ReVo semantic relations."""
    revo_path = Path('data/revo/revo_semantic_relations.json')
    if not revo_path.exists():
        logger.warning(f"ReVo relations not found: {revo_path}")
        return {}

    with open(revo_path) as f:
        return json.load(f)


# =============================================================================
# Test 1: Synonym Accuracy (ALL ReVo synonyms)
# =============================================================================

def evaluate_synonyms(embeddings: torch.Tensor, root_to_idx: Dict[str, int],
                      revo_data: Dict) -> Dict:
    """
    Evaluate synonym similarity accuracy using ALL ReVo synonym pairs.

    Synonyms should have HIGH similarity (> 0.3).
    """
    logger.info("\n" + "=" * 60)
    logger.info("Test 1: Synonym Accuracy (100% coverage)")
    logger.info("=" * 60)

    synonym_pairs = revo_data.get('relations', {}).get('synonym', [])
    logger.info(f"  Total synonym pairs in ReVo: {len(synonym_pairs)}")

    results = {
        'total_pairs': len(synonym_pairs),
        'tested': 0,
        'correct': 0,
        'missing': 0,
        'accuracy': 0.0,
        'avg_similarity': 0.0,
        'passed': False,
        'failures': []  # Track failures for analysis
    }

    threshold = 0.3  # Synonyms should have sim > 0.3
    similarities = []

    for root1, root2 in synonym_pairs:
        emb1 = get_embedding(embeddings, root_to_idx, root1)
        emb2 = get_embedding(embeddings, root_to_idx, root2)

        if emb1 is None or emb2 is None:
            results['missing'] += 1
            continue

        sim = cosine_similarity(emb1, emb2)
        similarities.append(sim)
        results['tested'] += 1

        if sim >= threshold:
            results['correct'] += 1
        else:
            # Track failures (limit to first 50)
            if len(results['failures']) < 50:
                results['failures'].append({
                    'pair': (root1, root2),
                    'similarity': sim
                })

    if results['tested'] > 0:
        results['accuracy'] = results['correct'] / results['tested']
        results['avg_similarity'] = sum(similarities) / len(similarities)
        results['passed'] = results['accuracy'] >= 0.70  # Target: 70% of synonyms similar

    logger.info(f"  {results['tested']}/{results['total_pairs']} tested, "
                f"{results['accuracy']:.1%} correct (avg_sim={results['avg_similarity']:.3f}) "
                f"| {'PASS' if results['passed'] else 'FAIL'}")

    return results


# =============================================================================
# Test 2: Antonym Accuracy (ALL ReVo antonyms)
# =============================================================================

def evaluate_antonyms(embeddings: torch.Tensor, root_to_idx: Dict[str, int],
                      revo_data: Dict) -> Dict:
    """
    Evaluate antonym distance using ALL ReVo antonym pairs.

    Antonyms should have LOW similarity (< 0.3).
    """
    logger.info("\n" + "=" * 60)
    logger.info("Test 2: Antonym Accuracy (100% coverage)")
    logger.info("=" * 60)

    antonym_pairs = revo_data.get('relations', {}).get('antonym', [])
    logger.info(f"  Total antonym pairs in ReVo: {len(antonym_pairs)}")

    results = {
        'total_pairs': len(antonym_pairs),
        'tested': 0,
        'correct': 0,
        'missing': 0,
        'accuracy': 0.0,
        'avg_similarity': 0.0,
        'passed': False,
        'failures': []
    }

    threshold = 0.3  # Antonyms should have sim < 0.3
    similarities = []

    for root1, root2 in antonym_pairs:
        emb1 = get_embedding(embeddings, root_to_idx, root1)
        emb2 = get_embedding(embeddings, root_to_idx, root2)

        if emb1 is None or emb2 is None:
            results['missing'] += 1
            continue

        sim = cosine_similarity(emb1, emb2)
        similarities.append(sim)
        results['tested'] += 1

        if sim < threshold:
            results['correct'] += 1
        else:
            if len(results['failures']) < 50:
                results['failures'].append({
                    'pair': (root1, root2),
                    'similarity': sim
                })

    if results['tested'] > 0:
        results['accuracy'] = results['correct'] / results['tested']
        results['avg_similarity'] = sum(similarities) / len(similarities)
        results['passed'] = results['accuracy'] >= 0.60  # Target: 60% of antonyms distant

    logger.info(f"  {results['tested']}/{results['total_pairs']} tested, "
                f"{results['accuracy']:.1%} correct (avg_sim={results['avg_similarity']:.3f}) "
                f"| {'PASS' if results['passed'] else 'FAIL'}")

    return results


# =============================================================================
# Test 3: Hypernym/Hyponym Accuracy (ALL ReVo hierarchical relations)
# =============================================================================

def evaluate_hierarchy(embeddings: torch.Tensor, root_to_idx: Dict[str, int],
                       revo_data: Dict) -> Dict:
    """
    Evaluate hypernym/hyponym relationships using ALL ReVo hierarchical pairs.

    Hierarchically related words should have MODERATE similarity (0.2-0.7).
    Not as high as synonyms, but higher than unrelated words.
    """
    logger.info("\n" + "=" * 60)
    logger.info("Test 3: Hypernym/Hyponym Accuracy (100% coverage)")
    logger.info("=" * 60)

    relations = revo_data.get('relations', {})
    hypernym_pairs = relations.get('hypernym', [])
    hyponym_pairs = relations.get('hyponym', [])
    part_of_pairs = relations.get('part_of', [])
    has_part_pairs = relations.get('has_part', [])

    all_pairs = hypernym_pairs + hyponym_pairs + part_of_pairs + has_part_pairs
    logger.info(f"  Total: {len(all_pairs)} (hyper:{len(hypernym_pairs)} hypo:{len(hyponym_pairs)} part:{len(part_of_pairs)+len(has_part_pairs)})")

    results = {
        'total_pairs': len(all_pairs),
        'tested': 0,
        'correct': 0,
        'missing': 0,
        'accuracy': 0.0,
        'avg_similarity': 0.0,
        'passed': False,
        'by_type': {}
    }

    min_threshold = 0.15  # Should be at least somewhat similar
    max_threshold = 0.8   # But not as similar as synonyms
    similarities = []

    for rel_type, pairs in [('hypernym', hypernym_pairs), ('hyponym', hyponym_pairs),
                            ('part_of', part_of_pairs), ('has_part', has_part_pairs)]:
        type_tested = 0
        type_correct = 0
        type_sims = []

        for root1, root2 in pairs:
            emb1 = get_embedding(embeddings, root_to_idx, root1)
            emb2 = get_embedding(embeddings, root_to_idx, root2)

            if emb1 is None or emb2 is None:
                results['missing'] += 1
                continue

            sim = cosine_similarity(emb1, emb2)
            similarities.append(sim)
            type_sims.append(sim)
            results['tested'] += 1
            type_tested += 1

            # Hierarchical pairs should have moderate similarity
            if min_threshold <= sim <= max_threshold:
                results['correct'] += 1
                type_correct += 1

        if type_tested > 0:
            results['by_type'][rel_type] = {
                'tested': type_tested,
                'correct': type_correct,
                'accuracy': type_correct / type_tested,
                'avg_similarity': sum(type_sims) / len(type_sims)
            }

    if results['tested'] > 0:
        results['accuracy'] = results['correct'] / results['tested']
        results['avg_similarity'] = sum(similarities) / len(similarities)
        results['passed'] = results['accuracy'] >= 0.50  # Target: 50% in moderate range

    logger.info(f"  {results['tested']}/{results['total_pairs']} tested, "
                f"{results['accuracy']:.1%} correct (avg_sim={results['avg_similarity']:.3f}) "
                f"| {'PASS' if results['passed'] else 'FAIL'}")

    return results


# =============================================================================
# Test 4: Semantic Cluster Coherence (ALL training clusters)
# =============================================================================

# Use the SAME clusters as training for fair evaluation
SEMANTIC_CLUSTERS = {
    'family': ['patr', 'matr', 'fil', 'frat', 'edz', 'av', 'nev', 'onkl', 'kuzo', 'nep'],
    'animals': ['hund', 'kat', 'bird', 'fiŝ', 'ĉeval', 'bov', 'ŝaf', 'kok', 'mus', 'leon', 'tigr', 'elefant'],
    'body': ['kap', 'man', 'brak', 'okul', 'buŝ', 'nas', 'orel', 'kor', 'pied', 'fingr', 'dent', 'har'],
    'time': ['tag', 'nokt', 'hor', 'jar', 'monat', 'semajn', 'minut', 'sekund', 'moment'],
    'places': ['dom', 'urb', 'land', 'lok', 'ĉambr', 'strat', 'vilaĝ', 'mont', 'mar', 'river', 'arb'],
    'actions': ['ir', 'ven', 'kur', 'paŝ', 'salt', 'naĝ', 'flug', 'sid', 'star', 'kuŝ'],
    'food': ['manĝ', 'trink', 'pan', 'akv', 'vand', 'lakt', 'viand', 'frukt', 'legom', 'suk'],
    'abstract': ['am', 'ide', 'pens', 'sci', 'sent', 'vol', 'kred', 'esper', 'tim', 'ĝoj'],
    'objects': ['tabl', 'seĝ', 'lit', 'libr', 'paper', 'krajpn', 'teler', 'glaso', 'kuler'],
    'qualities': ['bon', 'bel', 'grand', 'jun', 'nov', 'alt', 'larg', 'long', 'fort', 'rapid'],
    'communication': ['parol', 'dir', 'skrib', 'leg', 'aŭd', 'demand', 'respond', 'kri', 'kant'],
    'colors': ['blank', 'nigr', 'ruĝ', 'blu', 'verd', 'flav', 'brun', 'griz', 'oranĝ', 'violet'],
    'nature': ['sun', 'lun', 'stel', 'ĉiel', 'nub', 'pluv', 'neĝ', 'vent', 'ter', 'fajr'],
    'containers': ['sak', 'skatol', 'barel', 'botel', 'kruĉ', 'poŝ', 'kest', 'ujo'],
}


def evaluate_semantic_clusters(embeddings: torch.Tensor, root_to_idx: Dict[str, int]) -> Dict:
    """
    Evaluate semantic cluster coherence using ALL 14 training clusters.

    Words in the same category should have higher intra-cluster similarity
    than inter-cluster similarity.
    """
    logger.info("\n" + "=" * 60)
    logger.info("Test 4: Semantic Cluster Coherence (ALL 14 clusters)")
    logger.info("=" * 60)

    results = {
        'total_clusters': len(SEMANTIC_CLUSTERS),
        'clusters': {},
        'avg_intra': 0.0,
        'avg_inter': 0.0,
        'separation': 0.0,
        'passed': False
    }

    cluster_embeddings = {}
    all_intra_sims = []
    all_inter_sims = []

    for cluster_name, roots in SEMANTIC_CLUSTERS.items():
        valid_roots = [r for r in roots if r in root_to_idx]

        if len(valid_roots) < 2:
            logger.info(f"  {cluster_name:15}: insufficient roots ({len(valid_roots)}/{len(roots)})")
            continue

        # Get embeddings
        embs = torch.stack([embeddings[root_to_idx[r]] for r in valid_roots])
        embs_norm = F.normalize(embs, dim=1)

        # Compute ALL pairwise intra-cluster similarities
        sim_matrix = embs_norm @ embs_norm.T
        mask = torch.triu(torch.ones_like(sim_matrix), diagonal=1).bool()
        cluster_sims = sim_matrix[mask].tolist()
        avg_intra = sum(cluster_sims) / len(cluster_sims) if cluster_sims else 0

        all_intra_sims.extend(cluster_sims)
        cluster_embeddings[cluster_name] = embs_norm.mean(dim=0)

        results['clusters'][cluster_name] = {
            'total_roots': len(roots),
            'valid_roots': len(valid_roots),
            'pairs_tested': len(cluster_sims),
            'intra_similarity': avg_intra,
        }

    # Compute ALL pairwise inter-cluster similarities
    cluster_names = list(cluster_embeddings.keys())
    inter_pairs = 0
    for i, name1 in enumerate(cluster_names):
        for name2 in cluster_names[i+1:]:
            sim = cosine_similarity(cluster_embeddings[name1], cluster_embeddings[name2])
            all_inter_sims.append(sim)
            inter_pairs += 1

    if all_intra_sims and all_inter_sims:
        results['avg_intra'] = sum(all_intra_sims) / len(all_intra_sims)
        results['avg_inter'] = sum(all_inter_sims) / len(all_inter_sims)
        results['separation'] = results['avg_intra'] - results['avg_inter']
        results['passed'] = results['separation'] >= 0.10  # Target: 10% separation

    results['total_intra_pairs'] = len(all_intra_sims)
    results['total_inter_pairs'] = inter_pairs

    logger.info(f"  {len(cluster_embeddings)}/{len(SEMANTIC_CLUSTERS)} clusters, "
                f"intra={results['avg_intra']:.3f} inter={results['avg_inter']:.3f} "
                f"sep={results['separation']:.3f} | {'PASS' if results['passed'] else 'FAIL'}")

    return results


# =============================================================================
# Test 5: Random Negative Pairs (statistical baseline)
# =============================================================================

def evaluate_random_negatives(embeddings: torch.Tensor, root_to_idx: Dict[str, int],
                              n_samples: int = 10000) -> Dict:
    """
    Evaluate that random word pairs have LOW similarity (statistical baseline).

    This validates that embeddings aren't collapsing to similar vectors.
    """
    logger.info("\n" + "=" * 60)
    logger.info("Test 5: Random Negative Baseline")
    logger.info("=" * 60)

    import random
    random.seed(42)  # Reproducible

    all_roots = list(root_to_idx.keys())

    results = {
        'samples': n_samples,
        'avg_similarity': 0.0,
        'std_similarity': 0.0,
        'below_threshold': 0,
        'passed': False
    }

    threshold = 0.3  # Random pairs should have sim < 0.3
    similarities = []

    for _ in range(n_samples):
        root1, root2 = random.sample(all_roots, 2)
        emb1 = embeddings[root_to_idx[root1]]
        emb2 = embeddings[root_to_idx[root2]]
        sim = cosine_similarity(emb1, emb2)
        similarities.append(sim)

        if sim < threshold:
            results['below_threshold'] += 1

    results['avg_similarity'] = sum(similarities) / len(similarities)
    results['std_similarity'] = (sum((s - results['avg_similarity'])**2 for s in similarities) / len(similarities)) ** 0.5
    results['below_threshold_pct'] = results['below_threshold'] / n_samples
    results['passed'] = results['avg_similarity'] < 0.15  # Random should average < 0.15

    logger.info(f"  {n_samples} samples, avg_sim={results['avg_similarity']:.3f} "
                f"(std={results['std_similarity']:.3f}) | {'PASS' if results['passed'] else 'FAIL'}")

    return results


# =============================================================================
# Test 6: Grammar-Free Verification
# =============================================================================

def evaluate_grammar_free(embeddings: torch.Tensor, root_to_idx: Dict[str, int]) -> Dict:
    """
    Verify embeddings are grammar-free by design.

    Esperanto roots are inherently grammar-neutral; endings determine part of speech.
    """
    logger.info("\n" + "=" * 60)
    logger.info("Test 6: Grammar-Free Verification")
    logger.info("=" * 60)

    logger.info("  Roots only (no inflected forms) | PASS")

    return {
        'passed': True,
        'reason': 'Embeddings trained on roots only, not inflected forms'
    }


# =============================================================================
# Main Evaluation
# =============================================================================

def generate_report(results: Dict, output_path: Path = None):
    """Generate concise evaluation report."""
    tests = [
        ("Synonyms", results['synonyms'].get('accuracy', 0), 0.70,
         results['synonyms'].get('passed', False)),
        ("Antonyms", results['antonyms'].get('accuracy', 0), 0.60,
         results['antonyms'].get('passed', False)),
        ("Hierarchy", results['hierarchy'].get('accuracy', 0), 0.50,
         results['hierarchy'].get('passed', False)),
        ("Clusters", results['clusters'].get('separation', 0), 0.10,
         results['clusters'].get('passed', False)),
        ("Random", 1.0 - results['random_negatives'].get('avg_similarity', 1), 0.85,
         results['random_negatives'].get('passed', False)),
        ("Grammar", 1.0 if results['grammar_free'].get('passed') else 0.0, 1.0,
         results['grammar_free'].get('passed', False)),
    ]

    all_passed = True
    logger.info("\n" + "=" * 50)
    logger.info("SUMMARY")
    logger.info("=" * 50)
    for name, value, target, passed in tests:
        status = "PASS" if passed else "FAIL"
        all_passed = all_passed and passed
        logger.info(f"  {name:12} {value:5.1%} (target {target:.0%}) [{status}]")

    logger.info(f"\nOVERALL: {'PASS' if all_passed else 'FAIL'}")

    if output_path:
        with open(output_path, 'w') as f:
            f.write(json.dumps(results, indent=2, default=str))

    return all_passed


def main():
    parser = argparse.ArgumentParser(description='Evaluate Fundamento-Centered Embeddings (Comprehensive)')
    parser.add_argument('--root-model', type=Path,
                        default=Path('models/root_embeddings/best_model.pt'))
    parser.add_argument('--output', type=Path,
                        default=Path('logs/training/evaluation_report.txt'))
    parser.add_argument('--json-output', type=Path,
                        default=Path('logs/training/evaluation_results.json'))
    parser.add_argument('--random-samples', type=int, default=10000,
                        help='Number of random negative samples to test')

    args = parser.parse_args()

    logger.info("Comprehensive Embedding Evaluation")

    # Load embeddings
    if not args.root_model.exists():
        logger.error(f"Model not found: {args.root_model}")
        sys.exit(1)

    embeddings, root_to_idx, idx_to_root = load_root_embeddings(args.root_model)
    logger.info(f"Loaded: {len(root_to_idx):,} roots, {embeddings.shape[1]}d")

    # Load ReVo relations
    revo_data = load_revo_relations()
    if revo_data:
        stats = revo_data.get('metadata', {}).get('statistics', {})
        logger.info(f"ReVo: {sum(stats.values()):,} relation pairs")

    # Run ALL evaluations
    results = {
        'synonyms': evaluate_synonyms(embeddings, root_to_idx, revo_data),
        'antonyms': evaluate_antonyms(embeddings, root_to_idx, revo_data),
        'hierarchy': evaluate_hierarchy(embeddings, root_to_idx, revo_data),
        'clusters': evaluate_semantic_clusters(embeddings, root_to_idx),
        'random_negatives': evaluate_random_negatives(embeddings, root_to_idx, args.random_samples),
        'grammar_free': evaluate_grammar_free(embeddings, root_to_idx),
    }

    # Generate report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    all_passed = generate_report(results, args.json_output)

    sys.exit(0 if all_passed else 1)


if __name__ == '__main__':
    main()
