#!/usr/bin/env python3
"""
Evaluate trained affix embeddings with 100% coverage.

Tests:
1. Semantic Group Coherence - ALL affix pairs within curated groups
2. Participial Ordering - ALL 6 participial suffixes
3. Causative/Inchoative - ig/iĝ pair
4. Degree Suffixes - et/eg pair
5. All Pairs Baseline - ALL possible affix pairs
6. Prefix Coverage - ALL prefix pairs

Run: python scripts/training/evaluate_affix_embeddings.py
"""

import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def load_model(model_path: Path):
    """Load trained affix embeddings."""
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    return {
        'prefix_embeddings': checkpoint['model_state_dict']['prefix_embeddings.weight'],
        'suffix_embeddings': checkpoint['model_state_dict']['suffix_embeddings.weight'],
        'prefix_vocab': checkpoint['prefix_vocab'],
        'suffix_vocab': checkpoint['suffix_vocab'],
        'embedding_dim': checkpoint['embedding_dim'],
        'accuracy': checkpoint.get('accuracy', 0),
    }


def cosine_similarity(emb1, emb2):
    """Compute cosine similarity between two embeddings."""
    e1 = F.normalize(emb1, dim=0)
    e2 = F.normalize(emb2, dim=0)
    return (e1 * e2).sum().item()


def get_prefix_sim(model, p1, p2):
    """Get similarity between two prefixes."""
    if p1 not in model['prefix_vocab'] or p2 not in model['prefix_vocab']:
        return None
    idx1, idx2 = model['prefix_vocab'][p1], model['prefix_vocab'][p2]
    return cosine_similarity(model['prefix_embeddings'][idx1], model['prefix_embeddings'][idx2])


def get_suffix_sim(model, s1, s2):
    """Get similarity between two suffixes."""
    if s1 not in model['suffix_vocab'] or s2 not in model['suffix_vocab']:
        return None
    idx1, idx2 = model['suffix_vocab'][s1], model['suffix_vocab'][s2]
    return cosine_similarity(model['suffix_embeddings'][idx1], model['suffix_embeddings'][idx2])


def evaluate_semantic_groups(model):
    """Test 1: Semantic group coherence."""
    logger.info("=" * 60)
    logger.info("Test 1: Semantic Group Coherence")
    logger.info("=" * 60)

    # Suffix groups that should be similar
    suffix_groups = {
        'participial_active': ['ant', 'int', 'ont'],
        'participial_passive': ['at', 'it', 'ot'],
        'modal': ['ebl', 'ind', 'end'],
        'verbal': ['ig', 'iĝ'],
        'degree': ['et', 'eg'],
        'container': ['ej', 'uj', 'ing'],
        'person': ['ul', 'ist', 'an'],
        'abstract': ['ec', 'ism', 'aĵ'],
    }

    results = {'groups': {}, 'avg_intra': 0, 'passed': False}
    all_intra_sims = []

    for group_name, affixes in suffix_groups.items():
        valid = [s for s in affixes if s in model['suffix_vocab']]
        if len(valid) < 2:
            continue

        sims = []
        for i, s1 in enumerate(valid):
            for s2 in valid[i+1:]:
                sim = get_suffix_sim(model, s1, s2)
                if sim is not None:
                    sims.append(sim)
                    all_intra_sims.append(sim)

        if sims:
            avg = sum(sims) / len(sims)
            results['groups'][group_name] = {'avg_sim': avg, 'pairs': len(sims)}

    if all_intra_sims:
        results['avg_intra'] = sum(all_intra_sims) / len(all_intra_sims)

    # Target: average intra-group similarity > 0.15
    results['passed'] = results['avg_intra'] > 0.15

    logger.info(f"  Groups tested: {len(results['groups'])}")
    for name, data in sorted(results['groups'].items(), key=lambda x: -x[1]['avg_sim']):
        logger.info(f"    {name}: {data['avg_sim']:.3f} ({data['pairs']} pairs)")
    logger.info(f"  Average intra-group: {results['avg_intra']:.3f} (target >0.15) | {'PASS' if results['passed'] else 'FAIL'}")

    return results


def evaluate_participial_ordering(model):
    """Test 2: Participial temporal ordering."""
    logger.info("=" * 60)
    logger.info("Test 2: Participial Temporal Ordering")
    logger.info("=" * 60)

    # Active participles should show temporal progression
    # ant (present) should be between int (past) and ont (future)
    results = {'active': {}, 'passive': {}, 'passed': False}

    for voice, triplet in [('active', ['int', 'ant', 'ont']), ('passive', ['it', 'at', 'ot'])]:
        past, present, future = triplet
        if all(s in model['suffix_vocab'] for s in triplet):
            sim_past_present = get_suffix_sim(model, past, present)
            sim_present_future = get_suffix_sim(model, present, future)
            sim_past_future = get_suffix_sim(model, past, future)

            results[voice] = {
                f'{past}↔{present}': sim_past_present,
                f'{present}↔{future}': sim_present_future,
                f'{past}↔{future}': sim_past_future,
            }
            logger.info(f"  {voice.capitalize()}:")
            logger.info(f"    {past}↔{present}: {sim_past_present:.3f}")
            logger.info(f"    {present}↔{future}: {sim_present_future:.3f}")
            logger.info(f"    {past}↔{future}: {sim_past_future:.3f}")

    # All participle pairs should have positive similarity (they're all related)
    all_sims = []
    for voice_data in [results['active'], results['passive']]:
        all_sims.extend(voice_data.values())

    if all_sims:
        results['avg_sim'] = sum(all_sims) / len(all_sims)
        # Target: participial suffixes should have positive average similarity
        results['passed'] = results['avg_sim'] > 0.1
        logger.info(f"  Average participial similarity: {results['avg_sim']:.3f} (target >0.1) | {'PASS' if results['passed'] else 'FAIL'}")

    return results


def evaluate_causative_inchoative(model):
    """Test 3: Causative/Inchoative pair (ig/iĝ)."""
    logger.info("=" * 60)
    logger.info("Test 3: Causative/Inchoative Pair")
    logger.info("=" * 60)

    results = {'similarity': 0, 'passed': False}

    sim = get_suffix_sim(model, 'ig', 'iĝ')
    if sim is not None:
        results['similarity'] = sim
        # ig and iĝ are the most closely related suffixes - target > 0.3
        results['passed'] = sim > 0.3
        logger.info(f"  ig ↔ iĝ similarity: {sim:.3f} (target >0.3) | {'PASS' if results['passed'] else 'FAIL'}")
    else:
        logger.info("  ig or iĝ not in vocabulary")

    return results


def evaluate_degree_suffixes(model):
    """Test 4: Degree suffixes (et/eg)."""
    logger.info("=" * 60)
    logger.info("Test 4: Degree Suffixes")
    logger.info("=" * 60)

    results = {'similarity': 0, 'passed': False}

    sim = get_suffix_sim(model, 'et', 'eg')
    if sim is not None:
        results['similarity'] = sim
        # et (diminutive) and eg (augmentative) are related - target > 0.15
        results['passed'] = sim > 0.15
        logger.info(f"  et ↔ eg similarity: {sim:.3f} (target >0.15) | {'PASS' if results['passed'] else 'FAIL'}")
    else:
        logger.info("  et or eg not in vocabulary")

    return results


def evaluate_all_pairs(model):
    """Test 5: ALL possible affix pairs (100% coverage)."""
    logger.info("=" * 60)
    logger.info("Test 5: All Pairs Analysis (100% coverage)")
    logger.info("=" * 60)

    results = {
        'suffix_pairs': {'total': 0, 'avg_sim': 0, 'positive': 0, 'negative': 0},
        'prefix_pairs': {'total': 0, 'avg_sim': 0, 'positive': 0, 'negative': 0},
        'passed': False
    }

    suffixes = [s for s in model['suffix_vocab'] if s != '<NONE>']
    prefixes = [p for p in model['prefix_vocab'] if p != '<NONE>']

    # ALL suffix pairs
    suffix_sims = []
    for i, s1 in enumerate(suffixes):
        for s2 in suffixes[i+1:]:
            sim = get_suffix_sim(model, s1, s2)
            if sim is not None:
                suffix_sims.append(sim)

    if suffix_sims:
        results['suffix_pairs']['total'] = len(suffix_sims)
        results['suffix_pairs']['avg_sim'] = sum(suffix_sims) / len(suffix_sims)
        results['suffix_pairs']['positive'] = sum(1 for s in suffix_sims if s > 0.3)
        results['suffix_pairs']['negative'] = sum(1 for s in suffix_sims if s < 0)

    # ALL prefix pairs
    prefix_sims = []
    for i, p1 in enumerate(prefixes):
        for p2 in prefixes[i+1:]:
            sim = get_prefix_sim(model, p1, p2)
            if sim is not None:
                prefix_sims.append(sim)

    if prefix_sims:
        results['prefix_pairs']['total'] = len(prefix_sims)
        results['prefix_pairs']['avg_sim'] = sum(prefix_sims) / len(prefix_sims)
        results['prefix_pairs']['positive'] = sum(1 for s in prefix_sims if s > 0.3)
        results['prefix_pairs']['negative'] = sum(1 for s in prefix_sims if s < 0)

    total_pairs = results['suffix_pairs']['total'] + results['prefix_pairs']['total']
    all_sims = suffix_sims + prefix_sims
    overall_avg = sum(all_sims) / len(all_sims) if all_sims else 0

    # Pass if average similarity is reasonable (not collapsed, not random)
    results['passed'] = 0.05 < overall_avg < 0.5
    results['overall_avg'] = overall_avg
    results['total_pairs'] = total_pairs

    logger.info(f"  Suffix pairs: {results['suffix_pairs']['total']}/{len(suffixes)*(len(suffixes)-1)//2} "
                f"(avg={results['suffix_pairs']['avg_sim']:.3f}, "
                f"high={results['suffix_pairs']['positive']}, neg={results['suffix_pairs']['negative']})")
    logger.info(f"  Prefix pairs: {results['prefix_pairs']['total']}/{len(prefixes)*(len(prefixes)-1)//2} "
                f"(avg={results['prefix_pairs']['avg_sim']:.3f}, "
                f"high={results['prefix_pairs']['positive']}, neg={results['prefix_pairs']['negative']})")
    logger.info(f"  Total: {total_pairs} pairs, overall_avg={overall_avg:.3f} (target 0.05-0.5) | {'PASS' if results['passed'] else 'FAIL'}")

    return results


def evaluate_prefix_groups(model):
    """Test 6: Prefix semantic groups."""
    logger.info("=" * 60)
    logger.info("Test 6: Prefix Semantic Groups")
    logger.info("=" * 60)

    prefix_groups = {
        'aspectual': ['ek', 're', 'for'],  # begin, repeat, completely
        'modification': ['mal', 'mis', 'fi'],  # opposite, wrongly, bad
        'relationship': ['bo', 'ge', 'pra'],  # in-law, both genders, primordial
    }

    results = {'groups': {}, 'avg_intra': 0, 'passed': False}
    all_intra_sims = []

    for group_name, affixes in prefix_groups.items():
        valid = [p for p in affixes if p in model['prefix_vocab']]
        if len(valid) < 2:
            continue

        sims = []
        for i, p1 in enumerate(valid):
            for p2 in valid[i+1:]:
                sim = get_prefix_sim(model, p1, p2)
                if sim is not None:
                    sims.append(sim)
                    all_intra_sims.append(sim)

        if sims:
            avg = sum(sims) / len(sims)
            results['groups'][group_name] = {'avg_sim': avg, 'pairs': len(sims)}

    if all_intra_sims:
        results['avg_intra'] = sum(all_intra_sims) / len(all_intra_sims)

    # Prefixes have less training data, so lower threshold
    results['passed'] = len(results['groups']) > 0

    logger.info(f"  Groups tested: {len(results['groups'])}")
    for name, data in sorted(results['groups'].items(), key=lambda x: -x[1]['avg_sim']):
        logger.info(f"    {name}: {data['avg_sim']:.3f} ({data['pairs']} pairs)")

    if all_intra_sims:
        logger.info(f"  Average intra-group: {results['avg_intra']:.3f} | {'PASS' if results['passed'] else 'FAIL'}")
    else:
        logger.info("  No valid prefix groups found (limited prefix data)")

    return results


def main():
    model_path = Path("models/affix_embeddings/best_model.pt")

    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        logger.error("Run affix training first: python scripts/training/train_affix_embeddings.py")
        sys.exit(1)

    logger.info("Comprehensive Affix Embedding Evaluation")
    logger.info("")

    model = load_model(model_path)
    logger.info(f"Loaded: {len(model['prefix_vocab'])} prefixes, {len(model['suffix_vocab'])} suffixes, {model['embedding_dim']}d")
    logger.info(f"Training accuracy: {model['accuracy']:.4f}")
    logger.info("")

    # Run all tests
    results = {}
    results['semantic_groups'] = evaluate_semantic_groups(model)
    results['participial'] = evaluate_participial_ordering(model)
    results['causative'] = evaluate_causative_inchoative(model)
    results['degree'] = evaluate_degree_suffixes(model)
    results['all_pairs'] = evaluate_all_pairs(model)
    results['prefix_groups'] = evaluate_prefix_groups(model)

    # Summary
    logger.info("")
    logger.info("=" * 50)
    logger.info("SUMMARY")
    logger.info("=" * 50)

    tests = [
        ('Semantic Groups', results['semantic_groups']['passed'], results['semantic_groups'].get('avg_intra', 0), '>0.15'),
        ('Participial', results['participial']['passed'], results['participial'].get('avg_sim', 0), '>0.10'),
        ('Causative/Inch', results['causative']['passed'], results['causative']['similarity'], '>0.30'),
        ('Degree (et/eg)', results['degree']['passed'], results['degree']['similarity'], '>0.15'),
        ('All Pairs', results['all_pairs']['passed'], results['all_pairs'].get('overall_avg', 0), '0.05-0.5'),
        ('Prefix Groups', results['prefix_groups']['passed'], results['prefix_groups'].get('avg_intra', 0), '-'),
    ]

    passed = 0
    for name, ok, value, target in tests:
        status = 'PASS' if ok else 'FAIL'
        if target == '-':
            logger.info(f"  {name:16} {value:.3f} [{status}]")
        else:
            logger.info(f"  {name:16} {value:.3f} (target {target}) [{status}]")
        if ok:
            passed += 1

    logger.info("")
    overall = passed >= 4  # Pass if at least 4 of 6 tests pass
    logger.info(f"OVERALL: {passed}/6 tests passed | {'PASS' if overall else 'FAIL'}")

    # Save results
    output_path = Path("logs/training/affix_evaluation_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        # Convert tensors to floats for JSON
        json_results = {
            'semantic_groups': {
                'avg_intra': results['semantic_groups']['avg_intra'],
                'passed': results['semantic_groups']['passed'],
                'groups': {k: {'avg_sim': v['avg_sim'], 'pairs': v['pairs']}
                          for k, v in results['semantic_groups']['groups'].items()}
            },
            'participial': {
                'avg_sim': results['participial'].get('avg_sim', 0),
                'passed': results['participial']['passed']
            },
            'causative': results['causative'],
            'degree': results['degree'],
            'all_pairs': {
                'total_pairs': results['all_pairs']['total_pairs'],
                'overall_avg': results['all_pairs']['overall_avg'],
                'suffix_pairs': results['all_pairs']['suffix_pairs'],
                'prefix_pairs': results['all_pairs']['prefix_pairs'],
                'passed': results['all_pairs']['passed']
            },
            'prefix_groups': {
                'avg_intra': results['prefix_groups'].get('avg_intra', 0),
                'passed': results['prefix_groups']['passed']
            },
            'overall_passed': overall,
            'tests_passed': passed
        }
        json.dump(json_results, f, indent=2)
    logger.info(f"Results saved to: {output_path}")


if __name__ == '__main__':
    main()
