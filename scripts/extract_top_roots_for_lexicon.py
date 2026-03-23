#!/usr/bin/env python3
"""
Extract Top Roots for Semantic Lexicon

Analyzes SVO triples to find the most frequent roots that need semantic
annotation for the plausibility scorer.

Strategy:
- Extract all unique roots from SVO triples
- Count frequencies (Zipf's law: top 500 = ~80% coverage)
- Output ranked list for manual annotation

Output:
- Top 500 roots with frequencies
- Suggested semantic categories based on affix patterns
- Roots already in Fundamento (priority for annotation)

Usage:
    python scripts/extract_top_roots_for_lexicon.py \
        --svo-triples data/semantic_types/svo_triples_quality.jsonl \
        --output data/lexicons/top_roots_for_annotation.jsonl \
        --limit 500
"""

import argparse
import json
import jsonlines
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_roots_from_triples(svo_file: Path) -> Counter:
    """Extract all roots from SVO triples with frequencies."""
    roots = Counter()

    logger.info(f"Reading SVO triples from {svo_file}")

    with jsonlines.open(svo_file) as reader:
        for triple in reader:
            # Count subject, verb, object roots
            roots[triple['subject_root']] += 1
            roots[triple['verb_root']] += 1
            roots[triple['object_root']] += 1

    logger.info(f"Found {len(roots)} unique roots")
    logger.info(f"Total occurrences: {sum(roots.values())}")

    return roots


def suggest_semantic_category(root: str, verb_count: int, subject_count: int, object_count: int) -> str:
    """Suggest semantic category based on usage patterns."""
    total = verb_count + subject_count + object_count

    if total == 0:
        return 'unknown'

    # Heuristics
    verb_ratio = verb_count / total
    subject_ratio = subject_count / total
    object_ratio = object_count / total

    if verb_ratio > 0.7:
        return 'action/verb'
    elif subject_ratio > 0.6:
        if root.endswith('o') or root.endswith('a'):
            return 'likely_agent/animate'
        return 'subject-heavy'
    elif object_ratio > 0.6:
        return 'likely_patient/object'
    else:
        return 'mixed_usage'


def analyze_by_role(svo_file: Path) -> Dict[str, Dict[str, int]]:
    """Analyze roots by their role (subject/verb/object)."""
    role_counts = {}

    with jsonlines.open(svo_file) as reader:
        for triple in reader:
            # Track role-specific counts
            for role_key, role_name in [
                ('subject_root', 'subject'),
                ('verb_root', 'verb'),
                ('object_root', 'object')
            ]:
                root = triple[role_key]
                if root not in role_counts:
                    role_counts[root] = {'subject': 0, 'verb': 0, 'object': 0}
                role_counts[root][role_name] += 1

    return role_counts


def main():
    parser = argparse.ArgumentParser(description='Extract top roots for semantic lexicon')
    parser.add_argument('--svo-triples', type=Path, required=True,
                       help='Input SVO triples file (JSONL)')
    parser.add_argument('--output', type=Path, required=True,
                       help='Output file for top roots (JSONL)')
    parser.add_argument('--limit', type=int, default=500,
                       help='Number of top roots to extract (default: 500)')

    args = parser.parse_args()

    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Extract root frequencies
    root_counts = extract_roots_from_triples(args.svo_triples)

    # Analyze by role
    logger.info("Analyzing roots by grammatical role...")
    role_counts = analyze_by_role(args.svo_triples)

    # Get top N roots
    top_roots = root_counts.most_common(args.limit)

    logger.info(f"Extracting top {args.limit} roots...")

    # Build output data
    output_data = []
    for rank, (root, total_count) in enumerate(top_roots, 1):
        role_data = role_counts.get(root, {'subject': 0, 'verb': 0, 'object': 0})

        # Suggest category
        category = suggest_semantic_category(
            root,
            role_data['verb'],
            role_data['subject'],
            role_data['object']
        )

        output_data.append({
            'rank': rank,
            'root': root,
            'total_count': total_count,
            'subject_count': role_data['subject'],
            'verb_count': role_data['verb'],
            'object_count': role_data['object'],
            'suggested_category': category,
            'animacy': 'TODO',  # To be filled manually
            'type': 'TODO',     # To be filled manually
            'notes': ''         # Optional notes
        })

    # Write output
    logger.info(f"Writing to {args.output}")
    with jsonlines.open(args.output, 'w') as writer:
        writer.write_all(output_data)

    # Print summary statistics
    print("\n" + "="*70)
    print("TOP ROOTS SUMMARY")
    print("="*70)
    print(f"Total unique roots: {len(root_counts)}")
    print(f"Top {args.limit} roots extracted")
    print(f"Coverage: {sum(r[1] for r in top_roots) / sum(root_counts.values()) * 100:.1f}%")

    print(f"\nTop 20 roots:")
    for rank, (root, count) in enumerate(top_roots[:20], 1):
        role_data = role_counts[root]
        print(f"  {rank:2d}. {root:15s} ({count:5d} total) "
              f"S:{role_data['subject']:4d} V:{role_data['verb']:4d} O:{role_data['object']:4d}")

    print(f"\nOutput written to: {args.output}")
    print("\nNext steps:")
    print("1. Open the output file")
    print("2. Fill in 'animacy' field: animate/inanimate/abstract")
    print("3. Fill in 'type' field: person/animal/object/action/quality/etc")
    print("4. Save as root lexicon JSON")


if __name__ == '__main__':
    main()
