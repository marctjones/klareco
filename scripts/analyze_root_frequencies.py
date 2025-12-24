#!/usr/bin/env python3
"""
Analyze root frequency distribution in M1 corpus.

Part of Issue #41 - determines optimal vocabulary size for morpheme-aware embeddings.

Usage:
    python scripts/analyze_root_frequencies.py \\
        --corpus data/corpus_enhanced_m1.jsonl \\
        --output /tmp/root_frequency_analysis.json
"""

import json
import argparse
from pathlib import Path
from collections import Counter
from typing import Any, List
from tqdm import tqdm


def extract_roots(obj: Any) -> List[str]:
    """
    Recursively extract all roots from AST structure.

    Args:
        obj: AST object (dict, list, or scalar)

    Returns:
        List of root strings
    """
    roots = []

    if isinstance(obj, dict):
        # Check if this node has a root
        if 'radiko' in obj and obj.get('radiko'):
            root = obj['radiko']
            # Filter out empty strings and special markers
            if root and root not in ['<UNK>', '<MASK>', '<PAD>']:
                roots.append(root)

        # Recursively process all values
        for value in obj.values():
            roots.extend(extract_roots(value))

    elif isinstance(obj, list):
        # Recursively process all items
        for item in obj:
            roots.extend(extract_roots(item))

    return roots


def analyze_frequencies(corpus_path: Path) -> dict:
    """
    Analyze root frequency distribution in corpus.

    Args:
        corpus_path: Path to corpus JSONL file

    Returns:
        Dictionary with analysis results
    """
    print(f"Reading corpus from: {corpus_path}")

    root_counts = Counter()
    total_words = 0
    total_sentences = 0

    # Count roots
    with open(corpus_path) as f:
        for line in tqdm(f, desc="Analyzing roots"):
            try:
                entry = json.loads(line)

                # Extract roots from AST
                roots = extract_roots(entry.get('ast', {}))
                root_counts.update(roots)
                total_words += len(roots)
                total_sentences += 1

            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON line: {e}")
                continue

    print(f"\nProcessed {total_sentences:,} sentences")
    print(f"Found {total_words:,} total word instances")
    print(f"Found {len(root_counts):,} unique roots")

    # Calculate statistics
    sorted_roots = root_counts.most_common()

    # Coverage at different vocabulary sizes
    coverage_stats = []
    cumulative_count = 0

    for rank, (root, count) in enumerate(sorted_roots, 1):
        cumulative_count += count
        coverage_pct = cumulative_count / total_words

        coverage_stats.append({
            'rank': rank,
            'root': root,
            'count': count,
            'frequency': count / total_words,
            'cumulative_coverage': coverage_pct
        })

    # Key milestones
    milestones = {}
    for rank in [10, 50, 100, 500, 1000, 5000, 10000]:
        if rank <= len(coverage_stats):
            milestones[f'top_{rank}'] = {
                'vocabulary_size': rank,
                'coverage': coverage_stats[rank - 1]['cumulative_coverage'],
                'last_root': coverage_stats[rank - 1]['root'],
                'last_count': coverage_stats[rank - 1]['count']
            }

    # Recommendation
    recommendation = None
    recommendation_reason = None

    if len(coverage_stats) >= 5000:
        cov_5k = coverage_stats[4999]['cumulative_coverage']
        if cov_5k >= 0.90:
            recommendation = 5000
            recommendation_reason = f"{cov_5k:.1%} coverage with 5K roots - excellent"

    if recommendation is None and len(coverage_stats) >= 10000:
        cov_10k = coverage_stats[9999]['cumulative_coverage']
        if cov_10k >= 0.95:
            recommendation = 10000
            recommendation_reason = f"{cov_10k:.1%} coverage with 10K roots - comprehensive"

    if recommendation is None:
        # Default to 5K or total, whichever is smaller
        recommendation = min(5000, len(coverage_stats))
        if len(coverage_stats) >= 5000:
            cov = coverage_stats[4999]['cumulative_coverage']
        else:
            cov = coverage_stats[-1]['cumulative_coverage']
        recommendation_reason = f"{cov:.1%} coverage (default)"

    # Build result
    result = {
        'total_roots': len(root_counts),
        'total_words': total_words,
        'total_sentences': total_sentences,
        'average_words_per_sentence': total_words / total_sentences if total_sentences > 0 else 0,
        'milestones': milestones,
        'recommendation': {
            'vocabulary_size': recommendation,
            'reason': recommendation_reason,
            'coverage': milestones.get(f'top_{recommendation}', {}).get('coverage', 0)
        },
        'top_100_roots': [
            {
                'rank': i + 1,
                'root': root,
                'count': count,
                'frequency': count / total_words,
                'cumulative_coverage': coverage_stats[i]['cumulative_coverage']
            }
            for i, (root, count) in enumerate(sorted_roots[:100])
        ]
    }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Analyze root frequency distribution in M1 corpus"
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        required=True,
        help="Path to corpus JSONL file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to output JSON file"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed statistics"
    )

    args = parser.parse_args()

    # Verify corpus exists
    if not args.corpus.exists():
        print(f"Error: Corpus not found at {args.corpus}")
        return 1

    # Analyze
    result = analyze_frequencies(args.corpus)

    # Save result
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Analysis saved to: {args.output}")

    # Print summary
    print("\n" + "=" * 60)
    print("ROOT FREQUENCY ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Total unique roots: {result['total_roots']:,}")
    print(f"Total word instances: {result['total_words']:,}")
    print(f"Total sentences: {result['total_sentences']:,}")
    print()

    print("Coverage at different vocabulary sizes:")
    for key, stats in sorted(result['milestones'].items()):
        size = stats['vocabulary_size']
        cov = stats['coverage']
        print(f"  Top {size:5,}: {cov:6.1%} coverage (last root: '{stats['last_root']}', count: {stats['last_count']:,})")

    print()
    print("RECOMMENDATION:")
    rec = result['recommendation']
    print(f"  Vocabulary size: {rec['vocabulary_size']:,} roots")
    print(f"  Coverage: {rec['coverage']:.1%}")
    print(f"  Reason: {rec['reason']}")

    if args.verbose:
        print("\nTop 20 most frequent roots:")
        for item in result['top_100_roots'][:20]:
            print(f"  {item['rank']:3}. {item['root']:15} - {item['count']:8,} ({item['frequency']:6.2%})")

    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
