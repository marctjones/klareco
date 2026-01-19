#!/usr/bin/env python3
"""
Analyze Corpus Quality by Source

Analyzes parse rates by source (books, articles) to help assign quality levels.
This helps identify which sources should be upgraded/downgraded from their default quality.

Quality Thresholds (for non-authoritative sources):
- GOLD:   parse_rate >= 0.98 (exceptional quality)
- SILVER: parse_rate >= 0.95 (high quality)
- BRONZE: parse_rate >= 0.90 (good quality)
- COPPER: parse_rate < 0.90  (fair quality, may want to exclude)

Usage:
    python scripts/analyze_corpus_quality.py \\
        --corpus data/enhanced_corpus/corpus_with_metadata.jsonl \\
        --output data/quality_report.txt
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def assign_quality_by_parse_rate(parse_rate: float, source_type: str) -> str:
    """Assign quality level based on parse rate."""
    # Authoritative sources are always GOLD
    if source_type in ['grammar_reference', 'pedagogical']:
        return 'GOLD'

    # For literary and encyclopedic sources, use parse rate
    if parse_rate >= 0.98:
        return 'GOLD'
    elif parse_rate >= 0.95:
        return 'SILVER'
    elif parse_rate >= 0.90:
        return 'BRONZE'
    else:
        return 'COPPER'


def analyze_corpus(corpus_path: Path, max_sentences: int = None) -> Dict:
    """Analyze parse rates by source."""

    # Track stats by source_name
    source_stats = defaultdict(lambda: {
        'parse_rates': [],
        'source_type': None,
        'count': 0
    })

    total_sentences = 0

    print(f"Analyzing corpus: {corpus_path}")
    print(f"Reading sentences...")

    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if max_sentences and total_sentences >= max_sentences:
                break

            try:
                entry = json.loads(line)

                # Extract metadata
                source = entry.get('source', {})
                source_name = source.get('source_name') or source.get('name', 'unknown')
                source_type = source.get('source_type', 'unknown')
                parse_rate = entry.get('parse_rate', 0.0)

                # Track stats
                source_stats[source_name]['parse_rates'].append(parse_rate)
                source_stats[source_name]['source_type'] = source_type
                source_stats[source_name]['count'] += 1

                total_sentences += 1

                if line_num % 100000 == 0:
                    print(f"  Processed {line_num:,} lines...")

            except json.JSONDecodeError:
                print(f"Warning: Invalid JSON at line {line_num}")
                continue
            except Exception as e:
                print(f"Warning: Error at line {line_num}: {e}")
                continue

    print(f"Total sentences analyzed: {total_sentences:,}")
    print(f"Unique sources found: {len(source_stats)}")

    # Calculate averages
    source_quality = {}
    for source_name, stats in source_stats.items():
        avg_parse_rate = sum(stats['parse_rates']) / len(stats['parse_rates'])
        min_parse_rate = min(stats['parse_rates'])
        max_parse_rate = max(stats['parse_rates'])

        # Assign quality based on parse rate
        suggested_quality = assign_quality_by_parse_rate(avg_parse_rate, stats['source_type'])

        source_quality[source_name] = {
            'avg_parse_rate': avg_parse_rate,
            'min_parse_rate': min_parse_rate,
            'max_parse_rate': max_parse_rate,
            'sentence_count': stats['count'],
            'source_type': stats['source_type'],
            'suggested_quality': suggested_quality
        }

    return source_quality


def generate_report(source_quality: Dict, output_path: Path):
    """Generate quality report grouped by suggested quality level."""

    # Group by suggested quality
    by_quality = defaultdict(list)
    for source_name, stats in source_quality.items():
        by_quality[stats['suggested_quality']].append((source_name, stats))

    # Write report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("CORPUS QUALITY ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write("Quality Thresholds:\n")
        f.write("  GOLD:   parse_rate >= 0.98 (exceptional quality)\n")
        f.write("  SILVER: parse_rate >= 0.95 (high quality)\n")
        f.write("  BRONZE: parse_rate >= 0.90 (good quality)\n")
        f.write("  COPPER: parse_rate < 0.90  (fair quality)\n\n")

        # Report each quality level
        for quality in ['GOLD', 'SILVER', 'BRONZE', 'COPPER']:
            if quality not in by_quality:
                continue

            sources = by_quality[quality]
            # Sort by parse rate descending
            sources.sort(key=lambda x: x[1]['avg_parse_rate'], reverse=True)

            f.write("-" * 80 + "\n")
            f.write(f"{quality} Quality ({len(sources)} sources)\n")
            f.write("-" * 80 + "\n\n")

            for source_name, stats in sources:
                f.write(f"{source_name}\n")
                f.write(f"  Parse rate: {stats['avg_parse_rate']:.4f} ")
                f.write(f"(min: {stats['min_parse_rate']:.4f}, max: {stats['max_parse_rate']:.4f})\n")
                f.write(f"  Sentences:  {stats['sentence_count']:,}\n")
                f.write(f"  Type:       {stats['source_type']}\n")
                f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("=" * 80 + "\n\n")

        f.write("1. Review COPPER sources - consider excluding if parse rate < 0.85\n")
        f.write("2. Review borderline cases (parse rate 0.89-0.91, 0.94-0.96, 0.97-0.99)\n")
        f.write("3. Check if any Wikipedia articles should be upgraded to SILVER/GOLD\n")
        f.write("4. Create quality_overrides.json for manual adjustments\n\n")

        # Calculate overall stats
        total_sentences = sum(s['sentence_count'] for _, s in source_quality.items())
        f.write(f"Total sources analyzed: {len(source_quality)}\n")
        f.write(f"Total sentences: {total_sentences:,}\n\n")

        # Distribution by quality
        f.write("Distribution by suggested quality:\n")
        for quality in ['GOLD', 'SILVER', 'BRONZE', 'COPPER']:
            if quality in by_quality:
                count = sum(s['sentence_count'] for _, s in by_quality[quality])
                pct = (count / total_sentences * 100) if total_sentences > 0 else 0
                f.write(f"  {quality:7s}: {count:8,} sentences ({pct:5.1f}%)\n")

    print(f"\nReport written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze corpus quality by source'
    )
    parser.add_argument(
        '--corpus',
        type=Path,
        required=True,
        help='Path to corpus file (JSONL)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/quality_report.txt'),
        help='Output report file'
    )
    parser.add_argument(
        '--max-sentences',
        type=int,
        help='Limit analysis to first N sentences (for testing)'
    )

    args = parser.parse_args()

    # Analyze corpus
    source_quality = analyze_corpus(args.corpus, args.max_sentences)

    # Generate report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    generate_report(source_quality, args.output)

    print("\nNext steps:")
    print("  1. Review the quality report")
    print("  2. Create data/quality_overrides.json for manual adjustments")
    print("  3. Rebuild corpus with: python scripts/build_unified_corpus.py --fresh")


if __name__ == '__main__':
    sys.exit(main())
