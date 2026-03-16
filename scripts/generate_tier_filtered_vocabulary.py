#!/usr/bin/env python3
"""
Generate tier-filtered vocabulary from classified database.

Extracts vocabulary from v2.1 Kuzu database based on tier classification.
This replaces the old clean_roots.json with a properly filtered version.

Default: Excludes tier0 (function words) and tier5 (garbage)
Includes: tier1a, tier1b, tier2, tier3, tier4

Usage:
    python scripts/generate_tier_filtered_vocabulary.py --kuzu data/indexes/v2.1_kuzu_index_full
    python scripts/generate_tier_filtered_vocabulary.py --kuzu data/indexes/v2.1_kuzu_index_full --tiers 1a,1b,2,3
    python scripts/generate_tier_filtered_vocabulary.py --kuzu data/indexes/v2.1_kuzu_index_full --min-ofteco 5
"""
import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import kuzu
except ImportError:
    print("ERROR: kuzu not installed")
    exit(1)


def generate_tier_filtered_vocab(db_path: Path, tiers: list, min_ofteco: int,
                                  output_path: Path, stats_output: Path = None):
    """Generate vocabulary filtered by tier classification."""

    print(f"\n{'='*60}")
    print("TIER-FILTERED VOCABULARY GENERATION")
    print(f"{'='*60}")

    print(f"\nDatabase: {db_path}")
    print(f"Output: {output_path}")
    print(f"Tiers: {', '.join(tiers)}")
    print(f"Min ofteco: {min_ofteco}")

    # Connect to database (read-only)
    db = kuzu.Database(str(db_path), read_only=True)
    conn = kuzu.Connection(db)

    # Build tier filter patterns
    tier_patterns = []
    for tier in tiers:
        if tier == '1a':
            tier_patterns.append("'tier1a_unua_libro'")
        elif tier == '1b':
            tier_patterns.append("'tier1b_fundamento'")
        elif tier == '2':
            tier_patterns.append("'tier2_revo'")
        elif tier == '3':
            tier_patterns.append("'tier3_korpuso'")
        elif tier == '4':
            tier_patterns.append("'tier4_propranomo'")
        else:
            print(f"WARNING: Unknown tier '{tier}', skipping")

    if not tier_patterns:
        print("ERROR: No valid tiers specified")
        return 1

    tier_filter = ', '.join(tier_patterns)

    # Query database
    print("\nQuerying database...")
    query = f"""
        MATCH (r:Radiko)
        WHERE r.nivelo IN [{tier_filter}]
          AND r.ofteco >= {min_ofteco}
        RETURN r.radiko, r.nivelo, r.fonto, r.ofteco
        ORDER BY r.ofteco DESC
    """

    result = conn.execute(query)

    # Collect roots with metadata
    vocabulary = {}
    tier_counts = defaultdict(int)
    source_counts = defaultdict(int)

    count = 0
    while result.has_next():
        radiko, nivelo, fonto, ofteco = result.get_next()

        vocabulary[radiko] = {
            'tier': nivelo,
            'source': fonto,
            'frequency': ofteco
        }

        tier_counts[nivelo] += 1
        if fonto:
            source_counts[fonto] += 1
        count += 1

        if count % 10000 == 0:
            print(f"  Processed {count:,} roots...", flush=True)

    print(f"  Total roots: {count:,}")

    # Statistics
    print(f"\n{'='*60}")
    print("VOCABULARY STATISTICS")
    print(f"{'='*60}")

    print(f"\nTotal vocabulary size: {len(vocabulary):,} roots")

    print("\nBy Tier:")
    for tier in sorted(tier_counts.keys()):
        count = tier_counts[tier]
        pct = 100 * count / len(vocabulary) if len(vocabulary) > 0 else 0
        print(f"  {tier:25s}: {count:7,} ({pct:5.1f}%)")

    print("\nBy Source:")
    for source in sorted(source_counts.keys()):
        count = source_counts[source]
        pct = 100 * count / len(vocabulary) if len(vocabulary) > 0 else 0
        print(f"  {source:15s}: {count:7,} ({pct:5.1f}%)")

    # Ofteco statistics
    ofteco_values = [v['frequency'] for v in vocabulary.values()]
    ofteco_values.sort()

    print("\nFrequency Distribution:")
    print(f"  Minimum: {min(ofteco_values):,}")
    print(f"  Median: {ofteco_values[len(ofteco_values)//2]:,}")
    print(f"  P90: {ofteco_values[int(len(ofteco_values)*0.9)]:,}")
    print(f"  P95: {ofteco_values[int(len(ofteco_values)*0.95)]:,}")
    print(f"  P99: {ofteco_values[int(len(ofteco_values)*0.99)]:,}")
    print(f"  Maximum: {max(ofteco_values):,}")

    # Save vocabulary
    print(f"\nSaving vocabulary to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(vocabulary, f, ensure_ascii=False, indent=2)

    print(f"✓ Vocabulary saved ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # Save statistics if requested
    if stats_output:
        stats = {
            'generated_at': str(Path(__file__).name),
            'database': str(db_path),
            'tiers_included': tiers,
            'min_ofteco': min_ofteco,
            'total_roots': len(vocabulary),
            'tier_counts': dict(tier_counts),
            'source_counts': dict(source_counts),
            'frequency_stats': {
                'min': min(ofteco_values),
                'median': ofteco_values[len(ofteco_values)//2],
                'p90': ofteco_values[int(len(ofteco_values)*0.9)],
                'p95': ofteco_values[int(len(ofteco_values)*0.95)],
                'p99': ofteco_values[int(len(ofteco_values)*0.99)],
                'max': max(ofteco_values)
            }
        }

        with open(stats_output, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"✓ Statistics saved to {stats_output}")

    # Validation checks
    print(f"\n{'='*60}")
    print("VALIDATION CHECKS")
    print(f"{'='*60}")

    # Check for function words (should be 0)
    function_word_samples = ['mi', 'kaj', 'la', 'de', 'en', 'al', 'kio', 'mal', 'iĝ']
    found_function_words = [w for w in function_word_samples if w in vocabulary]

    if found_function_words:
        print(f"\n❌ FAIL: Found function words in vocabulary:")
        print(f"  {', '.join(found_function_words)}")
        print(f"  These should be excluded (tier0)")
        return 1
    else:
        print(f"\n✓ PASS: No function words in vocabulary (tier0 excluded)")

    # Check for known good words
    good_word_samples = ['hund', 'dom', 'amik', 'bon', 'vid', 'parol']
    found_good_words = [w for w in good_word_samples if w in vocabulary]

    if len(found_good_words) >= 4:
        print(f"✓ PASS: Found expected content words: {', '.join(found_good_words)}")
    else:
        print(f"⚠ WARNING: Only found {len(found_good_words)}/6 expected content words")

    # Check vocabulary size is reasonable
    if len(vocabulary) < 1000:
        print(f"❌ FAIL: Vocabulary too small ({len(vocabulary)} roots)")
        return 1
    elif len(vocabulary) > 200000:
        print(f"⚠ WARNING: Vocabulary very large ({len(vocabulary)} roots)")
    else:
        print(f"✓ PASS: Vocabulary size reasonable ({len(vocabulary):,} roots)")

    print(f"\n{'='*60}")
    print("✓ VOCABULARY GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"\nNext steps:")
    print(f"  1. Use this vocabulary in training:")
    print(f"     ./scripts/train_roots.sh")
    print(f"  2. Validate embeddings exclude function words")
    print(f"  3. Compare model quality vs old vocabulary")

    return 0


def main():
    parser = argparse.ArgumentParser(description='Generate tier-filtered vocabulary')
    parser.add_argument('--kuzu', type=Path, required=True,
                        help='Kuzu database path')
    parser.add_argument('--tiers', type=str, default='1a,1b,2,3,4',
                        help='Comma-separated tier list (default: 1a,1b,2,3,4)')
    parser.add_argument('--min-ofteco', type=int, default=1,
                        help='Minimum frequency threshold (default: 1)')
    parser.add_argument('--output', type=Path,
                        default=Path('data/vocabularies/tier_filtered_roots.json'),
                        help='Output vocabulary file')
    parser.add_argument('--stats', type=Path,
                        default=Path('data/vocabularies/tier_filtered_stats.json'),
                        help='Output statistics file')

    args = parser.parse_args()

    # Parse tiers
    tiers = [t.strip() for t in args.tiers.split(',')]

    return generate_tier_filtered_vocab(
        db_path=args.kuzu,
        tiers=tiers,
        min_ofteco=args.min_ofteco,
        output_path=args.output,
        stats_output=args.stats
    )


if __name__ == '__main__':
    sys.exit(main())
