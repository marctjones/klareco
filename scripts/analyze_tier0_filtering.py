#!/usr/bin/env python3
"""
Analyze Tier0 Filtering in M1 Training Data Generation

This script simulates the filtering logic from prepare_m1_training_data_semantic.py
to understand why tier0 sentences are being filtered out.
"""

import json
import sys
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def extract_svo_triple(ast):
    """Extract (subject_root, verb_root, object_root) from AST."""
    if not all(k in ast for k in ['subjekto', 'verbo', 'objekto']):
        return None

    try:
        subjekto = ast['subjekto']
        verbo = ast['verbo']
        objekto = ast['objekto']

        subj_root = subjekto.get('kerno', {}).get('radiko') if isinstance(subjekto, dict) else None
        verb_root = verbo.get('radiko') if isinstance(verbo, dict) else None
        obj_root = objekto.get('kerno', {}).get('radiko') if isinstance(objekto, dict) else None

        if subj_root and verb_root and obj_root:
            return (subj_root.lower(), verb_root.lower(), obj_root.lower())
    except (AttributeError, KeyError, TypeError):
        pass

    return None


def main():
    corpus_path = Path('data/enhanced_corpus/corpus_full_with_tier0.jsonl')

    if not corpus_path.exists():
        print(f"❌ Corpus not found: {corpus_path}")
        return 1

    print("=" * 70)
    print("Tier0 Filtering Analysis")
    print("=" * 70)
    print(f"Corpus: {corpus_path}")
    print()

    # Track tier0 statistics
    stats = {
        'tier0_total': 0,
        'tier0_with_parse_rate': 0,
        'tier0_with_ast': 0,
        'tier0_with_svo': 0,
        'tier0_parse_rates': [],
        'tier0_sources': Counter(),
        'tier0_first_position': None,
        'tier0_last_position': None,
    }

    # Track overall statistics
    total_lines = 0
    tier0_positions = []

    min_parse_rate = 0.0  # As used in training script

    print("Scanning corpus...")
    with open(corpus_path) as f:
        for i, line in enumerate(f):
            if i % 500000 == 0 and i > 0:
                print(f"  Processed {i:,} sentences, found {stats['tier0_total']:,} tier0")

            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            total_lines += 1
            source = entry.get('source', {})

            if source.get('tier') == 0:
                stats['tier0_total'] += 1
                tier0_positions.append(i)

                if stats['tier0_first_position'] is None:
                    stats['tier0_first_position'] = i

                stats['tier0_last_position'] = i
                stats['tier0_sources'][source.get('name', 'unknown')] += 1

                # Check parse rate filter
                parse_rate = entry.get('parse_rate', 0)
                stats['tier0_parse_rates'].append(parse_rate)

                if parse_rate >= min_parse_rate:
                    stats['tier0_with_parse_rate'] += 1

                    # Check AST filter
                    ast = entry.get('ast')
                    if ast:
                        stats['tier0_with_ast'] += 1

                        # Check S-V-O filter
                        triple = extract_svo_triple(ast)
                        if triple:
                            stats['tier0_with_svo'] += 1

    print(f"✓ Scanned {total_lines:,} total sentences")
    print()

    # Print results
    print("=" * 70)
    print("RESULTS: Tier0 Filtering Stages")
    print("=" * 70)
    print()

    print(f"1. Total tier0 sentences in corpus: {stats['tier0_total']:,}")
    print()

    print(f"2. After parse_rate >= {min_parse_rate} filter: {stats['tier0_with_parse_rate']:,}")
    filtered_parse = stats['tier0_total'] - stats['tier0_with_parse_rate']
    pct_parse = 100 * stats['tier0_with_parse_rate'] / stats['tier0_total'] if stats['tier0_total'] > 0 else 0
    print(f"   Filtered out: {filtered_parse:,} ({100-pct_parse:.1f}%)")
    print()

    print(f"3. After AST existence filter: {stats['tier0_with_ast']:,}")
    filtered_ast = stats['tier0_with_parse_rate'] - stats['tier0_with_ast']
    pct_ast = 100 * stats['tier0_with_ast'] / stats['tier0_with_parse_rate'] if stats['tier0_with_parse_rate'] > 0 else 0
    print(f"   Filtered out: {filtered_ast:,} ({100-pct_ast:.1f}%)")
    print()

    print(f"4. After S-V-O triple extraction filter: {stats['tier0_with_svo']:,}")
    filtered_svo = stats['tier0_with_ast'] - stats['tier0_with_svo']
    pct_svo = 100 * stats['tier0_with_svo'] / stats['tier0_with_ast'] if stats['tier0_with_ast'] > 0 else 0
    print(f"   Filtered out: {filtered_svo:,} ({100-pct_svo:.1f}%)")
    print()

    print("=" * 70)
    print(f"FINAL: {stats['tier0_with_svo']:,} tier0 triples should be extracted")
    print("       (but training data has 0)")
    print("=" * 70)
    print()

    # Additional diagnostics
    print("Tier0 Sources:")
    for source, count in stats['tier0_sources'].most_common():
        print(f"  {source}: {count:,}")
    print()

    print("Tier0 Position in Corpus:")
    print(f"  First tier0 at line: {stats['tier0_first_position']:,}")
    print(f"  Last tier0 at line: {stats['tier0_last_position']:,}")
    print(f"  Total corpus lines: {total_lines:,}")
    pct_before_first = 100 * stats['tier0_first_position'] / total_lines if total_lines > 0 else 0
    print(f"  Tier0 starts at: {pct_before_first:.1f}% through corpus")
    print()

    if stats['tier0_parse_rates']:
        print("Tier0 Parse Rates:")
        print(f"  Min: {min(stats['tier0_parse_rates']):.3f}")
        print(f"  Max: {max(stats['tier0_parse_rates']):.3f}")
        print(f"  Mean: {sum(stats['tier0_parse_rates'])/len(stats['tier0_parse_rates']):.3f}")
        print()

    # Hypothesis determination
    print("=" * 70)
    print("HYPOTHESIS:")
    print("=" * 70)

    if stats['tier0_with_svo'] > 0:
        print("✓ Tier0 HAS extractable S-V-O triples")
        print()
        print("Possible causes:")
        print("  1. --max-triples limit reached before processing tier0")
        if pct_before_first > 50:
            print(f"     LIKELY! Tier0 appears {pct_before_first:.1f}% through corpus")
            print(f"     Script may hit 200K limit before reaching tier0")
        else:
            print(f"     Unlikely. Tier0 appears early ({pct_before_first:.1f}% through)")
        print()
        print("  2. Training data was generated from different corpus")
        print("     Check: Was corpus_with_metadata.jsonl used instead?")
        print()
        print("  3. Training data was generated before tier0 was added")
        print("     Check: Compare corpus timestamp vs training data timestamp")
    else:
        if stats['tier0_with_ast'] == 0:
            print("❌ Tier0 has NO ASTs!")
            print("   → Need to re-parse tier0 data")
        else:
            print("❌ Tier0 has ASTs but NO S-V-O triples!")
            print("   → Tier0 sentences lack subject-verb-object structure")
            print("   → May be questions, fragments, or commands")
            print("   → Need to modify triple extraction for Q&A format")

    print("=" * 70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
