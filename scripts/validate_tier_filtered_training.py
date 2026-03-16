#!/usr/bin/env python3
"""
Validate that tier-filtered vocabulary is being used correctly in training.

Checks:
1. Vocabulary file exists and is recent
2. No function words in vocabulary
3. Vocabulary size is reasonable (50K-100K)
4. Training scripts point to correct file

Usage:
    python scripts/validate_tier_filtered_training.py
"""
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta

def validate_vocabulary():
    """Validate tier-filtered vocabulary."""
    print("=" * 60)
    print("TIER-FILTERED TRAINING VALIDATION")
    print("=" * 60)

    vocab_path = Path('data/vocabularies/tier_filtered_roots.json')
    stats_path = Path('data/vocabularies/tier_filtered_stats.json')

    # Check vocabulary exists
    print("\n1. Checking vocabulary file...")
    if not vocab_path.exists():
        print(f"  ❌ FAIL: {vocab_path} not found")
        print(f"  Run: python scripts/generate_tier_filtered_vocabulary.py --kuzu data/indexes/v2.1_kuzu_index_full")
        return False

    print(f"  ✓ PASS: Vocabulary file exists")

    # Check vocabulary is recent (within last 7 days)
    mod_time = datetime.fromtimestamp(vocab_path.stat().st_mtime)
    age_days = (datetime.now() - mod_time).days

    if age_days > 7:
        print(f"  ⚠ WARNING: Vocabulary is {age_days} days old")
        print(f"  Consider regenerating if classification changed recently")
    else:
        print(f"  ✓ PASS: Vocabulary is recent ({age_days} days old)")

    # Load and check vocabulary
    print("\n2. Checking vocabulary contents...")
    with open(vocab_path) as f:
        vocab = json.load(f)

    print(f"  Vocabulary size: {len(vocab):,} roots")

    if len(vocab) < 50000:
        print(f"  ⚠ WARNING: Vocabulary seems small")
    elif len(vocab) > 100000:
        print(f"  ⚠ WARNING: Vocabulary seems large")
    else:
        print(f"  ✓ PASS: Vocabulary size reasonable")

    # Check for function words (should be none)
    print("\n3. Checking for function words...")
    function_words = ['mi', 'vi', 'li', 'kaj', 'sed', 'la', 'de', 'en', 'al',
                      'kio', 'tio', 'ĉio', 'mal', 'ne', 'iĝ', 'ig', 'ej']

    found_function_words = [w for w in function_words if w in vocab]

    if found_function_words:
        print(f"  ❌ FAIL: Found function words: {', '.join(found_function_words)}")
        return False
    else:
        print(f"  ✓ PASS: No function words in vocabulary (tier0 excluded)")

    # Check for expected content words
    print("\n4. Checking for expected content words...")
    expected_words = ['hund', 'kat', 'dom', 'amik', 'bon', 'bel', 'grand',
                      'parol', 'veni', 'iri', 'vid', 'sci']

    found_content_words = [w for w in expected_words if w in vocab]

    if len(found_content_words) < 8:
        print(f"  ⚠ WARNING: Only found {len(found_content_words)}/{len(expected_words)} expected words")
    else:
        print(f"  ✓ PASS: Found {len(found_content_words)}/{len(expected_words)} expected words")

    # Check tier distribution
    print("\n5. Checking tier distribution...")
    tier_counts = {}
    for root, data in vocab.items():
        tier = data.get('tier', 'unknown')
        tier_counts[tier] = tier_counts.get(tier, 0) + 1

    for tier in sorted(tier_counts.keys()):
        count = tier_counts[tier]
        pct = 100 * count / len(vocab)
        print(f"  {tier:25s}: {count:7,} ({pct:5.1f}%)")

    # Check if tier0 is present (should not be)
    tier0_count = sum(count for tier, count in tier_counts.items() if tier.startswith('tier0_'))
    if tier0_count > 0:
        print(f"  ❌ FAIL: Found {tier0_count} tier0 (function word) entries")
        return False
    else:
        print(f"  ✓ PASS: No tier0 entries")

    # Check if tier5 is present (should not be)
    tier5_count = tier_counts.get('tier5_rubaĵo', 0)
    if tier5_count > 0:
        print(f"  ❌ FAIL: Found {tier5_count} tier5 (garbage) entries")
        return False
    else:
        print(f"  ✓ PASS: No tier5 entries")

    # Check statistics file
    print("\n6. Checking statistics file...")
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)

        print(f"  ✓ Statistics file exists")
        print(f"  Generated: {stats.get('generated_at', 'unknown')}")
        print(f"  Tiers included: {', '.join(stats.get('tiers_included', []))}")
        print(f"  Min ofteco: {stats.get('min_ofteco', 'unknown')}")
    else:
        print(f"  ⚠ WARNING: Statistics file not found")

    # Check training script
    print("\n7. Checking training script configuration...")
    train_script = Path('scripts/train_roots.sh')

    if not train_script.exists():
        print(f"  ⚠ WARNING: Training script not found")
    else:
        with open(train_script) as f:
            content = f.read()

        if 'tier_filtered_roots.json' in content:
            print(f"  ✓ PASS: Training script configured for tier-filtered vocabulary")
        else:
            print(f"  ⚠ WARNING: Training script may not be using tier-filtered vocabulary")
            print(f"  Check: {train_script}")

    print("\n" + "=" * 60)
    print("✓ VALIDATION COMPLETE")
    print("=" * 60)

    print("\nNext steps:")
    print("  1. Train root embeddings:")
    print("     ./scripts/train_roots.sh --fresh")
    print("  2. Validate no embedding collapse:")
    print("     python tests/test_embedding_quality.py")
    print("  3. Compare with old model quality")

    return True


if __name__ == '__main__':
    success = validate_vocabulary()
    sys.exit(0 if success else 1)
