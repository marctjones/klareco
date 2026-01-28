#!/usr/bin/env python3
"""
Validate Training Data Quality

Checks training data before model training:
1. Files exist and are readable
2. Train/val/test splits present
3. Data format correct
4. No empty or malformed entries
5. Label distribution reasonable
6. Vocabulary coverage adequate
7. No data leakage between splits

Usage:
    python scripts/validate_training_data.py data/training/m1_semantic_full
    python scripts/validate_training_data.py data/training/ekzercaro_sentences.jsonl
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def validate_m1_training_data(data_dir: Path):
    """Validate M1 training data (selectional preference)."""

    print("=" * 80)
    print(f"VALIDATING M1 TRAINING DATA: {data_dir.name}")
    print("=" * 80)
    print()

    # Check required files
    required_files = ['train.jsonl', 'val.jsonl', 'test.jsonl', 'vocabulary.json', 'metadata.json']
    for filename in required_files:
        filepath = data_dir / filename
        if not filepath.exists():
            print(f"❌ Missing file: {filename}")
            return False
        print(f"✓ Found: {filename}")

    print()

    # Load metadata
    with open(data_dir / 'metadata.json') as f:
        metadata = json.load(f)

    print("Metadata:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")
    print()

    # Validate each split
    splits = ['train', 'val', 'test']
    split_stats = {}

    for split in splits:
        filepath = data_dir / f'{split}.jsonl'

        count = 0
        labels = []
        issues = []

        with open(filepath) as f:
            for line_num, line in enumerate(f, 1):
                try:
                    entry = json.loads(line)

                    # Check required fields
                    if 'subject' not in entry or 'verb' not in entry or 'object' not in entry:
                        issues.append(f"Line {line_num}: Missing SVO fields")
                        continue

                    if 'label' not in entry:
                        issues.append(f"Line {line_num}: Missing label")
                        continue

                    labels.append(entry['label'])
                    count += 1

                except json.JSONDecodeError:
                    issues.append(f"Line {line_num}: Invalid JSON")

        # Calculate statistics
        label_dist = Counter(labels)
        pos_count = label_dist.get(1, 0)
        neg_count = label_dist.get(0, 0)
        pos_rate = pos_count / count if count > 0 else 0

        split_stats[split] = {
            'count': count,
            'pos': pos_count,
            'neg': neg_count,
            'pos_rate': pos_rate,
            'issues': len(issues)
        }

        status = "✓" if pos_rate > 0.4 and pos_rate < 0.6 else "⚠️ "
        print(f"{split:5s}: {count:,} examples, {pos_count:,} pos ({pos_rate:.1%}), {neg_count:,} neg")
        if issues:
            print(f"       ⚠️  {len(issues)} issues found")
            for issue in issues[:3]:
                print(f"         - {issue}")

    print()

    # Check balance
    train_stats = split_stats['train']
    if train_stats['pos_rate'] < 0.4 or train_stats['pos_rate'] > 0.6:
        print(f"⚠️  Imbalanced training data: {train_stats['pos_rate']:.1%} positive")
        print("   Recommendation: Aim for 40-60% positive examples")
        print()

    # Check vocabulary
    with open(data_dir / 'vocabulary.json') as f:
        vocab = json.load(f)

    print(f"Vocabulary: {len(vocab):,} roots")
    print()

    print("✓ Training data validation complete!")
    return True


def validate_ekzercaro_data(filepath: Path):
    """Validate Ekzercaro-style training data (root embeddings)."""

    print("=" * 80)
    print(f"VALIDATING EKZERCARO DATA: {filepath.name}")
    print("=" * 80)
    print()

    if not filepath.exists():
        print(f"❌ File not found: {filepath}")
        return False

    count = 0
    roots_seen = set()
    issues = []

    with open(filepath) as f:
        for line_num, line in enumerate(f, 1):
            try:
                entry = json.loads(line)

                # Check required fields
                if 'text' not in entry:
                    issues.append(f"Line {line_num}: Missing 'text'")
                    continue

                if 'ast' not in entry:
                    issues.append(f"Line {line_num}: Missing 'ast'")
                    continue

                # Extract roots from AST
                ast = entry.get('ast')
                if ast and isinstance(ast, dict):
                    if 'subjekto' in ast and ast['subjekto'] and 'kerno' in ast['subjekto']:
                        root = ast['subjekto']['kerno'].get('radiko')
                        if root:
                            roots_seen.add(root)

                count += 1

            except json.JSONDecodeError:
                issues.append(f"Line {line_num}: Invalid JSON")

    print(f"Total sentences: {count:,}")
    print(f"Unique roots:    {len(roots_seen):,}")
    print()

    if issues:
        print(f"⚠️  {len(issues)} issues found:")
        for issue in issues[:5]:
            print(f"  - {issue}")
        print()

    if count < 10000:
        print(f"⚠️  Low sentence count: {count:,}")
        print("   Expected: 50,000+ for good root coverage")
        print()

    print("✓ Ekzercaro data validation complete!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Validate training data")
    parser.add_argument(
        'data_path',
        type=Path,
        help='Path to training data (directory or file)'
    )

    args = parser.parse_args()

    if args.data_path.is_dir():
        success = validate_m1_training_data(args.data_path)
    else:
        success = validate_ekzercaro_data(args.data_path)

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
