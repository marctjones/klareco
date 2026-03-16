#!/usr/bin/env python3
"""
Clean corpus-validated roots by removing garbage patterns.

Removes:
- Pure numbers (0, 123, etc.)
- Single characters
- Non-alphabetic content
- Mixed alphanumeric garbage (abc123, etc.)
- Common OCR errors and typos

Usage:
    python scripts/clean_corpus_roots.py
"""
import argparse
import json
import re
from pathlib import Path


def is_valid_esperanto_root(root: str) -> bool:
    """Check if root looks like a valid Esperanto word root."""
    # Must be at least 2 characters
    if len(root) < 2:
        return False

    # Must contain only alphabetic characters (including Esperanto letters)
    # Allow: a-z, ĉ, ĝ, ĥ, ĵ, ŝ, ŭ
    esperanto_pattern = re.compile(r'^[a-zĉĝĥĵŝŭ]+$', re.IGNORECASE)
    if not esperanto_pattern.match(root):
        return False

    # Reject if contains numbers
    if any(c.isdigit() for c in root):
        return False

    # Reject if all caps (likely abbreviation)
    if root.isupper() and len(root) > 1:
        return False

    # Reject common garbage patterns
    garbage_patterns = [
        r'^x+$',  # xxx, xxxx
        r'^[aeiou]+$',  # aaaa, eeeee (repeated vowels)
        r'^(.)\1{3,}$',  # aaaa, bbbb (any repeated char 4+ times)
    ]

    for pattern in garbage_patterns:
        if re.match(pattern, root, re.IGNORECASE):
            return False

    # Accept if it looks linguistic
    return True


def main():
    parser = argparse.ArgumentParser(description='Clean corpus roots')
    parser.add_argument('--vocab-dir', type=Path,
                       default=Path('data/vocabularies'),
                       help='Vocabulary directory')

    args = parser.parse_args()

    # Load corpus-validated roots
    corpus_file = args.vocab_dir / 'corpus_validated_roots.json'
    with open(corpus_file, 'r', encoding='utf-8') as f:
        corpus_data = json.load(f)

    print(f"Original corpus roots: {len(corpus_data):,}")

    # Filter valid roots
    valid_roots = {}
    invalid_roots = {}

    for root, data in corpus_data.items():
        if is_valid_esperanto_root(root):
            valid_roots[root] = data
        else:
            invalid_roots[root] = data

    print(f"Valid roots: {len(valid_roots):,}")
    print(f"Invalid roots (filtered): {len(invalid_roots):,}")

    # Show sample invalid roots
    print("\nSample invalid roots (first 50):")
    for i, root in enumerate(sorted(invalid_roots.keys())[:50]):
        print(f"  {root}")

    # Save cleaned corpus roots
    clean_file = args.vocab_dir / 'corpus_validated_roots_clean.json'
    with open(clean_file, 'w', encoding='utf-8') as f:
        json.dump(valid_roots, f, indent=2, ensure_ascii=False)

    print(f"\nSaved cleaned corpus: {clean_file}")
    print(f"  Clean roots: {len(valid_roots):,}")

    # Save garbage roots for inspection
    garbage_file = args.vocab_dir / 'corpus_garbage.json'
    with open(garbage_file, 'w', encoding='utf-8') as f:
        json.dump(invalid_roots, f, indent=2, ensure_ascii=False)

    print(f"Saved garbage roots: {garbage_file}")
    print(f"  Garbage entries: {len(invalid_roots):,}")

    # Statistics
    print("\n=== Statistics ===")
    kept_pct = 100 * len(valid_roots) / len(corpus_data)
    removed_pct = 100 * len(invalid_roots) / len(corpus_data)
    print(f"Kept: {kept_pct:.1f}%")
    print(f"Removed: {removed_pct:.1f}%")


if __name__ == '__main__':
    main()
