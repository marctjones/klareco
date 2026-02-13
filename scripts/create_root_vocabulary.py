#!/usr/bin/env python3
"""
Create Root Vocabulary for Synthetic Example Generation.

Extracts high-frequency roots from corpus for generating synthetic training examples.
"""

import sys
import json
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))


def extract_roots_from_corpus(corpus_path: Path, output_path: Path, min_frequency: int = 10, max_roots: int = 1000, max_sentences: int = None):
    """
    Extract frequently occurring roots from corpus.

    Args:
        corpus_path: Path to unified_corpus.jsonl
        output_path: Path to save root_vocab.json
        min_frequency: Minimum occurrences to include
        max_roots: Maximum roots to extract
        max_sentences: Optional limit on sentences to process (for testing)
    """
    print("="*60)
    print("CREATING ROOT VOCABULARY")
    print("="*60)
    print(f"Input: {corpus_path}")
    print(f"Output: {output_path}")
    print(f"Min frequency: {min_frequency}")
    print(f"Max roots: {max_roots}")
    if max_sentences:
        print(f"Max sentences: {max_sentences:,} (limited for speed)")
    print()

    root_counter = Counter()
    total_sentences = 0
    total_words = 0
    filtered_count = 0

    print("Extracting roots from corpus...")
    print("(Filtering: proper names, non-Esperanto letters, invalid patterns)")
    print()

    with open(corpus_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue

            # Stop if max_sentences reached
            if max_sentences and total_sentences >= max_sentences:
                print(f"\n✓ Reached limit of {max_sentences:,} sentences")
                break

            try:
                sentence_data = json.loads(line)
                ast = sentence_data.get('ast')

                if not ast:
                    continue

                total_sentences += 1

                # Extract roots from all words in sentence
                roots = extract_roots_from_ast(ast)
                total_words += len(roots)
                root_counter.update(roots)

                if line_num % 10000 == 0:
                    print(f"  Processed {line_num:,} sentences, {total_words:,} words, {len(root_counter):,} unique roots")

            except json.JSONDecodeError:
                continue

    print()
    print(f"✓ Processed {total_sentences:,} sentences")
    print(f"✓ Found {total_words:,} words")
    print(f"✓ Identified {len(root_counter):,} unique roots (after filtering)")
    print()

    # Filter by frequency and take top N
    filtered_roots = [
        root for root, count in root_counter.most_common()
        if count >= min_frequency
    ][:max_roots]

    print(f"Filtered roots:")
    print(f"  Min frequency {min_frequency}: {len(filtered_roots):,} roots")
    print(f"  Taking top {max_roots}: {len(filtered_roots[:max_roots]):,} roots")
    print()

    # Show top 20
    print("Top 20 most frequent roots:")
    for i, (root, count) in enumerate(root_counter.most_common(20), 1):
        print(f"  {i:2d}. {root:15s} ({count:,} occurrences)")
    print()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(filtered_roots, f, ensure_ascii=False, indent=2)

    print(f"✓ Saved {len(filtered_roots):,} roots to {output_path}")
    print()
    print("Next step: Generate synthetic examples")
    print("  ./scripts/generate_entity_training_data.sh --fresh")


def is_valid_esperanto_root(root: str) -> bool:
    """
    Validate if root is likely a real Esperanto root.

    Filters out:
    - Proper names (capitalized)
    - Non-Esperanto letters
    - Too short/long
    - Numbers
    - Foreign words
    """
    if not root or len(root) < 2:
        return False

    # Skip if too long (Esperanto roots rarely exceed 10 letters)
    if len(root) > 10:
        return False

    # Skip if starts with capital (likely proper name)
    if root[0].isupper():
        return False

    # Valid Esperanto letters (including ĉ, ĝ, ĥ, ĵ, ŝ, ŭ)
    valid_chars = set('abcdefghijklmnoprstuvzĉĝĥĵŝŭ')

    # Check all characters are valid Esperanto letters
    if not all(c.lower() in valid_chars for c in root):
        return False

    # Skip if contains numbers
    if any(c.isdigit() for c in root):
        return False

    # Skip common parse error patterns
    error_patterns = ['xxx', 'zzz', 'aaa', 'qqq']
    if any(pattern in root.lower() for pattern in error_patterns):
        return False

    return True


def extract_roots_from_ast(ast):
    """Extract all roots from AST recursively."""
    roots = []

    if isinstance(ast, dict):
        # Extract root from word node
        if ast.get('tipo') == 'vorto' and 'radiko' in ast:
            root = ast['radiko']

            # Validate root
            if is_valid_esperanto_root(root):
                roots.append(root.lower())  # Normalize to lowercase

        # Recurse into structure
        if ast.get('tipo') == 'frazo':
            for key in ['subjekto', 'verbo', 'objekto']:
                if ast.get(key):
                    roots.extend(extract_roots_from_ast(ast[key]))
            for alia in ast.get('aliaj', []):
                roots.extend(extract_roots_from_ast(alia))

        elif ast.get('tipo') == 'vortgrupo':
            if ast.get('kerno'):
                roots.extend(extract_roots_from_ast(ast['kerno']))
            for priskribo in ast.get('priskriboj', []):
                roots.extend(extract_roots_from_ast(priskribo))

    return roots


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Create root vocabulary from corpus')
    parser.add_argument(
        '--corpus',
        type=Path,
        default=Path('data/corpus/unified_corpus.jsonl'),
        help='Path to corpus'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/vocabularies/root_vocab.json'),
        help='Output path for root vocabulary'
    )
    parser.add_argument(
        '--min-frequency',
        type=int,
        default=10,
        help='Minimum frequency to include root'
    )
    parser.add_argument(
        '--max-roots',
        type=int,
        default=1000,
        help='Maximum roots to extract'
    )
    parser.add_argument(
        '--max-sentences',
        type=int,
        default=None,
        help='Limit sentences to process (for testing, default: all)'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Quick test mode (100K sentences only)'
    )

    args = parser.parse_args()

    if not args.corpus.exists():
        print(f"ERROR: Corpus not found: {args.corpus}")
        sys.exit(1)

    # Test mode: quick run on 100K sentences
    max_sentences = args.max_sentences
    if args.test:
        max_sentences = 100000
        print("⚡ TEST MODE: Processing 100K sentences only")
        print()

    extract_roots_from_corpus(
        args.corpus,
        args.output,
        args.min_frequency,
        args.max_roots,
        max_sentences
    )
