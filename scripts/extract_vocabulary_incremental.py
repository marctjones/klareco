#!/usr/bin/env python3
"""
Extract Missing Roots Incrementally from Large Corpus

Processes files one at a time to avoid memory issues.
Aggregates results across all files.
"""

import re
import sys
from pathlib import Path
from collections import Counter

# Add klareco to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.parser import KNOWN_ROOTS, KNOWN_PREFIXES, KNOWN_SUFFIXES


def extract_root_from_word(word: str) -> str:
    """
    Attempt to extract root from an Esperanto word by stripping endings.
    """
    word = word.lower().strip()

    if not word or len(word) < 2:
        return None

    # Skip numbers
    if word.isdigit():
        return None

    # Remove accusative -n
    if word.endswith('n') and len(word) > 2:
        word = word[:-1]

    # Remove plural -j
    if word.endswith('j') and len(word) > 2:
        word = word[:-1]

    # Remove verb endings (check longest first)
    verb_endings = ['antis', 'intis', 'ontis', 'atis', 'itis', 'otis',
                    'ante', 'inte', 'onte', 'ate', 'ite', 'ote',
                    'anta', 'inta', 'onta', 'ata', 'ita', 'ota',
                    'as', 'is', 'os', 'us', 'u', 'i']

    for ending in verb_endings:
        if word.endswith(ending) and len(word) > len(ending) + 1:
            return word[:-len(ending)]

    # Remove noun endings
    if word.endswith('o') and len(word) > 2:
        return word[:-1]

    # Remove adjective endings
    if word.endswith('a') and len(word) > 2:
        # But not articles
        if word not in ['la', 'na']:
            return word[:-1]

    # Remove adverb endings
    if word.endswith('e') and len(word) > 2:
        # But not common particles
        if word not in ['de', 'ke', 'se', 'ĉe', 'ne', 'je']:
            return word[:-1]

    # If no ending matched, return as-is
    return word


def clean_root(root: str) -> str:
    """Clean extracted root by removing known affixes."""
    if not root or len(root) < 2:
        return None

    # Remove prefixes
    for prefix in KNOWN_PREFIXES:
        if root.startswith(prefix) and len(root) > len(prefix) + 1:
            root = root[len(prefix):]
            break

    # Remove suffixes (check longest first)
    sorted_suffixes = sorted(KNOWN_SUFFIXES, key=len, reverse=True)
    for suffix in sorted_suffixes:
        if root.endswith(suffix) and len(root) > len(suffix) + 1:
            root = root[:-len(suffix)]
            # Check for more suffixes
            for suffix2 in sorted_suffixes:
                if root.endswith(suffix2) and len(root) > len(suffix2) + 1:
                    root = root[:-len(suffix2)]
                    break
            break

    # Skip if too short
    if len(root) < 2:
        return None

    # Skip if it's a particle or correlative (already covered)
    common_particles = {'la', 'kaj', 'sed', 'aŭ', 'se', 'ke', 'ne', 'ja', 'nu',
                        'jes', 'en', 'de', 'al', 'el', 'da', 'pri', 'pro', 'kun'}
    if root in common_particles:
        return None

    return root


def process_file_incrementally(file_path: Path) -> Counter:
    """
    Process a single file and return root counts.
    Memory-efficient: processes in chunks without loading entire file.
    """
    root_counts = Counter()

    try:
        # Read file in chunks
        chunk_size = 1024 * 1024  # 1MB at a time
        text_buffer = ""

        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break

                text_buffer += chunk

                # Process complete words (keep incomplete word at end for next iteration)
                words = re.findall(r'[a-zA-ZĉĝĥĵŝŭĈĜĤĴŜŬ]+', text_buffer)

                # Keep last word for next chunk (might be incomplete)
                if words:
                    text_buffer = words[-1]
                    words = words[:-1]
                else:
                    text_buffer = ""

                # Extract roots from words
                for word in words:
                    word = word.lower()
                    if len(word) >= 2:
                        root = extract_root_from_word(word)
                        if root:
                            cleaned = clean_root(root)
                            if cleaned:
                                root_counts[cleaned] += 1

        # Process final buffer
        if text_buffer:
            root = extract_root_from_word(text_buffer.lower())
            if root:
                cleaned = clean_root(root)
                if cleaned:
                    root_counts[cleaned] += 1

    except Exception as e:
        print(f"  ⚠ Error processing {file_path.name}: {e}")

    return root_counts


def extract_missing_roots_incremental(corpus_dir: Path, min_frequency: int = 3):
    """
    Extract all roots from corpus incrementally (one file at a time).

    Args:
        corpus_dir: Directory with text files
        min_frequency: Minimum occurrence count to include root
    """
    print(f"\n{'='*70}")
    print("Extracting Missing Roots (Incremental Processing)")
    print(f"{'='*70}\n")

    print(f"Corpus directory: {corpus_dir}")
    print(f"Minimum frequency: {min_frequency}")
    print(f"Known roots: {len(KNOWN_ROOTS):,}")

    # Find all text files (including files without .txt extension)
    files = sorted([f for f in corpus_dir.iterdir() if f.is_file()])
    print(f"\nFound {len(files)} files")
    print(f"\nProcessing files incrementally (memory-efficient)...")

    # Process each file and aggregate results
    total_root_counts = Counter()

    for i, file_path in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] Processing {file_path.name}...", end=" ", flush=True)

        file_root_counts = process_file_incrementally(file_path)
        total_root_counts.update(file_root_counts)

        unique_roots = len(file_root_counts)
        print(f"✓ ({unique_roots:,} unique roots)")

    print(f"\n{'='*70}")
    print("Aggregating Results")
    print(f"{'='*70}\n")

    print(f"Total unique roots extracted: {len(total_root_counts):,}")

    # Filter for missing roots
    missing_roots = {}
    for root, count in total_root_counts.items():
        if count >= min_frequency and root not in KNOWN_ROOTS:
            missing_roots[root] = count

    print(f"Missing roots (>={min_frequency} occurrences): {len(missing_roots):,}")

    # Sort by frequency
    sorted_missing = sorted(missing_roots.items(), key=lambda x: x[1], reverse=True)

    # Statistics
    print(f"\n{'='*70}")
    print("Statistics")
    print(f"{'='*70}\n")
    print(f"Total unique roots extracted: {len(total_root_counts):,}")
    print(f"Already in vocabulary: {len([r for r in total_root_counts if r in KNOWN_ROOTS]):,}")
    print(f"Missing roots: {len(missing_roots):,}")
    print(f"  High frequency (>=10): {len([r for r, c in missing_roots.items() if c >= 10]):,}")
    print(f"  Medium frequency (5-9): {len([r for r, c in missing_roots.items() if 5 <= c < 10]):,}")
    print(f"  Low frequency (3-4): {len([r for r, c in missing_roots.items() if 3 <= c < 5]):,}")

    # Show top missing roots
    print(f"\n{'='*70}")
    print(f"Top 100 Missing Roots (by frequency)")
    print(f"{'='*70}\n")
    for i, (root, count) in enumerate(sorted_missing[:100], 1):
        print(f"{i:3}. {root:20} ({count:6,} occurrences)")

    # Save to file
    output_path = corpus_dir.parent / f'{corpus_dir.name}_missing_roots.txt'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"# Missing Roots from {corpus_dir.name}\n")
        f.write(f"# Minimum frequency: {min_frequency}\n")
        f.write(f"# Total: {len(missing_roots)} roots\n\n")

        for root, count in sorted_missing:
            f.write(f"{root:20} # {count:,} occurrences\n")

    print(f"\n💾 Saved missing roots to: {output_path}")

    # Generate Python set for easy merging
    python_output = corpus_dir.parent / f'{corpus_dir.name}_missing_roots.py'
    with open(python_output, 'w', encoding='utf-8') as f:
        f.write(f'"""Missing roots extracted from {corpus_dir.name}."""\n\n')
        f.write(f'# Total: {len(missing_roots)} roots\n')
        f.write(f'# Minimum frequency: {min_frequency}\n\n')
        f.write(f'{corpus_dir.name.upper()}_MISSING_ROOTS = {{\n')

        roots_list = [root for root, count in sorted_missing]
        for i, root in enumerate(sorted(roots_list)):
            f.write(f'    "{root}"')
            if i < len(roots_list) - 1:
                f.write(',')
            f.write('\n')

        f.write('}\n')

    print(f"💾 Saved Python set to: {python_output}")
    print(f"\nTo use: Add {corpus_dir.name.upper()}_MISSING_ROOTS to parser's KNOWN_ROOTS")

    return missing_roots


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Extract missing roots incrementally from large corpus'
    )
    parser.add_argument(
        '--corpus-dir',
        type=str,
        required=True,
        help='Directory with text files to process'
    )
    parser.add_argument(
        '--min-frequency',
        type=int,
        default=3,
        help='Minimum occurrence count (default: 3)'
    )

    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    corpus_dir = project_root / args.corpus_dir

    if not corpus_dir.exists():
        print(f"❌ Corpus directory not found: {corpus_dir}")
        return

    extract_missing_roots_incremental(corpus_dir, args.min_frequency)


if __name__ == '__main__':
    main()
