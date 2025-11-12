"""
Extract Missing Roots from Literary Corpus

Analyzes cleaned literary texts to find Esperanto roots that aren't in our vocabulary.
Uses intelligent heuristics to extract likely roots from failed parse attempts.
"""

import re
from pathlib import Path
from collections import Counter
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


def extract_words_from_file(file_path: Path) -> list:
    """Extract all words from a text file."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
    except Exception as e:
        print(f"❌ Could not read {file_path}: {e}")
        return []

    # Extract words (sequences of Esperanto letters)
    words = re.findall(r'[a-zA-ZĉĝĥĵŝŭĈĜĤĴŜŬ]+', text)
    return [w.lower() for w in words if len(w) >= 2]


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


def extract_missing_roots(corpus_dir: Path, min_frequency: int = 2):
    """
    Extract all roots from corpus that aren't in KNOWN_ROOTS.

    Args:
        corpus_dir: Directory with cleaned text files
        min_frequency: Minimum occurrence count to include root
    """
    print(f"\n{'='*70}")
    print("Extracting Missing Literary Roots")
    print(f"{'='*70}\n")

    print(f"Corpus directory: {corpus_dir}")
    print(f"Minimum frequency: {min_frequency}")
    print(f"Known roots: {len(KNOWN_ROOTS):,}")

    # Find all text files
    files = list(corpus_dir.glob('*.txt'))
    print(f"\nFound {len(files)} text files")

    # Extract all words from all files
    all_words = []
    for file_path in files:
        words = extract_words_from_file(file_path)
        all_words.extend(words)
        print(f"  {file_path.name}: {len(words):,} words")

    print(f"\nTotal words: {len(all_words):,}")

    # Extract roots
    print("\nExtracting roots...")
    extracted_roots = []
    for word in all_words:
        root = extract_root_from_word(word)
        if root:
            cleaned = clean_root(root)
            if cleaned:
                extracted_roots.append(cleaned)

    # Count frequencies
    root_counts = Counter(extracted_roots)
    print(f"  Unique roots found: {len(root_counts):,}")

    # Filter for missing roots
    missing_roots = {}
    for root, count in root_counts.items():
        if count >= min_frequency and root not in KNOWN_ROOTS:
            missing_roots[root] = count

    print(f"  Missing roots (>={min_frequency} occurrences): {len(missing_roots):,}")

    # Sort by frequency
    sorted_missing = sorted(missing_roots.items(), key=lambda x: x[1], reverse=True)

    # Statistics
    print(f"\n{'='*70}")
    print("Statistics")
    print(f"{'='*70}\n")
    print(f"Total words analyzed: {len(all_words):,}")
    print(f"Unique roots extracted: {len(root_counts):,}")
    print(f"Already in vocabulary: {len([r for r in root_counts if r in KNOWN_ROOTS]):,}")
    print(f"Missing roots: {len(missing_roots):,}")

    # Show top missing roots
    print(f"\n{'='*70}")
    print(f"Top 50 Missing Roots (by frequency)")
    print(f"{'='*70}\n")
    for i, (root, count) in enumerate(sorted_missing[:50], 1):
        print(f"{i:3}. {root:15} ({count:4} occurrences)")

    # Save to file
    output_path = corpus_dir.parent / 'literary_missing_roots.txt'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"# Missing Literary Roots (>={min_frequency} occurrences)\n")
        f.write(f"# Total: {len(missing_roots)} roots\n\n")

        for root, count in sorted_missing:
            f.write(f"{root:20} # {count} occurrences\n")

    print(f"\n💾 Saved missing roots to: {output_path}")

    # Generate Python set for easy merging
    python_output = corpus_dir.parent / 'literary_missing_roots.py'
    with open(python_output, 'w', encoding='utf-8') as f:
        f.write('"""Missing literary roots extracted from cleaned corpus."""\n\n')
        f.write('LITERARY_ROOTS = {\n')

        roots_list = [root for root, count in sorted_missing]
        for i, root in enumerate(sorted(roots_list)):
            f.write(f'    "{root}"')
            if i < len(roots_list) - 1:
                f.write(',')
            if (i + 1) % 5 == 0:
                f.write('\n')

        f.write('\n}\n')

    print(f"💾 Saved Python set to: {python_output}")
    print(f"\nTo use: Add LITERARY_ROOTS to parser's KNOWN_ROOTS")

    return missing_roots


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Extract missing literary roots from cleaned corpus'
    )
    parser.add_argument(
        '--corpus-dir',
        type=str,
        default='data/cleaned',
        help='Directory with cleaned text files (default: data/cleaned)'
    )
    parser.add_argument(
        '--min-frequency',
        type=int,
        default=2,
        help='Minimum occurrence count (default: 2)'
    )

    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    corpus_dir = project_root / args.corpus_dir

    if not corpus_dir.exists():
        print(f"❌ Corpus directory not found: {corpus_dir}")
        return

    extract_missing_roots(corpus_dir, args.min_frequency)


if __name__ == '__main__':
    main()
