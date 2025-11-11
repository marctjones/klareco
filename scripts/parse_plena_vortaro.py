"""
Parse Plena Vortaro de Esperanto to extract comprehensive vocabulary.

This extracts roots from the monolingual Esperanto dictionary.
Format is different from Gutenberg (Esperanto→Esperanto definitions).
"""

import re
from pathlib import Path
from collections import Counter


def extract_roots_from_plena_vortaro(pv_path: Path) -> set:
    """
    Extract all roots from Plena Vortaro.

    The format is approximately:
    radiko. Difino de la vorto...

    We extract the root (word before the first space/period/comma).
    """
    roots = set()

    print("Reading Plena Vortaro...")
    with open(pv_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()

    print("Extracting roots...")

    # Strategy: Find all Esperanto words in the text
    # Then extract roots by removing grammatical endings

    # Find all words (sequences of Esperanto letters)
    words = re.findall(r'[a-zA-ZĉĝĥĵŝŭĈĜĤĴŜŬ]+', text)

    print(f"Found {len(words)} total words in dictionary")

    for word in words:
        word = word.lower()

        # Skip very short words
        if len(word) < 2:
            continue

        # Skip if all uppercase (likely acronym)
        if word.isupper():
            continue

        # Try to extract root
        root = extract_root(word)
        if root and len(root) >= 2:
            roots.add(root)

    return roots


def extract_root(word: str) -> str:
    """
    Extract root from Esperanto word by removing grammatical endings.
    """
    word = word.strip().lower()

    if not word or len(word) < 2:
        return None

    # Remove accusative -n
    if word.endswith('n') and len(word) > 2:
        word_without_n = word[:-1]
        # Check if it's a valid word without -n
        if has_valid_ending(word_without_n):
            word = word_without_n

    # Remove plural -j
    if word.endswith('j') and len(word) > 2:
        word_without_j = word[:-1]
        if has_valid_ending(word_without_j):
            word = word_without_j

    # Remove grammatical endings
    # Verb endings (longest first)
    verb_endings = ['antaj', 'intaj', 'ontaj', 'ataj', 'itaj', 'otaj',
                    'anta', 'inta', 'onta', 'ata', 'ita', 'ota',
                    'ante', 'inte', 'onte', 'ate', 'ite', 'ote',
                    'as', 'is', 'os', 'us', 'i', 'u']

    for ending in verb_endings:
        if word.endswith(ending) and len(word) > len(ending) + 1:
            root = word[:-len(ending)]
            if len(root) >= 2:
                return root

    # Noun endings
    if word.endswith('o') and len(word) > 2:
        return word[:-1]

    # Adjective endings
    if word.endswith('a') and len(word) > 2:
        # But not articles
        if word not in ['la', 'na']:
            return word[:-1]

    # Adverb endings
    if word.endswith('e') and len(word) > 2:
        # But not common particles
        if word not in ['de', 'ke', 'se', 'ĉe', 'ne', 'je']:
            return word[:-1]

    # If no ending matched, return as-is (might be a root, particle, etc.)
    return word


def has_valid_ending(word: str) -> bool:
    """Check if word has a valid Esperanto grammatical ending."""
    if not word:
        return False

    valid_endings = ['o', 'a', 'e', 'as', 'is', 'os', 'us', 'u', 'i',
                     'oj', 'aj', 'on', 'an',
                     'anta', 'inta', 'onta', 'ata', 'ita', 'ota']

    return any(word.endswith(ending) for ending in valid_endings)


def clean_and_deduplicate(roots: set, existing_roots: set) -> set:
    """
    Clean extracted roots and remove duplicates with existing vocabulary.
    """
    cleaned = set()

    for root in roots:
        # Skip very short
        if len(root) < 2:
            continue

        # Skip numbers
        if root.isdigit():
            continue

        # Skip if it's actually a pronoun or correlative (already have these)
        pronouns = {'mi', 'vi', 'li', 'ŝi', 'ĝi', 'si', 'ni', 'ili', 'oni'}
        if root in pronouns:
            continue

        # Skip common particles (already have these)
        particles = {'kaj', 'sed', 'aŭ', 'se', 'ke', 'ĉar', 'ne', 'jes',
                     'la', 'jen', 'ja', 'nu'}
        if root in particles:
            continue

        cleaned.add(root)

    # Find new roots (not in existing vocabulary)
    new_roots = cleaned - existing_roots

    return cleaned, new_roots


def main():
    """Main entry point."""
    project_root = Path(__file__).parent.parent
    pv_path = project_root / 'data' / 'grammar' / 'plena_vortaro.txt'

    if not pv_path.exists():
        print(f"❌ Plena Vortaro not found at {pv_path}")
        print("Run: wget https://archive.org/download/Plena_Vortaro_de_Esperanto/PV_djvu.txt")
        return

    print("="*70)
    print("Plena Vortaro Root Extraction")
    print("="*70)
    print()

    # Extract roots
    pv_roots = extract_roots_from_plena_vortaro(pv_path)
    print(f"\n✓ Extracted {len(pv_roots)} unique roots from Plena Vortaro")

    # Load existing roots
    print("\nLoading existing vocabulary...")
    from data.extracted_vocabulary import DICTIONARY_ROOTS
    existing_roots = set(DICTIONARY_ROOTS)
    print(f"  Existing roots from Gutenberg: {len(existing_roots)}")

    # Clean and deduplicate
    print("\nCleaning and deduplicating...")
    cleaned_roots, new_roots = clean_and_deduplicate(pv_roots, existing_roots)

    print(f"  Total cleaned roots: {len(cleaned_roots)}")
    print(f"  New roots (not in Gutenberg): {len(new_roots)}")
    print(f"  Overlap with Gutenberg: {len(cleaned_roots - new_roots)}")

    # Merge with existing
    merged_roots = existing_roots | cleaned_roots
    print(f"\n✓ Merged vocabulary: {len(merged_roots)} roots")
    print(f"  Increase: {len(merged_roots) - len(existing_roots)} roots")
    print(f"  Growth: {(len(merged_roots) / len(existing_roots) - 1) * 100:.1f}%")

    # Show some examples of new roots
    print(f"\n📚 Sample of new roots from Plena Vortaro (first 50):")
    for i, root in enumerate(sorted(new_roots)[:50]):
        print(f"  {root}", end=',  ' if (i+1) % 10 != 0 else '\n')

    # Save merged vocabulary
    output_path = project_root / 'data' / 'merged_vocabulary.py'
    print(f"\n💾 Saving merged vocabulary to {output_path}...")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('"""Merged vocabulary from Gutenberg + Plena Vortaro."""\n\n')
        f.write('MERGED_ROOTS = {\n')

        sorted_roots = sorted(merged_roots)
        for i, root in enumerate(sorted_roots):
            f.write(f'    "{root}"')
            if i < len(sorted_roots) - 1:
                f.write(',')
            if (i + 1) % 5 == 0:
                f.write('\n')

        f.write('\n}\n')

    print(f"✓ Saved {len(merged_roots)} roots to {output_path}")

    # Statistics
    print(f"\n{'='*70}")
    print("Final Statistics")
    print(f"{'='*70}")
    print(f"Gutenberg dictionary:  {len(existing_roots):6} roots")
    print(f"Plena Vortaro:         {len(cleaned_roots):6} roots")
    print(f"Merged vocabulary:     {len(merged_roots):6} roots")
    print(f"{'='*70}")

    print("\n✅ Plena Vortaro parsing complete!")
    print("\nNext steps:")
    print("1. Update parser.py to use MERGED_ROOTS")
    print("2. Run: python scripts/test_literary_parsing.py")
    print("3. Compare before/after success rates")


if __name__ == '__main__':
    main()
