"""
Extract Tolkien-Specific Vocabulary

Analyzes Tolkien texts to identify fantasy vocabulary that's missing.
"""

from pathlib import Path
from collections import Counter
import re
from klareco.parser import parse_word


def extract_tolkien_failures(file_path: Path, num_sentences: int = 50):
    """Extract all parse failures from a Tolkien text."""
    print(f"\n{'='*70}")
    print(f"Extracting Failures: {file_path.name}")
    print(f"{'='*70}\n")

    # Read file
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
    except Exception as e:
        print(f"❌ Could not read file: {e}")
        return []

    # Skip headers
    text = text[3000:]

    # Extract sentences
    sentences = re.split(r'\n\n+|[.!?]+\s+', text)

    # Filter for valid Esperanto sentences
    valid_sentences = []
    for sent in sentences:
        sent = sent.strip()
        if (sent and len(sent) > 10 and len(sent) < 200 and
            re.search(r'[a-zA-ZĉĝĥĵŝŭĈĜĤĴŜŬ]{3,}', sent)):
            valid_sentences.append(sent)
            if len(valid_sentences) >= num_sentences:
                break

    print(f"Testing {len(valid_sentences)} sentences...")

    # Extract all words and try parsing
    failures = []
    for sentence in valid_sentences:
        words = re.findall(r'[a-zA-ZĉĝĥĵŝŭĈĜĤĴŜŬ]+', sentence)
        for word in words:
            if len(word) < 2:
                continue
            try:
                parse_word(word)
            except Exception as e:
                error_msg = str(e)
                # Extract the remaining stem from error message
                if "Restaĵo:" in error_msg:
                    stem = error_msg.split("Restaĵo: '")[1].split("'")[0]
                    failures.append((word.lower(), stem))
                elif "havas neniun konatan finaĵon" in error_msg:
                    failures.append((word.lower(), "no_ending"))

    return failures


def categorize_tolkien_roots(failures):
    """Categorize Tolkien-specific roots."""
    failure_counts = Counter([stem for word, stem in failures])

    print(f"\n{'='*70}")
    print("Tolkien-Specific Roots Needed")
    print(f"{'='*70}\n")

    # Categorize by pattern
    proper_nouns = []  # Capitalized names
    fantasy_races = []  # elf, dwarf, orc, hobbit
    fantasy_items = []  # silmaril, ring-related
    fantasy_places = []  # place names
    common_words = []  # normal vocabulary

    for stem, count in failure_counts.most_common():
        if count < 2:  # Skip rare errors
            continue

        # Categorize
        if stem in ['melkor', 'feanor', 'erendil', 'gandalf', 'bilb', 'frodo',
                    'gollum', 'aragorn', 'legol', 'gimli', 'galadriel', 'elrond']:
            proper_nouns.append((stem, count))
        elif stem in ['nold', 'ork', 'hobit', 'elf', 'balrog', 'nazgul',
                      'entan', 'trol', 'uruk']:
            fantasy_races.append((stem, count))
        elif stem in ['silmaril', 'palantir', 'mithril', 'andul', 'mordor',
                      'angband', 'ising', 'rivendel']:
            fantasy_items.append((stem, count))
        elif stem in ['mez', 'ter', 'ŝir', 'bagin']:
            fantasy_places.append((stem, count))
        else:
            common_words.append((stem, count))

    print(f"Proper Nouns ({len(proper_nouns)}):")
    for stem, count in proper_nouns:
        print(f"  {stem:20} ({count} occurrences)")

    print(f"\nFantasy Races ({len(fantasy_races)}):")
    for stem, count in fantasy_races:
        print(f"  {stem:20} ({count} occurrences)")

    print(f"\nFantasy Items/Places ({len(fantasy_items)}):")
    for stem, count in fantasy_items:
        print(f"  {stem:20} ({count} occurrences)")

    print(f"\nCommon Words ({len(common_words)}):")
    for stem, count in common_words[:20]:  # Top 20
        print(f"  {stem:20} ({count} occurrences)")

    return {
        'proper_nouns': proper_nouns,
        'fantasy_races': fantasy_races,
        'fantasy_items': fantasy_items,
        'common_words': common_words
    }


def generate_tolkien_vocabulary(categorized):
    """Generate Tolkien vocabulary extension."""
    all_roots = set()

    for category in categorized.values():
        for stem, count in category:
            if stem != "no_ending" and len(stem) >= 2:
                all_roots.add(stem)

    return sorted(all_roots)


def main():
    """Main entry point."""
    project_root = Path(__file__).parent.parent

    # Test both Tolkien works
    hobbit_path = project_root / 'data/cleaned/cleaned_la_hobito.txt'
    lotr_path = project_root / 'data/cleaned/cleaned_la_mastro_de_l_ringoj.txt'

    all_failures = []

    if hobbit_path.exists():
        failures = extract_tolkien_failures(hobbit_path, num_sentences=100)
        all_failures.extend(failures)

    if lotr_path.exists():
        failures = extract_tolkien_failures(lotr_path, num_sentences=100)
        all_failures.extend(failures)

    # Categorize
    categorized = categorize_tolkien_roots(all_failures)

    # Generate vocabulary
    tolkien_roots = generate_tolkien_vocabulary(categorized)

    print(f"\n{'='*70}")
    print(f"Total Tolkien Roots: {len(tolkien_roots)}")
    print(f"{'='*70}\n")

    # Save to file
    output_path = project_root / 'data' / 'tolkien_vocabulary.py'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('"""Tolkien-specific vocabulary for fantasy texts."""\n\n')
        f.write('TOLKIEN_ROOTS = {\n')

        for i, root in enumerate(tolkien_roots):
            f.write(f'    "{root}"')
            if i < len(tolkien_roots) - 1:
                f.write(',')
            if (i + 1) % 5 == 0:
                f.write('\n')

        f.write('\n}\n')

    print(f"💾 Saved to: {output_path}")

    # Also save categorized version for reference
    ref_path = project_root / 'data' / 'tolkien_vocabulary_categorized.txt'
    with open(ref_path, 'w', encoding='utf-8') as f:
        f.write("# Tolkien-Specific Vocabulary (Categorized)\n\n")

        f.write("## Proper Nouns (Character Names)\n")
        for stem, count in categorized['proper_nouns']:
            f.write(f"{stem:20} # {count} occurrences\n")

        f.write("\n## Fantasy Races\n")
        for stem, count in categorized['fantasy_races']:
            f.write(f"{stem:20} # {count} occurrences\n")

        f.write("\n## Fantasy Items/Places\n")
        for stem, count in categorized['fantasy_items']:
            f.write(f"{stem:20} # {count} occurrences\n")

        f.write("\n## Common Words\n")
        for stem, count in categorized['common_words'][:50]:
            f.write(f"{stem:20} # {count} occurrences\n")

    print(f"💾 Categorized reference: {ref_path}")


if __name__ == '__main__':
    main()
