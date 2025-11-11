"""
Extract Esperanto roots from the Gutenberg English-Esperanto dictionary.

This script parses the dictionary and extracts roots, handling:
- The x-system (cx=ĉ, gx=ĝ, sx=ŝ, ux=ŭ, jx=ĵ, hx=ĥ)
- Removing grammatical endings to get pure roots
- Filtering out multi-word expressions
- Categorizing roots by part of speech
"""

import re
from pathlib import Path

def convert_x_system(text):
    """Convert x-system to proper Esperanto characters."""
    replacements = {
        'cx': 'ĉ',
        'gx': 'ĝ',
        'sx': 'ŝ',
        'ux': 'ŭ',
        'jx': 'ĵ',
        'hx': 'ĥ',
        'Cx': 'Ĉ',
        'Gx': 'Ĝ',
        'Sx': 'Ŝ',
        'Ux': 'Ŭ',
        'Jx': 'Ĵ',
        'Hx': 'Ĥ',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

def extract_root(word):
    """Extract the root from an Esperanto word by removing endings."""
    word = word.strip().lower()

    # Remove trailing punctuation
    word = word.rstrip('.,;:!?')

    # Skip if multi-word or has spaces
    if ' ' in word or '(' in word or '[' in word:
        return None

    # Skip if it's not a valid word
    if not word or len(word) < 2:
        return None

    # Handle verb endings
    verb_endings = ['as', 'is', 'os', 'us', 'anta', 'inta', 'onta', 'ata', 'ita', 'ota', 'ante', 'inte', 'onte', 'ate', 'ite', 'ote']
    for ending in verb_endings:
        if word.endswith(ending) and len(word) > len(ending) + 1:
            root = word[:-len(ending)]
            if len(root) >= 2:
                return root

    # Handle infinitive -i
    if word.endswith('i') and len(word) > 2:
        # But not if it's a pronoun or known exception
        if word not in ['mi', 'li', 'si', 'ĉi']:
            root = word[:-1]
            if len(root) >= 2:
                return root

    # Handle noun -o, -oj, -on, -ojn
    if word.endswith('ojn') and len(word) > 4:
        return word[:-3]
    if word.endswith('oj') and len(word) > 3:
        return word[:-2]
    if word.endswith('on') and len(word) > 3:
        return word[:-2]
    if word.endswith('o') and len(word) > 2:
        return word[:-1]

    # Handle adjective -a, -aj, -an, -ajn
    if word.endswith('ajn') and len(word) > 4:
        return word[:-3]
    if word.endswith('aj') and len(word) > 3:
        return word[:-2]
    if word.endswith('an') and len(word) > 3:
        return word[:-2]
    if word.endswith('a') and len(word) > 2:
        return word[:-1]

    # Handle adverb -e
    if word.endswith('e') and len(word) > 2:
        return word[:-1]

    # If no ending matched, return the word itself (might be a root, preposition, etc.)
    return word

def extract_vocabulary(dict_path):
    """Extract all vocabulary from the dictionary."""
    roots = set()
    prepositions = set()
    conjunctions = set()
    particles = set()

    with open(dict_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Skip header and non-entry lines
            if '=' not in line:
                continue

            # Extract Esperanto part (after =)
            parts = line.split('=')
            if len(parts) < 2:
                continue

            esperanto_part = parts[1].strip()

            # Convert x-system to proper characters
            esperanto_part = convert_x_system(esperanto_part)

            # Remove parenthetical notes
            esperanto_part = re.sub(r'\[.*?\]', '', esperanto_part)
            esperanto_part = re.sub(r'\(.*?\)', '', esperanto_part)

            # Split by commas, slashes, or "or"
            words = re.split(r'[,/]| or | aux ', esperanto_part)

            for word in words:
                word = word.strip().rstrip('.')

                # Skip empty or multi-word expressions
                if not word or ' ' in word:
                    continue

                # Check for prepositions (from the English side)
                english_part = parts[0].strip()
                if '(prep' in english_part.lower():
                    prepositions.add(word.lower())
                    continue

                # Check for conjunctions
                if '(conj' in english_part.lower() or english_part.lower() in ['and', 'or', 'but', 'if', 'because']:
                    conjunctions.add(word.lower())
                    continue

                # Extract root
                root = extract_root(word)
                if root and len(root) >= 2:
                    roots.add(root)

    return sorted(roots), sorted(prepositions), sorted(conjunctions), sorted(particles)

def main():
    dict_path = Path(__file__).parent.parent / 'data' / 'grammar' / 'gutenberg_dict.txt'

    print("Extracting vocabulary from Esperanto dictionary...")
    roots, preps, conjs, parts = extract_vocabulary(dict_path)

    print(f"\n✅ Extracted {len(roots)} roots")
    print(f"✅ Extracted {len(preps)} prepositions")
    print(f"✅ Extracted {len(conjs)} conjunctions")
    print(f"✅ Extracted {len(parts)} particles")

    # Show some samples
    print(f"\n📚 Sample roots (first 50):")
    for i, root in enumerate(roots[:50]):
        print(f"  {root}", end=',  ' if (i+1) % 10 != 0 else '\n')

    print(f"\n\n🔗 Prepositions: {', '.join(preps[:20])}")
    print(f"🔗 Conjunctions: {', '.join(conjs[:15])}")

    # Save to file for easy import
    output_path = Path(__file__).parent.parent / 'data' / 'extracted_vocabulary.py'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('"""Extracted vocabulary from Gutenberg English-Esperanto Dictionary."""\n\n')
        f.write('DICTIONARY_ROOTS = {\n')
        for i, root in enumerate(roots):
            f.write(f'    "{root}"')
            if i < len(roots) - 1:
                f.write(',')
            if (i + 1) % 5 == 0:
                f.write('\n')
        f.write('\n}\n\n')

        f.write('PREPOSITIONS = {\n')
        for i, prep in enumerate(preps):
            f.write(f'    "{prep}"')
            if i < len(preps) - 1:
                f.write(',')
            if (i + 1) % 5 == 0:
                f.write('\n')
        f.write('\n}\n\n')

        f.write('CONJUNCTIONS = {\n')
        for i, conj in enumerate(conjs):
            f.write(f'    "{conj}"')
            if i < len(conjs) - 1:
                f.write(',')
            if (i + 1) % 5 == 0:
                f.write('\n')
        f.write('\n}\n')

    print(f"\n💾 Saved vocabulary to: {output_path}")
    print(f"\nTotal vocabulary size: {len(roots) + len(preps) + len(conjs) + len(parts)}")

if __name__ == '__main__':
    main()
