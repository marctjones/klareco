"""
Semantic Categorization and Frequency Analysis for Esperanto Roots

This script analyzes vocabulary to:
1. Categorize roots by semantic type (verb, noun, adjective base)
2. Calculate frequency from corpus and text files
3. Generate enriched vocabulary with metadata
"""

import json
import re
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Set


def analyze_dictionary_for_categories(dict_path: Path) -> Dict[str, str]:
    """
    Analyze Gutenberg dictionary to infer root categories from English definitions.

    Returns dict mapping root -> category (verb/noun/adjective/unknown)
    """
    categories = {}

    with open(dict_path, 'r', encoding='utf-8') as f:
        for line in f:
            if '=' not in line:
                continue

            parts = line.split('=', 1)
            if len(parts) != 2:
                continue

            english = parts[0].strip().lower()
            esperanto = parts[1].strip()

            # Convert x-system
            esperanto = convert_x_system(esperanto)

            # Extract root from Esperanto word
            words = re.split(r'[,/;]| or | aux ', esperanto)
            for word in words:
                word = word.strip().rstrip('.').lower()
                if not word or ' ' in word or len(word) < 2:
                    continue

                root = extract_root(word)
                if not root or len(root) < 2:
                    continue

                # Infer category from English part of speech markers
                category = infer_category_from_english(english)

                # Only set if we don't have it yet (first occurrence wins)
                if root not in categories:
                    categories[root] = category

    return categories


def infer_category_from_english(english: str) -> str:
    """Infer semantic category from English definition."""
    english_lower = english.lower()

    # Check for explicit POS markers
    if '(v' in english_lower or 'verb' in english_lower:
        return 'verb'
    if '(n' in english_lower or 'noun' in english_lower:
        return 'noun'
    if '(adj' in english_lower or 'adjective' in english_lower:
        return 'adjective'
    if '(adv' in english_lower or 'adverb' in english_lower:
        return 'adverb'
    if '(prep' in english_lower or 'preposition' in english_lower:
        return 'preposition'
    if '(conj' in english_lower or 'conjunction' in english_lower:
        return 'conjunction'

    # Heuristics based on common English verb patterns
    verb_patterns = [
        r'^to\s+',  # "to run", "to eat"
        r'ing\s*$',  # "running", "eating"
        r'ed\s*$',   # "walked", "talked"
    ]
    for pattern in verb_patterns:
        if re.search(pattern, english_lower):
            return 'verb'

    # Heuristics for adjectives
    adjective_endings = ['ful', 'less', 'ous', 'ive', 'able', 'ible', 'al', 'ic']
    for ending in adjective_endings:
        if english_lower.endswith(ending):
            return 'adjective'

    # Common verbs (action words)
    action_words = ['run', 'walk', 'eat', 'drink', 'sleep', 'work', 'play',
                    'read', 'write', 'speak', 'think', 'feel', 'see', 'hear',
                    'make', 'take', 'give', 'get', 'go', 'come', 'say', 'tell']
    first_word = english_lower.split()[0] if english_lower else ''
    if first_word in action_words:
        return 'verb'

    # Common adjectives
    common_adjectives = ['good', 'bad', 'big', 'small', 'hot', 'cold', 'new', 'old',
                         'happy', 'sad', 'beautiful', 'ugly', 'fast', 'slow', 'easy',
                         'hard', 'strong', 'weak', 'rich', 'poor', 'full', 'empty']
    if first_word in common_adjectives:
        return 'adjective'

    return 'unknown'


def convert_x_system(text: str) -> str:
    """Convert x-system to proper Esperanto characters."""
    replacements = {
        'cx': 'ĉ', 'gx': 'ĝ', 'sx': 'ŝ', 'ux': 'ŭ', 'jx': 'ĵ', 'hx': 'ĥ',
        'Cx': 'Ĉ', 'Gx': 'Ĝ', 'Sx': 'Ŝ', 'Ux': 'Ŭ', 'Jx': 'Ĵ', 'Hx': 'Ĥ',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def extract_root(word: str) -> str:
    """Extract root from Esperanto word."""
    word = word.lower().strip().rstrip('.,;:!?')

    if not word or len(word) < 2 or ' ' in word:
        return None

    # Remove accusative -n
    if word.endswith('n') and len(word) > 2:
        word = word[:-1]

    # Remove plural -j
    if word.endswith('j') and len(word) > 2:
        word = word[:-1]

    # Remove verb endings
    verb_endings = ['as', 'is', 'os', 'us', 'i', 'u']
    for ending in verb_endings:
        if word.endswith(ending) and len(word) > len(ending) + 1:
            return word[:-len(ending)]

    # Remove noun/adjective/adverb endings
    if word.endswith('o') and len(word) > 2:
        return word[:-1]
    if word.endswith('a') and len(word) > 2:
        return word[:-1]
    if word.endswith('e') and len(word) > 2:
        return word[:-1]

    return word


def calculate_frequency_from_corpus(corpus_path: Path) -> Counter:
    """Calculate root frequency from test corpus."""
    frequencies = Counter()

    with open(corpus_path, 'r', encoding='utf-8') as f:
        corpus = json.load(f)

    for item in corpus:
        text = item if isinstance(item, str) else item.get('esperanto', '')
        words = re.findall(r"[a-zA-ZĉĝĥĵŝŭĈĜĤĴŜŬ]+", text.lower())

        for word in words:
            root = extract_root(word)
            if root and len(root) >= 2:
                frequencies[root] += 1

    return frequencies


def calculate_frequency_from_texts(data_dir: Path) -> Counter:
    """Calculate root frequency from all Esperanto text files."""
    frequencies = Counter()

    # Find Esperanto text files
    esperanto_files = []
    for ext in ['*.txt', '*.eo']:
        esperanto_files.extend(data_dir.glob(f'**/{ext}'))

    # Filter for likely Esperanto content
    esperanto_files = [f for f in esperanto_files
                       if 'esperanto' in f.name.lower() or 'eo' in f.stem]

    for file_path in esperanto_files:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()

            words = re.findall(r"[a-zA-ZĉĝĥĵŝŭĈĜĤĴŜŬ]+", text.lower())

            for word in words:
                root = extract_root(word)
                if root and len(root) >= 2:
                    frequencies[root] += 1
        except Exception as e:
            print(f"Warning: Could not read {file_path}: {e}")

    return frequencies


def categorize_by_usage_patterns(root: str, corpus_contexts: List[str]) -> str:
    """
    Infer category by analyzing how root is used in corpus.
    Look for patterns like: root + verb endings, root + noun endings, etc.
    """
    verb_endings = {'as', 'is', 'os', 'us', 'i', 'u', 'ante', 'inte', 'onte'}
    noun_endings = {'o', 'oj', 'on', 'ojn'}
    adj_endings = {'a', 'aj', 'an', 'ajn'}

    verb_count = 0
    noun_count = 0
    adj_count = 0

    for context in corpus_contexts:
        words = context.lower().split()
        for word in words:
            if root not in word:
                continue

            # Check what ending it has
            for ending in verb_endings:
                if word.endswith(ending) and word.replace(ending, '') == root:
                    verb_count += 1

            for ending in noun_endings:
                if word.endswith(ending) and word.replace(ending, '') == root:
                    noun_count += 1

            for ending in adj_endings:
                if word.endswith(ending) and word.replace(ending, '') == root:
                    adj_count += 1

    # Return most common usage
    if verb_count > noun_count and verb_count > adj_count:
        return 'verb'
    elif noun_count > adj_count:
        return 'noun'
    elif adj_count > 0:
        return 'adjective'

    return 'unknown'


def generate_enriched_vocabulary(output_path: Path):
    """
    Generate enriched vocabulary file with categories and frequencies.
    """
    project_root = Path(__file__).parent.parent
    dict_path = project_root / 'data' / 'grammar' / 'gutenberg_dict.txt'
    corpus_path = project_root / 'data' / 'test_corpus.json'
    data_dir = project_root / 'data'

    print("=== Generating Enriched Vocabulary ===\n")

    # 1. Analyze dictionary for categories
    print("1. Analyzing dictionary for semantic categories...")
    categories = analyze_dictionary_for_categories(dict_path)
    print(f"   Categorized {len(categories)} roots")

    # Count by category
    category_counts = Counter(categories.values())
    for cat, count in category_counts.most_common():
        print(f"   - {cat}: {count}")

    # 2. Calculate frequencies from corpus
    print("\n2. Calculating frequencies from test corpus...")
    corpus_freq = calculate_frequency_from_corpus(corpus_path)
    print(f"   Found {len(corpus_freq)} unique roots in corpus")

    # 3. Calculate frequencies from all texts
    print("\n3. Calculating frequencies from text files...")
    text_freq = calculate_frequency_from_texts(data_dir)
    print(f"   Found {len(text_freq)} unique roots in texts")

    # 4. Combine frequencies
    print("\n4. Combining frequency data...")
    combined_freq = corpus_freq + text_freq

    # 5. Load existing vocabulary
    from data.extracted_vocabulary import DICTIONARY_ROOTS

    # 6. Generate enriched vocabulary
    print("\n5. Generating enriched vocabulary...")
    enriched = {}

    for root in DICTIONARY_ROOTS:
        enriched[root] = {
            'root': root,
            'category': categories.get(root, 'unknown'),
            'corpus_frequency': corpus_freq.get(root, 0),
            'text_frequency': text_freq.get(root, 0),
            'total_frequency': combined_freq.get(root, 0),
        }

    # 7. Sort by frequency
    sorted_roots = sorted(enriched.items(),
                         key=lambda x: x[1]['total_frequency'],
                         reverse=True)

    # 8. Save to file
    print(f"\n6. Saving enriched vocabulary to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dict(sorted_roots), f, indent=2, ensure_ascii=False)

    # 9. Generate statistics
    print("\n=== Statistics ===")
    print(f"Total roots: {len(enriched)}")
    print(f"Categorized: {len([r for r in enriched.values() if r['category'] != 'unknown'])}")
    print(f"Unknown category: {len([r for r in enriched.values() if r['category'] == 'unknown'])}")

    print("\nCategory distribution:")
    cats = Counter(r['category'] for r in enriched.values())
    for cat, count in cats.most_common():
        pct = count / len(enriched) * 100
        print(f"  {cat}: {count} ({pct:.1f}%)")

    print("\nTop 20 most frequent roots:")
    for root, data in sorted_roots[:20]:
        print(f"  {root:15} {data['category']:12} (corpus: {data['corpus_frequency']:3}, "
              f"texts: {data['text_frequency']:4}, total: {data['total_frequency']:5})")

    print("\nCommon roots (frequency >= 5):")
    common = [r for r, d in enriched.items() if d['total_frequency'] >= 5]
    print(f"  {len(common)} roots appear 5+ times")

    print("\nRare roots (frequency = 0):")
    rare = [r for r, d in enriched.items() if d['total_frequency'] == 0]
    print(f"  {len(rare)} roots never appear in corpus/texts")

    print(f"\n✓ Enriched vocabulary saved to {output_path}")
    return enriched


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate enriched vocabulary with categories and frequencies'
    )
    parser.add_argument(
        '--output', '-o',
        default='data/enriched_vocabulary.json',
        help='Output file path (default: data/enriched_vocabulary.json)'
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    generate_enriched_vocabulary(output_path)


if __name__ == '__main__':
    main()
