"""
Vocabulary Validation Tool for Klareco

This tool validates and manages Esperanto vocabulary by:
1. Extracting roots from multiple sources (dictionaries, corpus texts)
2. Cross-validating vocabulary across different resources
3. Identifying missing roots needed for corpus coverage
4. Generating reports on vocabulary coverage and quality
5. Detecting potential parsing improvements from vocabulary updates
"""

import json
import re
from pathlib import Path
from collections import Counter, defaultdict
from typing import Set, List, Dict, Tuple


def load_parser_vocabulary() -> Dict[str, Set[str]]:
    """Load current vocabulary from parser."""
    # Import parser vocabulary
    from klareco.parser import (
        KNOWN_ROOTS,
        KNOWN_PREFIXES,
        KNOWN_SUFFIXES,
        KNOWN_CONJUNCTIONS,
        KNOWN_PREPOSITIONS,
        KNOWN_CORRELATIVES,
        KNOWN_PARTICLES,
        KNOWN_PRONOUNS
    )

    return {
        'roots': set(KNOWN_ROOTS),
        'prefixes': set(KNOWN_PREFIXES),
        'suffixes': set(KNOWN_SUFFIXES),
        'conjunctions': set(KNOWN_CONJUNCTIONS),
        'prepositions': set(KNOWN_PREPOSITIONS),
        'correlatives': set(KNOWN_CORRELATIVES),
        'particles': set(KNOWN_PARTICLES),
        'pronouns': set(KNOWN_PRONOUNS),
    }


def load_gutenberg_dictionary(dict_path: Path) -> Dict[str, Set[str]]:
    """Load vocabulary from Gutenberg English-Esperanto dictionary."""
    from data.extracted_vocabulary import DICTIONARY_ROOTS, PREPOSITIONS, CONJUNCTIONS

    return {
        'roots': set(DICTIONARY_ROOTS),
        'prepositions': set(PREPOSITIONS),
        'conjunctions': set(CONJUNCTIONS),
    }


def extract_words_from_text(text: str) -> List[str]:
    """Extract all words from Esperanto text."""
    # Remove punctuation and split
    words = re.findall(r"[a-zA-ZĉĝĥĵŝŭĈĜĤĴŜŬ]+", text.lower())
    return words


def extract_root_from_word(word: str) -> str:
    """
    Extract root from Esperanto word using simple heuristics.
    This is a simplified version of the parser's extract_root logic.
    """
    word = word.lower().strip()

    # Skip if too short
    if len(word) < 2:
        return None

    # Remove accusative -n
    if word.endswith('n') and len(word) > 2:
        word = word[:-1]

    # Remove plural -j
    if word.endswith('j') and len(word) > 2:
        word = word[:-1]

    # Remove common endings
    endings = ['as', 'is', 'os', 'us', 'i', 'u', 'o', 'a', 'e']
    for ending in endings:
        if word.endswith(ending) and len(word) > len(ending) + 1:
            return word[:-len(ending)]

    return word


def analyze_corpus_coverage(corpus_path: Path, parser_vocab: Dict[str, Set[str]]) -> Dict:
    """Analyze how well parser vocabulary covers the test corpus."""
    with open(corpus_path, 'r', encoding='utf-8') as f:
        corpus = json.load(f)

    all_words = []
    all_roots = []
    missing_roots = Counter()

    for item in corpus:
        # Handle both string format and object format
        text = item if isinstance(item, str) else item.get('esperanto', '')
        words = extract_words_from_text(text)
        all_words.extend(words)

        for word in words:
            root = extract_root_from_word(word)
            if root:
                all_roots.append(root)
                if root not in parser_vocab['roots']:
                    missing_roots[root] += 1

    total_words = len(all_words)
    unique_words = len(set(all_words))
    unique_roots = len(set(all_roots))
    covered_roots = len([r for r in set(all_roots) if r in parser_vocab['roots']])

    coverage_pct = (covered_roots / unique_roots * 100) if unique_roots > 0 else 0

    return {
        'total_words': total_words,
        'unique_words': unique_words,
        'unique_roots': unique_roots,
        'covered_roots': covered_roots,
        'coverage_pct': coverage_pct,
        'missing_roots': dict(missing_roots.most_common(50)),
    }


def analyze_text_files(data_dir: Path, parser_vocab: Dict[str, Set[str]]) -> Dict:
    """Analyze Esperanto text files in data directory."""
    esperanto_files = list(data_dir.glob('**/*.txt'))

    # Filter for likely Esperanto files (exclude English docs, logs, etc.)
    esperanto_files = [f for f in esperanto_files
                       if 'esperanto' in f.name.lower() or 'eo' in f.stem]

    all_roots = Counter()
    missing_roots = Counter()

    for file_path in esperanto_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            words = extract_words_from_text(text)
            for word in words:
                root = extract_root_from_word(word)
                if root and len(root) >= 2:
                    all_roots[root] += 1
                    if root not in parser_vocab['roots']:
                        missing_roots[root] += 1
        except Exception as e:
            print(f"Warning: Could not read {file_path}: {e}")

    return {
        'files_analyzed': len(esperanto_files),
        'unique_roots_found': len(all_roots),
        'missing_roots': dict(missing_roots.most_common(100)),
    }


def compare_vocabularies(vocab1: Dict[str, Set[str]], vocab2: Dict[str, Set[str]],
                        name1: str, name2: str) -> Dict:
    """Compare two vocabulary sets."""
    comparison = {}

    for key in vocab1.keys():
        if key in vocab2:
            set1 = vocab1[key]
            set2 = vocab2[key]

            only_in_1 = set1 - set2
            only_in_2 = set2 - set1
            in_both = set1 & set2

            comparison[key] = {
                f'only_in_{name1}': len(only_in_1),
                f'only_in_{name2}': len(only_in_2),
                'in_both': len(in_both),
                'examples_only_in_' + name1: list(only_in_1)[:10],
                'examples_only_in_' + name2: list(only_in_2)[:10],
            }

    return comparison


def generate_report(output_path: Path = None) -> str:
    """Generate comprehensive vocabulary validation report."""
    project_root = Path(__file__).parent.parent

    print("=== Klareco Vocabulary Validation Report ===\n")

    # 1. Load vocabularies
    print("Loading vocabularies...")
    parser_vocab = load_parser_vocabulary()
    dict_vocab = load_gutenberg_dictionary(project_root / 'data' / 'grammar' / 'gutenberg_dict.txt')

    # 2. Parser vocabulary stats
    print("\n--- Parser Vocabulary Statistics ---")
    for key, items in parser_vocab.items():
        print(f"  {key.capitalize()}: {len(items)}")
    total_parser = sum(len(v) for v in parser_vocab.values())
    print(f"  TOTAL: {total_parser}")

    # 3. Dictionary vocabulary stats
    print("\n--- Gutenberg Dictionary Statistics ---")
    for key, items in dict_vocab.items():
        print(f"  {key.capitalize()}: {len(items)}")
    total_dict = sum(len(v) for v in dict_vocab.values())
    print(f"  TOTAL: {total_dict}")

    # 4. Corpus coverage analysis
    corpus_path = project_root / 'data' / 'test_corpus.json'
    if corpus_path.exists():
        print("\n--- Test Corpus Coverage Analysis ---")
        corpus_analysis = analyze_corpus_coverage(corpus_path, parser_vocab)
        print(f"  Total words in corpus: {corpus_analysis['total_words']}")
        print(f"  Unique words: {corpus_analysis['unique_words']}")
        print(f"  Unique roots: {corpus_analysis['unique_roots']}")
        print(f"  Covered roots: {corpus_analysis['covered_roots']}")
        print(f"  Coverage: {corpus_analysis['coverage_pct']:.1f}%")

        if corpus_analysis['missing_roots']:
            print(f"\n  Top 10 missing roots in corpus:")
            for root, count in list(corpus_analysis['missing_roots'].items())[:10]:
                print(f"    {root}: {count} occurrences")

    # 5. Text files analysis
    data_dir = project_root / 'data'
    if data_dir.exists():
        print("\n--- Esperanto Text Files Analysis ---")
        text_analysis = analyze_text_files(data_dir, parser_vocab)
        print(f"  Files analyzed: {text_analysis['files_analyzed']}")
        print(f"  Unique roots found: {text_analysis['unique_roots_found']}")

        if text_analysis['missing_roots']:
            print(f"\n  Top 20 missing roots across all texts:")
            for root, count in list(text_analysis['missing_roots'].items())[:20]:
                print(f"    {root}: {count} occurrences")

    # 6. Vocabulary comparison
    print("\n--- Parser vs Dictionary Comparison ---")
    comparison = compare_vocabularies(parser_vocab, dict_vocab, 'parser', 'dictionary')
    for key, stats in comparison.items():
        if stats:
            print(f"\n  {key.capitalize()}:")
            print(f"    In both: {stats['in_both']}")
            print(f"    Only in parser: {stats['only_in_parser']}")
            print(f"    Only in dictionary: {stats['only_in_dictionary']}")

    # 7. Quality recommendations
    print("\n--- Recommendations ---")
    if corpus_analysis['coverage_pct'] < 95:
        print(f"  ⚠️  Corpus coverage is {corpus_analysis['coverage_pct']:.1f}% - consider adding missing roots")
    else:
        print(f"  ✅ Excellent corpus coverage: {corpus_analysis['coverage_pct']:.1f}%")

    print(f"\n  Total vocabulary size: {total_parser} items")
    if total_parser > 8000:
        print(f"  ✅ Comprehensive vocabulary (8000+ items)")

    print("\n=== End of Report ===\n")

    # Save report to file if requested
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            # Save detailed JSON report
            report_data = {
                'parser_vocab': {k: len(v) for k, v in parser_vocab.items()},
                'dict_vocab': {k: len(v) for k, v in dict_vocab.items()},
                'corpus_coverage': corpus_analysis if corpus_path.exists() else {},
                'text_analysis': text_analysis if data_dir.exists() else {},
                'comparison': comparison,
            }
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        print(f"Detailed report saved to: {output_path}")

    return "Report generated successfully"


def main():
    """Main entry point for vocabulary validation."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Validate and analyze Klareco vocabulary coverage'
    )
    parser.add_argument('--output', '-o', help='Save detailed report to JSON file')
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else None
    generate_report(output_path)


if __name__ == '__main__':
    main()
