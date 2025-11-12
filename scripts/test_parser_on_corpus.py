#!/usr/bin/env python3
"""
Test the parser on Gutenberg corpus and analyze failures.

This script:
1. Loads sentences from the Gutenberg corpus
2. Attempts to parse each one
3. Tracks success/failure rates
4. Categorizes failure types
5. Reports most common missing roots/affixes
"""

import json
import sys
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

# Add klareco to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.parser import parse


def analyze_error(error_msg: str, sentence: str) -> Tuple[str, str]:
    """Categorize the error and extract the problematic word."""
    if "Ne povis trovi validan radikon" in error_msg:
        # Extract the word
        import re
        match = re.search(r"'([^']+)'", error_msg)
        word = match.group(1) if match else "unknown"
        return ("missing_root", word)
    elif "Nekonata prefikso" in error_msg:
        match = re.search(r"'([^']+)'", error_msg)
        word = match.group(1) if match else "unknown"
        return ("unknown_prefix", word)
    elif "Nekonata sufikso" in error_msg:
        match = re.search(r"'([^']+)'", error_msg)
        word = match.group(1) if match else "unknown"
        return ("unknown_suffix", word)
    elif "Nekonata vortofino" in error_msg:
        return ("unknown_ending", "")
    else:
        return ("other", "")


def test_corpus(
    sentences: List[str],
    max_test: int = 100,
    verbose: bool = False
) -> Dict:
    """Test parser on corpus sentences."""

    results = {
        'total': 0,
        'success': 0,
        'failed': 0,
        'error_types': Counter(),
        'missing_roots': Counter(),
        'unknown_prefixes': Counter(),
        'unknown_suffixes': Counter(),
        'failed_sentences': []
    }

    for i, sentence in enumerate(sentences[:max_test]):
        results['total'] += 1

        # Clean sentence (remove extra whitespace, newlines)
        sentence = ' '.join(sentence.split())

        if verbose and i % 10 == 0:
            print(f"Testing {i}/{max_test}...", end='\r')

        try:
            ast = parse(sentence)
            results['success'] += 1
        except Exception as e:
            results['failed'] += 1
            error_msg = str(e)
            error_type, word = analyze_error(error_msg, sentence)

            results['error_types'][error_type] += 1

            if error_type == 'missing_root':
                results['missing_roots'][word.lower()] += 1
            elif error_type == 'unknown_prefix':
                results['unknown_prefixes'][word] += 1
            elif error_type == 'unknown_suffix':
                results['unknown_suffixes'][word] += 1

            # Store failed sentence for analysis
            if len(results['failed_sentences']) < 20:
                results['failed_sentences'].append({
                    'sentence': sentence[:100] + '...' if len(sentence) > 100 else sentence,
                    'error': error_msg[:150]
                })

    if verbose:
        print()  # Clear the progress line

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Test parser on Gutenberg corpus')
    parser.add_argument('--source', help='Filter by source (zamenhof, grammar, etc.)')
    parser.add_argument('--max-test', type=int, default=500, help='Max sentences to test')
    parser.add_argument('--verbose', action='store_true', help='Show progress')
    parser.add_argument('--quality', choices=['high', 'any'], default='high')

    args = parser.parse_args()

    # Load corpus
    corpus_file = Path(__file__).parent.parent / 'data' / 'gutenberg_sentences.json'
    print(f"Loading corpus from {corpus_file}")

    with open(corpus_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Filter sentences
    sentences = []
    for entry in data:
        if args.source and entry['source'] != args.source:
            continue
        sentences.extend(entry['sentences'])

    # Apply quality filter
    if args.quality == 'high':
        from filter_gutenberg_corpus import is_high_quality
        sentences = [s for s in sentences if is_high_quality(s)]

    print(f"Testing {min(args.max_test, len(sentences))} sentences from corpus")
    print(f"Source filter: {args.source or 'all'}")
    print(f"Quality filter: {args.quality}")
    print()

    # Run tests
    results = test_corpus(sentences, max_test=args.max_test, verbose=args.verbose)

    # Print results
    print("=" * 70)
    print("PARSER TEST RESULTS")
    print("=" * 70)
    print(f"Total tested: {results['total']}")
    print(f"Successful:   {results['success']} ({results['success']/results['total']*100:.1f}%)")
    print(f"Failed:       {results['failed']} ({results['failed']/results['total']*100:.1f}%)")
    print()

    print("ERROR TYPES:")
    print("-" * 70)
    for error_type, count in results['error_types'].most_common():
        print(f"  {error_type:20s}: {count:4d} ({count/results['failed']*100:.1f}%)")
    print()

    if results['missing_roots']:
        print("TOP 30 MISSING ROOTS:")
        print("-" * 70)
        for word, count in results['missing_roots'].most_common(30):
            print(f"  {word:20s}: {count:3d} occurrences")
        print()

    if results['unknown_prefixes']:
        print("UNKNOWN PREFIXES:")
        print("-" * 70)
        for word, count in results['unknown_prefixes'].most_common(10):
            print(f"  {word:20s}: {count:3d} occurrences")
        print()

    if results['unknown_suffixes']:
        print("UNKNOWN SUFFIXES:")
        print("-" * 70)
        for word, count in results['unknown_suffixes'].most_common(10):
            print(f"  {word:20s}: {count:3d} occurrences")
        print()

    print("SAMPLE FAILURES (first 10):")
    print("-" * 70)
    for i, failure in enumerate(results['failed_sentences'][:10], 1):
        print(f"{i}. {failure['sentence']}")
        print(f"   Error: {failure['error']}")
        print()


if __name__ == '__main__':
    main()
