#!/usr/bin/env python3
"""
Filter the Gutenberg Esperanto corpus by various criteria.

Usage:
    # Get only Zamenhof's works
    python filter_gutenberg_corpus.py --source zamenhof

    # Get short sentences (good for testing)
    python filter_gutenberg_corpus.py --max-words 20 --min-words 5

    # Get Zamenhof + grammar books, moderate length
    python filter_gutenberg_corpus.py --source zamenhof,grammar --max-words 30

    # Get high-quality subset for parser testing
    python filter_gutenberg_corpus.py --quality high --limit 1000
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict


def count_words(sentence: str) -> int:
    """Count words in a sentence."""
    return len(sentence.split())


def is_high_quality(sentence: str) -> bool:
    """Check if sentence is high quality for parser training."""
    # Remove formatting artifacts
    if '_' in sentence or '[' in sentence or ']' in sentence:
        return False

    # Check for metadata patterns
    if sentence.startswith(('ANTAŬPAROLO', 'ENHAVO', 'ĈAPITRO')):
        return False

    # Must be a reasonable length
    words = count_words(sentence)
    if words < 5 or words > 50:
        return False

    # Must contain esperanto characters or common words
    esperanto_chars = 'ĉĝĥĵŝŭĈĜĤĴŜŬ'
    common_words = ['la', 'de', 'kaj', 'en', 'estas', 'al']

    has_esperanto = any(c in sentence for c in esperanto_chars)
    has_common = any(f' {word} ' in f' {sentence.lower()} ' for word in common_words)

    return has_esperanto or has_common


def filter_corpus(
    data: List[Dict],
    sources: List[str] = None,
    min_words: int = None,
    max_words: int = None,
    quality: str = None,
    limit: int = None
) -> List[str]:
    """Filter corpus by criteria."""

    filtered = []

    for entry in data:
        # Filter by source
        if sources and entry['source'] not in sources:
            continue

        for sentence in entry['sentences']:
            # Filter by word count
            word_count = count_words(sentence)
            if min_words and word_count < min_words:
                continue
            if max_words and word_count > max_words:
                continue

            # Filter by quality
            if quality == 'high' and not is_high_quality(sentence):
                continue

            filtered.append(sentence)

            # Check limit
            if limit and len(filtered) >= limit:
                return filtered

    return filtered


def main():
    parser = argparse.ArgumentParser(
        description='Filter Gutenberg Esperanto corpus',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--source',
        help='Comma-separated list of sources: zamenhof,grammar,historical,original'
    )
    parser.add_argument(
        '--min-words',
        type=int,
        help='Minimum word count'
    )
    parser.add_argument(
        '--max-words',
        type=int,
        help='Maximum word count'
    )
    parser.add_argument(
        '--quality',
        choices=['high', 'any'],
        default='any',
        help='Quality filter: high = clean sentences only, any = all'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Maximum number of sentences to return'
    )
    parser.add_argument(
        '--output',
        help='Output file (default: stdout)'
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show statistics only, do not output sentences'
    )

    args = parser.parse_args()

    # Load corpus
    corpus_file = Path(__file__).parent.parent / 'data' / 'gutenberg_sentences.json'
    with open(corpus_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Parse sources
    sources = None
    if args.source:
        sources = args.source.split(',')

    # Apply filters
    filtered = filter_corpus(
        data,
        sources=sources,
        min_words=args.min_words,
        max_words=args.max_words,
        quality=args.quality,
        limit=args.limit
    )

    # Show statistics
    if args.stats or not args.output:
        print(f"Filtered corpus statistics:")
        print(f"  Total sentences: {len(filtered)}")
        if filtered:
            word_counts = [count_words(s) for s in filtered]
            print(f"  Average words per sentence: {sum(word_counts) / len(word_counts):.1f}")
            print(f"  Min/max words: {min(word_counts)}/{max(word_counts)}")
        print()

    # Output sentences
    if not args.stats:
        if args.output:
            output_file = Path(args.output)
            output_file.write_text('\n'.join(filtered), encoding='utf-8')
            print(f"Wrote {len(filtered)} sentences to {output_file}")
        else:
            print("Sample sentences (first 10):")
            print()
            for i, sentence in enumerate(filtered[:10], 1):
                print(f"{i}. {sentence}")
                print()


if __name__ == '__main__':
    main()
