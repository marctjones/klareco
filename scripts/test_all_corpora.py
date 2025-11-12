#!/usr/bin/env python3
"""
Test parser on ALL available Esperanto corpora.

Tests word-level success rates on:
- Gutenberg collection (Zamenhof, etc.)
- Tolkien (Hobbit, Lord of the Rings)
- Wikipedia
- Literary works (Poe, etc.)
"""

import json
import sys
import random
from pathlib import Path
from collections import defaultdict, Counter

# Add klareco to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.parser import parse


def sample_sentences_from_text(text: str, num_samples: int = 100) -> list:
    """Extract sample sentences from a large text file."""
    # Split on sentence boundaries
    import re

    # First, try to clean up HTML/XML if present
    text = re.sub(r'<[^>]+>', ' ', text)  # Remove HTML tags
    text = re.sub(r'http\S+', ' ', text)  # Remove URLs

    sentences = re.split(r'(?<=[.!?])\s+', text)

    # Filter sentences
    esperanto_chars = set('ĉĝĥĵŝŭĈĜĤĴŜŬ')
    filtered = []

    for s in sentences:
        s = s.strip()
        # Must be reasonable length
        if not (20 < len(s) < 500):
            continue
        # Must have at least some Esperanto characteristics
        # (either special chars or common words)
        has_eo_chars = any(c in s for c in esperanto_chars)
        common_eo_words = ['la', 'kaj', 'estas', 'de', 'en']
        has_common = any(f' {w} ' in f' {s.lower()} ' for w in common_eo_words)

        if has_eo_chars or has_common:
            filtered.append(s)

    # Random sample if we have more than needed
    if len(filtered) > num_samples:
        filtered = random.sample(filtered, num_samples)

    return filtered


def test_corpus_file(filepath: Path, num_samples: int = 100, verbose: bool = False):
    """Test parser on a single corpus file."""
    if verbose:
        print(f"  Loading {filepath.name}...", end=" ")

    try:
        text = filepath.read_text(encoding='utf-8')
    except Exception as e:
        print(f"  ✗ Failed to read: {e}")
        return None

    # Extract sentences
    sentences = sample_sentences_from_text(text, num_samples)

    if verbose:
        print(f"extracted {len(sentences)} sentences")

    results = {
        'file': filepath.name,
        'total_sentences': 0,
        'successful_sentences': 0,
        'total_words': 0,
        'esperanto_words': 0,
        'non_esperanto_words': 0,
        'categories': Counter()
    }

    for i, sentence in enumerate(sentences):
        results['total_sentences'] += 1

        try:
            ast = parse(sentence)
            results['successful_sentences'] += 1

            stats = ast.get('parse_statistics', {})
            results['total_words'] += stats.get('total_words', 0)
            results['esperanto_words'] += stats.get('esperanto_words', 0)
            results['non_esperanto_words'] += stats.get('non_esperanto_words', 0)

            for category, count in stats.get('categories', {}).items():
                results['categories'][category] += count

        except Exception as e:
            # Should not happen with graceful handling
            if verbose:
                print(f"    Unexpected error: {e}")
            continue

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Test parser on all Esperanto corpora')
    parser.add_argument('--samples', type=int, default=100, help='Sentences per file')
    parser.add_argument('--verbose', action='store_true', help='Show progress')

    args = parser.parse_args()

    data_dir = Path(__file__).parent.parent / 'data'

    # Define corpus categories
    corpora = {
        'Gutenberg (tested)': {
            'files': [],
            'note': 'Already tested via gutenberg_sentences.json'
        },
        'Tolkien': {
            'files': [
                data_dir / 'clean_corpus' / 'la_hobito.txt',
                data_dir / 'clean_corpus' / 'la_mastro_de_l_ringoj.txt'
            ]
        },
        'Poe Stories': {
            'files': [
                data_dir / 'clean_corpus' / 'kadavrejo_strato.txt',
                data_dir / 'clean_corpus' / 'la_korvo.txt',
                data_dir / 'clean_corpus' / 'puto_kaj_pendolo.txt',
                data_dir / 'clean_corpus' / 'ses_noveloj.txt',
                data_dir / 'clean_corpus' / 'usxero_domo.txt',
            ]
        },
        'Other Classics': {
            'files': [
                data_dir / 'clean_corpus' / 'alicio.txt',
                data_dir / 'clean_corpus' / 'frankenstejno.txt',
                data_dir / 'clean_corpus' / 'jekyll_hyde.txt',
                data_dir / 'clean_corpus' / 'milito_de_la_mondoj.txt',
                data_dir / 'clean_corpus' / 'sorcxisto_de_oz.txt',
            ]
        },
        'Wikipedia': {
            'files': [
                data_dir / 'clean_corpus' / 'wikipedia.txt'
            ],
            'note': 'Sampling from 540MB cleaned file'
        }
    }

    all_results = []

    print("=" * 70)
    print("TESTING PARSER ON ALL ESPERANTO CORPORA")
    print("=" * 70)
    print(f"Sampling {args.samples} sentences per file")
    print()

    for category, info in corpora.items():
        if not info.get('files'):
            print(f"{category}:")
            print(f"  {info.get('note', 'No files')}")
            print()
            continue

        print(f"{category}:")
        if 'note' in info:
            print(f"  {info['note']}")

        for filepath in info['files']:
            if not filepath.exists():
                print(f"  ✗ Not found: {filepath.name}")
                continue

            result = test_corpus_file(filepath, num_samples=args.samples, verbose=args.verbose)

            if result:
                all_results.append({
                    'category': category,
                    **result
                })

                word_rate = result['esperanto_words'] / result['total_words'] if result['total_words'] > 0 else 0
                sentence_rate = result['successful_sentences'] / result['total_sentences'] if result['total_sentences'] > 0 else 0

                print(f"  ✓ {result['file']}")
                print(f"    Sentences: {result['successful_sentences']}/{result['total_sentences']} with AST ({sentence_rate*100:.1f}%)")
                print(f"    Words: {result['esperanto_words']}/{result['total_words']} Esperanto ({word_rate*100:.1f}%)")

        print()

    # Aggregate results by category
    print("=" * 70)
    print("SUMMARY BY CATEGORY")
    print("=" * 70)

    category_stats = defaultdict(lambda: {
        'total_words': 0,
        'esperanto_words': 0,
        'files': 0
    })

    for result in all_results:
        cat = result['category']
        category_stats[cat]['total_words'] += result['total_words']
        category_stats[cat]['esperanto_words'] += result['esperanto_words']
        category_stats[cat]['files'] += 1

    for category, stats in category_stats.items():
        rate = stats['esperanto_words'] / stats['total_words'] if stats['total_words'] > 0 else 0
        print(f"{category}:")
        print(f"  Files tested: {stats['files']}")
        print(f"  Total words: {stats['total_words']:,}")
        print(f"  Esperanto words: {stats['esperanto_words']:,} ({rate*100:.1f}%)")
        print()

    # Overall statistics
    total_words = sum(r['total_words'] for r in all_results)
    total_esperanto = sum(r['esperanto_words'] for r in all_results)
    overall_rate = total_esperanto / total_words if total_words > 0 else 0

    print("=" * 70)
    print("OVERALL STATISTICS")
    print("=" * 70)
    print(f"Total files tested: {len(all_results)}")
    print(f"Total words analyzed: {total_words:,}")
    print(f"Esperanto words: {total_esperanto:,} ({overall_rate*100:.1f}%)")
    print(f"Non-Esperanto: {total_words - total_esperanto:,} ({(1-overall_rate)*100:.1f}%)")

    # Save detailed results
    output_file = Path(__file__).parent.parent / 'data' / 'corpus_test_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'samples_per_file': args.samples,
            'results': all_results,
            'category_summary': dict(category_stats),
            'overall': {
                'total_words': total_words,
                'esperanto_words': total_esperanto,
                'success_rate': overall_rate
            }
        }, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nDetailed results saved to: {output_file}")


if __name__ == '__main__':
    main()
