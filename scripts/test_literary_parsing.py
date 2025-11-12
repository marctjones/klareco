"""
Test parser on real literary texts.

This script tests how well the parser handles complex literary Esperanto
from actual books (Poe, Tolkien, etc.)
"""

import re
from pathlib import Path
from collections import Counter


def extract_sentences(text: str, max_sentences: int = 20) -> list:
    """Extract sentences from text."""
    # Split on sentence endings (handles both traditional and cleaned corpus format)
    # Traditional: ". " / Cleaned corpus: "\n\n"
    sentences = re.split(r'\n\n+|[.!?]+\s+', text)

    # Filter for valid Esperanto sentences
    valid_sentences = []
    for sent in sentences:
        sent = sent.strip()
        # Must have Esperanto characters and be reasonable length
        if (sent and
            len(sent) > 10 and
            len(sent) < 200 and
            re.search(r'[a-zA-ZĉĝĥĵŝŭĈĜĤĴŜŬ]{3,}', sent)):
            valid_sentences.append(sent)
            if len(valid_sentences) >= max_sentences:
                break

    return valid_sentences


def test_parse_sentence(sentence: str) -> dict:
    """Test parsing a single sentence."""
    from klareco.parser import parse

    try:
        ast = parse(sentence)
        return {
            'success': True,
            'sentence': sentence,
            'ast': ast,
            'error': None
        }
    except Exception as e:
        return {
            'success': False,
            'sentence': sentence,
            'ast': None,
            'error': str(e)
        }


def test_literary_file(file_path: Path, book_name: str, num_sentences: int = 10):
    """Test parser on sentences from a literary work."""
    print(f"\n{'='*70}")
    print(f"Testing: {book_name}")
    print(f"File: {file_path.name}")
    print(f"{'='*70}\n")

    # Read file
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
    except Exception as e:
        print(f"❌ Could not read file: {e}")
        return

    # Skip first 3000 chars (headers, copyright, licenses)
    text = text[3000:]

    # Extract sentences
    sentences = extract_sentences(text, num_sentences)
    print(f"Extracted {len(sentences)} sentences for testing\n")

    if not sentences:
        print("❌ No valid sentences found")
        return

    # Test parsing
    results = []
    for i, sentence in enumerate(sentences, 1):
        print(f"\n--- Sentence {i}/{len(sentences)} ---")
        print(f"Text: {sentence[:100]}{'...' if len(sentence) > 100 else ''}")

        result = test_parse_sentence(sentence)
        results.append(result)

        if result['success']:
            print("✅ PARSED SUCCESSFULLY")
        else:
            print(f"❌ PARSE FAILED: {result['error'][:100]}")

    # Summary
    success_count = sum(1 for r in results if r['success'])
    fail_count = len(results) - success_count
    success_rate = (success_count / len(results) * 100) if results else 0

    print(f"\n{'='*70}")
    print(f"Summary for {book_name}:")
    print(f"  Parsed: {success_count}/{len(results)} ({success_rate:.1f}%)")
    print(f"  Failed: {fail_count}/{len(results)}")
    print(f"{'='*70}")

    # Analyze errors
    if fail_count > 0:
        print(f"\n⚠️  Common error patterns:")
        error_types = Counter()
        for r in results:
            if not r['success']:
                # Categorize error
                error = r['error']
                if 'radikon' in error:
                    error_types['Unknown root'] += 1
                elif 'finaĵon' in error:
                    error_types['Unknown ending'] += 1
                elif 'syntax' in error.lower():
                    error_types['Syntax error'] += 1
                else:
                    error_types['Other'] += 1

        for error_type, count in error_types.most_common():
            print(f"  - {error_type}: {count}")

    return {
        'book': book_name,
        'total': len(results),
        'success': success_count,
        'fail': fail_count,
        'success_rate': success_rate
    }


def main():
    """Test parser on multiple literary works."""
    project_root = Path(__file__).parent.parent

    print("="*70)
    print("Literary Text Parsing Test")
    print("="*70)
    print("\nTesting parser on real literary Esperanto texts...")

    # Define books to test (Esperanto only!)
    books = [
        (project_root / 'data/cleaned/cleaned_usxero_domo.txt',
         "The Fall of the House of Usher (Poe)"),
        (project_root / 'data/cleaned/cleaned_la_korvo.txt',
         "The Raven (Poe)"),
        (project_root / 'data/cleaned/cleaned_puto_kaj_pendolo.txt',
         "The Pit and the Pendulum (Poe)"),
        (project_root / 'data/cleaned/cleaned_la_hobito.txt',
         "The Hobbit (Tolkien)"),
        (project_root / 'data/cleaned/cleaned_la_mastro_de_l_ringoj.txt',
         "The Lord of the Rings (Tolkien)"),
        (project_root / 'data/cleaned/cleaned_ses_noveloj.txt',
         "Six Tales (Poe)"),
    ]

    # Test each book
    all_results = []
    for file_path, book_name in books:
        if file_path.exists():
            result = test_literary_file(file_path, book_name, num_sentences=10)
            if result:
                all_results.append(result)
        else:
            print(f"\n❌ File not found: {file_path}")

    # Overall summary
    print(f"\n\n{'='*70}")
    print("OVERALL SUMMARY")
    print(f"{'='*70}\n")

    if all_results:
        total_sentences = sum(r['total'] for r in all_results)
        total_success = sum(r['success'] for r in all_results)
        overall_rate = (total_success / total_sentences * 100) if total_sentences > 0 else 0

        print(f"Books tested: {len(all_results)}")
        print(f"Total sentences: {total_sentences}")
        print(f"Successfully parsed: {total_success} ({overall_rate:.1f}%)")
        print(f"Failed to parse: {total_sentences - total_success}")

        print(f"\nPer-book results:")
        for result in all_results:
            print(f"  {result['book']:50} {result['success_rate']:5.1f}%")

        print(f"\n{'='*70}")
        if overall_rate >= 90:
            print("✅ EXCELLENT: Parser handles literary texts very well!")
        elif overall_rate >= 70:
            print("⚠️  GOOD: Parser handles most literary texts, some gaps remain")
        elif overall_rate >= 50:
            print("⚠️  MODERATE: Parser has significant gaps with literary texts")
        else:
            print("❌ NEEDS WORK: Parser struggles with complex literary texts")
        print(f"{'='*70}")
    else:
        print("No results to summarize")


if __name__ == '__main__':
    main()
