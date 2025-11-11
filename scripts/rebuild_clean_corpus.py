"""
Rebuild Clean Corpus from Scratch

This script rebuilds the cleaned corpus using proper language detection:
1. Processes all files in data/corpora/
2. Uses lingua to detect language per sentence
3. Filters to keep only Esperanto sentences
4. Saves cleaned versions to data/cleaned/
5. Generates before/after quality report
"""

import re
from pathlib import Path
from typing import List, Dict
from collections import Counter


def detect_language_batch(texts: List[str]) -> List[str]:
    """
    Detect language for a batch of texts.
    Returns list of ISO language codes.
    """
    from lingua import LanguageDetectorBuilder, Language

    # Build detector with common languages
    detector = LanguageDetectorBuilder.from_languages(
        Language.ESPERANTO,
        Language.ENGLISH,
        Language.FRENCH,
        Language.GERMAN,
        Language.SPANISH,
        Language.ITALIAN,
        Language.PORTUGUESE,
        Language.RUSSIAN
    ).build()

    results = []
    for text in texts:
        if not text or len(text.strip()) < 5:
            results.append('unknown')
            continue

        lang = detector.detect_language_of(text)
        if lang:
            results.append(lang.iso_code_639_1.name.lower())
        else:
            results.append('unknown')

    return results


def extract_sentences(text: str) -> List[str]:
    """Extract sentences from text."""
    # Split on sentence boundaries
    sentences = re.split(r'[.!?]+\s+', text)

    # Filter and clean
    cleaned = []
    for sent in sentences:
        sent = sent.strip()
        # Must have letters and be reasonable length
        if sent and len(sent) > 10 and re.search(r'[a-zA-ZĉĝĥĵŝŭĈĜĤĴŜŬ]+', sent):
            cleaned.append(sent)

    return cleaned


def clean_file(input_path: Path, output_path: Path, confidence_threshold: float = 0.7) -> Dict:
    """
    Clean a single file by filtering to Esperanto-only content.

    Args:
        input_path: Original file
        output_path: Cleaned output file
        confidence_threshold: Minimum confidence for Esperanto classification

    Returns:
        Statistics dict
    """
    from lingua import LanguageDetectorBuilder, Language

    print(f"\n{'='*70}")
    print(f"Cleaning: {input_path.name}")
    print(f"{'='*70}")

    # Build detector with ALL languages for better accuracy
    detector = LanguageDetectorBuilder.from_all_languages().build()

    # Read file
    try:
        with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
    except Exception as e:
        print(f"❌ Could not read file: {e}")
        return None

    # Extract sentences
    sentences = extract_sentences(text)
    total_sentences = len(sentences)

    if total_sentences == 0:
        print(f"⚠️  No sentences found")
        return {
            'file': input_path.name,
            'total_sentences': 0,
            'kept_sentences': 0,
            'kept_percentage': 0.0,
            'languages_found': Counter()
        }

    print(f"  Total sentences: {total_sentences}")
    print(f"  Filtering with confidence threshold {confidence_threshold}...")

    # Filter Esperanto with confidence
    esperanto_sentences = []
    non_esperanto = []
    language_counts = Counter()

    for sent in sentences:
        # Get confidence values for all languages
        confidences = detector.compute_language_confidence_values(sent)

        # Check if Esperanto has highest confidence above threshold
        if confidences:
            top = confidences[0]
            language_counts[top.language.name] += 1

            if (top.language == Language.ESPERANTO and
                top.value >= confidence_threshold):
                esperanto_sentences.append(sent)
            else:
                non_esperanto.append((sent[:80], top.language.name, top.value))

    kept_count = len(esperanto_sentences)
    kept_pct = (kept_count / total_sentences * 100) if total_sentences > 0 else 0

    print(f"\n  Kept: {kept_count} Esperanto sentences ({kept_pct:.1f}%)")
    print(f"  Filtered out: {len(non_esperanto)} non-Esperanto sentences")

    # Show language breakdown
    print(f"\n  Languages found:")
    for lang, count in language_counts.most_common(5):
        print(f"    {lang}: {count}")

    # Show sample filtered sentences
    if non_esperanto[:3]:
        print(f"\n  Sample filtered sentences:")
        for sent, lang, conf in non_esperanto[:3]:
            print(f"    [{lang} {conf:.2f}] {sent}...")

    # Write cleaned text
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if esperanto_sentences:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(esperanto_sentences))
        print(f"\n  ✓ Cleaned text saved to: {output_path.name}")
    else:
        print(f"\n  ⚠️  No Esperanto sentences found - no output file created")

    return {
        'file': input_path.name,
        'total_sentences': total_sentences,
        'kept_sentences': kept_count,
        'kept_percentage': kept_pct,
        'languages_found': language_counts
    }


def rebuild_corpus(corpus_dir: Path, output_dir: Path, confidence_threshold: float = 0.7):
    """
    Rebuild entire corpus by cleaning all files.

    Args:
        corpus_dir: Directory with original corpus files
        output_dir: Directory for cleaned output files
        confidence_threshold: Minimum confidence for Esperanto classification
    """
    print(f"\n{'='*70}")
    print(f"Rebuilding Clean Corpus")
    print(f"{'='*70}")
    print(f"\nSource: {corpus_dir}")
    print(f"Output: {output_dir}")
    print(f"Confidence threshold: {confidence_threshold}")

    # Find all text files
    files = sorted(corpus_dir.glob('*.txt'))
    print(f"\nFound {len(files)} corpus files to clean")

    # Process each file
    results = []
    for input_path in files:
        output_path = output_dir / f"cleaned_{input_path.name}"

        result = clean_file(input_path, output_path, confidence_threshold)
        if result:
            results.append(result)

    # Generate summary report
    print(f"\n\n{'='*70}")
    print("CLEANING SUMMARY")
    print(f"{'='*70}\n")

    total_input_sentences = sum(r['total_sentences'] for r in results)
    total_output_sentences = sum(r['kept_sentences'] for r in results)
    overall_kept_pct = (total_output_sentences / total_input_sentences * 100) if total_input_sentences > 0 else 0

    print(f"Files processed: {len(results)}")
    print(f"Total input sentences: {total_input_sentences:,}")
    print(f"Total output sentences: {total_output_sentences:,}")
    print(f"Overall kept: {overall_kept_pct:.1f}%")

    print(f"\n{'='*70}")
    print("Per-file Results:")
    print(f"{'='*70}\n")

    for result in sorted(results, key=lambda r: r['kept_percentage'], reverse=True):
        pct = result['kept_percentage']
        status = '✅' if pct >= 95 else '⚠️ ' if pct >= 80 else '❌'
        print(f"{status} {result['file']:40} {pct:5.1f}% kept ({result['kept_sentences']}/{result['total_sentences']})")

    # Quality breakdown
    excellent = sum(1 for r in results if r['kept_percentage'] >= 95)
    good = sum(1 for r in results if 80 <= r['kept_percentage'] < 95)
    moderate = sum(1 for r in results if 50 <= r['kept_percentage'] < 80)
    poor = sum(1 for r in results if r['kept_percentage'] < 50)

    print(f"\nQuality breakdown:")
    print(f"  Excellent (≥95%): {excellent}")
    print(f"  Good (80-95%): {good}")
    print(f"  Moderate (50-80%): {moderate}")
    print(f"  Poor (<50%): {poor}")

    # All languages found across corpus
    all_languages = Counter()
    for r in results:
        all_languages.update(r['languages_found'])

    print(f"\nAll languages found in corpus:")
    for lang, count in all_languages.most_common(10):
        print(f"  {lang}: {count:,}")

    print(f"\n{'='*70}")
    print("✅ Corpus cleaning complete!")
    print(f"{'='*70}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Rebuild clean corpus from scratch using language detection'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.7,
        help='Confidence threshold for Esperanto classification (default: 0.7)'
    )
    parser.add_argument(
        '--corpus-dir',
        type=str,
        default='data/corpora',
        help='Directory with original corpus files (default: data/corpora)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/cleaned',
        help='Directory for cleaned output (default: data/cleaned)'
    )

    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    corpus_dir = project_root / args.corpus_dir
    output_dir = project_root / args.output_dir

    if not corpus_dir.exists():
        print(f"❌ Corpus directory not found: {corpus_dir}")
        return

    rebuild_corpus(corpus_dir, output_dir, args.threshold)


if __name__ == '__main__':
    main()
