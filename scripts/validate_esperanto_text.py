"""
Esperanto Text Validator

This tool validates that text files contain only Esperanto:
1. Uses lingua language detector for sentence-level detection
2. Identifies and reports non-Esperanto content
3. Calculates Esperanto purity percentage
4. Can filter files to create Esperanto-only versions
"""

import re
from pathlib import Path
from typing import List, Tuple, Dict
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
        Language.ITALIAN
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


def validate_file(file_path: Path, sample_size: int = 100) -> Dict:
    """
    Validate that a file contains Esperanto text.

    Returns dict with:
    - total_sentences: int
    - esperanto_sentences: int
    - non_esperanto_sentences: int
    - esperanto_percentage: float
    - language_breakdown: Counter
    - sample_non_esperanto: List[str]
    """
    print(f"\nValidating: {file_path.name}")

    # Read file
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
    except Exception as e:
        print(f"❌ Could not read file: {e}")
        return None

    # Extract sentences
    sentences = extract_sentences(text)

    if not sentences:
        print(f"⚠️  No sentences found")
        return {
            'total_sentences': 0,
            'esperanto_sentences': 0,
            'non_esperanto_sentences': 0,
            'esperanto_percentage': 0.0,
            'language_breakdown': Counter(),
            'sample_non_esperanto': []
        }

    # Sample sentences if too many
    if len(sentences) > sample_size:
        import random
        sample = random.sample(sentences, sample_size)
        print(f"  Sampling {sample_size} of {len(sentences)} sentences")
    else:
        sample = sentences

    # Detect languages
    print(f"  Detecting languages...")
    languages = detect_language_batch(sample)

    # Count by language
    lang_counts = Counter(languages)
    esperanto_count = lang_counts.get('eo', 0)
    total = len(sample)
    esperanto_pct = (esperanto_count / total * 100) if total > 0 else 0

    # Find non-Esperanto examples
    non_eo_examples = []
    for sent, lang in zip(sample, languages):
        if lang != 'eo' and len(non_eo_examples) < 10:
            non_eo_examples.append((sent[:100], lang))

    # Scale to full file
    total_sentences = len(sentences)
    esperanto_sentences = int(total_sentences * esperanto_pct / 100)
    non_esperanto_sentences = total_sentences - esperanto_sentences

    return {
        'total_sentences': total_sentences,
        'esperanto_sentences': esperanto_sentences,
        'non_esperanto_sentences': non_esperanto_sentences,
        'esperanto_percentage': esperanto_pct,
        'language_breakdown': lang_counts,
        'sample_non_esperanto': non_eo_examples
    }


def print_validation_report(result: Dict, file_path: Path):
    """Print validation report."""
    if not result:
        return

    print(f"\n{'='*70}")
    print(f"Validation Report: {file_path.name}")
    print(f"{'='*70}")

    print(f"\nTotal sentences: {result['total_sentences']}")
    print(f"Esperanto: {result['esperanto_sentences']} ({result['esperanto_percentage']:.1f}%)")
    print(f"Non-Esperanto: {result['non_esperanto_sentences']}")

    print(f"\nLanguage breakdown (from sample):")
    for lang, count in result['language_breakdown'].most_common():
        print(f"  {lang}: {count}")

    if result['sample_non_esperanto']:
        print(f"\nSample non-Esperanto sentences:")
        for sent, lang in result['sample_non_esperanto'][:5]:
            print(f"  [{lang}] {sent}...")

    # Grade
    pct = result['esperanto_percentage']
    if pct >= 95:
        print(f"\n✅ EXCELLENT - File is {pct:.1f}% Esperanto")
    elif pct >= 80:
        print(f"\n⚠️  GOOD - File is {pct:.1f}% Esperanto (some cleanup needed)")
    elif pct >= 50:
        print(f"\n⚠️  MODERATE - File is {pct:.1f}% Esperanto (significant cleanup needed)")
    else:
        print(f"\n❌ POOR - File is only {pct:.1f}% Esperanto (major cleanup needed)")


def filter_esperanto_only(file_path: Path, output_path: Path, threshold: float = 0.7):
    """
    Filter file to keep only Esperanto sentences.

    Args:
        file_path: Input file
        output_path: Output file
        threshold: Confidence threshold (0.0-1.0)
    """
    from lingua import LanguageDetectorBuilder, Language, ConfidenceValue

    print(f"\n{'='*70}")
    print(f"Filtering: {file_path.name}")
    print(f"{'='*70}")

    # Build detector
    detector = LanguageDetectorBuilder.from_all_languages().build()

    # Read file
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()

    # Extract sentences
    sentences = extract_sentences(text)
    print(f"\nTotal sentences: {len(sentences)}")

    # Filter Esperanto with confidence
    esperanto_sentences = []
    non_esperanto = []

    print(f"Filtering with confidence threshold {threshold}...")

    for sent in sentences:
        # Get confidence values for all languages
        confidences = detector.compute_language_confidence_values(sent)

        # Check if Esperanto has highest confidence above threshold
        if confidences:
            top = confidences[0]
            if (top.language == Language.ESPERANTO and
                top.value >= threshold):
                esperanto_sentences.append(sent)
            else:
                non_esperanto.append((sent[:80], top.language.name, top.value))

    print(f"\nKept: {len(esperanto_sentences)} Esperanto sentences")
    print(f"Filtered out: {len(non_esperanto)} non-Esperanto sentences")

    if non_esperanto[:5]:
        print(f"\nSample filtered sentences:")
        for sent, lang, conf in non_esperanto[:5]:
            print(f"  [{lang} {conf:.2f}] {sent}...")

    # Write filtered text
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(esperanto_sentences))

    print(f"\n✓ Filtered text saved to: {output_path}")

    # Calculate stats
    kept_pct = len(esperanto_sentences) / len(sentences) * 100 if sentences else 0
    print(f"  Kept {kept_pct:.1f}% of sentences")


def validate_directory(dir_path: Path, pattern: str = '*.txt'):
    """Validate all text files in a directory."""
    print(f"\n{'='*70}")
    print(f"Validating Directory: {dir_path}")
    print(f"{'='*70}")

    files = list(dir_path.glob(pattern))
    print(f"\nFound {len(files)} files matching '{pattern}'")

    results = []
    for file_path in files:
        result = validate_file(file_path, sample_size=50)
        if result:
            results.append((file_path, result))

    # Summary
    print(f"\n\n{'='*70}")
    print("Summary")
    print(f"{'='*70}\n")

    for file_path, result in results:
        pct = result['esperanto_percentage']
        status = '✅' if pct >= 95 else '⚠️ ' if pct >= 80 else '❌'
        print(f"{status} {file_path.name:50} {pct:5.1f}% Esperanto")

    # Overall stats
    if results:
        avg_pct = sum(r[1]['esperanto_percentage'] for r in results) / len(results)
        print(f"\nAverage Esperanto purity: {avg_pct:.1f}%")

        excellent = sum(1 for _, r in results if r['esperanto_percentage'] >= 95)
        good = sum(1 for _, r in results if 80 <= r['esperanto_percentage'] < 95)
        moderate = sum(1 for _, r in results if 50 <= r['esperanto_percentage'] < 80)
        poor = sum(1 for _, r in results if r['esperanto_percentage'] < 50)

        print(f"\nFile quality breakdown:")
        print(f"  Excellent (≥95%): {excellent}")
        print(f"  Good (80-95%): {good}")
        print(f"  Moderate (50-80%): {moderate}")
        print(f"  Poor (<50%): {poor}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Validate and filter Esperanto text files'
    )
    parser.add_argument(
        'file',
        nargs='?',
        help='File to validate'
    )
    parser.add_argument(
        '--dir',
        help='Validate all files in directory'
    )
    parser.add_argument(
        '--filter',
        action='store_true',
        help='Filter to Esperanto-only sentences'
    )
    parser.add_argument(
        '--output', '-o',
        help='Output file for filtered text'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.7,
        help='Confidence threshold for filtering (default: 0.7)'
    )

    args = parser.parse_args()

    project_root = Path(__file__).parent.parent

    if args.dir:
        # Validate directory
        dir_path = Path(args.dir)
        if not dir_path.is_absolute():
            dir_path = project_root / dir_path
        validate_directory(dir_path)

    elif args.file:
        file_path = Path(args.file)
        if not file_path.is_absolute():
            file_path = project_root / file_path

        if args.filter:
            # Filter mode
            if not args.output:
                output_path = file_path.parent / f"filtered_{file_path.name}"
            else:
                output_path = Path(args.output)
                if not output_path.is_absolute():
                    output_path = project_root / output_path

            filter_esperanto_only(file_path, output_path, args.threshold)
        else:
            # Validate mode
            result = validate_file(file_path)
            if result:
                print_validation_report(result, file_path)
    else:
        # Default: validate cleaned directory
        cleaned_dir = project_root / 'data' / 'cleaned'
        if cleaned_dir.exists():
            validate_directory(cleaned_dir)
        else:
            print(f"Default directory not found: {cleaned_dir}")
            print("Use --file or --dir to specify what to validate")


if __name__ == '__main__':
    main()
