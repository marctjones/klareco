#!/usr/bin/env python3
"""
Extract clean Esperanto sentences from Project Gutenberg texts.

This script:
1. Removes Project Gutenberg headers/footers
2. Cleans up formatting (extra whitespace, page numbers, etc.)
3. Extracts individual sentences
4. Filters out non-Esperanto content (titles, metadata, etc.)
5. Creates a clean sentence corpus for parser training
"""

import re
import json
from pathlib import Path
from typing import List, Dict
import unicodedata


def strip_gutenberg_headers(text: str) -> str:
    """Remove Project Gutenberg header and footer."""
    # Find START marker
    start_match = re.search(r'\*\*\* START OF [^\*]+\*\*\*', text)
    if start_match:
        text = text[start_match.end():]

    # Find END marker
    end_match = re.search(r'\*\*\* END OF [^\*]+\*\*\*', text)
    if end_match:
        text = text[:end_match.start()]

    return text


def clean_text(text: str) -> str:
    """Clean up formatting issues."""
    # Normalize unicode (convert CX-system to Unicode)
    text = normalize_cx_system(text)

    # Remove page numbers and section markers
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    text = re.sub(r'\[Ilustrajxo:[^\]]+\]', '', text)
    text = re.sub(r'\[Footnote[^\]]+\]', '', text)

    # Remove multiple blank lines
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)

    # Remove lines with only special characters
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # Skip lines that are just decoration
        if stripped and not re.match(r'^[\*\-=_\s]+$', stripped):
            cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)


def normalize_cx_system(text: str) -> str:
    """Convert CX-system (CX, GX, etc.) to Unicode (ĉ, ĝ, etc.)."""
    replacements = {
        'CX': 'Ĉ', 'cx': 'ĉ',
        'GX': 'Ĝ', 'gx': 'ĝ',
        'HX': 'Ĥ', 'hx': 'ĥ',
        'JX': 'Ĵ', 'jx': 'ĵ',
        'SX': 'Ŝ', 'sx': 'ŝ',
        'UX': 'Ŭ', 'ux': 'ŭ',
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def extract_sentences(text: str) -> List[str]:
    """Extract individual sentences from text."""
    # Split on sentence boundaries
    # Esperanto uses . ! ? like English
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-ZĈĜĤĴŜŬ])', text)

    cleaned_sentences = []
    for sentence in sentences:
        # Clean up the sentence
        s = sentence.strip()

        # Skip if too short or too long
        if len(s) < 10 or len(s) > 500:
            continue

        # Skip if it doesn't end with punctuation
        if not s[-1] in '.!?':
            continue

        # Skip if it has too many non-Esperanto characters
        if not is_likely_esperanto(s):
            continue

        cleaned_sentences.append(s)

    return cleaned_sentences


def is_likely_esperanto(text: str) -> bool:
    """Check if text is likely to be Esperanto content."""
    # Skip if too many uppercase letters (likely a title/header)
    upper_ratio = sum(1 for c in text if c.isupper()) / len(text)
    if upper_ratio > 0.3:
        return False

    # Skip if it looks like metadata (has year, numbers, etc.)
    if re.match(r'^\d{4}', text):  # Starts with year
        return False

    # Skip if it has HTML or formatting codes
    if '<' in text or '>' in text or '{' in text:
        return False

    # Must contain at least one Esperanto letter
    esperanto_chars = 'ĉĝĥĵŝŭĈĜĤĴŜŬ'
    if not any(c in text for c in esperanto_chars):
        # Or at least have common Esperanto words
        common_words = ['la', 'de', 'kaj', 'en', 'estas', 'al', 'mi', 'vi']
        if not any(f' {word} ' in f' {text.lower()} ' for word in common_words):
            return False

    return True


def process_file(filepath: Path, source_name: str) -> Dict:
    """Process a single Gutenberg file."""
    print(f"\nProcessing: {filepath.name}")

    # Read file
    text = filepath.read_text(encoding='utf-8')

    # Clean text
    text = strip_gutenberg_headers(text)
    text = clean_text(text)

    # Extract sentences
    sentences = extract_sentences(text)

    print(f"  Extracted {len(sentences)} sentences")

    return {
        'source': source_name,
        'file': filepath.name,
        'sentence_count': len(sentences),
        'sentences': sentences
    }


def main():
    """Process all Gutenberg files and create a corpus."""
    # Input/output directories
    input_dir = Path(__file__).parent.parent / "data" / "gutenberg_esperanto"
    output_dir = Path(__file__).parent.parent / "data"

    if not input_dir.exists():
        print(f"Error: {input_dir} does not exist")
        print("Run download_gutenberg_esperanto.py first")
        return

    # Process each category
    categories = {
        'zamenhof': ['08224', '20006', '11307'],
        'grammar': ['47855', '52556', '24525'],
        'historical': ['57184', '26359', '38240'],
        'original': ['42028', '25311', '42774', '48896', '76273', '23670']
    }

    all_results = []
    total_sentences = 0

    print("=" * 70)
    print("EXTRACTING SENTENCES FROM GUTENBERG TEXTS")
    print("=" * 70)

    for category, file_ids in categories.items():
        print(f"\n{category.upper()}")
        print("-" * 70)

        for file_id in file_ids:
            # Find the file (it has a longer name)
            files = list(input_dir.glob(f"{file_id}_*.txt"))
            if not files:
                print(f"  Warning: No file found for ID {file_id}")
                continue

            filepath = files[0]
            result = process_file(filepath, category)
            all_results.append(result)
            total_sentences += result['sentence_count']

    # Save results
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    # Save as JSON with all sentences
    output_file = output_dir / "gutenberg_sentences.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"Saved JSON: {output_file}")

    # Save as plain text (one sentence per line)
    txt_file = output_dir / "gutenberg_sentences.txt"
    with open(txt_file, 'w', encoding='utf-8') as f:
        for result in all_results:
            for sentence in result['sentences']:
                f.write(sentence + '\n')
    print(f"Saved TXT: {txt_file}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Files processed: {len(all_results)}")
    print(f"Total sentences: {total_sentences}")
    print(f"\nBy category:")
    for category in categories.keys():
        cat_sentences = sum(r['sentence_count'] for r in all_results if r['source'] == category)
        print(f"  {category:12s}: {cat_sentences:6d} sentences")

    # Show sample sentences
    print("\n" + "=" * 70)
    print("SAMPLE SENTENCES (from Zamenhof)")
    print("=" * 70)
    zamenhof_results = [r for r in all_results if r['source'] == 'zamenhof']
    if zamenhof_results and zamenhof_results[0]['sentences']:
        for i, sentence in enumerate(zamenhof_results[0]['sentences'][:5]):
            print(f"{i+1}. {sentence}")


if __name__ == "__main__":
    main()
