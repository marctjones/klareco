#!/usr/bin/env python3
"""
Properly segment corpus into sentences (not lines).

This script fixes the core issue where corpus was split by lines instead of
sentences, resulting in fragmented text. It:
1. Reads raw text files
2. Joins hyphenated words across lines
3. Splits on sentence boundaries (., !, ?)
4. Handles abbreviations and edge cases
5. Outputs JSONL with source metadata

Usage:
    python scripts/segment_corpus_sentences.py --input data/clean_corpus --output data/corpus_sentences.jsonl
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Tuple

# Common Esperanto abbreviations that don't end sentences
ABBREVIATIONS = {
    'sro', 'sro.', 'D-ro', 'D-ro.', 'Prof', 'Prof.',
    'k', 'k.', 'ktp', 'ktp.', 'k.t.p', 'k.t.p.',
    'ekz', 'ekz.', 'n-ro', 'n-ro.', 'p', 'p.', 'pp', 'pp.',
    'vol', 'vol.', 'red', 'red.', 'trad', 'trad.',
    'resp', 'resp.', 'eld', 'eld.', 'eldonejo', 'eldon.',
    'Mr', 'Mr.', 'Mrs', 'Mrs.', 'Dr', 'Dr.',
}

# Texts to process
TEXTS_TO_PROCESS = [
    ("la_mastro_de_l_ringoj.txt", "La Mastro de l' Ringoj (Lord of the Rings)", "la_mastro_de_l_ringoj"),
    ("la_hobito.txt", "La Hobito (The Hobbit)", "la_hobito"),
    ("ses_noveloj.txt", "Ses Noveloj (Six Stories)", "ses_noveloj"),
    ("la_korvo.txt", "La Korvo (The Raven)", "la_korvo"),
    ("puto_kaj_pendolo.txt", "Puto kaj Pendolo (Pit and Pendulum)", "puto_kaj_pendolo"),
    ("usxero_domo.txt", "Usxero Domo (Fall of House of Usher)", "usxero_domo"),
    ("kadavrejo_strato.txt", "Kadavrejo Strato (Rue Morgue)", "kadavrejo_strato"),
]


def read_file_with_line_joining(filepath: Path) -> str:
    """
    Read file and join hyphenated words across lines.

    Example:
        "kaj li estis konten-\n"
        "ta kun tio."

        becomes:

        "kaj li estis kontenta kun tio."
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Join hyphenated words across lines
    text = ""
    for i, line in enumerate(lines):
        # If line ends with hyphen, join with next line
        if line.rstrip().endswith('-') and i + 1 < len(lines):
            # Remove hyphen and newline, join with next line
            text += line.rstrip()[:-1]  # Remove trailing hyphen
            # Don't add newline, will join with next line
        else:
            text += line

    return text


def is_sentence_boundary(text: str, pos: int) -> bool:
    """
    Check if position is a true sentence boundary.

    Args:
        text: Full text
        pos: Position of punctuation (., !, ?)

    Returns:
        True if this is a sentence boundary
    """
    if pos >= len(text) - 1:
        return True

    # Get context before punctuation
    start = max(0, pos - 20)
    before = text[start:pos]

    # Get character after punctuation
    after_char = text[pos + 1] if pos + 1 < len(text) else ''

    # Check if it's an abbreviation
    # Look for common abbreviation patterns before the period
    words_before = before.split()
    if words_before:
        last_word = words_before[-1]
        if last_word in ABBREVIATIONS or last_word + '.' in ABBREVIATIONS:
            return False

    # If next character is lowercase, probably not a sentence boundary
    # (except after dialog punctuation)
    if after_char.islower() and text[pos] == '.':
        return False

    # If next character is a digit, probably not a sentence boundary
    if after_char.isdigit():
        return False

    # If followed by whitespace or uppercase, likely a boundary
    if after_char in [' ', '\n', '\t', ''] or after_char.isupper():
        return True

    return True


def segment_sentences(text: str) -> List[str]:
    """
    Split text into sentences.

    Args:
        text: Full text content

    Returns:
        List of sentences
    """
    sentences = []
    current = ""

    i = 0
    while i < len(text):
        char = text[i]
        current += char

        # Check for sentence-ending punctuation
        if char in '.!?':
            # Check if this is a true sentence boundary
            if is_sentence_boundary(text, i):
                # Clean up the sentence
                sentence = current.strip()

                # Skip very short "sentences" (likely metadata or artifacts)
                if len(sentence) >= 10 and not is_metadata_line(sentence):
                    sentences.append(sentence)

                current = ""

        i += 1

    # Add any remaining text
    if current.strip() and len(current.strip()) >= 10:
        if not is_metadata_line(current.strip()):
            sentences.append(current.strip())

    return sentences


def is_metadata_line(line: str) -> bool:
    """
    Check if line is likely metadata (headers, copyright, etc.).
    """
    line = line.strip()

    # Very short lines
    if len(line) < 15:
        return True

    # All caps (likely headers)
    if line.isupper() and len(line) < 100:
        return True

    # Copyright/attribution patterns
    metadata_patterns = [
        r'^©',
        r'^Produced by',
        r'^Project Gutenberg',
        r'^\*\*\*',
        r'^---',
        r'^===',
        r'^\[Ilustraĵo:',
        r'^Distributed',
        r'^Sezonoj',
        r'^Enkonduko, mapoj',
        r'^\d{4,5}_',  # Gutenberg IDs
    ]

    for pattern in metadata_patterns:
        if re.match(pattern, line):
            return True

    return False


def process_file(
    filepath: Path,
    source_id: str,
    source_name: str
) -> List[dict]:
    """
    Process a single text file into sentences with metadata.

    Args:
        filepath: Path to text file
        source_id: Source identifier (e.g., "la_mastro_de_l_ringoj")
        source_name: Display name (e.g., "La Mastro de l' Ringoj")

    Returns:
        List of sentence dicts with metadata
    """
    print(f"📖 Processing: {source_name}")

    # Read file with line joining
    text = read_file_with_line_joining(filepath)

    # Segment into sentences
    sentences = segment_sentences(text)

    # Create entries with metadata
    entries = []
    for sentence in sentences:
        entry = {
            "sentence": sentence,
            "source": source_id,
            "source_name": source_name,
        }
        entries.append(entry)

    print(f"   ✅ Extracted {len(entries):,} sentences")

    return entries


def main():
    parser = argparse.ArgumentParser(description="Segment corpus into proper sentences")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/clean_corpus"),
        help="Directory with source text files"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/corpus_sentences.jsonl"),
        help="Output JSONL file"
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=10,
        help="Minimum sentence length in characters (default: 10)"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("CORPUS SENTENCE SEGMENTATION")
    print("=" * 70)
    print()

    all_entries = []

    for filename, display_name, source_id in TEXTS_TO_PROCESS:
        filepath = args.input / filename

        if not filepath.exists():
            print(f"⚠️  File not found: {filepath}")
            continue

        entries = process_file(filepath, source_id, display_name)
        all_entries.extend(entries)

    print()
    print("=" * 70)
    print(f"Writing {len(all_entries):,} sentences to {args.output}")

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        for entry in all_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"✅ Done!")
    print()

    # Print statistics
    print("=" * 70)
    print("STATISTICS")
    print("=" * 70)

    source_counts = {}
    for entry in all_entries:
        source = entry['source_name']
        source_counts[source] = source_counts.get(source, 0) + 1

    for source, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"{source:50s} {count:6,} sentences")

    print()
    print(f"{'TOTAL':50s} {len(all_entries):6,} sentences")
    print()


if __name__ == "__main__":
    main()
