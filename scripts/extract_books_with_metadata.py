#!/usr/bin/env python3
"""
Extract book sentences with chapter/section metadata.

Features:
- Detects chapter/section markers (all-caps lines, numbered chapters)
- Tracks sentence position within chapters
- Progress indicators
- Error logging with context
- Handles multiple books in one run
"""

import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Iterator, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('logs/books_extraction.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def detect_chapter_marker(line: str) -> Optional[dict]:
    """
    Detect if a line is a chapter/section marker.

    Patterns:
    - ALL CAPS with 4+ letters: "ENKONDUKO", "PROLOGO"
    - Numbered chapters: "CHAPTER 1", "Ĉapitro 5"
    - Roman numerals: "I.", "XII."

    Returns:
        dict with keys: chapter_name, chapter_type, chapter_number
        or None if not a chapter marker
    """
    line = line.strip()

    # Skip empty lines
    if not line:
        return None

    # Pattern 1: ALL CAPS (4+ letters)
    if line.isupper() and len(re.sub(r'[^A-ZĈĜĤĴŜŬ]', '', line)) >= 4:
        # Skip if it looks like a page number or header
        if re.match(r'^[0-9]+$', line):
            return None
        if re.match(r'^[IVX]+$', line):  # Standalone roman numerals
            return None

        return {
            'chapter_name': line,
            'chapter_type': 'section',
            'chapter_number': None
        }

    # Pattern 2: "CHAPTER N" or "Ĉapitro N"
    match = re.match(r'^(CHAPTER|ĈAPITRO|Chapter|Ĉapitro)\s+([0-9]+|[IVX]+)', line, re.IGNORECASE)
    if match:
        return {
            'chapter_name': line,
            'chapter_type': 'chapter',
            'chapter_number': match.group(2)
        }

    # Pattern 3: Roman numeral followed by period: "I.", "XII."
    match = re.match(r'^([IVX]+)\.\s*(.*)$', line)
    if match:
        return {
            'chapter_name': match.group(2) or f"Ĉapitro {match.group(1)}",
            'chapter_type': 'chapter',
            'chapter_number': match.group(1)
        }

    return None


def extract_sentences_from_text(text: str, min_words: int = 3, max_words: int = 150) -> list[str]:
    """
    Extract sentences from text with proper handling of abbreviations.

    Args:
        text: Input text
        min_words: Minimum words per sentence
        max_words: Maximum words per sentence

    Returns:
        List of sentences
    """
    # Pre-process: protect common abbreviations by replacing periods temporarily
    abbrev_map = {
        'D-ro.': 'D-ro▁',
        'S-ro.': 'S-ro▁',
        'S-ino.': 'S-ino▁',
        'd-ro.': 'd-ro▁',
        's-ro.': 's-ro▁',
        'ktp.': 'ktp▁',
        'k.t.p.': 'k▁t▁p▁',
        'ekz.': 'ekz▁',
        'n-ro.': 'n-ro▁',
        'vol.': 'vol▁',
        'p.K.': 'p▁K▁',
        'a.K.': 'a▁K▁',
    }

    # Replace abbreviations
    protected_text = text
    for abbrev, replacement in abbrev_map.items():
        protected_text = protected_text.replace(abbrev, replacement)

    # Split on sentence boundaries
    # . ! ? followed by whitespace and capital letter, or just followed by whitespace/end
    # but not after digit (to preserve decimals like 1.5)
    pattern = r'(?<!\d)[.!?]+'
    potential_sentences = re.split(pattern, protected_text)

    # Restore abbreviations
    potential_sentences = [s.replace('▁', '.') for s in potential_sentences]

    # Clean and filter
    result = []
    for sent in potential_sentences:
        sent = sent.strip()

        # Skip empty
        if not sent:
            continue

        # Count words
        words = sent.split()
        if len(words) < min_words or len(words) > max_words:
            continue

        # Skip if it's a chapter marker
        if detect_chapter_marker(sent):
            continue

        # Filter out sentences that are mostly non-alphabetic (page numbers, etc.)
        alpha_chars = sum(c.isalpha() or c.isspace() for c in sent)
        total_chars = len(sent)
        if total_chars > 0 and alpha_chars / total_chars < 0.7:
            continue

        # Filter out sentences that start with numbers (likely page markers)
        if sent and sent[0].isdigit():
            continue

        result.append(sent)

    return result


def process_book_file(
    input_file: Path,
    book_name: str,
    source_id: str
) -> Iterator[dict]:
    """
    Process a single book file and yield sentences with metadata.

    Args:
        input_file: Path to cleaned book text file
        book_name: Human-readable book name
        source_id: Source identifier (e.g., 'la_mastro_de_l_ringoj')

    Yields:
        dict with keys: text, source, source_name, chapter, chapter_number,
                       sentence_in_chapter, paragraph
    """
    logger.info(f"Processing: {book_name}")
    logger.info(f"  File: {input_file}")

    current_chapter = None
    current_chapter_number = None
    sentence_in_chapter = 0
    paragraph_number = 0
    total_sentences = 0

    with open(input_file, 'r', encoding='utf-8') as f:
        current_paragraph = []

        for line_num, line in enumerate(f, 1):
            line = line.strip()

            # Skip empty lines (but they mark paragraph boundaries)
            if not line:
                # Process accumulated paragraph
                if current_paragraph:
                    paragraph_text = ' '.join(current_paragraph)
                    sentences = extract_sentences_from_text(paragraph_text)

                    for sentence in sentences:
                        sentence_in_chapter += 1
                        total_sentences += 1

                        yield {
                            'text': sentence,
                            'source': source_id,
                            'source_name': book_name,
                            'chapter': current_chapter,
                            'chapter_number': current_chapter_number,
                            'sentence_in_chapter': sentence_in_chapter,
                            'paragraph': paragraph_number,
                            'line_number': line_num
                        }

                    paragraph_number += 1
                    current_paragraph = []

                continue

            # Check if this line is a chapter marker
            chapter_info = detect_chapter_marker(line)
            if chapter_info:
                # Save previous paragraph before starting new chapter
                if current_paragraph:
                    paragraph_text = ' '.join(current_paragraph)
                    sentences = extract_sentences_from_text(paragraph_text)

                    for sentence in sentences:
                        sentence_in_chapter += 1
                        total_sentences += 1

                        yield {
                            'text': sentence,
                            'source': source_id,
                            'source_name': book_name,
                            'chapter': current_chapter,
                            'chapter_number': current_chapter_number,
                            'sentence_in_chapter': sentence_in_chapter,
                            'paragraph': paragraph_number,
                            'line_number': line_num
                        }

                    current_paragraph = []

                # Start new chapter
                current_chapter = chapter_info['chapter_name']
                current_chapter_number = chapter_info['chapter_number']
                sentence_in_chapter = 0
                paragraph_number = 0

                logger.info(f"  ✓ Chapter detected: '{current_chapter}' (line {line_num})")
                continue

            # Regular line - add to current paragraph
            current_paragraph.append(line)

        # Process final paragraph
        if current_paragraph:
            paragraph_text = ' '.join(current_paragraph)
            sentences = extract_sentences_from_text(paragraph_text)

            for sentence in sentences:
                sentence_in_chapter += 1
                total_sentences += 1

                yield {
                    'text': sentence,
                    'source': source_id,
                    'source_name': book_name,
                    'chapter': current_chapter,
                    'chapter_number': current_chapter_number,
                    'sentence_in_chapter': sentence_in_chapter,
                    'paragraph': paragraph_number,
                    'line_number': line_num
                }

    logger.info(f"  ✓ Completed: {total_sentences:,} sentences extracted")


def process_all_books(
    cleaned_dir: Path,
    output_file: Path,
    books: list[tuple[str, str, str]]
):
    """
    Process all book files and write to output.

    Args:
        cleaned_dir: Directory containing cleaned text files
        output_file: Output JSONL file
        books: List of (filename, book_name, source_id) tuples
    """
    logger.info("=" * 60)
    logger.info("Starting book extraction")
    logger.info(f"Output: {output_file}")
    logger.info(f"Books to process: {len(books)}")
    logger.info("=" * 60)

    start_time = time.time()
    total_sentences = 0
    total_errors = 0

    with open(output_file, 'w', encoding='utf-8') as out:
        for filename, book_name, source_id in books:
            input_file = cleaned_dir / filename

            if not input_file.exists():
                logger.warning(f"⚠ File not found: {input_file}")
                continue

            try:
                book_sentences = 0

                for entry in process_book_file(input_file, book_name, source_id):
                    out.write(json.dumps(entry, ensure_ascii=False) + '\n')
                    book_sentences += 1
                    total_sentences += 1

                    # Progress indicator every 1000 sentences
                    if total_sentences % 1000 == 0:
                        elapsed = time.time() - start_time
                        rate = total_sentences / elapsed
                        logger.info(f"Progress: {total_sentences:,} sentences ({rate:.0f} sentences/sec)")

                logger.info(f"✓ {book_name}: {book_sentences:,} sentences")

            except Exception as e:
                total_errors += 1
                logger.error(f"Error processing {book_name}: {e}", exc_info=True)

    # Final summary
    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("Book extraction complete!")
    logger.info(f"Total sentences extracted: {total_sentences:,}")
    logger.info(f"Errors encountered: {total_errors}")
    logger.info(f"Time elapsed: {elapsed/60:.1f} minutes")
    logger.info(f"Rate: {total_sentences/(elapsed/60):.0f} sentences/min")
    logger.info("=" * 60)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Extract books with chapter metadata')
    parser.add_argument('--cleaned-dir', type=Path, default=Path('data/cleaned'),
                        help='Directory with cleaned text files')
    parser.add_argument('--output', type=Path, default=Path('data/extracted/books_sentences.jsonl'),
                        help='Output JSONL file')

    args = parser.parse_args()

    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Create logs directory
    Path('logs').mkdir(exist_ok=True)

    # Books to process
    BOOKS = [
        ('cleaned_la_mastro_de_l_ringoj.txt', 'La Mastro de l\' Ringoj', 'la_mastro_de_l_ringoj'),
        ('cleaned_la_hobito.txt', 'La Hobito', 'la_hobito'),
        ('cleaned_kadavrejo_strato.txt', 'Kadavrejo Strato', 'kadavrejo_strato'),
        ('cleaned_la_korvo.txt', 'La Korvo', 'la_korvo'),
        ('cleaned_puto_kaj_pendolo.txt', 'Puto kaj Pendolo', 'puto_kaj_pendolo'),
        ('cleaned_ses_noveloj.txt', 'Ses Noveloj', 'ses_noveloj'),
        ('cleaned_usxero_domo.txt', 'Usxero Domo', 'usxero_domo'),
    ]

    process_all_books(
        cleaned_dir=args.cleaned_dir,
        output_file=args.output,
        books=BOOKS
    )
