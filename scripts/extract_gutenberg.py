#!/usr/bin/env python3
"""
Extract book sentences with chapter/section metadata.

Features:
- Detects chapter/section markers (all-caps lines, numbered chapters)
- Tracks sentence position within chapters
- Progress indicators
- Error logging with context
- Handles multiple books in one run
- Checkpoint support for restartability (--fresh to start over)
"""

import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Iterator, Optional, Dict, Any

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


# =============================================================================
# Checkpoint Support
# =============================================================================

def load_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
    """Load checkpoint if exists."""
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load checkpoint: {e}")
    return {'completed_books': [], 'total_sentences': 0}


def save_checkpoint(checkpoint_path: Path, state: Dict[str, Any]):
    """Atomically save checkpoint."""
    temp_path = checkpoint_path.with_suffix('.tmp')
    try:
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2)
        temp_path.rename(checkpoint_path)
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")
        if temp_path.exists():
            temp_path.unlink()


# =============================================================================
# Chapter/Section Detection
# =============================================================================

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
    books: list[tuple[Path, str, str]],
    fresh: bool = False
):
    """
    Process all book files and write to output.

    Args:
        cleaned_dir: Directory containing cleaned text files (unused if books have full paths)
        output_file: Output JSONL file
        books: List of (file_path, book_name, source_id) tuples
        fresh: If True, start fresh ignoring any checkpoint
    """
    # Checkpoint file next to output
    checkpoint_path = output_file.with_suffix('.checkpoint.json')

    # Load or initialize checkpoint
    if fresh:
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            logger.info("Fresh start requested, deleted existing checkpoint")
        if output_file.exists():
            output_file.unlink()
            logger.info("Fresh start requested, deleted existing output")
        checkpoint = {'completed_books': [], 'total_sentences': 0}
    else:
        checkpoint = load_checkpoint(checkpoint_path)

    completed_books = set(checkpoint.get('completed_books', []))
    total_sentences = checkpoint.get('total_sentences', 0)

    # Filter to only books not yet completed
    remaining_books = [
        (f, name, sid) for f, name, sid in books
        if sid not in completed_books
    ]

    logger.info("=" * 60)
    logger.info("Starting book extraction")
    logger.info(f"Output: {output_file}")
    logger.info(f"Total books: {len(books)}")
    logger.info(f"Already completed: {len(completed_books)}")
    logger.info(f"Remaining to process: {len(remaining_books)}")
    logger.info("=" * 60)

    if not remaining_books:
        logger.info("All books already processed! Use --fresh to reprocess.")
        return

    start_time = time.time()
    total_errors = 0
    session_sentences = 0

    # Append mode if resuming, write mode if fresh
    mode = 'a' if completed_books else 'w'

    with open(output_file, mode, encoding='utf-8') as out:
        for input_file, book_name, source_id in remaining_books:
            # Handle both Path objects and string filenames (backwards compatibility)
            if isinstance(input_file, str):
                input_file = cleaned_dir / input_file

            if not input_file.exists():
                logger.warning(f"⚠ File not found: {input_file}")
                continue

            try:
                book_sentences = 0

                for entry in process_book_file(input_file, book_name, source_id):
                    out.write(json.dumps(entry, ensure_ascii=False) + '\n')
                    book_sentences += 1
                    total_sentences += 1
                    session_sentences += 1

                    # Progress indicator every 1000 sentences
                    if session_sentences % 1000 == 0:
                        elapsed = time.time() - start_time
                        rate = session_sentences / elapsed
                        logger.info(f"Progress: {total_sentences:,} total sentences ({rate:.0f} sentences/sec)")

                logger.info(f"✓ {book_name}: {book_sentences:,} sentences")

                # Mark book as completed and save checkpoint
                completed_books.add(source_id)
                checkpoint = {
                    'completed_books': list(completed_books),
                    'total_sentences': total_sentences,
                    'last_book': source_id,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                save_checkpoint(checkpoint_path, checkpoint)

            except Exception as e:
                total_errors += 1
                logger.error(f"Error processing {book_name}: {e}", exc_info=True)

    # Final summary
    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("Book extraction complete!")
    logger.info(f"Total sentences extracted: {total_sentences:,}")
    logger.info(f"This session: {session_sentences:,} sentences")
    logger.info(f"Errors encountered: {total_errors}")
    logger.info(f"Time elapsed: {elapsed/60:.1f} minutes")
    if elapsed > 0:
        logger.info(f"Rate: {session_sentences/(elapsed/60):.0f} sentences/min")
    logger.info("=" * 60)

    # Clean up checkpoint on successful completion
    if total_errors == 0 and len(remaining_books) > 0:
        logger.info("All books processed successfully, checkpoint retained for reference")


def discover_books(cleaned_dir: Path) -> list[tuple[Path, str, str]]:
    """
    Dynamically discover all Esperanto text files in the cleaned directory.

    Returns:
        List of (full_path, book_name, source_id) tuples
    """
    # Files to skip (non-book content, copyrighted, etc.)
    SKIP_FILES = {
        'wikipedia.txt',           # Wikipedia has separate extraction
        'la_mastro_de_l_ringoj',   # Lord of the Rings - copyrighted
        'la_hobito',               # The Hobbit - copyrighted
    }

    books = []

    # Look in both data/cleaned and data/cleaned/eo
    search_dirs = [cleaned_dir, cleaned_dir / 'eo']

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue

        for txt_file in sorted(search_dir.glob('*.txt')):
            filename = txt_file.name

            # Skip files in skip list (check partial matches)
            skip = False
            for skip_pattern in SKIP_FILES:
                if skip_pattern in filename.lower():
                    skip = True
                    logger.info(f"Skipping {filename} (in skip list)")
                    break
            if skip:
                continue

            # Generate source_id from filename
            # Remove .txt extension and Gutenberg ID prefix if present
            name_parts = filename.replace('.txt', '')

            # Handle Gutenberg format: "12345_Book_Title.txt"
            if re.match(r'^\d+_', name_parts):
                # Remove numeric prefix
                name_parts = re.sub(r'^\d+_', '', name_parts)

            # Handle "cleaned_" prefix
            name_parts = name_parts.replace('cleaned_', '')

            # Create source_id (lowercase, underscores)
            source_id = name_parts.lower().replace(' ', '_')

            # Create human-readable name (replace underscores with spaces, title case)
            book_name = name_parts.replace('_', ' ')

            # Return full path instead of just filename
            books.append((txt_file, book_name, source_id))

    return books


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Extract books with chapter metadata')
    parser.add_argument('--cleaned-dir', type=Path, default=Path('data/cleaned'),
                        help='Directory with cleaned text files')
    parser.add_argument('--output', type=Path, default=Path('data/extracted/books_sentences.jsonl'),
                        help='Output JSONL file')
    parser.add_argument('--list-only', action='store_true',
                        help='Only list discovered books, do not extract')
    parser.add_argument('--fresh', action='store_true',
                        help='Start fresh, ignore checkpoint and delete existing output')

    args = parser.parse_args()

    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Create logs directory
    Path('logs').mkdir(exist_ok=True)

    # Dynamically discover all books
    BOOKS = discover_books(args.cleaned_dir)

    logger.info(f"Discovered {len(BOOKS)} books to process")

    if args.list_only:
        print(f"\nDiscovered {len(BOOKS)} books:")
        for file_path, book_name, source_id in BOOKS:
            print(f"  {file_path.name} -> {book_name} ({source_id})")
        sys.exit(0)

    process_all_books(
        cleaned_dir=args.cleaned_dir,
        output_file=args.output,
        books=BOOKS,
        fresh=args.fresh
    )
