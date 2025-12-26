#!/usr/bin/env python3
"""
Extract well-formed sentences from hard-wrapped cleaned texts.

This handles:
1. Unwrapping hard-wrapped lines (75-char wrapping)
2. Splitting on sentence boundaries
3. Handling Esperanto abbreviations
4. Optional AST generation for quality control
5. Memory-efficient streaming for large files
6. Batch processing for AST generation
"""

import re
import gc
import signal
import json
from pathlib import Path
from typing import List, Dict, Optional, Iterator
from datetime import datetime
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# File to log problem sentences
PROBLEM_SENTENCES_LOG = Path(__file__).parent.parent / "data" / "problem_sentences.jsonl"


class ParseTimeout(Exception):
    """Raised when parsing takes too long."""
    pass


def _timeout_handler(signum, frame):
    raise ParseTimeout("Parse timed out")


def parse_with_timeout(text: str, timeout_seconds: int = 30):
    """
    Parse text with a timeout to avoid getting stuck on problem sentences.

    Args:
        text: Text to parse
        timeout_seconds: Maximum time to allow for parsing (default: 5s)

    Returns:
        Parsed AST or None if timeout/error

    Raises:
        ParseTimeout: If parsing exceeds timeout
    """
    from klareco.parser import parse

    # Set up signal handler for timeout
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_seconds)

    try:
        result = parse(text)
        signal.alarm(0)  # Cancel alarm
        return result
    finally:
        signal.alarm(0)  # Ensure alarm is cancelled
        signal.signal(signal.SIGALRM, old_handler)  # Restore old handler


def log_problem_sentence(sentence: str, source_file: str, para_num: int, error: str):
    """Log a problem sentence for later analysis."""
    # Ensure directory exists
    PROBLEM_SENTENCES_LOG.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        'timestamp': datetime.now().isoformat(),
        'sentence': sentence,
        'source_file': source_file,
        'paragraph': para_num,
        'error': error,
        'word_count': len(sentence.split()),
    }
    with open(PROBLEM_SENTENCES_LOG, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')


def unwrap_paragraphs(text: str) -> List[str]:
    """
    Unwrap hard-wrapped paragraphs.

    Args:
        text: Text with hard-wrapped lines

    Returns:
        List of unwrapped paragraphs
    """
    paragraphs = []
    current_lines = []

    for line in text.split('\n'):
        line = line.strip()

        # Empty line = paragraph boundary
        if not line:
            if current_lines:
                # Join lines with space
                paragraphs.append(' '.join(current_lines))
                current_lines = []
        else:
            current_lines.append(line)

    # Don't forget last paragraph
    if current_lines:
        paragraphs.append(' '.join(current_lines))

    return paragraphs


def split_sentences(text: str) -> List[str]:
    """
    Split paragraph into sentences.

    Handles:
    - Sentence terminators: . ! ?
    - Esperanto abbreviations: D-ro, S-ro, k.t.p., etc.
    - Ellipsis: ... (don't split)

    Args:
        text: Paragraph text

    Returns:
        List of sentences
    """
    # Common Esperanto abbreviations that shouldn't trigger splits
    abbreviations = [
        'D-ro',      # Doktoro (Doctor)
        'S-ro',      # Sinjoro (Mister)
        'S-ino',     # Sinjorino (Mrs)
        'D-rino',    # Doktorino (female doctor)
        'k.t.p',     # kaj tiel plu (etc.)
        'k.a',       # kaj aliaj (and others)
        'n-ro',      # numero (number)
        'p.K',       # post Kristo (AD)
        'a.K',       # antaŭ Kristo (BC)
        'k.c',       # kaj ceteraj (and the rest)
        'ekz',       # ekzemple (for example)
        't.e',       # tio estas (that is)
    ]

    # Protect abbreviations temporarily
    protected = text
    for i, abbrev in enumerate(abbreviations):
        protected = protected.replace(abbrev, f'<ABBREV{i}>')

    # Split on sentence terminators followed by space and capital letter
    # or end of string
    sentences = re.split(r'([.!?])\s+(?=[A-ZĈĜĤĴŜŬ]|$)', protected)

    # Rejoin sentences with their punctuation
    result = []
    i = 0
    while i < len(sentences):
        if i + 1 < len(sentences) and sentences[i+1] in '.!?':
            # Sentence with punctuation
            sent = (sentences[i] + sentences[i+1]).strip()
            if sent:
                result.append(sent)
            i += 2
        else:
            # Last fragment (shouldn't happen with our regex)
            if sentences[i].strip():
                result.append(sentences[i].strip())
            i += 1

    # Restore abbreviations
    for i, abbrev in enumerate(abbreviations):
        result = [s.replace(f'<ABBREV{i}>', abbrev) for s in result]

    return [s for s in result if s]


def extract_sentences_from_file(
    file_path: Path,
    min_words: int = 3,
    max_words: int = 100,
    with_ast: bool = False,
    batch_size: int = 50
) -> List[Dict]:
    """
    Extract sentences from a cleaned text file.

    Args:
        file_path: Path to cleaned text file
        min_words: Minimum words per sentence (default: 3)
        max_words: Maximum words per sentence (default: 100)
        with_ast: Generate AST for each sentence (slower)
        batch_size: Process ASTs in batches to reduce memory (default: 50)

    Returns:
        List of sentence dictionaries with metadata
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Extract paragraphs and sentences
    paragraphs = unwrap_paragraphs(text)
    results = []

    # Import parser only if needed
    if with_ast:
        from klareco.parser import parse

    for para_num, para in enumerate(paragraphs, 1):
        sentences = split_sentences(para)

        for sent_num, sent in enumerate(sentences, 1):
            word_count = len(sent.split())

            # Filter by length
            if word_count < min_words or word_count > max_words:
                continue

            # Skip likely metadata/headers
            if sent.isupper() and word_count < 10:
                continue

            # Skip lines with lots of special chars (likely errors)
            special_ratio = sum(1 for c in sent if not c.isalnum() and c not in ' .,;:!?-—\'\"') / len(sent)
            if special_ratio > 0.3:
                continue

            entry = {
                'text': sent,
                'paragraph': para_num,
                'sentence_in_para': sent_num,
                'word_count': word_count,
            }

            # Optional: generate AST for quality control
            if with_ast:
                try:
                    ast = parse(sent)
                    entry['ast'] = ast
                    entry['parse_success'] = True
                    stats = ast.get('parse_statistics', {})
                    entry['parse_rate'] = stats.get('success_rate', 0.0)
                except Exception as e:
                    entry['ast'] = None
                    entry['parse_success'] = False
                    entry['parse_rate'] = 0.0
                    entry['parse_error'] = str(e)

            results.append(entry)

            # Periodically clear memory during AST generation
            if with_ast and len(results) % batch_size == 0:
                gc.collect()

    return results


def extract_sentences_streaming(
    file_path: Path,
    min_words: int = 3,
    max_words: int = 100,
    with_ast: bool = False,
    batch_size: int = 50,
    chunk_size_mb: int = 50,
    parse_timeout: int = 30,
    start_byte: int = 0,
) -> Iterator[Dict]:
    """
    Extract sentences from a file using streaming/generator pattern for memory efficiency.

    For very large files (like Wikipedia), reads and processes in chunks to avoid loading
    the entire file into memory at once.

    Args:
        file_path: Path to cleaned text file
        min_words: Minimum words per sentence (default: 3)
        max_words: Maximum words per sentence (default: 100)
        with_ast: Generate AST for each sentence (slower)
        batch_size: Trigger GC every N sentences (default: 50)
        chunk_size_mb: Process file in chunks of this size in MB (default: 50MB)
        parse_timeout: Max seconds to wait for parsing a sentence (default: 30)
        start_byte: Byte position to start from for resumption (default: 0)

    Yields:
        Sentence dictionaries with metadata
    """
    source_file = str(file_path.name)
    file_size = file_path.stat().st_size
    file_size_mb = file_size / (1024 * 1024)

    # For large files (>100MB), process in chunks (supports resumption via start_byte)
    if file_size_mb > 100:
        yield from _extract_sentences_chunked(
            file_path, min_words, max_words, with_ast, batch_size, chunk_size_mb, parse_timeout,
            start_byte=start_byte
        )
    else:
        # For smaller files, load entire file (faster)
        # Note: start_byte is ignored for small files since they load instantly
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Add byte tracking to each entry (for small files, we've read the whole file)
        for entry in _process_text_streaming(
            text, min_words, max_words, with_ast, batch_size,
            start_para=1, source_file=source_file, parse_timeout=parse_timeout
        ):
            entry['_byte_position'] = file_size  # Already read whole file
            entry['_file_size'] = file_size
            yield entry


def _process_text_streaming(
    text: str,
    min_words: int,
    max_words: int,
    with_ast: bool,
    batch_size: int,
    start_para: int = 1,
    source_file: str = "unknown",
    parse_timeout: int = 30
) -> Iterator[Dict]:
    """Helper to process text and yield sentences."""
    # We use parse_with_timeout instead of direct parse import

    # Extract paragraphs and sentences
    paragraphs = unwrap_paragraphs(text)
    count = 0

    for para_num, para in enumerate(paragraphs, start_para):
        # Skip English-language sections in Wikipedia (between <div lang="en"> and </div>)
        if '<div lang="en">' in para or para.strip().startswith('This part of Wikipedia'):
            continue

        sentences = split_sentences(para)

        for sent_num, sent in enumerate(sentences, 1):
            word_count = len(sent.split())

            # Filter by length
            if word_count < min_words or word_count > max_words:
                continue

            # Skip likely metadata/headers
            if sent.isupper() and word_count < 10:
                continue

            # Skip lines that look like English or other non-Esperanto text
            # Esperanto uses ĉ, ĝ, ĥ, ĵ, ŝ, ŭ and has no 'q', 'w', 'x', 'y' in native words
            # If sentence has English patterns or no Esperanto characteristics, skip
            if any(indicator in sent.lower() for indicator in ['http://', 'https://', 'www.', '.com', '.net', '.org']):
                # URLs - likely metadata, but let's keep if it's in an Esperanto sentence
                url_word_count = sum(1 for word in sent.split() if 'http' in word or 'www.' in word or '.com' in word)
                if url_word_count > word_count * 0.3:  # If >30% of words are URLs, skip
                    continue

            # Skip lines with lots of special chars (likely errors)
            special_ratio = sum(1 for c in sent if not c.isalnum() and c not in ' .,;:!?-—\'\"ĉĝĥĵŝŭĈĜĤĴŜŬ') / len(sent)
            if special_ratio > 0.3:
                continue

            entry = {
                'text': sent,
                'paragraph': para_num,
                'sentence_in_para': sent_num,
                'word_count': word_count,
            }

            # Optional: generate AST for quality control
            if with_ast:
                try:
                    ast = parse_with_timeout(sent, timeout_seconds=parse_timeout)
                    entry['ast'] = ast
                    entry['parse_success'] = True
                    stats = ast.get('parse_statistics', {})
                    entry['parse_rate'] = stats.get('success_rate', 0.0)
                except ParseTimeout:
                    # Log the problem sentence for later analysis
                    log_problem_sentence(sent, source_file, para_num, "timeout")
                    entry['ast'] = None
                    entry['parse_success'] = False
                    entry['parse_rate'] = 0.0
                    entry['parse_error'] = f"timeout after {parse_timeout}s"
                except Exception as e:
                    # Log other errors too
                    log_problem_sentence(sent, source_file, para_num, str(e))
                    entry['ast'] = None
                    entry['parse_success'] = False
                    entry['parse_rate'] = 0.0
                    entry['parse_error'] = str(e)

            count += 1
            yield entry

            # Periodically clear memory
            if count % batch_size == 0:
                gc.collect()


def _extract_sentences_chunked(
    file_path: Path,
    min_words: int,
    max_words: int,
    with_ast: bool,
    batch_size: int,
    chunk_size_mb: int,
    parse_timeout: int = 30,
    start_byte: int = 0,
) -> Iterator[Dict]:
    """
    Process very large files in chunks to avoid memory issues.
    Reads file chunk by chunk, maintaining paragraph boundaries.

    Each yielded entry includes '_byte_position' and '_file_size' for progress tracking.

    Args:
        start_byte: Byte position to start from (for resumption). Will seek to this
                   position and then find the next paragraph boundary before processing.
    """
    source_file = str(file_path.name)
    file_size = file_path.stat().st_size
    chunk_size_bytes = chunk_size_mb * 1024 * 1024
    para_num = 1
    overlap_buffer = ""  # Store incomplete paragraph from previous chunk
    bytes_read = 0  # Track position in file

    with open(file_path, 'r', encoding='utf-8') as f:
        # If resuming, seek to start position and find next paragraph boundary
        if start_byte > 0:
            f.seek(start_byte)
            # Read past any partial line/paragraph to find clean boundary
            # First, skip to end of current line
            f.readline()
            # Then find next paragraph boundary (empty line)
            while True:
                line = f.readline()
                if not line:  # EOF
                    return  # Nothing left to process
                if line.strip() == '':
                    break  # Found paragraph boundary
            bytes_read = f.tell()
            # Note: para_num will be wrong but we don't use it for resumption

        while True:
            # Read chunk
            chunk = f.read(chunk_size_bytes)
            if not chunk:
                # Process any remaining buffer
                if overlap_buffer.strip():
                    for entry in _process_text_streaming(
                        overlap_buffer, min_words, max_words, with_ast, batch_size, para_num,
                        source_file=source_file, parse_timeout=parse_timeout
                    ):
                        entry['_byte_position'] = bytes_read
                        entry['_file_size'] = file_size
                        yield entry
                break

            bytes_read = f.tell()  # Track byte position after reading chunk

            # Combine with overlap from previous chunk
            text = overlap_buffer + chunk

            # Find last complete paragraph boundary (double newline or end of chunk)
            last_para_boundary = text.rfind('\n\n')

            if last_para_boundary > 0:
                # Process complete paragraphs
                complete_text = text[:last_para_boundary]
                overlap_buffer = text[last_para_boundary:].lstrip()

                # Count paragraphs before processing
                para_count_before = para_num

                # Calculate byte position at start of this chunk's text
                # bytes_read is position after reading chunk, so subtract chunk size
                chunk_start_byte = bytes_read - len(chunk.encode('utf-8'))
                complete_text_bytes = len(complete_text.encode('utf-8'))

                # Process this chunk
                chars_processed = 0
                for entry in _process_text_streaming(
                    complete_text, min_words, max_words, with_ast, batch_size, para_num,
                    source_file=source_file, parse_timeout=parse_timeout
                ):
                    # Interpolate byte position based on character position in text
                    # Find where this sentence ends in complete_text
                    sent_text = entry.get('text', '')
                    sent_end_pos = complete_text.find(sent_text, chars_processed)
                    if sent_end_pos >= 0:
                        chars_processed = sent_end_pos + len(sent_text)
                        # Estimate byte position (assuming roughly 1 byte per char for ASCII,
                        # but this is approximate for UTF-8)
                        progress_ratio = chars_processed / len(complete_text) if complete_text else 0
                        entry['_byte_position'] = chunk_start_byte + int(complete_text_bytes * progress_ratio)
                    else:
                        entry['_byte_position'] = bytes_read
                    entry['_file_size'] = file_size
                    yield entry
                    # Track paragraph number from entries
                    if entry['paragraph'] > para_num:
                        para_num = entry['paragraph']

                # If no entries were yielded but we had text, still increment para counter
                if para_num == para_count_before and complete_text.strip():
                    para_num += complete_text.count('\n\n') + 1

            else:
                # No paragraph boundary found, keep entire text as overlap
                # This handles cases where a single paragraph is larger than chunk_size
                overlap_buffer = text

            # Clear chunk from memory
            del chunk
            del text
            gc.collect()


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description='Extract sentences from cleaned texts')
    parser.add_argument('input_file', type=Path, help='Input cleaned text file')
    parser.add_argument('--output', type=Path, help='Output JSONL file (default: stdout)')
    parser.add_argument('--min-words', type=int, default=3, help='Minimum words (default: 3)')
    parser.add_argument('--max-words', type=int, default=100, help='Maximum words (default: 100)')
    parser.add_argument('--with-ast', action='store_true', help='Generate AST (slower)')
    parser.add_argument('--stats', action='store_true', help='Print statistics')

    args = parser.parse_args()

    # Extract sentences
    sentences = extract_sentences_from_file(
        args.input_file,
        min_words=args.min_words,
        max_words=args.max_words,
        with_ast=args.with_ast
    )

    # Write output
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            for sent in sentences:
                f.write(json.dumps(sent, ensure_ascii=False) + '\n')
        print(f"✅ Wrote {len(sentences)} sentences to {args.output}", file=sys.stderr)
    else:
        for sent in sentences:
            print(json.dumps(sent, ensure_ascii=False))

    # Print statistics
    if args.stats:
        print(f"\n=== STATISTICS ===", file=sys.stderr)
        print(f"Total sentences: {len(sentences)}", file=sys.stderr)
        if args.with_ast and sentences:
            parsed = sum(1 for s in sentences if s.get('parse_success'))
            avg_rate = sum(s.get('parse_rate', 0) for s in sentences) / len(sentences)
            print(f"Successfully parsed: {parsed}/{len(sentences)} ({100*parsed/len(sentences):.1f}%)", file=sys.stderr)
            print(f"Average parse rate: {avg_rate:.2f}", file=sys.stderr)


if __name__ == '__main__':
    main()
