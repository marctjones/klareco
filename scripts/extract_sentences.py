#!/usr/bin/env python3
"""
Extract well-formed sentences from hard-wrapped cleaned texts.

This handles:
1. Unwrapping hard-wrapped lines (75-char wrapping)
2. Splitting on sentence boundaries
3. Handling Esperanto abbreviations
4. Optional AST generation for quality control
"""

import re
from pathlib import Path
from typing import List, Dict, Optional
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


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
    with_ast: bool = False
) -> List[Dict]:
    """
    Extract sentences from a cleaned text file.

    Args:
        file_path: Path to cleaned text file
        min_words: Minimum words per sentence (default: 3)
        max_words: Maximum words per sentence (default: 100)
        with_ast: Generate AST for each sentence (slower)

    Returns:
        List of sentence dictionaries with metadata
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Extract paragraphs and sentences
    paragraphs = unwrap_paragraphs(text)
    results = []

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
                    from klareco.parser import parse
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

    return results


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
