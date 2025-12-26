#!/usr/bin/env python3
"""
Extract Ekzercaro sentences from Fundamento de Esperanto.

Phase 0.2 of Fundamento-Centered Training (Issue #67)

Input: data/raw/fundamento/fundamento_de_esperanto.txt
Output: data/training/ekzercaro_sentences.jsonl

The Ekzercaro (lines ~2072-5295) contains 42 exercises written by Zamenhof
demonstrating canonical usage of Esperanto roots. These are the highest-quality
training examples available.

Format of output:
{
    "text": "La patro kaj la filo.",
    "exercise_number": 1,
    "line_number": 2080,
    "roots": ["patr", "fil"],
    "confidence": "authoritative"
}
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, List

# Add klareco to path for parser
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def setup_file_logging(log_path: Path):
    """Add file handler for logging."""
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)


def extract_roots_from_text(text: str) -> List[str]:
    """
    Extract probable roots from Esperanto text.

    This is a simple heuristic approach. For production, use the parser.
    """
    # Remove punctuation and lowercase
    clean = re.sub(r'[^\w\s]', ' ', text.lower())
    words = clean.split()

    roots = []
    for word in words:
        # Strip common endings to approximate root
        for ending in ['on', 'ojn', 'oj', 'an', 'aj', 'ajn', 'o', 'a', 'e', 'i',
                       'as', 'is', 'os', 'us', 'u', 'n']:
            if word.endswith(ending) and len(word) - len(ending) >= 2:
                root = word[:-len(ending)]
                if root and root not in ['l', 'k', 'la', 'kaj']:  # Skip articles/conjunctions
                    roots.append(root)
                    break
        else:
            # No ending matched - might be a root itself or foreign word
            if len(word) >= 2 and word not in ['la', 'kaj', 'en', 'de', 'al', 'kun', 'por', 'pri']:
                roots.append(word)

    return list(set(roots))


def is_sentence(text: str) -> bool:
    """Check if text looks like a sentence (not a header or instruction)."""
    text = text.strip()
    if not text:
        return False

    # Too short
    if len(text) < 5:
        return False

    # Headers and instructions often start with numbers or are all caps
    if text[0].isdigit() and not text[0:2].isdigit():
        return False

    # All uppercase suggests header
    if text.isupper():
        return False

    # Has at least one Esperanto word
    has_eo_word = bool(re.search(r'\b[A-Za-zĉĝĥĵŝŭĈĜĤĴŜŬ]{2,}', text))

    return has_eo_word


def extract_sentences_from_line(line: str) -> List[str]:
    """Split a line into individual sentences."""
    # Split on common sentence terminators
    parts = re.split(r'[.!?]+', line)
    sentences = []

    for part in parts:
        part = part.strip()
        if is_sentence(part):
            # Re-add period for complete sentence
            sentences.append(part + '.')

    return sentences


def extract_ekzercaro(fundamento_path: Path,
                      start_line: int = 2072,
                      end_line: int = 5295) -> List[dict]:
    """
    Extract Ekzercaro sentences from Fundamento.

    Args:
        fundamento_path: Path to fundamento_de_esperanto.txt
        start_line: Line where Ekzercaro starts (1-indexed)
        end_line: Line where Ekzercaro ends

    Returns:
        List of sentence dictionaries
    """
    logger.info(f"Reading Fundamento from {fundamento_path}")

    with open(fundamento_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    logger.info(f"Extracting Ekzercaro from lines {start_line}-{end_line}")

    sentences = []
    current_exercise = 0
    exercise_pattern = re.compile(r'^§?\s*(\d+)\s*[.\)]?\s*$')

    for line_num, line in enumerate(lines[start_line-1:end_line], start=start_line):
        line = line.strip()

        if not line:
            continue

        # Check for exercise number
        match = exercise_pattern.match(line)
        if match:
            current_exercise = int(match.group(1))
            logger.debug(f"Found exercise {current_exercise} at line {line_num}")
            continue

        # Skip obvious headers
        if line.startswith('EKZERCARO') or line.startswith('GRAMATI'):
            continue

        # Extract sentences from this line
        line_sentences = extract_sentences_from_line(line)

        for sentence_text in line_sentences:
            roots = extract_roots_from_text(sentence_text)

            sentences.append({
                'text': sentence_text,
                'exercise_number': current_exercise,
                'line_number': line_num,
                'roots': roots,
                'confidence': 'authoritative',
                'source': 'ekzercaro'
            })

    logger.info(f"Extracted {len(sentences)} sentences from {current_exercise} exercises")

    return sentences


def try_parse_with_klareco(sentences: List[dict]) -> List[dict]:
    """
    Try to parse sentences with Klareco parser for better root extraction.
    Falls back gracefully if parser not available.
    """
    try:
        from klareco.parser import parse_sentence
        logger.info("Using Klareco parser for accurate root extraction")

        parsed_count = 0
        for sent in sentences:
            try:
                ast = parse_sentence(sent['text'])
                if ast and ast.get('tipo') == 'frazo':
                    # Extract roots from AST
                    roots = extract_roots_from_ast(ast)
                    if roots:
                        sent['roots'] = roots
                        sent['parsed'] = True
                        parsed_count += 1
            except Exception as e:
                logger.debug(f"Parse error for '{sent['text'][:50]}': {e}")

        logger.info(f"Successfully parsed {parsed_count}/{len(sentences)} sentences")

    except ImportError:
        logger.warning("Klareco parser not available - using heuristic root extraction")

    return sentences


def extract_roots_from_ast(ast: dict) -> List[str]:
    """Extract roots from a parsed AST."""
    roots = []

    def visit(node):
        if isinstance(node, dict):
            if 'radiko' in node:
                roots.append(node['radiko'])
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(ast)
    return list(set(roots))


def main():
    parser = argparse.ArgumentParser(description='Extract Ekzercaro sentences from Fundamento')
    parser.add_argument('--input', type=Path,
                        default=Path('data/raw/fundamento/fundamento_de_esperanto.txt'),
                        help='Path to Fundamento text file')
    parser.add_argument('--output', type=Path,
                        default=Path('data/training/ekzercaro_sentences.jsonl'),
                        help='Output JSONL file')
    parser.add_argument('--log-dir', type=Path,
                        default=Path('logs/training'),
                        help='Directory for log files')
    parser.add_argument('--start-line', type=int, default=2072,
                        help='Line where Ekzercaro starts')
    parser.add_argument('--end-line', type=int, default=5295,
                        help='Line where Ekzercaro ends')
    parser.add_argument('--use-parser', action='store_true',
                        help='Use Klareco parser for root extraction')
    parser.add_argument('--dry-run', action='store_true',
                        help='Parse and report but do not write output')

    args = parser.parse_args()

    # Setup logging
    args.log_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_dir / f'extract_ekzercaro_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    setup_file_logging(log_path)

    logger.info("=" * 60)
    logger.info("Phase 0.2: Ekzercaro Extraction")
    logger.info("=" * 60)

    # Validate input
    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    # Extract sentences
    sentences = extract_ekzercaro(args.input, args.start_line, args.end_line)

    # Optionally use parser for better roots
    if args.use_parser:
        sentences = try_parse_with_klareco(sentences)

    # Report statistics
    exercises = set(s['exercise_number'] for s in sentences)
    all_roots = set()
    for s in sentences:
        all_roots.update(s['roots'])

    logger.info(f"\nStatistics:")
    logger.info(f"  Total sentences: {len(sentences)}")
    logger.info(f"  Exercises covered: {len(exercises)}")
    logger.info(f"  Unique roots found: {len(all_roots)}")

    # Sample output
    logger.info("\nSample sentences:")
    for sent in sentences[:5]:
        logger.info(f"  Ex {sent['exercise_number']}: {sent['text'][:60]}")
        logger.info(f"    Roots: {sent['roots'][:5]}")

    if args.dry_run:
        logger.info("\nDry run - not writing output")
        return

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp_path = args.output.with_suffix('.tmp')

    try:
        with open(temp_path, 'w', encoding='utf-8') as f:
            for sent in sentences:
                f.write(json.dumps(sent, ensure_ascii=False) + '\n')
        temp_path.rename(args.output)
        logger.info(f"\nOutput written to {args.output}")
    except Exception as e:
        logger.error(f"Failed to write output: {e}")
        if temp_path.exists():
            temp_path.unlink()
        sys.exit(1)

    logger.info(f"Log saved to {log_path}")
    logger.info("Phase 0.2 complete!")


if __name__ == '__main__':
    main()
