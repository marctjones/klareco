#!/usr/bin/env python3
"""
Parse entire Esperanto corpus into AST dataset.

This script reads the clean corpus (547MB) and parses every sentence into
a structured AST, saving them as JSONL for training the GNN encoder.

Usage:
    python scripts/parse_corpus_to_asts.py --input data/clean_corpus/ --output data/ast_corpus/

Output:
    - JSONL files (one AST per line)
    - Statistics: success rate, unknown words, processing time
    - Error log for failed sentences
"""

import argparse
import json
import re
import sys
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.parser import parse
from klareco.logging_config import setup_logging


def extract_sentences(text: str) -> List[str]:
    """
    Extract sentences from text.

    Args:
        text: Raw text from corpus file

    Returns:
        List of sentences (cleaned and filtered)
    """
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)

    # Filter and clean
    filtered = []
    esperanto_chars = set('ĉĝĥĵŝŭĈĜĤĴŜŬ')
    common_eo_words = ['la', 'kaj', 'estas', 'de', 'en', 'mi', 'vi', 'li', 'ŝi']

    for s in sentences:
        s = s.strip()

        # Length filter
        if not (20 < len(s) < 500):
            continue

        # Quality filter: has Esperanto markers
        has_eo_chars = any(c in s for c in esperanto_chars)
        has_common = any(f' {w} ' in f' {s.lower()} ' for w in common_eo_words)

        if has_eo_chars or has_common:
            filtered.append(s)

    return filtered


def parse_corpus_file(
    file_path: Path,
    output_dir: Path,
    error_log: Path,
    stats: Dict
) -> int:
    """
    Parse a single corpus file into ASTs.

    Args:
        file_path: Path to input text file
        output_dir: Directory to save AST JSONL
        error_log: File to log parsing errors
        stats: Dictionary to accumulate statistics

    Returns:
        Number of ASTs successfully parsed
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Processing {file_path.name}...")

    # Read text
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Extract sentences
    sentences = extract_sentences(text)
    stats['total_sentences'] += len(sentences)
    logger.info(f"  Extracted {len(sentences)} sentences")

    # Output file
    output_file = output_dir / f"{file_path.stem}_asts.jsonl"

    # Parse and save
    successful = 0
    failed = 0

    with open(output_file, 'w', encoding='utf-8') as out_f, \
         open(error_log, 'a', encoding='utf-8') as err_f:

        for i, sentence in enumerate(sentences, 1):
            try:
                # Parse to AST
                ast = parse(sentence)

                # Save as JSON line
                json.dump({
                    'source_file': file_path.name,
                    'sentence_id': i,
                    'sentence': sentence,
                    'ast': ast
                }, out_f, ensure_ascii=False)
                out_f.write('\n')

                successful += 1

                # Track statistics from AST
                if 'parse_statistics' in ast:
                    ps = ast['parse_statistics']
                    stats['total_words'] += ps.get('total_words', 0)
                    stats['esperanto_words'] += ps.get('esperanto_words', 0)
                    stats['non_esperanto_words'] += ps.get('non_esperanto_words', 0)

                    # Track failure categories
                    for category, count in ps.get('categories', {}).items():
                        stats['categories'][category] += count

            except Exception as e:
                # Log error
                err_f.write(f"{file_path.name}:{i} - {sentence[:100]}\n")
                err_f.write(f"  Error: {str(e)}\n\n")
                failed += 1
                stats['failed_sentences'] += 1

            # Progress update every 100 sentences
            if i % 100 == 0:
                logger.info(f"    Processed {i}/{len(sentences)} sentences ({successful} successful, {failed} failed)")

    stats['successful_sentences'] += successful
    logger.info(f"  Complete: {successful} successful, {failed} failed")

    return successful


def main():
    """Parse entire corpus into AST dataset."""
    parser = argparse.ArgumentParser(description='Parse Esperanto corpus to ASTs')
    parser.add_argument('--input', type=str, default='data/clean_corpus',
                        help='Input corpus directory')
    parser.add_argument('--output', type=str, default='data/ast_corpus',
                        help='Output directory for AST JSONL files')
    parser.add_argument('--error-log', type=str, default='corpus_parsing_errors.log',
                        help='Error log file')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    args = parser.parse_args()

    # Setup logging
    setup_logging(level=logging.DEBUG if args.debug else logging.INFO)
    logger = logging.getLogger(__name__)

    # Create output directory
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    error_log = Path(args.error_log)
    if error_log.exists():
        error_log.unlink()  # Clear previous error log

    # Statistics
    stats = {
        'total_sentences': 0,
        'successful_sentences': 0,
        'failed_sentences': 0,
        'total_words': 0,
        'esperanto_words': 0,
        'non_esperanto_words': 0,
        'categories': defaultdict(int),
        'files_processed': 0,
        'start_time': datetime.now(),
    }

    # Process all text files
    logger.info("="*70)
    logger.info("CORPUS PARSING - PHASE 3")
    logger.info("="*70)
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Error log: {error_log}")
    logger.info("")

    text_files = sorted(input_dir.glob('*.txt'))
    if not text_files:
        logger.error(f"No .txt files found in {input_dir}")
        return 1

    logger.info(f"Found {len(text_files)} corpus files")
    logger.info("")

    # Process each file
    for file_path in text_files:
        try:
            parse_corpus_file(file_path, output_dir, error_log, stats)
            stats['files_processed'] += 1
        except Exception as e:
            logger.error(f"Failed to process {file_path.name}: {e}")

    # Final statistics
    stats['end_time'] = datetime.now()
    stats['duration'] = (stats['end_time'] - stats['start_time']).total_seconds()

    logger.info("")
    logger.info("="*70)
    logger.info("PARSING COMPLETE")
    logger.info("="*70)
    logger.info(f"Files processed: {stats['files_processed']}/{len(text_files)}")
    logger.info(f"Total sentences: {stats['total_sentences']:,}")
    logger.info(f"Successful: {stats['successful_sentences']:,} ({stats['successful_sentences']/stats['total_sentences']*100:.1f}%)")
    logger.info(f"Failed: {stats['failed_sentences']:,} ({stats['failed_sentences']/stats['total_sentences']*100:.1f}%)")
    logger.info("")
    logger.info(f"Total words: {stats['total_words']:,}")
    logger.info(f"Esperanto words: {stats['esperanto_words']:,} ({stats['esperanto_words']/stats['total_words']*100:.1f}%)")
    logger.info(f"Non-Esperanto: {stats['non_esperanto_words']:,} ({stats['non_esperanto_words']/stats['total_words']*100:.1f}%)")
    logger.info("")

    if stats['categories']:
        logger.info("Non-Esperanto word categories:")
        for category, count in sorted(stats['categories'].items(), key=lambda x: -x[1]):
            logger.info(f"  {category}: {count:,}")
        logger.info("")

    logger.info(f"Duration: {stats['duration']:.1f}s")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Error log: {error_log}")

    # Save statistics as JSON
    stats_file = output_dir / 'parsing_statistics.json'
    stats_serializable = {
        k: v for k, v in stats.items()
        if k not in ['start_time', 'end_time']
    }
    stats_serializable['duration_seconds'] = stats['duration']
    stats_serializable['categories'] = dict(stats['categories'])

    with open(stats_file, 'w') as f:
        json.dump(stats_serializable, f, indent=2)

    logger.info(f"Statistics saved: {stats_file}")
    logger.info("")
    logger.info("✅ Phase 3 corpus parsing complete!")

    return 0


if __name__ == '__main__':
    sys.exit(main())
