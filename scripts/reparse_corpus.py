#!/usr/bin/env python3
"""
Reparse training corpus with updated parser.

This script reads an existing corpus JSONL file and re-parses each sentence
with the current parser, preserving metadata (source, original text, etc.).

Usage:
    python scripts/reparse_corpus.py \
        --input data/corpus_with_sources_v2.jsonl \
        --output data/corpus_with_sources_v3.jsonl

    # Resume from checkpoint (default)
    python scripts/reparse_corpus.py --input ... --output ...

    # Start fresh, ignoring checkpoint
    python scripts/reparse_corpus.py --input ... --output ... --fresh

The script:
1. Reads existing corpus entries
2. Re-parses the original text with updated parser
3. Preserves metadata (source, sentence_id, etc.)
4. Writes new JSONL with updated ASTs
5. Reports statistics on new fields added
6. Supports checkpointing for restartability
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from collections import Counter
import logging
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.parser import parse
from klareco.logging_config import setup_logging


def save_checkpoint(checkpoint_path: Path, state: dict):
    """Atomically save checkpoint state."""
    temp_path = checkpoint_path.with_suffix('.tmp')
    try:
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(state, f)
        temp_path.rename(checkpoint_path)
    except Exception as e:
        logging.error(f"Failed to save checkpoint: {e}")
        if temp_path.exists():
            temp_path.unlink()


def load_checkpoint(checkpoint_path: Path) -> dict:
    """Load checkpoint state if it exists."""
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"Failed to load checkpoint: {e}")
    return None


def reparse_corpus(input_path: Path, output_path: Path, batch_size: int = 10000,
                   resume: bool = True):
    """
    Reparse corpus with updated parser.

    Args:
        input_path: Path to existing corpus JSONL
        output_path: Path for output JSONL
        batch_size: Report progress every batch_size lines
        resume: If True, resume from checkpoint if available
    """
    logger = logging.getLogger(__name__)

    # Checkpoint file
    checkpoint_path = output_path.with_suffix('.checkpoint.json')

    # Statistics
    total_lines = 0
    successful = 0
    failed = 0
    start_line = 0

    # New field statistics
    new_fields = Counter()
    fraztipo_counts = Counter()
    participle_count = 0
    compound_count = 0
    elision_count = 0
    correlative_decomposed = 0

    # Try to resume from checkpoint
    if resume:
        checkpoint = load_checkpoint(checkpoint_path)
        if checkpoint:
            start_line = checkpoint.get('last_line', 0)
            total_lines = checkpoint.get('total_lines', 0)
            successful = checkpoint.get('successful', 0)
            failed = checkpoint.get('failed', 0)
            new_fields = Counter(checkpoint.get('new_fields', {}))
            fraztipo_counts = Counter(checkpoint.get('fraztipo_counts', {}))
            participle_count = checkpoint.get('participle_count', 0)
            compound_count = checkpoint.get('compound_count', 0)
            elision_count = checkpoint.get('elision_count', 0)
            correlative_decomposed = checkpoint.get('correlative_decomposed', 0)
            logger.info(f"Resuming from line {start_line:,}")

    start_time = time.time()

    # Get total line count for progress
    logger.info(f"Counting lines in {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        total_expected = sum(1 for _ in f)
    logger.info(f"Total lines: {total_expected:,}")

    if start_line >= total_expected:
        logger.info("Already completed! Nothing to do.")
        return

    # Open output file in append mode if resuming
    write_mode = 'a' if start_line > 0 else 'w'

    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, write_mode, encoding='utf-8') as outfile:

        for line_num, line in enumerate(infile, 1):
            # Skip already processed lines if resuming
            if line_num <= start_line:
                continue

            try:
                entry = json.loads(line.strip())

                # Get the original text to reparse
                text = entry.get('text') or entry.get('sentence') or entry.get('original_text')

                if not text:
                    logger.warning(f"Line {line_num}: No text field found")
                    outfile.write(line)  # Keep original
                    failed += 1
                    total_lines += 1
                    continue

                # Reparse with updated parser
                try:
                    new_ast = parse(text)

                    # Update the entry with new AST
                    entry['ast'] = new_ast
                    entry['reparsed_at'] = datetime.now().isoformat()
                    entry['parser_version'] = 'v3'  # After Stage 0 fixes

                    # Track new fields added
                    if new_ast.get('fraztipo'):
                        new_fields['fraztipo'] += 1
                        fraztipo_counts[new_ast['fraztipo']] += 1

                    if new_ast.get('demandotipo'):
                        new_fields['demandotipo'] += 1

                    # Check word-level new fields
                    for word in _extract_words(new_ast):
                        if word.get('participo_voĉo'):
                            participle_count += 1
                        if word.get('kunmetitaj_radikoj'):
                            compound_count += 1
                        if word.get('elidita'):
                            elision_count += 1
                        if word.get('korelativo_prefikso'):
                            correlative_decomposed += 1

                    successful += 1

                except Exception as e:
                    # Keep original AST if reparse fails
                    entry['reparse_error'] = str(e)
                    failed += 1

                # Write updated entry
                outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')
                total_lines += 1

                # Progress report and checkpoint
                if line_num % batch_size == 0:
                    elapsed = time.time() - start_time
                    rate = (line_num - start_line) / elapsed if elapsed > 0 else 0
                    remaining = total_expected - line_num
                    eta = remaining / rate if rate > 0 else 0
                    logger.info(
                        f"Progress: {line_num:,}/{total_expected:,} ({line_num/total_expected*100:.1f}%) | "
                        f"Rate: {rate:.0f}/s | ETA: {eta/60:.1f}m | "
                        f"Success: {successful:,} | Failed: {failed:,}"
                    )

                    # Save checkpoint
                    save_checkpoint(checkpoint_path, {
                        'last_line': line_num,
                        'total_lines': total_lines,
                        'successful': successful,
                        'failed': failed,
                        'new_fields': dict(new_fields),
                        'fraztipo_counts': dict(fraztipo_counts),
                        'participle_count': participle_count,
                        'compound_count': compound_count,
                        'elision_count': elision_count,
                        'correlative_decomposed': correlative_decomposed,
                    })

            except json.JSONDecodeError as e:
                logger.error(f"Line {line_num}: JSON decode error: {e}")
                outfile.write(line)  # Keep original line
                failed += 1
                total_lines += 1

    # Final statistics
    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("REPARSE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total lines processed: {total_lines:,}")
    if total_lines > 0:
        logger.info(f"Successful: {successful:,} ({successful/total_lines*100:.1f}%)")
        logger.info(f"Failed: {failed:,} ({failed/total_lines*100:.1f}%)")
    logger.info(f"Time elapsed: {elapsed/60:.1f} minutes")
    if elapsed > 0:
        logger.info(f"Average rate: {(total_lines - (start_line if resume else 0))/elapsed:.0f} lines/second")
    logger.info("")
    logger.info("NEW FIELDS ADDED:")
    logger.info(f"  fraztipo: {new_fields['fraztipo']:,}")
    logger.info(f"    - demando: {fraztipo_counts['demando']:,}")
    logger.info(f"    - ordono: {fraztipo_counts['ordono']:,}")
    logger.info(f"    - deklaro: {fraztipo_counts['deklaro']:,}")
    logger.info(f"  demandotipo: {new_fields['demandotipo']:,}")
    logger.info(f"  participles (voĉo/tempo): {participle_count:,}")
    logger.info(f"  compound words (kunmetitaj_radikoj): {compound_count:,}")
    logger.info(f"  elided words: {elision_count:,}")
    logger.info(f"  correlatives decomposed: {correlative_decomposed:,}")

    # Remove checkpoint file on successful completion
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        logger.info("Checkpoint file removed (completed successfully)")


def _extract_words(ast: dict) -> list:
    """Extract all word nodes from an AST."""
    words = []

    def _visit(node):
        if isinstance(node, dict):
            if node.get('tipo') == 'vorto':
                words.append(node)
            elif node.get('tipo') == 'vortgrupo':
                if node.get('kerno'):
                    _visit(node['kerno'])
                for desc in node.get('priskriboj', []):
                    _visit(desc)
            else:
                for value in node.values():
                    _visit(value)
        elif isinstance(node, list):
            for item in node:
                _visit(item)

    _visit(ast)
    return words


def main():
    parser = argparse.ArgumentParser(description='Reparse corpus with updated parser')
    parser.add_argument('--input', required=True, type=Path,
                        help='Input corpus JSONL file')
    parser.add_argument('--output', required=True, type=Path,
                        help='Output corpus JSONL file')
    parser.add_argument('--batch-size', type=int, default=10000,
                        help='Report progress every N lines')
    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    parser.add_argument('--fresh', action='store_true',
                        help='Start fresh, ignoring any existing checkpoint')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Run reparse
    reparse_corpus(args.input, args.output, args.batch_size, resume=not args.fresh)


if __name__ == '__main__':
    main()
