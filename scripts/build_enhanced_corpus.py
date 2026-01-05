#!/usr/bin/env python3
"""
Build enhanced corpus from extracted JSONL files.

This script:
- Reads extracted sentences from books_sentences.jsonl and wikipedia_sentences.jsonl
- Parses each sentence to AST using the Esperanto parser
- Filters by parse quality (configurable min_parse_rate)
- Assigns tiers based on source
- Outputs unified corpus JSONL with metadata

Usage:
    python scripts/build_enhanced_corpus.py --stage all
    python scripts/build_enhanced_corpus.py --stage books  # Only books
    python scripts/build_enhanced_corpus.py --stage wiki   # Only Wikipedia

The script supports checkpointing for restartability.
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional
from collections import Counter

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.parser import parse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('logs/corpus_building.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Tier assignments for sources
TIER_MAP = {
    # Tier 1: Fundamento (handled separately)
    'fundamento_ekzercaro': 1,

    # Tier 2: Fundamenta Krestomatio
    'fundamenta_krestomatio': 2,

    # Tier 3: Early classics
    'gerda_malaperis': 3,

    # Tier 5: Literature (most books)
    # Default for books

    # Tier 6: Wikipedia
    'wikipedia': 6,
}

# Authoritative sources get higher weight
AUTHORITATIVE_SOURCES = {
    'fundamenta_krestomatio',
    'fundamento_ekzercaro',
    'gerda_malaperis',
    'dokumentoj_de_esperanto',
    'dua_libro',
}


def get_tier(source: str) -> int:
    """Get tier for a source."""
    if source in TIER_MAP:
        return TIER_MAP[source]
    # Default tier 5 for literature
    return 5


def get_weight(source: str) -> float:
    """Get training weight for a source."""
    tier = get_tier(source)
    if tier == 1:
        return 10.0
    elif tier == 2:
        return 5.0
    elif tier == 3:
        return 3.0
    elif tier == 5:
        return 1.0 if source in AUTHORITATIVE_SOURCES else 0.8
    else:  # tier 6 (Wikipedia)
        return 0.5


def calculate_parse_rate(ast: dict) -> float:
    """Calculate parse success rate from AST."""
    if not ast:
        return 0.0

    stats = ast.get('parse_statistics', {})
    if stats:
        return stats.get('success_rate', 0.0)

    # Count words and successful parses
    total_words = 0
    successful = 0

    def count_words(node):
        nonlocal total_words, successful
        if isinstance(node, dict):
            if node.get('tipo') == 'vorto':
                total_words += 1
                if node.get('parse_status') == 'success':
                    successful += 1
            for value in node.values():
                count_words(value)
        elif isinstance(node, list):
            for item in node:
                count_words(item)

    count_words(ast)

    if total_words == 0:
        return 0.0
    return successful / total_words


def load_checkpoint(checkpoint_path: Path) -> Optional[dict]:
    """Load checkpoint if exists."""
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
    return None


def save_checkpoint(checkpoint_path: Path, state: dict):
    """Atomically save checkpoint."""
    temp_path = checkpoint_path.with_suffix('.tmp')
    try:
        with open(temp_path, 'w') as f:
            json.dump(state, f)
        temp_path.rename(checkpoint_path)
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")
        if temp_path.exists():
            temp_path.unlink()


def process_extracted_file(
    input_file: Path,
    output_file: Path,
    min_parse_rate: float,
    checkpoint_path: Path,
    stage: str,
) -> dict:
    """
    Process an extracted JSONL file and add to corpus.

    Returns statistics dict.
    """
    if not input_file.exists():
        logger.warning(f"Input file not found: {input_file}")
        return {'skipped': True}

    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path)
    start_line = 0
    stats = Counter()

    if checkpoint and checkpoint.get('stage') == stage:
        start_line = checkpoint.get('line', 0)
        stats = Counter(checkpoint.get('stats', {}))
        logger.info(f"Resuming from line {start_line}")

    # Count total lines for progress
    total_lines = sum(1 for _ in open(input_file, 'r', encoding='utf-8'))
    logger.info(f"Processing {input_file.name}: {total_lines:,} sentences")

    # Open output in append mode if resuming
    mode = 'a' if start_line > 0 else 'w'

    start_time = time.time()
    last_checkpoint_time = start_time

    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, mode, encoding='utf-8') as f_out:

        for line_num, line in enumerate(f_in):
            # Skip already processed lines
            if line_num < start_line:
                continue

            try:
                entry = json.loads(line.strip())
            except json.JSONDecodeError:
                stats['json_errors'] += 1
                continue

            text = entry.get('text', '')
            if not text or len(text) < 5:
                stats['too_short'] += 1
                continue

            # Parse to AST
            try:
                ast = parse(text)
            except Exception as e:
                stats['parse_errors'] += 1
                continue

            # Calculate parse rate
            parse_rate = calculate_parse_rate(ast)

            # Filter by parse rate
            if parse_rate < min_parse_rate:
                stats['low_quality'] += 1
                continue

            # Get source info
            source_id = entry.get('source', 'unknown')
            source_name = entry.get('source_name', source_id)

            # Build corpus entry
            corpus_entry = {
                'text': text,
                'source': {
                    'tier': get_tier(source_id),
                    'name': source_id,
                    'source_name': source_name,
                    'chapter': entry.get('chapter'),
                    'weight': get_weight(source_id),
                },
                'ast': ast,
                'parse_rate': parse_rate,
                'word_count': len(text.split()),
            }

            # Add optional metadata
            if entry.get('article_title'):
                corpus_entry['source']['article_title'] = entry['article_title']
            if entry.get('section'):
                corpus_entry['source']['section'] = entry['section']

            f_out.write(json.dumps(corpus_entry, ensure_ascii=False) + '\n')
            stats['processed'] += 1
            stats[f'source:{source_id}'] += 1

            # Progress and checkpoint
            if (line_num + 1) % 1000 == 0:
                elapsed = time.time() - start_time
                rate = (line_num + 1 - start_line) / elapsed
                pct = (line_num + 1) / total_lines * 100
                logger.info(f"Progress: {line_num + 1:,}/{total_lines:,} ({pct:.1f}%) - {rate:.0f} sent/sec")

                # Checkpoint every 30 seconds
                if time.time() - last_checkpoint_time > 30:
                    save_checkpoint(checkpoint_path, {
                        'stage': stage,
                        'line': line_num + 1,
                        'stats': dict(stats),
                    })
                    last_checkpoint_time = time.time()

    # Final checkpoint
    save_checkpoint(checkpoint_path, {
        'stage': stage,
        'line': total_lines,
        'stats': dict(stats),
        'completed': True,
    })

    elapsed = time.time() - start_time
    logger.info(f"Completed {stage}: {stats['processed']:,} sentences in {elapsed/60:.1f} min")

    return dict(stats)


def merge_corpus_files(
    output_dir: Path,
    final_output: Path,
    min_parse_rate: float,
):
    """Merge individual corpus files into final unified corpus."""
    logger.info("Merging corpus files...")

    books_file = output_dir / 'books_corpus.jsonl'
    wiki_file = output_dir / 'wikipedia_corpus.jsonl'

    total = 0
    sources = Counter()
    tier_counts = Counter()

    with open(final_output, 'w', encoding='utf-8') as f_out:
        for corpus_file in [books_file, wiki_file]:
            if not corpus_file.exists():
                continue

            with open(corpus_file, 'r', encoding='utf-8') as f_in:
                for line in f_in:
                    entry = json.loads(line)
                    f_out.write(line)
                    total += 1

                    source = entry.get('source', {})
                    sources[source.get('name', 'unknown')] += 1
                    tier_counts[source.get('tier', 5)] += 1

    # Write metadata
    meta_file = output_dir / 'corpus_metadata.json'
    with open(meta_file, 'w') as f:
        json.dump({
            'created': datetime.now().isoformat(),
            'total_entries': total,
            'min_parse_rate': min_parse_rate,
            'tier_counts': dict(tier_counts),
            'source_counts': dict(sources.most_common(50)),
        }, f, indent=2)

    logger.info(f"Merged corpus: {total:,} entries")
    logger.info(f"Metadata saved: {meta_file}")


def main():
    parser = argparse.ArgumentParser(description='Build enhanced corpus from extracted JSONL')
    parser.add_argument('--stage', choices=['all', 'books', 'wiki', 'merge'],
                        default='all', help='Which stage to run')
    parser.add_argument('--min-parse-rate', type=float, default=0.5,
                        help='Minimum parse rate to include sentence')
    parser.add_argument('--output-dir', type=Path, default=Path('data/enhanced_corpus'),
                        help='Output directory')
    parser.add_argument('--fresh', action='store_true',
                        help='Start fresh, ignore checkpoints')

    args = parser.parse_args()

    # Create directories
    args.output_dir.mkdir(parents=True, exist_ok=True)
    Path('logs').mkdir(exist_ok=True)

    checkpoint_path = args.output_dir / 'checkpoint.json'

    if args.fresh and checkpoint_path.exists():
        checkpoint_path.unlink()
        logger.info("Starting fresh (checkpoint removed)")

    logger.info("=" * 60)
    logger.info("Enhanced Corpus Builder")
    logger.info(f"Stage: {args.stage}")
    logger.info(f"Min parse rate: {args.min_parse_rate}")
    logger.info(f"Output: {args.output_dir}")
    logger.info("=" * 60)

    # Process books
    if args.stage in ['all', 'books']:
        books_input = Path('data/extracted/books_sentences.jsonl')
        books_output = args.output_dir / 'books_corpus.jsonl'

        stats = process_extracted_file(
            books_input,
            books_output,
            args.min_parse_rate,
            checkpoint_path,
            'books',
        )
        logger.info(f"Books stats: {stats}")

    # Process Wikipedia
    if args.stage in ['all', 'wiki']:
        wiki_input = Path('data/extracted/wikipedia_sentences.jsonl')
        wiki_output = args.output_dir / 'wikipedia_corpus.jsonl'

        stats = process_extracted_file(
            wiki_input,
            wiki_output,
            args.min_parse_rate,
            checkpoint_path,
            'wiki',
        )
        logger.info(f"Wikipedia stats: {stats}")

    # Merge into final corpus
    if args.stage in ['all', 'merge']:
        final_output = args.output_dir / 'corpus_with_metadata.jsonl'
        merge_corpus_files(args.output_dir, final_output, args.min_parse_rate)

    logger.info("=" * 60)
    logger.info("Corpus building complete!")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
