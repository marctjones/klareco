#!/usr/bin/env python3
"""
Build enhanced corpus with full citation metadata.

This script:
1. Extracts Wikipedia articles with titles and sections
2. Extracts books with chapter detection
3. Parses all sentences to ASTs
4. Combines into single corpus with rich metadata
5. Progress tracking and error logging
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

# Add parent to path
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


def parse_sentence_with_retry(text: str, max_retries: int = 2) -> Optional[dict]:
    """
    Parse sentence to AST with retries.

    Args:
        text: Sentence to parse
        max_retries: Number of retry attempts

    Returns:
        AST dict or None if parsing fails
    """
    for attempt in range(max_retries + 1):
        try:
            ast = parse(text)
            return ast
        except Exception as e:
            if attempt == max_retries:
                logger.debug(f"Parse failed after {max_retries} retries: {text[:50]}... Error: {e}")
                return None
            time.sleep(0.1)  # Brief delay before retry

    return None


def calculate_parse_rate(ast: dict) -> float:
    """
    Calculate parse success rate from AST.

    Args:
        ast: Parsed AST

    Returns:
        Parse rate (0.0 to 1.0)
    """
    if not ast:
        return 0.0

    stats = ast.get('parse_statistics', {})
    return stats.get('success_rate', 0.0)


def process_extracted_sentences(
    input_file: Path,
    output_file: Path,
    min_parse_rate: float = 0.5,
    checkpoint_file: Optional[Path] = None,
    checkpoint_interval: int = 1000
):
    """
    Process extracted sentences: parse to AST and filter by quality.

    Args:
        input_file: Input JSONL with extracted sentences
        output_file: Output JSONL with parsed ASTs
        min_parse_rate: Minimum parse rate to include (0.0-1.0)
        checkpoint_file: Checkpoint file for resuming
        checkpoint_interval: Save checkpoint every N sentences
    """
    logger.info(f"Processing: {input_file}")
    logger.info(f"Output: {output_file}")
    logger.info(f"Min parse rate: {min_parse_rate}")

    # Load checkpoint
    last_processed = 0
    if checkpoint_file and checkpoint_file.exists():
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
            last_processed = checkpoint.get('last_processed', 0)
            logger.info(f"Resuming from line {last_processed}")

    # Open files
    mode = 'a' if last_processed > 0 else 'w'

    total_processed = 0
    total_included = 0
    total_filtered = 0
    total_errors = 0

    start_time = time.time()

    with open(input_file, 'r', encoding='utf-8') as inp, \
         open(output_file, mode, encoding='utf-8') as out:

        for line_num, line in enumerate(inp, 1):
            # Skip already processed lines
            if line_num <= last_processed:
                continue

            try:
                entry = json.loads(line)
                text = entry.get('text', '')

                if not text:
                    continue

                # Parse to AST
                ast = parse_sentence_with_retry(text)

                if ast:
                    parse_rate = calculate_parse_rate(ast)

                    # Filter by parse quality
                    if parse_rate >= min_parse_rate:
                        # Add AST to entry
                        entry['ast'] = ast
                        entry['parse_rate'] = parse_rate

                        # Calculate word count
                        entry['word_count'] = len(text.split())

                        out.write(json.dumps(entry, ensure_ascii=False) + '\n')
                        total_included += 1
                    else:
                        total_filtered += 1
                        if total_filtered % 100 == 0:
                            logger.debug(f"Filtered {total_filtered} low-quality parses (rate < {min_parse_rate})")
                else:
                    total_filtered += 1

                total_processed += 1

                # Progress indicator
                if total_processed % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = total_processed / elapsed
                    include_pct = (total_included / total_processed * 100) if total_processed > 0 else 0

                    logger.info(
                        f"Progress: {total_processed:,} processed, "
                        f"{total_included:,} included ({include_pct:.1f}%), "
                        f"{rate:.0f} sent/sec"
                    )

                # Save checkpoint
                if checkpoint_file and total_processed % checkpoint_interval == 0:
                    with open(checkpoint_file, 'w') as f:
                        json.dump({
                            'last_processed': line_num,
                            'total_processed': total_processed,
                            'total_included': total_included,
                            'total_filtered': total_filtered
                        }, f)

            except Exception as e:
                total_errors += 1
                logger.error(f"Error processing line {line_num}: {e}")

                if total_errors % 10 == 0:
                    logger.warning(f"⚠ {total_errors} errors encountered")

    # Final summary
    elapsed = time.time() - start_time
    include_pct = (total_included / total_processed * 100) if total_processed > 0 else 0

    logger.info("=" * 60)
    logger.info("Processing complete!")
    logger.info(f"Total processed: {total_processed:,}")
    logger.info(f"Included (parse rate ≥ {min_parse_rate}): {total_included:,} ({include_pct:.1f}%)")
    logger.info(f"Filtered out: {total_filtered:,}")
    logger.info(f"Errors: {total_errors}")
    logger.info(f"Time elapsed: {elapsed/60:.1f} minutes")
    logger.info(f"Rate: {total_processed/(elapsed/60):.0f} sentences/min")
    logger.info("=" * 60)


def combine_sources(
    wikipedia_file: Path,
    books_file: Path,
    output_file: Path
):
    """
    Combine Wikipedia and books into single corpus file.

    Args:
        wikipedia_file: Wikipedia sentences JSONL
        books_file: Books sentences JSONL
        output_file: Combined output JSONL
    """
    logger.info("=" * 60)
    logger.info("Combining sources into final corpus")
    logger.info("=" * 60)

    total_sentences = 0

    with open(output_file, 'w', encoding='utf-8') as out:
        # Add Wikipedia
        if wikipedia_file.exists():
            logger.info(f"Adding Wikipedia from: {wikipedia_file}")
            wiki_count = 0

            with open(wikipedia_file, 'r', encoding='utf-8') as f:
                for line in f:
                    out.write(line)
                    wiki_count += 1
                    total_sentences += 1

                    if wiki_count % 10000 == 0:
                        logger.info(f"  {wiki_count:,} Wikipedia sentences added")

            logger.info(f"✓ Wikipedia: {wiki_count:,} sentences")
        else:
            logger.warning(f"⚠ Wikipedia file not found: {wikipedia_file}")

        # Add books
        if books_file.exists():
            logger.info(f"Adding books from: {books_file}")
            books_count = 0

            with open(books_file, 'r', encoding='utf-8') as f:
                for line in f:
                    out.write(line)
                    books_count += 1
                    total_sentences += 1

                    if books_count % 10000 == 0:
                        logger.info(f"  {books_count:,} book sentences added")

            logger.info(f"✓ Books: {books_count:,} sentences")
        else:
            logger.warning(f"⚠ Books file not found: {books_file}")

    logger.info("=" * 60)
    logger.info(f"✓ Combined corpus created: {total_sentences:,} total sentences")
    logger.info(f"Output: {output_file}")
    logger.info("=" * 60)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Build enhanced corpus with full metadata')
    parser.add_argument('--stage', choices=['extract', 'parse', 'combine', 'all'], default='all',
                        help='Which stage to run')
    parser.add_argument('--min-parse-rate', type=float, default=0.5,
                        help='Minimum parse rate to include sentence (0.0-1.0)')
    parser.add_argument('--output-dir', type=Path, default=Path('data/enhanced_corpus'),
                        help='Output directory')

    args = parser.parse_args()

    # Create directories
    args.output_dir.mkdir(parents=True, exist_ok=True)
    Path('logs').mkdir(exist_ok=True)

    # Define file paths
    wiki_extracted = Path('data/extracted/wikipedia_sentences.jsonl')
    books_extracted = Path('data/extracted/books_sentences.jsonl')
    wiki_parsed = args.output_dir / 'wikipedia_parsed.jsonl'
    books_parsed = args.output_dir / 'books_parsed.jsonl'
    combined = args.output_dir / 'corpus_with_metadata.jsonl'

    # Run stages
    if args.stage in ['extract', 'all']:
        logger.info("Stage 1: Extraction")
        logger.info("Please run the extraction scripts separately:")
        logger.info("  1. ./scripts/run_wikipedia_extraction.sh")
        logger.info("  2. ./scripts/run_books_extraction.sh")
        logger.info("")

        if args.stage == 'extract':
            sys.exit(0)

    if args.stage in ['parse', 'all']:
        logger.info("=" * 60)
        logger.info("Stage 2: Parsing to AST")
        logger.info("=" * 60)

        # Parse Wikipedia
        if wiki_extracted.exists():
            logger.info("\nParsing Wikipedia sentences...")
            process_extracted_sentences(
                input_file=wiki_extracted,
                output_file=wiki_parsed,
                min_parse_rate=args.min_parse_rate,
                checkpoint_file=args.output_dir / 'wiki_parse_checkpoint.json',
                checkpoint_interval=1000
            )
        else:
            logger.warning(f"⚠ Wikipedia extraction not found: {wiki_extracted}")

        # Parse books
        if books_extracted.exists():
            logger.info("\nParsing book sentences...")
            process_extracted_sentences(
                input_file=books_extracted,
                output_file=books_parsed,
                min_parse_rate=args.min_parse_rate,
                checkpoint_file=args.output_dir / 'books_parse_checkpoint.json',
                checkpoint_interval=1000
            )
        else:
            logger.warning(f"⚠ Books extraction not found: {books_extracted}")

    if args.stage in ['combine', 'all']:
        logger.info("\n" + "=" * 60)
        logger.info("Stage 3: Combining sources")
        logger.info("=" * 60)

        combine_sources(
            wikipedia_file=wiki_parsed,
            books_file=books_parsed,
            output_file=combined
        )

    logger.info("\n✓ All stages complete!")
    logger.info(f"Final corpus: {combined}")
