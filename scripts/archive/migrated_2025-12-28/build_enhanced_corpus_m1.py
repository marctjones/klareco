#!/usr/bin/env python3
"""
Build M1 Enhanced Corpus with ASTs and Metadata.

Input: Raw extracted sentences (data/extracted/*.jsonl)
Output: Enhanced corpus with AST structures (data/corpus_enhanced_m1.jsonl)

Progress: Logged to data/corpus_enhanced_m1_build.log
Checkpoint: data/corpus_enhanced_m1_progress.json (for smart restart)
"""

import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Set
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.parser import parse


def setup_logging(log_file: Path) -> logging.Logger:
    """Setup logging to both console and file."""

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler (INFO level)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # File handler (DEBUG level)
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    return logging.getLogger(__name__)


def process_sentence(
    sent_text: str,
    metadata: dict,
    sentence_num: int,
    logger: logging.Logger
) -> Optional[Dict]:
    """
    Parse sentence and build enhanced entry.

    Args:
        sent_text: Raw sentence text
        metadata: Source metadata (article, section, etc.)
        sentence_num: Global sentence number (for logging)
        logger: Logger instance

    Returns:
        Enhanced entry dict or None if parsing failed
    """

    try:
        # Parse to AST
        ast = parse(sent_text)

        if ast is None:
            logger.debug(f"Sentence {sentence_num}: Parse returned None")
            return None

        # Get parse statistics
        stats = ast.get('parse_statistics', {})
        success_rate = stats.get('success_rate', 0.0)
        total_words = stats.get('total_words', 0)

        # Log low-quality parses (but still include them)
        if success_rate < 0.5:
            logger.debug(
                f"Sentence {sentence_num}: Low parse quality ({success_rate:.2f}) - "
                f"{total_words} words - '{sent_text[:50]}...'"
            )

        # Build enhanced entry
        enhanced = {
            'text': sent_text,
            'ast': ast,
            'metadata': metadata,
            'parse_quality': success_rate,
            'word_count': total_words,
        }

        return enhanced

    except Exception as e:
        logger.warning(
            f"Sentence {sentence_num}: Exception during parsing - {e} - "
            f"'{sent_text[:50]}...'"
        )
        return None


def count_lines(file_path: Path) -> int:
    """Count lines in file (for progress bar)."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)


def load_progress(progress_file: Path) -> Dict:
    """Load progress checkpoint."""
    if not progress_file.exists():
        return {
            'files_completed': [],
            'current_file': None,
            'lines_processed': 0,
            'total_success': 0,
            'total_filtered': 0,
        }

    with open(progress_file, 'r') as f:
        return json.load(f)


def save_progress(progress_file: Path, progress: Dict):
    """Save progress checkpoint atomically."""
    temp_file = progress_file.with_suffix('.json.tmp')
    try:
        with open(temp_file, 'w') as f:
            json.dump(progress, f, indent=2)
        temp_file.rename(progress_file)
    except Exception as e:
        if temp_file.exists():
            temp_file.unlink()
        raise e


def get_processed_lines(output_file: Path) -> int:
    """Count how many lines are already in output file."""
    if not output_file.exists():
        return 0
    return count_lines(output_file)


def process_file(
    input_file: Path,
    output_file: Path,
    logger: logging.Logger,
    global_sentence_num: int,
    progress_file: Path,
    skip_lines: int = 0,
    min_parse_quality: float = 0.0,
) -> tuple:
    """
    Process one input file and append to output.

    Args:
        input_file: Input JSONL file
        output_file: Output JSONL file (append mode)
        logger: Logger instance
        global_sentence_num: Starting sentence number
        progress_file: Progress checkpoint file
        skip_lines: Number of lines to skip (already processed)
        min_parse_quality: Minimum parse quality to include (0.0 = include all)

    Returns:
        (processed_count, success_count, filtered_count, new_global_num)
    """

    logger.info(f"Processing {input_file.name}...")

    # Count lines for progress bar
    total_lines = count_lines(input_file)
    logger.info(f"  Total sentences: {total_lines:,}")

    if skip_lines > 0:
        logger.info(f"  Resuming from line {skip_lines + 1:,} (skipping {skip_lines:,} already processed)")

    processed = 0
    success = 0
    filtered = 0
    current_num = global_sentence_num
    checkpoint_interval = 10000  # Save progress every 10K sentences

    # Open output in append mode
    with open(output_file, 'a', encoding='utf-8') as out_f:

        # Open input
        with open(input_file, 'r', encoding='utf-8') as in_f:

            # Progress bar
            pbar = tqdm(
                in_f,
                total=total_lines,
                desc=f"  {input_file.name}",
                unit=" sentences",
                ncols=100,
                initial=skip_lines
            )

            for line_num, line in enumerate(pbar):
                # Skip already processed lines
                if line_num < skip_lines:
                    continue
                try:
                    # Parse JSON
                    sent_data = json.loads(line)

                    # Extract text and metadata
                    text = sent_data.get('text', '').strip()
                    if not text:
                        continue

                    metadata = {
                        'source': sent_data.get('source'),
                        'source_name': sent_data.get('source_name'),
                        'article_title': sent_data.get('article_title'),
                        'article_id': sent_data.get('article_id'),
                        'section': sent_data.get('section'),
                        'section_level': sent_data.get('section_level'),
                        'timestamp': sent_data.get('timestamp'),
                    }

                    # Process sentence
                    enhanced = process_sentence(text, metadata, current_num, logger)

                    processed += 1
                    current_num += 1

                    if enhanced:
                        # Check quality filter
                        if enhanced['parse_quality'] >= min_parse_quality:
                            # Write to output
                            out_f.write(json.dumps(enhanced, ensure_ascii=False) + '\n')
                            success += 1
                        else:
                            filtered += 1

                    # Update progress bar with stats
                    pbar.set_postfix({
                        'success': f"{success:,}",
                        'filtered': f"{filtered:,}",
                        'rate': f"{100*success/processed:.1f}%" if processed > 0 else "0.0%"
                    })

                    # Save checkpoint periodically
                    if (line_num + 1) % checkpoint_interval == 0:
                        try:
                            progress = {
                                'current_file': input_file.name,
                                'lines_processed': line_num + 1,
                                'total_success': success,
                                'total_filtered': filtered,
                                'last_update': datetime.now().isoformat()
                            }
                            save_progress(progress_file, progress)
                        except Exception as e:
                            logger.warning(f"Failed to save checkpoint: {e}")

                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON at line {line_num + 1}: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Error processing line {line_num + 1}: {e}")
                    continue

            pbar.close()

    logger.info(f"  Processed: {processed:,}")
    if processed > 0:
        logger.info(f"  Success: {success:,} ({100*success/processed:.1f}%)")
        logger.info(f"  Filtered: {filtered:,} ({100*filtered/processed:.1f}%)")

    return processed, success, filtered, current_num


def main():
    """Main entry point."""

    # Setup paths
    data_dir = Path(__file__).parent.parent / 'data'
    extracted_dir = data_dir / 'extracted'
    output_file = data_dir / 'corpus_enhanced_m1.jsonl'
    log_file = data_dir / 'corpus_enhanced_m1_build.log'
    progress_file = data_dir / 'corpus_enhanced_m1_progress.json'

    # Setup logging
    logger = setup_logging(log_file)

    # Log start
    logger.info("=" * 80)
    logger.info("Starting M1 Enhanced Corpus Build")
    logger.info(f"Start time: {datetime.now()}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Progress file: {progress_file}")
    logger.info("=" * 80)

    # Input files
    input_files = [
        extracted_dir / 'wikipedia_sentences.jsonl',
        extracted_dir / 'books_sentences.jsonl',
    ]

    # Check input files exist
    for input_file in input_files:
        if not input_file.exists():
            logger.error(f"Input file not found: {input_file}")
            logger.error("Please run corpus extraction first!")
            return 1
        logger.info(f"Input: {input_file} ({input_file.stat().st_size / 1024 / 1024:.1f} MB)")

    # Load progress (for restart)
    progress = load_progress(progress_file)
    existing_lines = get_processed_lines(output_file)

    if existing_lines > 0:
        logger.warning(f"Output file already exists with {existing_lines:,} lines")
        logger.warning("Will resume from last checkpoint")
        logger.info(f"Last checkpoint: {progress.get('current_file', 'Unknown')} at line {progress.get('lines_processed', 0):,}")
    else:
        logger.info(f"Output: {output_file}")

    # Configuration
    min_parse_quality = 0.0  # Include all parses (we can filter later)
    logger.info(f"Min parse quality: {min_parse_quality}")
    logger.info("")

    # Process each file
    total_processed = 0
    total_success = 0
    total_filtered = 0
    global_num = 1

    for input_file in input_files:
        # Check if this file was already completed
        if input_file.name in progress.get('files_completed', []):
            logger.info(f"Skipping {input_file.name} (already completed)")
            continue

        # Calculate skip lines for resume
        skip_lines = 0
        if progress.get('current_file') == input_file.name:
            skip_lines = progress.get('lines_processed', 0)

        processed, success, filtered, global_num = process_file(
            input_file,
            output_file,
            logger,
            global_num,
            progress_file,
            skip_lines,
            min_parse_quality
        )

        total_processed += processed
        total_success += success
        total_filtered += filtered

        # Mark file as completed
        files_completed = progress.get('files_completed', [])
        if input_file.name not in files_completed:
            files_completed.append(input_file.name)
            progress['files_completed'] = files_completed
            progress['current_file'] = None
            progress['lines_processed'] = 0
            save_progress(progress_file, progress)

    # Clean up progress file on successful completion
    if progress_file.exists():
        progress_file.unlink()
        logger.info("Progress checkpoint removed (build complete)")

    # Final statistics
    logger.info("")
    logger.info("=" * 80)
    logger.info("Build Complete!")
    logger.info("=" * 80)
    if total_processed > 0:
        logger.info(f"Total processed: {total_processed:,}")
        logger.info(f"Total success: {total_success:,} ({100*total_success/total_processed:.1f}%)")
        logger.info(f"Total filtered: {total_filtered:,} ({100*total_filtered/total_processed:.1f}%)")
    logger.info(f"Output file: {output_file}")
    if output_file.exists():
        logger.info(f"Output size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
    logger.info(f"End time: {datetime.now()}")
    logger.info(f"Log file: {log_file}")
    logger.info("=" * 80)

    # Parse quality distribution
    logger.info("")
    logger.info("Analyzing parse quality distribution...")

    quality_buckets = {
        '0.0-0.3': 0,
        '0.3-0.5': 0,
        '0.5-0.7': 0,
        '0.7-0.9': 0,
        '0.9-1.0': 0,
    }

    with open(output_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Analyzing", unit=" sentences"):
            entry = json.loads(line)
            quality = entry.get('parse_quality', 0.0)

            if quality < 0.3:
                quality_buckets['0.0-0.3'] += 1
            elif quality < 0.5:
                quality_buckets['0.3-0.5'] += 1
            elif quality < 0.7:
                quality_buckets['0.5-0.7'] += 1
            elif quality < 0.9:
                quality_buckets['0.7-0.9'] += 1
            else:
                quality_buckets['0.9-1.0'] += 1

    logger.info("Parse quality distribution:")
    for bucket, count in quality_buckets.items():
        pct = 100 * count / total_success if total_success > 0 else 0
        logger.info(f"  {bucket}: {count:,} ({pct:.1f}%)")

    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Review log file for any issues")
    logger.info("  2. Build FAISS index: python scripts/index_corpus.py --corpus data/corpus_enhanced_m1.jsonl --output data/corpus_index_m1")
    logger.info("  3. Test retrieval: python scripts/demo_rag.py --index data/corpus_index_m1 --interactive")

    return 0


if __name__ == "__main__":
    sys.exit(main())
