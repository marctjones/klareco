#!/usr/bin/env python3
"""
Build Unified Corpus from All Extracted Sources

This script:
1. Reads extracted sentences from all sources (GOLD/SILVER/BRONZE quality)
2. Parses each sentence to AST with current parser
3. Calculates parse_rate from AST statistics
4. Preserves quality and source metadata
5. Outputs unified corpus in standard format

Quality System (replaces old tier 0-6 numbering):
- GOLD: Authoritative (PMEG, Krestomatio, Lingvaj Respondoj) ~22K sentences
- SILVER: Literary (Gutenberg - high language quality) ~380K sentences
- BRONZE: Encyclopedic (Wikipedia - variable quality) ~3.8M sentences

Output format:
{
    "text": "...",
    "source": {
        "quality": "GOLD|SILVER|BRONZE",
        "name": "pmeg",
        "source_type": "grammar_reference|literary|encyclopedia|...",
        ...
    },
    "ast": {...},
    "parse_rate": 0.85
}

Usage:
    python scripts/parse/build_unified_corpus.py \\
        --output data/enhanced_corpus/corpus_with_metadata.jsonl \\
        --fresh

    python scripts/parse/build_unified_corpus.py --resume  # Continue from checkpoint
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco import parser

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Source file configurations
# Quality levels based on Esperanto language quality:
#   GOLD: Authoritative (expert-written) OR parse_rate >= 0.98
#   SILVER: Literary (published books) OR parse_rate >= 0.95
#   BRONZE: Encyclopedic (crowd-sourced) OR parse_rate >= 0.90
#   COPPER: parse_rate < 0.90 (may be excluded)
#
# Default quality is baseline, but can be upgraded/downgraded by:
#   1. Parse rate (automated)
#   2. Manual overrides (data/quality_overrides.json)
SOURCE_CONFIGS = {
    # GOLD: Authoritative grammar and Q&A (always GOLD)
    'authoritative_grammar': {
        'path': 'data/extracted/eo/tier0_filtered/grammar/*.jsonl',
        'base_quality': 'GOLD',
        'allow_parse_rate_adjustment': False,  # Always GOLD
        'text_field': 'sentence',
    },
    'authoritative_literary': {
        'path': 'data/extracted/eo/tier0_filtered/literary/*.jsonl',
        'base_quality': 'GOLD',
        'allow_parse_rate_adjustment': False,  # Always GOLD
        'text_field': 'sentence',
    },
    # SILVER: Gutenberg books (can be adjusted by parse rate or overrides)
    'gutenberg': {
        'path': 'data/extracted/books_sentences.jsonl',
        'base_quality': 'SILVER',
        'allow_parse_rate_adjustment': True,
        'text_field': 'text',
    },
    # BRONZE: Wikipedia (can be adjusted by parse rate or overrides)
    'wikipedia': {
        'path': 'data/extracted/wikipedia_sentences.jsonl',
        'base_quality': 'BRONZE',
        'allow_parse_rate_adjustment': True,
        'text_field': 'text',
        'source_name_field': 'article_title',
    },
}


# Quality overrides and exclusions
_quality_overrides = None
_quality_exclusions = None


def load_quality_overrides(overrides_path: Path) -> tuple:
    """Load quality overrides and exclusions from JSON config."""
    if not overrides_path.exists():
        logger.info(f"No quality overrides file found at {overrides_path}")
        return {}, {}

    try:
        with open(overrides_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # Load overrides (only enabled ones)
        overrides = {}
        for key, override in config.get('overrides', {}).items():
            if key.startswith('_'):  # Skip examples and comments
                continue
            if override.get('enabled', True):
                source_name = override['source_name']
                overrides[source_name] = override['quality']

        # Load exclusions (only enabled ones)
        exclusions = set()
        for key, exclusion in config.get('exclude', {}).items():
            if key.startswith('_'):  # Skip examples and comments
                continue
            if exclusion.get('enabled', True):
                exclusions.add(exclusion['source_name'])

        logger.info(f"Loaded {len(overrides)} quality overrides and {len(exclusions)} exclusions")
        return overrides, exclusions

    except Exception as e:
        logger.warning(f"Failed to load quality overrides: {e}")
        return {}, {}


def calculate_quality_by_parse_rate(parse_rate: float, base_quality: str, allow_adjustment: bool) -> str:
    """Calculate quality based on parse rate."""
    if not allow_adjustment:
        return base_quality

    # Use parse rate thresholds
    if parse_rate >= 0.98:
        return 'GOLD'
    elif parse_rate >= 0.95:
        return 'SILVER'
    elif parse_rate >= 0.90:
        return 'BRONZE'
    else:
        return 'COPPER'


def determine_final_quality(
    source_name: str,
    parse_rate: float,
    base_quality: str,
    allow_adjustment: bool,
    overrides: dict
) -> str:
    """Determine final quality using hybrid approach: parse rate + overrides."""
    # Check for manual override first
    if source_name in overrides:
        return overrides[source_name]

    # Use parse rate if adjustment allowed
    if allow_adjustment:
        return calculate_quality_by_parse_rate(parse_rate, base_quality, allow_adjustment)

    # Otherwise use base quality
    return base_quality


def is_excluded(source_name: str, exclusions: set) -> bool:
    """Check if source should be excluded."""
    return source_name in exclusions


def load_checkpoint(checkpoint_path: Path) -> Dict[str, int]:
    """Load processing checkpoint."""
    if not checkpoint_path.exists():
        return {}

    try:
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load checkpoint: {e}")
        return {}


def save_checkpoint(checkpoint_path: Path, completed: Dict[str, int]):
    """Atomically save checkpoint."""
    temp_path = checkpoint_path.with_suffix('.tmp')
    try:
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(completed, f, indent=2, ensure_ascii=False)
        temp_path.rename(checkpoint_path)
        logger.debug(f"Checkpoint saved: {len(completed)} files completed")
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")
        if temp_path.exists():
            temp_path.unlink()


def format_authoritative_entry(entry: dict, config: dict) -> dict:
    """Format authoritative (GOLD) extracted sentence to unified corpus format."""
    text = entry['sentence']

    # Build source metadata (GOLD sources have rich metadata)
    # Note: quality will be determined after parsing (hybrid approach)
    source = {
        'name': entry['source'],
        'source_type': entry['source_type'],
        'source_name': entry['source_title'],
        'author': entry['author'],
        'year': entry['year'],
        '_base_quality': config['base_quality'],
        '_allow_adjustment': config['allow_parse_rate_adjustment'],
    }

    # Add optional fields
    if 'translator' in entry and entry['translator']:
        source['translator'] = entry['translator']
    if 'sentence_type' in entry:
        source['sentence_type'] = entry['sentence_type']
    if 'version' in entry:
        source['version'] = entry['version']

    return text, source


def format_encyclopedic_entry(entry: dict, config: dict) -> dict:
    """Format encyclopedic (BRONZE) extracted sentence to unified corpus format."""
    text = entry['text']

    # Note: quality will be determined after parsing (hybrid approach)
    source = {
        'name': 'wikipedia',
        'source_type': 'encyclopedia',
        'source_name': entry.get('article_title', 'Unknown Article'),
        'article_id': entry.get('article_id'),
        'section': entry.get('section'),
        'section_level': entry.get('section_level'),
        '_base_quality': config['base_quality'],
        '_allow_adjustment': config['allow_parse_rate_adjustment'],
    }

    return text, source


def format_literary_entry(entry: dict, config: dict) -> dict:
    """Format literary (SILVER) book sentence to unified corpus format."""
    text = entry['text']

    # Note: quality will be determined after parsing (hybrid approach)
    source = {
        'name': 'gutenberg',
        'source_type': 'literary',
        'source_name': entry.get('source_name', entry.get('source', 'Unknown Book')),
        'chapter': entry.get('chapter'),
        'chapter_number': entry.get('chapter_number'),
        '_base_quality': config['base_quality'],
        '_allow_adjustment': config['allow_parse_rate_adjustment'],
    }

    return text, source


def format_entry(entry: dict, config: dict, source_type: str) -> Optional[tuple]:
    """Format entry based on source type."""
    try:
        if source_type.startswith('authoritative'):  # GOLD quality
            return format_authoritative_entry(entry, config)
        elif source_type == 'gutenberg':  # SILVER quality
            return format_literary_entry(entry, config)
        elif source_type == 'wikipedia':  # BRONZE quality
            return format_encyclopedic_entry(entry, config)
        else:
            logger.warning(f"Unknown source type: {source_type}")
            return None
    except KeyError as e:
        logger.warning(f"Missing required field in entry: {e}")
        return None


def parse_and_build_entry(text: str, source: dict, overrides: dict, exclusions: set) -> Optional[dict]:
    """Parse sentence to AST and build unified corpus entry with hybrid quality."""
    # Extract quality determination parameters
    source_name = source.get('source_name', 'unknown')
    base_quality = source.pop('_base_quality', 'BRONZE')
    allow_adjustment = source.pop('_allow_adjustment', True)

    # Check if source is excluded
    if is_excluded(source_name, exclusions):
        return None

    # Parse to AST
    try:
        ast = parser.parse(text)
    except Exception as e:
        logger.debug(f"Parse failed for: {text[:50]}... Error: {e}")
        # Create minimal AST for unparseable sentences
        ast = {
            'tipo': 'frazo',
            'parse_statistics': {
                'success_rate': 0.0,
                'total_words': len(text.split()),
                'esperanto_words': 0,
                'non_esperanto_words': len(text.split()),
            },
            'error': str(e)
        }

    # Extract parse rate from AST statistics
    parse_rate = ast.get('parse_statistics', {}).get('success_rate', 0.0)

    # Determine final quality using hybrid approach
    final_quality = determine_final_quality(
        source_name,
        parse_rate,
        base_quality,
        allow_adjustment,
        overrides
    )

    # Add quality to source metadata
    source['quality'] = final_quality

    return {
        'text': text,
        'source': source,
        'ast': ast,
        'parse_rate': parse_rate
    }


def process_source_files(
    source_type: str,
    config: dict,
    output_file: Path,
    completed_files: Dict[str, int],
    resume: bool,
    overrides: dict,
    exclusions: set,
    checkpoint_path: Path = None,
    global_sentence_count: List[int] = None
) -> tuple:
    """Process all files for a source type.

    Args:
        global_sentence_count: List with one element [count] for tracking total sentences
                              across all sources (mutable for checkpoint updates)
    """
    from glob import glob

    path_pattern = config['path']
    files = sorted(glob(path_pattern))

    if not files:
        logger.warning(f"No files found for {source_type}: {path_pattern}")
        return 0, 0

    total_processed = 0
    total_failed = 0

    # Checkpoint interval (every 150K sentences)
    CHECKPOINT_INTERVAL = 150000
    last_checkpoint_count = global_sentence_count[0] if global_sentence_count else 0

    with open(output_file, 'a', encoding='utf-8') as out_f:
        for file_path in files:
            file_name = Path(file_path).name
            file_key = f"{source_type}:{file_name}"

            # Skip if already processed (when resuming)
            if resume and file_key in completed_files:
                count = completed_files[file_key]
                logger.info(f"  Skipping {file_name} (already processed: {count} sentences)")
                total_processed += count
                # Update global count for skipped files
                if global_sentence_count is not None:
                    global_sentence_count[0] += count
                continue

            logger.info(f"  Processing {file_name}...")

            file_processed = 0
            file_failed = 0

            with open(file_path, 'r', encoding='utf-8') as in_f:
                for line_num, line in enumerate(in_f, 1):
                    try:
                        entry = json.loads(line)

                        # Format entry based on source type
                        result = format_entry(entry, config, source_type)
                        if result is None:
                            file_failed += 1
                            continue

                        text, source = result

                        # Skip empty sentences
                        if not text or not text.strip():
                            file_failed += 1
                            continue

                        # Parse and build corpus entry (with hybrid quality)
                        corpus_entry = parse_and_build_entry(text, source, overrides, exclusions)

                        # Skip if excluded
                        if corpus_entry is None:
                            file_failed += 1
                            continue

                        # Track low parse rate
                        if corpus_entry['parse_rate'] < 0.3:
                            file_failed += 1

                        # Write to corpus
                        out_f.write(json.dumps(corpus_entry, ensure_ascii=False) + '\n')
                        file_processed += 1

                        # Update global count
                        if global_sentence_count is not None:
                            global_sentence_count[0] += 1

                            # Save checkpoint every 150K sentences
                            if checkpoint_path and global_sentence_count[0] - last_checkpoint_count >= CHECKPOINT_INTERVAL:
                                logger.info(f"    ⏸️  Checkpoint: {global_sentence_count[0]:,} sentences processed (saving...)")
                                # Update completed files with current progress
                                completed_files[file_key] = file_processed
                                save_checkpoint(checkpoint_path, completed_files)
                                last_checkpoint_count = global_sentence_count[0]

                        # Progress update
                        if file_processed % 10000 == 0:
                            logger.info(f"    Processed {file_processed:,} sentences...")

                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON at {file_name}:{line_num}")
                        file_failed += 1
                    except Exception as e:
                        logger.error(f"Error processing {file_name}:{line_num}: {e}")
                        file_failed += 1

            logger.info(f"    Added: {file_processed:,} sentences")
            if file_failed > 0:
                parse_fail_pct = (file_failed / (file_processed + file_failed)) * 100 if file_processed + file_failed > 0 else 0
                logger.info(f"    Low quality/failed: {file_failed:,} ({parse_fail_pct:.1f}%)")

            total_processed += file_processed
            total_failed += file_failed

            # Save checkpoint after each file
            completed_files[file_key] = file_processed
            if checkpoint_path:
                save_checkpoint(checkpoint_path, completed_files)

    return total_processed, total_failed


def build_corpus(
    output_file: Path,
    resume: bool = False,
    fresh: bool = False,
    overrides_path: Path = None
):
    """Build unified corpus from all extracted sources with hybrid quality."""
    logger.info("=" * 80)
    logger.info("BUILD UNIFIED CORPUS (Hybrid Quality System)")
    logger.info("=" * 80)
    logger.info("")

    # Load quality overrides and exclusions
    if overrides_path is None:
        overrides_path = Path('config/quality_overrides.json')

    overrides, exclusions = load_quality_overrides(overrides_path)

    logger.info("Quality System:")
    logger.info("  GOLD:   parse_rate >= 0.98 (exceptional)")
    logger.info("  SILVER: parse_rate >= 0.95 (high quality)")
    logger.info("  BRONZE: parse_rate >= 0.90 (good quality)")
    logger.info("  COPPER: parse_rate < 0.90  (fair quality)")
    logger.info("")

    # Checkpoint management
    checkpoint_path = output_file.parent / '.build_corpus_checkpoint.json'
    completed_files = {}

    if fresh and checkpoint_path.exists():
        logger.info("Fresh start requested - ignoring checkpoint")
        checkpoint_path.unlink()
        if output_file.exists():
            backup_path = output_file.with_suffix(f'.backup_{output_file.stat().st_mtime:.0f}.jsonl')
            logger.info(f"Backing up existing corpus to: {backup_path}")
            output_file.rename(backup_path)
    elif resume:
        completed_files = load_checkpoint(checkpoint_path)
        if completed_files:
            logger.info(f"Resuming from checkpoint: {len(completed_files)} files already processed")
        else:
            logger.info("No checkpoint found - starting from beginning")

    # Create output directory
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Create/truncate output file if fresh start
    if not resume or fresh:
        output_file.touch()
        output_file.write_text('')  # Truncate

    # Process each source type
    total_stats = defaultdict(int)

    # Global sentence counter for periodic checkpoints (use list for mutability)
    global_sentence_count = [0]

    for source_type, config in SOURCE_CONFIGS.items():
        logger.info("")
        logger.info(f"Processing {source_type} (base: {config['base_quality']})...")
        logger.info(f"  Pattern: {config['path']}")

        processed, failed = process_source_files(
            source_type,
            config,
            output_file,
            completed_files,
            resume,
            overrides,
            exclusions,
            checkpoint_path,
            global_sentence_count
        )

        # Note: actual quality distribution will be calculated after processing
        total_stats['total_processed'] += processed
        total_stats['total_failed'] += failed

        # Save checkpoint after each source type
        save_checkpoint(checkpoint_path, completed_files)

    # Summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("BUILD COMPLETE")
    logger.info("=" * 80)
    logger.info("")

    # Count actual quality distribution in corpus
    logger.info("Calculating actual quality distribution...")
    quality_counts = defaultdict(int)
    final_count = 0

    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            quality = entry.get('source', {}).get('quality', 'UNKNOWN')
            quality_counts[quality] += 1
            final_count += 1

    logger.info(f"Total sentences in corpus: {final_count:,}")
    logger.info("")
    logger.info("Breakdown by actual quality (after hybrid assessment):")
    for quality in ['GOLD', 'SILVER', 'BRONZE', 'COPPER']:
        count = quality_counts.get(quality, 0)
        if count > 0:
            pct = (count / final_count * 100) if final_count > 0 else 0
            logger.info(f"  {quality}: {count:,} sentences ({pct:.1f}%)")

    logger.info("")
    logger.info(f"Output: {output_file}")
    logger.info(f"Size: {output_file.stat().st_size / (1024**3):.2f} GB")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Rebuild Kuzu index:")
    logger.info("     ./scripts/index_kuzu.sh --fresh")
    logger.info("  2. Train M1 with quality priority:")
    logger.info("     ./scripts/train_m1_semantic_tier_priority.sh")
    logger.info("")

    # Clean up checkpoint on success
    if checkpoint_path.exists():
        checkpoint_path.unlink()


def main():
    parser_arg = argparse.ArgumentParser(
        description='Build unified corpus from all extracted sources'
    )
    parser_arg.add_argument(
        '--output',
        type=Path,
        default=Path('data/enhanced_corpus/corpus_with_metadata.jsonl'),
        help='Output corpus file'
    )
    parser_arg.add_argument(
        '--resume',
        action='store_true',
        help='Resume from checkpoint (skip already-processed files)'
    )
    parser_arg.add_argument(
        '--fresh',
        action='store_true',
        help='Start fresh, ignore checkpoint and backup existing corpus'
    )
    parser_arg.add_argument(
        '--overrides',
        type=Path,
        default=Path('config/quality_overrides.json'),
        help='Path to quality overrides config (JSON)'
    )

    args = parser_arg.parse_args()

    # Validate flags
    if args.resume and args.fresh:
        logger.error("Cannot use both --resume and --fresh flags")
        return 1

    try:
        build_corpus(
            args.output,
            resume=args.resume,
            fresh=args.fresh,
            overrides_path=args.overrides
        )
        return 0
    except Exception as e:
        logger.error(f"Build failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit(main())
