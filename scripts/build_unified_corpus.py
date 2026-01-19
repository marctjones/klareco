#!/usr/bin/env python3
"""
Build Unified Corpus from All Extracted Sources

This script:
1. Reads extracted sentences from all sources (tier0-6)
2. Parses each sentence to AST with current parser
3. Calculates parse_rate from AST statistics
4. Preserves tier, quality, and source metadata
5. Outputs unified corpus in standard format

Output format:
{
    "text": "...",
    "source": {
        "tier": 0-6,
        "name": "pmeg",
        "quality": "authoritative|high|medium|low",
        "source_type": "grammar_reference|literary|encyclopedia|...",
        ...
    },
    "ast": {...},
    "parse_rate": 0.85
}

Usage:
    python scripts/build_unified_corpus.py \\
        --output data/enhanced_corpus/corpus_with_metadata.jsonl \\
        --fresh

    python scripts/build_unified_corpus.py --resume  # Continue from checkpoint
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
SOURCE_CONFIGS = {
    # Tier 0: Authoritative grammar and Q&A
    'tier0_grammar': {
        'path': 'data/extracted/eo/tier0_filtered/grammar/*.jsonl',
        'tier': 0,
        'quality': 'authoritative',
        'text_field': 'sentence',
    },
    'tier0_literary': {
        'path': 'data/extracted/eo/tier0_filtered/literary/*.jsonl',
        'tier': 0,
        'quality': 'authoritative',
        'text_field': 'sentence',
    },
    # Tier 5: Wikipedia (high coverage, medium quality)
    'wikipedia': {
        'path': 'data/extracted/wikipedia_sentences.jsonl',
        'tier': 5,
        'quality': 'medium',
        'text_field': 'text',
        'source_name_field': 'article_title',
    },
    # Tier 6: Gutenberg books (natural text, medium-low quality)
    'gutenberg': {
        'path': 'data/extracted/books_sentences.jsonl',
        'tier': 6,
        'quality': 'medium',
        'text_field': 'text',
    },
}


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


def format_tier0_entry(entry: dict, config: dict) -> dict:
    """Format tier0 extracted sentence to unified corpus format."""
    text = entry['sentence']

    # Build source metadata (tier0 has rich metadata)
    source = {
        'tier': entry['tier'],
        'name': entry['source'],
        'quality': entry['quality'],
        'source_type': entry['source_type'],
        'source_name': entry['source_title'],
        'author': entry['author'],
        'year': entry['year'],
    }

    # Add optional fields
    if 'translator' in entry and entry['translator']:
        source['translator'] = entry['translator']
    if 'sentence_type' in entry:
        source['sentence_type'] = entry['sentence_type']
    if 'version' in entry:
        source['version'] = entry['version']

    return text, source


def format_wikipedia_entry(entry: dict, config: dict) -> dict:
    """Format Wikipedia extracted sentence to unified corpus format."""
    text = entry['text']

    source = {
        'tier': config['tier'],
        'name': 'wikipedia',
        'quality': config['quality'],
        'source_type': 'encyclopedia',
        'source_name': entry.get('article_title', 'Unknown Article'),
        'article_id': entry.get('article_id'),
        'section': entry.get('section'),
        'section_level': entry.get('section_level'),
    }

    return text, source


def format_gutenberg_entry(entry: dict, config: dict) -> dict:
    """Format Gutenberg book sentence to unified corpus format."""
    text = entry['text']

    source = {
        'tier': config['tier'],
        'name': 'gutenberg',
        'quality': config['quality'],
        'source_type': 'literary',
        'source_name': entry.get('source_name', entry.get('source', 'Unknown Book')),
        'chapter': entry.get('chapter'),
        'chapter_number': entry.get('chapter_number'),
    }

    return text, source


def format_entry(entry: dict, config: dict, source_type: str) -> Optional[tuple]:
    """Format entry based on source type."""
    try:
        if source_type.startswith('tier0'):
            return format_tier0_entry(entry, config)
        elif source_type == 'wikipedia':
            return format_wikipedia_entry(entry, config)
        elif source_type == 'gutenberg':
            return format_gutenberg_entry(entry, config)
        else:
            logger.warning(f"Unknown source type: {source_type}")
            return None
    except KeyError as e:
        logger.warning(f"Missing required field in entry: {e}")
        return None


def parse_and_build_entry(text: str, source: dict) -> dict:
    """Parse sentence to AST and build unified corpus entry."""
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
    resume: bool
) -> tuple:
    """Process all files for a source type."""
    from glob import glob

    path_pattern = config['path']
    files = sorted(glob(path_pattern))

    if not files:
        logger.warning(f"No files found for {source_type}: {path_pattern}")
        return 0, 0

    total_processed = 0
    total_failed = 0

    with open(output_file, 'a', encoding='utf-8') as out_f:
        for file_path in files:
            file_name = Path(file_path).name
            file_key = f"{source_type}:{file_name}"

            # Skip if already processed (when resuming)
            if resume and file_key in completed_files:
                count = completed_files[file_key]
                logger.info(f"  Skipping {file_name} (already processed: {count} sentences)")
                total_processed += count
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

                        # Parse and build corpus entry
                        corpus_entry = parse_and_build_entry(text, source)

                        # Track low parse rate
                        if corpus_entry['parse_rate'] < 0.3:
                            file_failed += 1

                        # Write to corpus
                        out_f.write(json.dumps(corpus_entry, ensure_ascii=False) + '\n')
                        file_processed += 1

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

    return total_processed, total_failed


def build_corpus(
    output_file: Path,
    resume: bool = False,
    fresh: bool = False
):
    """Build unified corpus from all extracted sources."""
    logger.info("=" * 80)
    logger.info("BUILD UNIFIED CORPUS")
    logger.info("=" * 80)
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

    for source_type, config in SOURCE_CONFIGS.items():
        logger.info("")
        logger.info(f"Processing {source_type} (Tier {config['tier']})...")
        logger.info(f"  Pattern: {config['path']}")

        processed, failed = process_source_files(
            source_type,
            config,
            output_file,
            completed_files,
            resume
        )

        total_stats[f'tier{config["tier"]}_processed'] += processed
        total_stats[f'tier{config["tier"]}_failed'] += failed
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

    # Count lines in final corpus
    final_count = sum(1 for _ in open(output_file, 'r', encoding='utf-8'))

    logger.info(f"Total sentences in corpus: {final_count:,}")
    logger.info("")
    logger.info("Breakdown by tier:")
    for key in sorted(total_stats.keys()):
        if key.startswith('tier') and key.endswith('_processed'):
            tier = key.split('_')[0]
            processed = total_stats[key]
            failed = total_stats.get(f"{tier}_failed", 0)
            pct = (processed / total_stats['total_processed'] * 100) if total_stats['total_processed'] > 0 else 0
            logger.info(f"  {tier}: {processed:,} sentences ({pct:.1f}%)")

    logger.info("")
    logger.info(f"Output: {output_file}")
    logger.info(f"Size: {output_file.stat().st_size / (1024**3):.2f} GB")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Rebuild Kuzu index:")
    logger.info("     ./scripts/index_kuzu.sh --fresh")
    logger.info("  2. Train M1 with tier priority:")
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

    args = parser_arg.parse_args()

    # Validate flags
    if args.resume and args.fresh:
        logger.error("Cannot use both --resume and --fresh flags")
        return 1

    try:
        build_corpus(
            args.output,
            resume=args.resume,
            fresh=args.fresh
        )
        return 0
    except Exception as e:
        logger.error(f"Build failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit(main())
