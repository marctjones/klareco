#!/usr/bin/env python3
"""
Merge filtered Tier 0 sentences into unified corpus.

This script:
1. Reads filtered Tier 0 JSONL files
2. Parses each sentence to generate AST
3. Formats with proper metadata (tier, source, weight)
4. Appends to unified corpus or creates new file

After running this, rebuild Kuzu index with:
    python scripts/index_kuzu.py --fresh
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco import parser

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Tier 0 weight (10x higher than standard corpus)
TIER0_WEIGHT = 10.0


def format_tier0_entry(sentence_data: dict) -> dict:
    """
    Format Tier 0 sentence into unified corpus structure.

    Input (Tier 0 JSONL):
    {
        "sentence": "...",
        "source": "pmeg",
        "source_title": "...",
        "author": "...",
        "year": 2024,
        "tier": 0,
        "quality": "authoritative",
        "source_type": "grammar_reference",
        "sentence_type": "explanation",  # Optional (grammar only)
        "sentence_id": 123
    }

    Output (Unified Corpus):
    {
        "text": "...",
        "source": {
            "tier": 0,
            "name": "pmeg",
            "source_name": "...",
            "weight": 10.0,
            "author": "...",
            "year": 2024,
            "quality": "authoritative",
            "source_type": "grammar_reference",
            "sentence_type": "explanation"  # If present
        },
        "ast": {...},
        "parse_rate": 0.85  # Copied from ast.parse_statistics.success_rate
    }
    """
    text = sentence_data['sentence']

    # Parse to AST
    try:
        ast = parser.parse(text)
    except Exception as e:
        logger.warning(f"Failed to parse: {text[:50]}... Error: {e}")
        # Create minimal AST for unparseable sentences
        ast = {
            'tipo': 'frazo',
            'parse_statistics': {'success_rate': 0.0},
            'error': str(e)
        }

    # Build source metadata
    source = {
        'tier': sentence_data['tier'],
        'name': sentence_data['source'],
        'source_name': sentence_data['source_title'],
        'weight': TIER0_WEIGHT,
        'author': sentence_data['author'],
        'year': sentence_data['year'],
        'quality': sentence_data['quality'],
        'source_type': sentence_data['source_type'],
    }

    # Add optional fields
    if 'translator' in sentence_data and sentence_data['translator']:
        source['translator'] = sentence_data['translator']

    if 'sentence_type' in sentence_data:  # Grammar works only
        source['sentence_type'] = sentence_data['sentence_type']

    if 'version' in sentence_data:  # PMEG only
        source['version'] = sentence_data['version']

    # Extract parse rate from AST statistics
    # This should match the structure used in regular corpus entries
    parse_rate = ast.get('parse_statistics', {}).get('success_rate', 0.0)

    return {
        'text': text,
        'source': source,
        'ast': ast,
        'parse_rate': parse_rate  # Add top-level parse_rate field for filtering
    }


def load_checkpoint(checkpoint_path: Path) -> Dict[str, int]:
    """Load checkpoint of already-processed files."""
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


def merge_tier0(
    tier0_dir: Path,
    output_file: Path,
    append: bool = False,
    resume: bool = False,
    fresh: bool = False
):
    """
    Merge Tier 0 filtered sentences into unified corpus.

    Args:
        tier0_dir: Directory with filtered Tier 0 JSONL files
        output_file: Output unified corpus file
        append: If True, append to existing file; if False, create new
        resume: If True, resume from checkpoint
        fresh: If True, ignore checkpoint and start over
    """
    logger.info("=" * 70)
    logger.info("MERGE TIER 0 INTO UNIFIED CORPUS")
    logger.info("=" * 70)
    logger.info("")

    # Checkpoint management
    checkpoint_path = output_file.parent / '.merge_checkpoint.json'
    completed_files = {}

    if fresh and checkpoint_path.exists():
        logger.info("Fresh start requested - ignoring checkpoint")
        checkpoint_path.unlink()
    elif resume:
        completed_files = load_checkpoint(checkpoint_path)
        if completed_files:
            logger.info(f"Resuming from checkpoint: {len(completed_files)} files already processed")
        else:
            logger.info("No checkpoint found - starting from beginning")

    # Find all filtered Tier 0 files
    literary_files = sorted((tier0_dir / 'literary').glob('*_sentences.jsonl'))
    grammar_files = sorted((tier0_dir / 'grammar').glob('*_sentences.jsonl'))

    all_files = list(literary_files) + list(grammar_files)

    if not all_files:
        logger.error(f"No Tier 0 JSONL files found in {tier0_dir}")
        return

    logger.info(f"Found {len(all_files)} Tier 0 files:")
    for f in all_files:
        status = "✓ (completed)" if f.name in completed_files else ""
        logger.info(f"  - {f.relative_to(tier0_dir)} {status}")
    logger.info("")

    # Check if appending to existing corpus
    existing_count = 0
    if append and output_file.exists():
        logger.info(f"Appending to existing corpus: {output_file}")
        with open(output_file, 'r', encoding='utf-8') as f:
            existing_count = sum(1 for _ in f)
        logger.info(f"Existing sentences: {existing_count:,}")
        logger.info("")
        mode = 'a'
    elif resume and output_file.exists():
        # When resuming, always append
        logger.info(f"Resuming to existing file: {output_file}")
        with open(output_file, 'r', encoding='utf-8') as f:
            existing_count = sum(1 for _ in f)
        logger.info(f"Existing sentences: {existing_count:,}")
        logger.info("")
        mode = 'a'
    else:
        if output_file.exists():
            logger.warning(f"Output file exists, will be overwritten: {output_file}")
        mode = 'w'

    # Process each file
    output_file.parent.mkdir(parents=True, exist_ok=True)

    total_added = 0
    total_parse_failures = 0

    with open(output_file, mode, encoding='utf-8') as out_f:
        for tier0_file in all_files:
            # Skip if already completed (when resuming)
            if resume and tier0_file.name in completed_files:
                sentences_count = completed_files[tier0_file.name]
                logger.info(f"Skipping {tier0_file.name} (already processed: {sentences_count} sentences)")
                total_added += sentences_count
                continue

            logger.info(f"Processing {tier0_file.name}...")

            file_added = 0
            file_failed = 0

            with open(tier0_file, 'r', encoding='utf-8') as in_f:
                for line in in_f:
                    sentence_data = json.loads(line)

                    # Format and parse
                    corpus_entry = format_tier0_entry(sentence_data)

                    # Check parse success
                    if corpus_entry['ast'].get('parse_statistics', {}).get('success_rate', 0) < 0.3:
                        file_failed += 1
                        total_parse_failures += 1

                    # Write to corpus
                    out_f.write(json.dumps(corpus_entry, ensure_ascii=False) + '\n')
                    file_added += 1
                    total_added += 1

            logger.info(f"  Added: {file_added:,} sentences")
            if file_failed > 0:
                logger.info(f"  Parse failures: {file_failed:,} ({file_failed/file_added*100:.1f}%)")
            logger.info("")

            # Save checkpoint after each file
            completed_files[tier0_file.name] = file_added
            save_checkpoint(checkpoint_path, completed_files)

    # Summary
    logger.info("=" * 70)
    logger.info("MERGE COMPLETE")
    logger.info("=" * 70)
    logger.info("")

    if append:
        logger.info(f"Existing sentences: {existing_count:,}")
    logger.info(f"Added Tier 0 sentences: {total_added:,}")
    logger.info(f"New total: {existing_count + total_added:,}")
    logger.info("")

    if total_parse_failures > 0:
        logger.info(f"Parse failures: {total_parse_failures:,} ({total_parse_failures/total_added*100:.1f}%)")
        logger.info("(These sentences still included with minimal AST)")
        logger.info("")

    logger.info(f"Output file: {output_file}")
    logger.info(f"File size: {output_file.stat().st_size / (1024**3):.2f} GB")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Rebuild Kuzu index:")
    logger.info("     python scripts/index_kuzu.py --fresh")
    logger.info("  2. Update README.md with new corpus statistics")
    logger.info("  3. Optionally retrain models with Tier 0 weights")


def main():
    parser_arg = argparse.ArgumentParser(
        description='Merge filtered Tier 0 sentences into unified corpus'
    )
    parser_arg.add_argument(
        '--tier0-dir',
        type=Path,
        default=Path('data/extracted/eo/tier0_filtered'),
        help='Directory with filtered Tier 0 JSONL files'
    )
    parser_arg.add_argument(
        '--output',
        type=Path,
        default=Path('data/enhanced_corpus/corpus_with_tier0.jsonl'),
        help='Output corpus file (default: new file, not overwriting existing)'
    )
    parser_arg.add_argument(
        '--append',
        action='store_true',
        help='Append to existing corpus file instead of creating new'
    )
    parser_arg.add_argument(
        '--resume',
        action='store_true',
        help='Resume from checkpoint (skip already-processed files)'
    )
    parser_arg.add_argument(
        '--fresh',
        action='store_true',
        help='Start fresh, ignore checkpoint'
    )

    args = parser_arg.parse_args()

    # Validate flags
    if args.resume and args.fresh:
        logger.error("Cannot use both --resume and --fresh flags")
        return 1

    try:
        merge_tier0(
            args.tier0_dir,
            args.output,
            append=args.append,
            resume=args.resume,
            fresh=args.fresh
        )
        return 0
    except Exception as e:
        logger.error(f"Merge failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit(main())
