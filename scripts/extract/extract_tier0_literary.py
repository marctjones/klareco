#!/usr/bin/env python3
"""
Extract sentences from Tier 0 literary works (Gutenberg books, Ekzercaro, Krestomatio).

Processes:
- Alice in Wonderland (Lewis Carroll)
- Fabeloj de Andersen (H.C. Andersen)
- Ekzercaro (Zamenhof exercises)
- Krestomatio (Zamenhof literary collection)

Output: JSONL with sentence + metadata (source, author, tier, quality)
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Source metadata
LITERARY_WORKS = {
    'alice': {
        'title': 'La Aventuroj de Alicio en Mirlando',
        'author': 'Lewis Carroll',
        'translator': 'E.L. Kearney',
        'year': 1910,
        'tier': 1,
        'quality': 'born_digital',
        'source_type': 'literary_translation',
    },
    'andersen': {
        'title': 'Fabeloj de Andersen',
        'author': 'Hans Christian Andersen',
        'translator': 'F. Skeel-Giörling',
        'year': 1907,
        'tier': 1,
        'quality': 'born_digital',
        'source_type': 'literary_translation',
    },
    'ekzercaro': {
        'title': 'Ekzercaro',
        'author': 'L.L. Zamenhof',
        'translator': None,
        'year': 1894,
        'tier': 0,
        'quality': 'authoritative',
        'source_type': 'exercises',
    },
    'krestomatio': {
        'title': 'Fundamenta Krestomatio',
        'author': 'L.L. Zamenhof',
        'translator': None,
        'year': 1903,
        'tier': 0,
        'quality': 'authoritative',
        'source_type': 'literary_collection',
    },
}


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using Esperanto sentence boundaries.

    Handles:
    - Standard sentence endings (. ! ?)
    - Abbreviations (D-ro, S-ro, k.t.p.)
    - Quotes and parentheses
    - Dialogue markers
    """
    # Replace common abbreviations with placeholders to avoid false splits
    abbreviations = {
        'D-ro': '<ABBR_DRO>',
        'S-ro': '<ABBR_SRO>',
        'S-ino': '<ABBR_SINO>',
        'k.t.p.': '<ABBR_KTP>',
        'k.a.': '<ABBR_KA>',
        'n-ro': '<ABBR_NRO>',
        'p-ro': '<ABBR_PRO>',
    }

    temp_text = text
    for abbr, placeholder in abbreviations.items():
        temp_text = temp_text.replace(abbr, placeholder)

    # Split on sentence boundaries
    # Match: . ! ? followed by whitespace and capital letter or quote
    sentences = re.split(r'([.!?])\s+(?=[A-ZĈĜĤĴŜŬ"\'\(])', temp_text)

    # Reconstruct sentences (split creates alternating text/punctuation)
    reconstructed = []
    for i in range(0, len(sentences) - 1, 2):
        sentence = sentences[i] + sentences[i + 1]
        reconstructed.append(sentence.strip())

    # Add last sentence if no punctuation at end
    if len(sentences) % 2 == 1:
        reconstructed.append(sentences[-1].strip())

    # Restore abbreviations
    final_sentences = []
    for sent in reconstructed:
        for placeholder, abbr in {v: k for k, v in abbreviations.items()}.items():
            sent = sent.replace(placeholder, abbr)

        # Only keep sentences with actual content
        if len(sent) > 5 and any(c.isalpha() for c in sent):
            final_sentences.append(sent)

    return final_sentences


def extract_from_text(
    text_path: Path,
    work_id: str,
    output_path: Path
) -> int:
    """Extract sentences from a single literary work."""

    if work_id not in LITERARY_WORKS:
        logger.error(f"Unknown work ID: {work_id}")
        return 0

    metadata = LITERARY_WORKS[work_id]
    logger.info(f"Extracting from: {metadata['title']}")

    # Read text
    try:
        text = text_path.read_text(encoding='utf-8')
    except Exception as e:
        logger.error(f"Failed to read {text_path}: {e}")
        return 0

    # Split into sentences
    sentences = split_into_sentences(text)
    logger.info(f"Found {len(sentences)} sentences")

    # Write JSONL
    output_path.parent.mkdir(parents=True, exist_ok=True)

    extracted_count = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, sentence in enumerate(sentences, 1):
            entry = {
                'sentence': sentence,
                'source': work_id,
                'source_title': metadata['title'],
                'author': metadata['author'],
                'translator': metadata['translator'],
                'year': metadata['year'],
                'tier': metadata['tier'],
                'quality': metadata['quality'],
                'source_type': metadata['source_type'],
                'sentence_id': i,
            }

            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            extracted_count += 1

    logger.info(f"Extracted {extracted_count} sentences to {output_path}")
    return extracted_count


def load_checkpoint(checkpoint_path: Path) -> Dict[str, int]:
    """Load checkpoint of already-processed works."""
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
        logger.debug(f"Checkpoint saved: {len(completed)} works completed")
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")
        if temp_path.exists():
            temp_path.unlink()


def extract_all_literary(
    input_dir: Path,
    output_dir: Path,
    resume: bool = False,
    fresh: bool = False
) -> Dict[str, int]:
    """Extract sentences from all Tier 0 literary works."""

    logger.info("=" * 70)
    logger.info("EXTRACT TIER 0 LITERARY WORKS")
    logger.info("=" * 70)

    # Checkpoint management
    checkpoint_path = output_dir / '.checkpoint.json'
    completed = {}

    if fresh and checkpoint_path.exists():
        logger.info("Fresh start requested - ignoring checkpoint")
        checkpoint_path.unlink()
    elif resume:
        completed = load_checkpoint(checkpoint_path)
        if completed:
            logger.info(f"Resuming from checkpoint: {len(completed)} works already completed")
        else:
            logger.info("No checkpoint found - starting from beginning")

    results = {}

    # Map work IDs to input files
    # Note: Using cleaned files from data/cleaned/eo/tier0/
    input_files = {
        'alice': input_dir / 'alice.txt',
        'andersen': input_dir / 'andersen.txt',
        'ekzercaro': input_dir / 'ekzercaro.txt',
        'krestomatio': input_dir / 'krestomatio.txt',
    }

    for work_id, input_path in input_files.items():
        # Skip if already completed (when resuming)
        if resume and work_id in completed:
            logger.info(f"Skipping {work_id} (already completed: {completed[work_id]} sentences)")
            results[work_id] = completed[work_id]
            continue

        if not input_path.exists():
            logger.warning(f"Input file not found: {input_path}")
            results[work_id] = 0
            continue

        output_path = output_dir / f"{work_id}_sentences.jsonl"
        count = extract_from_text(input_path, work_id, output_path)
        results[work_id] = count

        # Save checkpoint after each work
        completed[work_id] = count
        save_checkpoint(checkpoint_path, completed)

    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("EXTRACTION COMPLETE")
    logger.info("=" * 70)
    total = sum(results.values())
    logger.info(f"Total sentences extracted: {total:,}")

    for work_id, count in results.items():
        status = "✓" if count > 0 else "✗"
        logger.info(f"  {status} {work_id}: {count:,} sentences")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Extract sentences from Tier 0 literary works'
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        default=Path('data/cleaned/eo/tier0'),
        help='Input directory with cleaned texts'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/extracted/eo/tier0/literary'),
        help='Output directory for JSONL files'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from checkpoint (skip already-processed works)'
    )
    parser.add_argument(
        '--fresh',
        action='store_true',
        help='Start fresh, ignore checkpoint'
    )

    args = parser.parse_args()

    # Validate flags
    if args.resume and args.fresh:
        logger.error("Cannot use both --resume and --fresh flags")
        return 1

    try:
        results = extract_all_literary(
            args.input_dir,
            args.output_dir,
            resume=args.resume,
            fresh=args.fresh
        )

        # Exit with error if nothing extracted
        if sum(results.values()) == 0:
            logger.error("No sentences extracted from any source")
            return 1

        return 0

    except Exception as e:
        logger.error(f"Extraction failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit(main())
