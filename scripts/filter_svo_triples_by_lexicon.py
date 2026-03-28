#!/usr/bin/env python3
"""
Filter SVO Triples by Lexicon Coverage

Keeps only triples where ALL three roles (subject, verb, object) have roots
in the ROOT_LEXICON. This ensures 100% semantic coverage for training.

Usage:
    python scripts/filter_svo_triples_by_lexicon.py \
        --input data/semantic_types/svo_triples_word_level_full.jsonl \
        --output data/semantic_types/svo_triples_word_level_filtered.jsonl
"""

import argparse
import jsonlines
from pathlib import Path
import sys
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.morphology.root_lexicon import ROOT_LEXICON

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def filter_triples(input_path: Path, output_path: Path):
    """
    Filter SVO triples to only include examples with full lexicon coverage.
    """
    logger.info(f"Loading triples from {input_path}")

    total = 0
    kept = 0
    skipped_subject = 0
    skipped_verb = 0
    skipped_object = 0

    filtered_triples = []

    with jsonlines.open(input_path) as reader:
        for triple in reader:
            total += 1

            # Extract roots
            subj_root = triple['subject']['root']
            verb_root = triple['verb']['root']
            obj_root = triple['object']['root']

            # Check if all roots are in lexicon
            subj_in = subj_root in ROOT_LEXICON
            verb_in = verb_root in ROOT_LEXICON
            obj_in = obj_root in ROOT_LEXICON

            if subj_in and verb_in and obj_in:
                filtered_triples.append(triple)
                kept += 1
            else:
                # Track what caused the skip
                if not subj_in:
                    skipped_subject += 1
                if not verb_in:
                    skipped_verb += 1
                if not obj_in:
                    skipped_object += 1

    logger.info(f"\nFiltering results:")
    logger.info(f"  Total triples: {total:,}")
    logger.info(f"  Kept (100% coverage): {kept:,} ({100*kept/total:.1f}%)")
    logger.info(f"  Skipped: {total - kept:,} ({100*(total-kept)/total:.1f}%)")
    logger.info(f"    - Missing subject root: {skipped_subject:,}")
    logger.info(f"    - Missing verb root: {skipped_verb:,}")
    logger.info(f"    - Missing object root: {skipped_object:,}")

    # Save filtered triples
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(output_path, mode='w') as writer:
        writer.write_all(filtered_triples)

    logger.info(f"\nFiltered triples saved to: {output_path}")
    logger.info(f"Lexicon size: {len(ROOT_LEXICON)} roots")

    # Sample some kept triples
    logger.info(f"\nSample of kept triples:")
    for i, triple in enumerate(filtered_triples[:5], 1):
        logger.info(f"  {i}. {triple['subject']['text']} → {triple['verb']['text']} → {triple['object']['text']}")


def main():
    parser = argparse.ArgumentParser(description='Filter SVO triples by lexicon coverage')
    parser.add_argument('--input', type=Path, required=True,
                       help='Input JSONL with word-level SVO triples')
    parser.add_argument('--output', type=Path, required=True,
                       help='Output JSONL with filtered triples')

    args = parser.parse_args()

    filter_triples(args.input, args.output)


if __name__ == '__main__':
    main()
