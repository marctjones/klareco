#!/usr/bin/env python3
"""
Build slot-based index from corpus.

Usage:
    # Test on small subset
    python scripts/index_slot_based.py \
        --corpus data/corpus/authoritative_corpus.jsonl \
        --output data/indexes/slot_test \
        --limit 1000

    # Full index
    python scripts/index_slot_based.py \
        --corpus data/corpus/unified_corpus.jsonl \
        --output data/indexes/slot_full

    # Resume from checkpoint
    python scripts/index_slot_based.py \
        --corpus data/corpus/unified_corpus.jsonl \
        --output data/indexes/slot_full \
        --resume
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.rag.slot_indexer import SlotBasedIndexer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Build slot-based index')
    parser.add_argument(
        '--corpus',
        type=Path,
        required=True,
        help='Path to corpus JSONL file'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output directory for index'
    )
    parser.add_argument(
        '--root-model',
        type=Path,
        default=Path('models/root_embeddings/best_model.pt'),
        help='Path to root embeddings model'
    )
    parser.add_argument(
        '--affix-model',
        type=Path,
        default=Path('models/affix_transforms_v2/best_model.pt'),
        help='Path to affix transforms model'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Batch size for processing'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of sentences (for testing)'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from checkpoint'
    )
    parser.add_argument(
        '--fresh',
        action='store_true',
        help='Start fresh (ignore checkpoint)'
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.corpus.exists():
        logger.error(f"Corpus not found: {args.corpus}")
        sys.exit(1)

    if not args.root_model.exists():
        logger.error(f"Root model not found: {args.root_model}")
        sys.exit(1)

    if not args.affix_model.exists():
        logger.error(f"Affix model not found: {args.affix_model}")
        sys.exit(1)

    # Create indexer
    logger.info("=" * 60)
    logger.info("Slot-Based Indexing")
    logger.info("=" * 60)
    logger.info(f"Corpus: {args.corpus}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Limit: {args.limit or 'None (full corpus)'}")

    indexer = SlotBasedIndexer(
        root_model_path=args.root_model,
        affix_model_path=args.affix_model,
        output_dir=args.output,
        batch_size=args.batch_size,
    )

    # Build index
    if args.limit:
        # For testing: create limited corpus file
        limited_corpus = args.output / f"corpus_limited_{args.limit}.jsonl"
        logger.info(f"Creating limited corpus: {limited_corpus}")

        with open(args.corpus) as f_in, open(limited_corpus, 'w') as f_out:
            for i, line in enumerate(f_in):
                if i >= args.limit:
                    break
                f_out.write(line)

        corpus_to_index = limited_corpus
    else:
        corpus_to_index = args.corpus

    stats = indexer.build_index(
        corpus_path=corpus_to_index,
        resume=not args.fresh,
    )

    logger.info("=" * 60)
    logger.info("Indexing Complete!")
    logger.info("=" * 60)
    logger.info(f"Success rate: {100 * stats['successful'] / stats['processed']:.1f}%")
    logger.info(f"Output: {args.output / 'slot_index.jsonl'}")


if __name__ == '__main__':
    main()
