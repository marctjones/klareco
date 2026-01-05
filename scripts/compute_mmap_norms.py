#!/usr/bin/env python3
"""
Compute and save pre-computed norms for mmap slot embeddings.

This provides a 20% speedup by avoiding repeated norm computations during retrieval.

Usage:
    python scripts/compute_mmap_norms.py --index data/indexes/slot_full

    # Or use the shell wrapper:
    ./scripts/compute_mmap_norms.sh
"""

import argparse
import logging
import numpy as np
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


def compute_and_save_norms(index_path: Path, force: bool = False):
    """
    Compute L2 norms for each slot embedding and save to {slot}_norms.npy.

    Args:
        index_path: Path to slot index directory
        force: If True, recompute even if norms already exist
    """
    mmap_dir = index_path / "mmap"

    if not mmap_dir.exists():
        logger.error(f"Mmap directory not found: {mmap_dir}")
        logger.error("Run index_slot_based.py first to create mmap files")
        return False

    slots = ['SUBJ', 'VERB', 'OBJ']

    logger.info(f"Computing norms for slot embeddings in {mmap_dir}")
    logger.info("")

    for slot in slots:
        emb_file = mmap_dir / f"{slot}.npy"
        norm_file = mmap_dir / f"{slot}_norms.npy"

        if not emb_file.exists():
            logger.warning(f"Skipping {slot}: embedding file not found")
            continue

        if norm_file.exists() and not force:
            logger.info(f"✓ {slot}: norms already exist (use --force to recompute)")
            continue

        logger.info(f"Processing {slot}...")

        # Load embeddings (memory-mapped)
        embeddings = np.load(emb_file, mmap_mode='r')
        num_docs, dim = embeddings.shape

        logger.info(f"  Loading {num_docs:,} vectors ({dim}d)...")

        # Compute norms in batches to avoid loading entire array
        batch_size = 100000
        norms = np.zeros(num_docs, dtype=np.float32)

        for start_idx in range(0, num_docs, batch_size):
            end_idx = min(start_idx + batch_size, num_docs)
            batch = embeddings[start_idx:end_idx]

            # Compute L2 norm for each vector
            norms[start_idx:end_idx] = np.linalg.norm(batch, axis=1)

            if (end_idx % 500000) == 0:
                logger.info(f"    Processed {end_idx:,} / {num_docs:,} vectors...")

        logger.info(f"  Computed {num_docs:,} norms")

        # Save norms
        logger.info(f"  Saving to {norm_file}...")
        np.save(norm_file, norms)

        # Verify
        file_size_mb = norm_file.stat().st_size / (1024 * 1024)
        logger.info(f"✓ {slot}: norms saved ({file_size_mb:.1f} MB)")
        logger.info("")

    logger.info("=" * 70)
    logger.info("Norm computation complete!")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Retrievers will now use pre-computed norms for 20% speedup:")
    logger.info("  - HNSWSlotRetriever")
    logger.info("  - HybridFAISSMmapRetriever")
    logger.info("  - ScaNNSlotRetriever")
    logger.info("  - MemoryMappedSlotRetriever")
    logger.info("")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Compute pre-computed norms for mmap slot embeddings (20% speedup)"
    )
    parser.add_argument(
        '--index',
        type=Path,
        default=Path('data/indexes/slot_full'),
        help='Path to slot index directory (default: data/indexes/slot_full)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Recompute norms even if they already exist'
    )

    args = parser.parse_args()

    success = compute_and_save_norms(args.index, force=args.force)

    if not success:
        exit(1)


if __name__ == '__main__':
    main()
