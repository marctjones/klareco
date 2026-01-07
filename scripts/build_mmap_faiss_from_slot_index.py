#!/usr/bin/env python3
"""
Build mmap arrays and FAISS index from existing slot_index.jsonl.

MEMORY-SAFE: Processes documents in streaming fashion, writes to disk incrementally.
RESTARTABLE: Saves checkpoints every 100K documents.

This script extracts embeddings from the slot index and creates:
1. mmap/ directory with SUBJ.npy, VERB.npy, OBJ.npy, full.npy + norms
2. faiss/ directory with full_embeddings.index

Usage:
    python scripts/build_mmap_faiss_from_slot_index.py --index-dir data/indexes/slot_hybrid
    python scripts/build_mmap_faiss_from_slot_index.py --index-dir data/indexes/slot_hybrid --mmap-only
    python scripts/build_mmap_faiss_from_slot_index.py --index-dir data/indexes/slot_hybrid --faiss-only
    python scripts/build_mmap_faiss_from_slot_index.py --index-dir data/indexes/slot_hybrid --fresh
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

CHECKPOINT_INTERVAL = 100000  # Save checkpoint every 100K docs


def count_lines(file_path: Path) -> int:
    """Count lines in file efficiently."""
    count = 0
    with open(file_path, 'rb') as f:
        for _ in f:
            count += 1
    return count


def load_checkpoint(checkpoint_file: Path) -> dict:
    """Load checkpoint if exists."""
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            return json.load(f)
    return {'phase': 'mmap', 'processed': 0}


def save_checkpoint(checkpoint_file: Path, data: dict):
    """Save checkpoint atomically."""
    temp_file = checkpoint_file.with_suffix('.tmp')
    with open(temp_file, 'w') as f:
        json.dump(data, f)
    temp_file.rename(checkpoint_file)


def build_mmap_arrays(index_dir: Path, checkpoint_file: Path, fresh: bool = False):
    """Build mmap arrays from slot_index.jsonl with checkpointing."""
    slot_index_file = index_dir / "slot_index.jsonl"
    mmap_dir = index_dir / "mmap"

    if not slot_index_file.exists():
        logger.error(f"Slot index not found: {slot_index_file}")
        sys.exit(1)

    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_file)
    if fresh:
        checkpoint = {'phase': 'mmap', 'processed': 0}

    if checkpoint.get('phase') == 'mmap_done':
        logger.info("Mmap arrays already built (checkpoint shows complete)")
        return True

    start_idx = checkpoint.get('processed', 0)

    # Count documents
    logger.info("Counting documents...")
    num_docs = count_lines(slot_index_file)
    logger.info(f"  Found {num_docs:,} documents")

    # Detect embedding dimension from first document
    with open(slot_index_file) as f:
        first_doc = json.loads(f.readline())
        emb_dim = len(first_doc.get('full_embedding', []))
        logger.info(f"  Embedding dimension: {emb_dim}")

    if emb_dim == 0:
        logger.error("No embeddings found in slot index")
        sys.exit(1)

    # Create mmap directory
    mmap_dir.mkdir(parents=True, exist_ok=True)

    slots = ['SUBJ', 'VERB', 'OBJ', 'full']

    # Check if we're resuming
    if start_idx > 0:
        logger.info(f"Resuming from document {start_idx:,}")
        # Load existing arrays
        arrays = {}
        for slot in slots:
            arr_file = mmap_dir / f"{slot}.npy"
            if arr_file.exists():
                arrays[slot] = np.load(arr_file, mmap_mode='r+')
                logger.info(f"  Loaded existing {slot}.npy for append")
            else:
                logger.error(f"Resume requested but {arr_file} not found")
                sys.exit(1)
    else:
        # Pre-allocate arrays using memory-mapped files (disk-backed)
        logger.info("Pre-allocating memory-mapped arrays on disk...")
        arrays = {}
        for slot in slots:
            arr_file = mmap_dir / f"{slot}.npy"
            # Create empty memory-mapped array
            arr = np.lib.format.open_memmap(
                arr_file, mode='w+', dtype=np.float32, shape=(num_docs, emb_dim)
            )
            arrays[slot] = arr
            logger.info(f"  Created {arr_file} ({arr.nbytes / (1024**3):.2f} GB)")

    # Read slot_index.jsonl and extract embeddings
    logger.info("Extracting embeddings from slot index...")
    start_time = time.time()
    last_checkpoint_time = start_time

    with open(slot_index_file) as f:
        # Skip to resume point
        for _ in range(start_idx):
            f.readline()

        for i, line in enumerate(f, start=start_idx):
            doc = json.loads(line)

            # Extract full embedding
            full_emb = doc.get('full_embedding')
            if full_emb:
                arrays['full'][i] = full_emb

            # Extract slot embeddings
            slots_data = doc.get('slots', {})
            for slot in ['SUBJ', 'VERB', 'OBJ']:
                slot_emb = slots_data.get(slot)
                if slot_emb:
                    arrays[slot][i] = slot_emb

            # Progress and checkpoint
            processed = i + 1
            if processed % 50000 == 0:
                elapsed = time.time() - start_time
                docs_per_sec = (processed - start_idx) / elapsed if elapsed > 0 else 0
                remaining = num_docs - processed
                eta_sec = remaining / docs_per_sec if docs_per_sec > 0 else 0
                eta_min = eta_sec / 60

                logger.info(
                    f"  [{processed:,}/{num_docs:,}] "
                    f"{100*processed/num_docs:.1f}% | "
                    f"{docs_per_sec:.0f} docs/sec | "
                    f"ETA: {eta_min:.1f} min"
                )

            if processed % CHECKPOINT_INTERVAL == 0:
                # Flush arrays to disk
                for arr in arrays.values():
                    arr.flush()

                checkpoint['processed'] = processed
                save_checkpoint(checkpoint_file, checkpoint)
                logger.info(f"  💾 Checkpoint saved at {processed:,}")
                last_checkpoint_time = time.time()

    logger.info(f"  Processed {num_docs:,}/{num_docs:,} documents (100%)")

    # Flush final arrays
    for arr in arrays.values():
        arr.flush()

    # Compute and save norms
    logger.info("Computing norms...")
    for slot in slots:
        logger.info(f"  Computing norms for {slot}...")
        arr = arrays[slot]

        # Compute norms in chunks to avoid memory issues
        chunk_size = 100000
        norms = np.zeros(num_docs, dtype=np.float32)

        for chunk_start in range(0, num_docs, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_docs)
            chunk = np.array(arr[chunk_start:chunk_end])  # Load chunk to RAM
            chunk_norms = np.linalg.norm(chunk, axis=1).astype(np.float32)
            chunk_norms[chunk_norms == 0] = 1.0  # Avoid division by zero
            norms[chunk_start:chunk_end] = chunk_norms

        norm_file = mmap_dir / f"{slot}_norms.npy"
        np.save(norm_file, norms)
        logger.info(f"    Saved {norm_file} ({norms.nbytes / (1024**2):.1f} MB)")

    # Mark mmap phase complete
    checkpoint['phase'] = 'mmap_done'
    checkpoint['processed'] = num_docs
    save_checkpoint(checkpoint_file, checkpoint)

    logger.info(f"✓ Mmap arrays saved to {mmap_dir}")
    return True


def build_faiss_index(index_dir: Path, checkpoint_file: Path):
    """Build FAISS index from mmap full embeddings."""
    try:
        import faiss
    except ImportError:
        logger.error("FAISS not installed. Install with: pip install faiss-cpu")
        sys.exit(1)

    checkpoint = load_checkpoint(checkpoint_file)
    if checkpoint.get('phase') == 'faiss_done':
        logger.info("FAISS index already built (checkpoint shows complete)")
        return True

    mmap_dir = index_dir / "mmap"
    faiss_dir = index_dir / "faiss"

    full_emb_file = mmap_dir / "full.npy"
    if not full_emb_file.exists():
        logger.error(f"Full embeddings not found: {full_emb_file}")
        logger.error("Run with --mmap-only first, or without flags to build both")
        sys.exit(1)

    # Load embeddings as memory-mapped (read-only)
    logger.info(f"Loading embeddings from {full_emb_file} (memory-mapped)...")
    embeddings_mmap = np.load(full_emb_file, mmap_mode='r')
    num_docs, emb_dim = embeddings_mmap.shape
    logger.info(f"  Shape: {num_docs:,} x {emb_dim}")

    # Process in chunks to normalize without loading all into RAM
    logger.info("Normalizing embeddings in chunks...")
    chunk_size = 500000

    # Create temporary normalized array
    norm_file = faiss_dir.parent / "normalized_temp.npy"
    faiss_dir.mkdir(parents=True, exist_ok=True)

    normalized = np.lib.format.open_memmap(
        norm_file, mode='w+', dtype=np.float32, shape=(num_docs, emb_dim)
    )

    for chunk_start in range(0, num_docs, chunk_size):
        chunk_end = min(chunk_start + chunk_size, num_docs)
        chunk = np.array(embeddings_mmap[chunk_start:chunk_end], dtype=np.float32)
        faiss.normalize_L2(chunk)
        normalized[chunk_start:chunk_end] = chunk
        logger.info(f"  Normalized {chunk_end:,}/{num_docs:,}")

    normalized.flush()

    # Build FAISS index
    logger.info(f"Building FAISS IndexFlatIP with dim={emb_dim}...")
    index = faiss.IndexFlatIP(emb_dim)

    # Add vectors in chunks
    logger.info(f"Adding {num_docs:,} vectors to index in chunks...")
    for chunk_start in range(0, num_docs, chunk_size):
        chunk_end = min(chunk_start + chunk_size, num_docs)
        chunk = np.array(normalized[chunk_start:chunk_end], dtype=np.float32)
        index.add(chunk)
        logger.info(f"  Added {chunk_end:,}/{num_docs:,} vectors")

    logger.info(f"  Index built with {index.ntotal:,} vectors")

    # Save index
    output_file = faiss_dir / "full_embeddings.index"
    logger.info(f"Saving FAISS index to {output_file}...")
    faiss.write_index(index, str(output_file))

    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    logger.info(f"  Saved: {output_file} ({file_size_mb:.1f} MB)")

    # Clean up temp file
    if norm_file.exists():
        norm_file.unlink()
        logger.info("  Cleaned up temporary normalized file")

    # Mark faiss phase complete
    checkpoint['phase'] = 'faiss_done'
    save_checkpoint(checkpoint_file, checkpoint)

    logger.info(f"✓ FAISS index saved to {faiss_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Build mmap arrays and FAISS index from slot_index.jsonl"
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        required=True,
        help="Path to slot index directory (containing slot_index.jsonl)"
    )
    parser.add_argument(
        "--mmap-only",
        action="store_true",
        help="Only build mmap arrays, skip FAISS"
    )
    parser.add_argument(
        "--faiss-only",
        action="store_true",
        help="Only build FAISS index (requires mmap/full.npy to exist)"
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Start fresh, ignore checkpoint"
    )

    args = parser.parse_args()

    if args.mmap_only and args.faiss_only:
        logger.error("Cannot specify both --mmap-only and --faiss-only")
        sys.exit(1)

    checkpoint_file = args.index_dir / "build_indexes_checkpoint.json"

    if args.fresh and checkpoint_file.exists():
        checkpoint_file.unlink()
        logger.info("Removed existing checkpoint (--fresh)")

    if not args.faiss_only:
        build_mmap_arrays(args.index_dir, checkpoint_file, args.fresh)

    if not args.mmap_only:
        build_faiss_index(args.index_dir, checkpoint_file)

    logger.info("Done!")


if __name__ == "__main__":
    main()
