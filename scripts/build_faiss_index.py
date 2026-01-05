#!/usr/bin/env python3
"""
Build FAISS index from existing embeddings.

Usage:
    python scripts/build_faiss_index.py
    python scripts/build_faiss_index.py --embeddings data/indexes/merged/embeddings.npy
"""

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import faiss
except ImportError:
    print("ERROR: FAISS not installed. Install with: pip install faiss-cpu")
    sys.exit(1)


def build_faiss_index(embeddings_path: Path, output_path: Path):
    """Build FAISS index from embeddings array."""

    print(f"Loading embeddings from {embeddings_path}...")
    if not embeddings_path.exists():
        print(f"ERROR: Embeddings file not found: {embeddings_path}")
        sys.exit(1)

    embeddings = np.load(embeddings_path)
    print(f"  Loaded {embeddings.shape[0]:,} embeddings of dimension {embeddings.shape[1]}")

    # Build FAISS index
    print("Building FAISS index...")
    print("  Normalizing embeddings (L2 norm for cosine similarity)...")
    faiss.normalize_L2(embeddings)

    print(f"  Creating IndexFlatIP (inner product) with dim={embeddings.shape[1]}...")
    index = faiss.IndexFlatIP(embeddings.shape[1])

    print(f"  Adding {len(embeddings):,} vectors to index...")
    index.add(embeddings)
    print(f"  Index built with {index.ntotal:,} vectors")

    # Save index
    print(f"Saving FAISS index to {output_path}...")
    faiss.write_index(index, str(output_path))

    # Verify
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Saved: {output_path} ({file_size_mb:.1f} MB)")
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index from embeddings")
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path("data/indexes/merged/embeddings.npy"),
        help="Path to embeddings .npy file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to output FAISS index (default: same dir as embeddings, faiss_index.bin)"
    )

    args = parser.parse_args()

    # Default output path: same directory as embeddings
    if args.output is None:
        args.output = args.embeddings.parent / "faiss_index.bin"

    build_faiss_index(args.embeddings, args.output)


if __name__ == "__main__":
    main()
