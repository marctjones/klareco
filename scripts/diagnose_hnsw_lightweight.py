#!/usr/bin/env python3
"""
Lightweight HNSW Index Diagnostic

This script diagnoses the HNSW ID mismatch issue WITHOUT loading the full retriever.
It only loads the minimum required components to avoid OOM.

Usage:
    python scripts/diagnose_hnsw_lightweight.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import hnswlib

# Configuration
INDEX_PATH = Path("data/indexes/slot_hybrid")
HNSW_FILE = INDEX_PATH / "hnsw" / "full_embeddings.hnsw"
JSONL_FILE = INDEX_PATH / "slot_index.jsonl"
OFFSETS_FILE = INDEX_PATH / "slot_index.offsets.npy"

# Global: load offsets once
_offsets = None

def get_offsets():
    global _offsets
    if _offsets is None:
        _offsets = np.load(OFFSETS_FILE)
    return _offsets

def load_doc_by_line(jsonl_file: Path, line_num: int) -> dict:
    """Load a single document by line number using offsets for O(1) access."""
    offsets = get_offsets()
    if line_num >= len(offsets):
        return None
    with open(jsonl_file) as f:
        f.seek(int(offsets[line_num]))
        line = f.readline()
        return json.loads(line)

def main():
    print("=" * 60)
    print("LIGHTWEIGHT HNSW DIAGNOSTIC")
    print("=" * 60)

    # Step 1: Get embedding dimension from first doc
    print("\n[1] Reading first document to get embedding dimension...")
    doc0 = load_doc_by_line(JSONL_FILE, 0)
    embedding_dim = len(doc0['full_embedding'])
    print(f"  Embedding dimension: {embedding_dim}")
    print(f"  Doc 0 text: {doc0['text'][:60]}...")

    # Step 2: Load HNSW index
    print("\n[2] Loading HNSW index...")
    index = hnswlib.Index(space='cosine', dim=embedding_dim)
    index.load_index(str(HNSW_FILE))
    index.set_ef(50)  # Search parameter

    hnsw_count = index.get_current_count()
    print(f"  HNSW vectors: {hnsw_count:,}")

    # Step 3: Test self-retrieval for a few documents
    print("\n[3] Self-retrieval test (docs 0, 1, 100, 1000)...")

    test_doc_ids = [0, 1, 100, 1000]
    for doc_id in test_doc_ids:
        doc = load_doc_by_line(JSONL_FILE, doc_id)
        if doc is None:
            print(f"  Doc {doc_id}: NOT FOUND in JSONL")
            continue

        emb = np.array(doc['full_embedding'], dtype=np.float32).reshape(1, -1)
        labels, distances = index.knn_query(emb, k=5)

        top1_id = labels[0][0]
        top1_dist = distances[0][0]

        if top1_id == doc_id:
            print(f"  Doc {doc_id}: ✓ Self-retrieval OK (dist={top1_dist:.6f})")
        else:
            # Check if doc_id is in top 5
            if doc_id in labels[0]:
                rank = list(labels[0]).index(doc_id) + 1
                print(f"  Doc {doc_id}: ✗ Self at rank {rank}, top-1={top1_id} (dist={top1_dist:.6f})")
            else:
                print(f"  Doc {doc_id}: ✗ NOT in top-5, top-1={top1_id} (dist={top1_dist:.6f})")
                # Show what top-1 is
                top1_doc = load_doc_by_line(JSONL_FILE, int(top1_id))
                if top1_doc:
                    print(f"           Top-1 text: {top1_doc['text'][:50]}...")

    # Step 4: Check HNSW ID range
    print("\n[4] Checking HNSW ID assignment pattern...")

    # Query with doc 0 embedding and look at returned IDs
    doc0_emb = np.array(doc0['full_embedding'], dtype=np.float32).reshape(1, -1)
    labels, _ = index.knn_query(doc0_emb, k=100)

    ids = labels[0]
    print(f"  Min ID in top-100: {ids.min()}")
    print(f"  Max ID in top-100: {ids.max()}")
    print(f"  Top-5 IDs: {ids[:5]}")

    # Step 5: Count how many docs are near ID 0 vs far
    near_zero = sum(1 for x in ids if x < 10000)
    far_from_zero = sum(1 for x in ids if x > 4000000)
    print(f"  IDs < 10000: {near_zero}")
    print(f"  IDs > 4M: {far_from_zero}")

    # Step 6: Check if there's an offset pattern
    print("\n[5] Checking for systematic ID offset...")

    # Try a document from the middle of the file
    mid_doc_id = 2000000
    mid_doc = load_doc_by_line(JSONL_FILE, mid_doc_id)
    if mid_doc:
        mid_emb = np.array(mid_doc['full_embedding'], dtype=np.float32).reshape(1, -1)
        labels, distances = index.knn_query(mid_emb, k=5)

        top1_id = labels[0][0]
        offset = top1_id - mid_doc_id
        print(f"  Doc {mid_doc_id} -> HNSW top-1 ID: {top1_id}")
        print(f"  Potential offset: {offset}")

        if top1_id == mid_doc_id:
            print("  ✓ Middle doc self-retrieves correctly")
        else:
            print(f"  ✗ Middle doc does NOT self-retrieve")

    # Step 7: Verify by checking a few more docs
    print("\n[6] Sampling 10 random docs for self-retrieval...")
    import random
    random.seed(42)
    sample_ids = random.sample(range(hnsw_count), 10)

    success_count = 0
    for doc_id in sample_ids:
        doc = load_doc_by_line(JSONL_FILE, doc_id)
        if doc is None:
            continue
        emb = np.array(doc['full_embedding'], dtype=np.float32).reshape(1, -1)
        labels, _ = index.knn_query(emb, k=1)
        if labels[0][0] == doc_id:
            success_count += 1

    print(f"  Self-retrieval success: {success_count}/10")

    if success_count < 5:
        print("\n" + "!" * 60)
        print("CRITICAL: HNSW index IDs do NOT match JSONL line numbers!")
        print("The index needs to be rebuilt with correct ID assignment.")
        print("!" * 60)

    print("\n" + "=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
