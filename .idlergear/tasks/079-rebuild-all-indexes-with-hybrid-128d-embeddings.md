---
id: 79
title: Rebuild all indexes with hybrid 128d embeddings
state: open
created: '2026-01-06T05:43:45.931938Z'
labels:
- enhancement
- embeddings
- indexing
priority: high
---
Regenerate FAISS, HNSW, ScaNN, and memory-mapped indexes using new hybrid embeddings.

**Prerequisites:**
- Task #78 complete (retrievers updated to use hybrid embeddings)
- Both models available:
  - `models/root_embeddings/best_model.pt` (linguistic)
  - `models/topical_embeddings/best_model.pt` (topical)

**Scripts to run:**
```bash
# Build all indexes with hybrid embeddings
./scripts/build_verified_indexes.sh --hybrid

# Or individually:
./scripts/build_faiss_index.sh --embedding-dim 128 --hybrid
./scripts/build_hnsw_index.sh --embedding-dim 128 --hybrid
./scripts/build_scann_index.sh --embedding-dim 128 --hybrid
./scripts/build_mmap_index.sh --embedding-dim 128 --hybrid
```

**Changes needed:**
1. Update build scripts to accept `--hybrid` flag
2. Pass both model paths to indexing code
3. Update index metadata to track embedding type and version
4. Validate index sizes (should be ~2x larger: 128d vs 64d)

**Success criteria:**
- All indexes rebuilt successfully
- Index search returns results
- Metadata correctly identifies hybrid embeddings
- No dimension mismatches at query time
