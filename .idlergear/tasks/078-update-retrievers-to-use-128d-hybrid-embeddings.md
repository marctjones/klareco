---
id: 78
title: Update retrievers to use 128d hybrid embeddings
state: closed
created: '2026-01-06T05:43:37.134938Z'
labels:
- enhancement
- embeddings
priority: high
---
## Task #78: Update Retrievers to Use 128d Hybrid Embeddings

**Status:** ✅ **COMPLETED**

### What Was Done

1. **Updated SlotBasedIndexer** (`klareco/rag/slot_indexer.py`):
   - Added `topical_model_path` and `use_hybrid` parameters
   - Loads HybridEmbeddings when `use_hybrid=True`
   - Uses 128d embeddings (64d linguistic + 64d topical)
   - Temporarily skips affix transforms in hybrid mode (need 128d versions - see Task #83)
   - Backward compatible: defaults to 64d linguistic-only mode

2. **Updated Indexing Script** (`scripts/index_slot_based.py`):
   - Added `--topical-model` argument
   - Added `--hybrid` flag
   - Validates that `--topical-model` is provided when using `--hybrid`
   - Logs embedding mode in output
   - Updated usage documentation

3. **Created Build Script** (`scripts/build_hybrid_indexes.sh`):
   - Master script to rebuild indexes with hybrid embeddings
   - Builds slot index with 128d embeddings
   - Builds HNSW index (FAISS, ScaNN, MemoryMapped auto-build on first use)
   - Includes prerequisite checks
   - Supports `--limit` flag for testing

4. **Created Documentation**:
   - `HYBRID_EMBEDDINGS_GUIDE.md` - Comprehensive guide
   - `QUICK_START_HYBRID.md` - Quick reference

### Testing Results

Tested with 100 sentences from unified corpus:
- ✅ Embeddings are correctly 128d (verified)
- ✅ 100% success rate
- ✅ HybridEmbeddings properly loaded (11K linguistic + 77K topical roots)

### Key Insight

**Retrievers don't need code changes!** They automatically detect embedding dimension from the data (`dim = embeddings.shape[1]`), so we only need to rebuild indexes with new 128d embeddings.

### Files Modified

1. `klareco/rag/slot_indexer.py` - Added hybrid embeddings support
2. `scripts/index_slot_based.py` - Added --hybrid flag
3. `scripts/build_hybrid_indexes.sh` - NEW master build script
4. `HYBRID_EMBEDDINGS_GUIDE.md` - NEW documentation
5. `QUICK_START_HYBRID.md` - NEW quick reference

### Scripts to Run (in separate terminal)

**Quick test (recommended first):**
```bash
./scripts/build_hybrid_indexes.sh --limit 1000
```

**Full build (8-11 hours):**
```bash
# In tmux/screen:
./scripts/build_hybrid_indexes.sh
```

**Verify embeddings:**
```bash
python -c "
import json
with open('data/indexes/slot_hybrid/slot_index.jsonl') as f:
    entry = json.loads(f.readline())
    print(f'Dimension: {len(entry[\"full_embedding\"])}d')
"
```

### Next Steps

- **Task #79**: Rebuild all indexes with hybrid embeddings (run script above)
- **Task #80**: Benchmark hybrid vs linguistic-only performance
- **Task #83**: Train affix transforms for 128d embeddings (future enhancement)

### Notes

- IdlerGear Note #79: Integration details and changes
- All changes maintain backward compatibility
- Default mode remains 64d linguistic-only unless `--hybrid` is explicitly used
