# Hybrid Embeddings Integration Guide

This guide covers the integration of hybrid embeddings (128d = 64d linguistic + 64d topical) into the Klareco retrieval pipeline.

## Overview

**What Changed:**
- SlotBasedIndexer now supports hybrid embeddings (linguistic + topical)
- All retrievers (FAISS, HNSW, MemoryMapped, ScaNN) auto-detect 128d embeddings
- Backward compatible: defaults to 64d linguistic-only mode

**Key Benefits:**
- Better coverage: 77K topical roots vs 11K linguistic roots
- Improved proper noun handling: proper nouns use topical embeddings only
- Contextual understanding: combines linguistic structure + topical semantics

## Prerequisites

You need these trained models:
```bash
models/root_embeddings/best_model.pt        # Linguistic (11K roots, 64d)
models/topical_embeddings/best_model.pt     # Topical (77K roots, 64d)
models/affix_transforms_v2/best_model.pt    # Affix transforms (64d - skipped in hybrid mode)
```

Check if they exist:
```bash
ls -lh models/root_embeddings/best_model.pt
ls -lh models/topical_embeddings/best_model.pt
```

## Building Hybrid Indexes

### Step 1: Test with Small Dataset (Recommended)

First test with 1,000 sentences to verify everything works:

```bash
./scripts/build_hybrid_indexes.sh --limit 1000
```

This creates:
- `data/indexes/slot_hybrid/slot_index.jsonl` (128d embeddings)
- `data/indexes/slot_hybrid/hnsw/` (HNSW index)

**Expected output:**
```
Hybrid Indexes Built Successfully!

Created:
  ✓ Slot index:  data/indexes/slot_hybrid/slot_index.jsonl (128d hybrid embeddings)
  ✓ HNSW index:  data/indexes/slot_hybrid/hnsw/

Auto-built on first use:
  • FAISS index:  data/indexes/slot_hybrid/faiss/
  • ScaNN index:  data/indexes/slot_hybrid/scann/
  • Mmap arrays:  data/indexes/slot_hybrid/mmap/
```

**Verify embeddings are 128d:**
```bash
python -c "
import json
with open('data/indexes/slot_hybrid/slot_index.jsonl') as f:
    entry = json.loads(f.readline())
    print(f'Embedding dimension: {len(entry[\"full_embedding\"])}')
"
# Expected: Embedding dimension: 128
```

### Step 2: Build Full Indexes

Once testing succeeds, build the full indexes (this will take hours):

```bash
# Run in a separate terminal with tmux/screen:
./scripts/build_hybrid_indexes.sh
```

**Progress monitoring:**
The script will log progress every 1,000 documents:
```
Processed 1,000 | Success: 1,000 | Failed: 0
Processed 2,000 | Success: 2,000 | Failed: 0
...
```

**Checkpointing:**
The script resumes automatically if interrupted. To start fresh:
```bash
rm -rf data/indexes/slot_hybrid
./scripts/build_hybrid_indexes.sh
```

## Testing Retrieval

### Interactive Testing

```bash
python scripts/demo_slot_retrieval.py --index data/indexes/slot_hybrid -i
```

Try queries that benefit from hybrid embeddings:
- **Proper nouns**: "Kiu estis Napoleono?" (Who was Napoleon?)
- **Technical terms**: "Kio estas algoritmo?" (What is an algorithm?)
- **Mixed content**: "Kie loĝas tigroj?" (Where do tigers live?)

### Compare Hybrid vs Linguistic-Only

```bash
# Test with linguistic-only (64d)
python scripts/demo_slot_retrieval.py --index data/indexes/slot_full -i

# Test with hybrid (128d)
python scripts/demo_slot_retrieval.py --index data/indexes/slot_hybrid -i
```

Look for improvements in:
- Proper noun queries (Paris, Napoleon, Esperanto, etc.)
- Technical/domain-specific terms
- Cross-lingual concepts

## Benchmarking

### Run Full Benchmark Suite

```bash
# In a separate terminal (takes ~30 minutes):
./scripts/benchmark_qa_all.sh
```

This benchmarks all retrievers (FAISS, HNSW, MemoryMapped, ScaNN) and generates:
- `benchmark_results/combined_YYYYMMDD_HHMMSS.json`
- `benchmark_results/report_YYYYMMDD_HHMMSS.html`

### Compare Hybrid vs Linguistic-Only

After running benchmarks for both index types, compare:

```python
import json

# Load results
with open('benchmark_results/linguistic_only.json') as f:
    ling_results = json.load(f)

with open('benchmark_results/hybrid.json') as f:
    hyb_results = json.load(f)

# Compare accuracy
print(f"Linguistic-only accuracy: {ling_results['accuracy']:.2%}")
print(f"Hybrid accuracy:          {hyb_results['accuracy']:.2%}")
print(f"Improvement:              +{(hyb_results['accuracy'] - ling_results['accuracy']):.2%}")
```

## Technical Details

### Embedding Architecture

**Linguistic Embeddings (64d):**
- 11,121 roots from Fundamento + corpus analysis
- Captures morphological relationships
- Good for grammatical patterns

**Topical Embeddings (64d):**
- 77,236 roots from Wikipedia + corpus
- Captures semantic/topical relationships
- Includes proper nouns and technical terms

**Hybrid Embeddings (128d):**
- Concatenates linguistic (64d) + topical (64d)
- Smart padding when root exists in only one vocabulary
- Auto-detects root type (content word vs proper noun)

### How Hybrid Mode Works

```python
# In SlotBasedIndexer
if use_hybrid and topical_model_path:
    # Load hybrid embeddings
    hybrid_model = HybridEmbeddings.from_checkpoints(
        linguistic_checkpoint=root_model_path,
        topical_checkpoint=topical_model_path,
        pad_missing=True,      # Zero-pad if root missing in one vocab
        default_mode='hybrid'  # Use 128d output
    )
    embedding_dim = 128
else:
    # Legacy mode: linguistic-only (64d)
    root_emb = load_root_model(root_model_path)
    embedding_dim = 64
```

### Backward Compatibility

Existing indexes continue to work:
- Default mode is linguistic-only (64d)
- Must explicitly use `--hybrid` flag for 128d embeddings
- Retrievers auto-detect dimension from data

### Known Limitations

1. **Affix transforms skipped in hybrid mode**
   - Current affix transforms are 64d
   - Need to train 128d versions (Task #83)
   - Temporarily skipped for hybrid embeddings

2. **Larger index size**
   - 128d embeddings are 2x larger than 64d
   - Index files will be ~2x bigger
   - Trade-off for better coverage and accuracy

## Directory Structure

```
data/indexes/
├── slot_full/              # Linguistic-only (64d) - existing
│   ├── slot_index.jsonl
│   ├── faiss/
│   ├── hnsw/
│   └── mmap/
└── slot_hybrid/            # Hybrid (128d) - new
    ├── slot_index.jsonl    # 128d embeddings
    ├── faiss/              # Auto-built on first use
    ├── hnsw/               # Built by script
    ├── scann/              # Auto-built on first use
    └── mmap/               # Auto-built on first use
```

## Troubleshooting

### Error: "Topical model not found"

```bash
# Check if topical model exists
ls models/topical_embeddings/best_model.pt

# If missing, train it:
./scripts/train_topical_embeddings.sh
```

### Error: "Dimension mismatch"

This means you're using a 64d retriever with 128d embeddings or vice versa. Make sure:
- Build new indexes with `--hybrid` flag
- Point retriever to the hybrid index directory

### Slow Performance

For large indexes (4M+ documents):
- Use HNSW or FAISS for fast approximate search
- Avoid MemoryMapped for large-scale queries
- Consider using `prefilter_n` to limit candidates

## Next Steps

After building hybrid indexes:

1. **Benchmark** (Task #80):
   ```bash
   ./scripts/benchmark_qa_all.sh
   ```

2. **Compare Results**:
   - Analyze accuracy improvements
   - Identify query types that benefit most
   - Document findings

3. **Train Affix Transforms for 128d** (Task #83):
   - Adapt affix training pipeline
   - Train 128d low-rank transforms
   - Integrate into hybrid mode

4. **Integrate into AST-Aware Retriever** (Task #82):
   - Update AST-aware retriever to use hybrid embeddings
   - Test structural matching with improved semantics

## Scripts Reference

| Script | Purpose | Usage |
|--------|---------|-------|
| `build_hybrid_indexes.sh` | Build all hybrid indexes | `./scripts/build_hybrid_indexes.sh [--limit N]` |
| `index_slot_based.py` | Build slot index | `python scripts/index_slot_based.py --hybrid --topical-model models/topical_embeddings/best_model.pt ...` |
| `build_hnsw_index.sh` | Build HNSW index | `./scripts/build_hnsw_index.sh data/indexes/slot_hybrid` |
| `demo_slot_retrieval.py` | Interactive retrieval demo | `python scripts/demo_slot_retrieval.py --index data/indexes/slot_hybrid -i` |
| `benchmark_qa_all.sh` | Full benchmark suite | `./scripts/benchmark_qa_all.sh` |

## Files Modified

1. **klareco/rag/slot_indexer.py**
   - Added `topical_model_path` and `use_hybrid` parameters
   - Loads HybridEmbeddings in hybrid mode
   - Backward compatible with linguistic-only mode

2. **scripts/index_slot_based.py**
   - Added `--hybrid` and `--topical-model` arguments
   - Validates prerequisites
   - Logs embedding mode

3. **scripts/build_hybrid_indexes.sh** (NEW)
   - Master script for building all hybrid indexes
   - Includes prerequisite checks
   - Supports `--limit` for testing

## Performance Expectations

**Build time (full corpus ~4.2M sentences):**
- Slot index creation: ~6-8 hours
- HNSW index build: ~2-3 hours
- Total: ~8-11 hours

**Memory usage:**
- Peak during HNSW build: ~40 GB
- Runtime (retrieval): ~4-6 GB

**Index sizes (approximate):**
- slot_index.jsonl: ~8 GB (128d) vs ~4 GB (64d)
- HNSW index: ~6 GB (128d) vs ~3 GB (64d)
- FAISS index: ~5 GB (128d) vs ~2.5 GB (64d)

## Support

For issues or questions:
- Check existing tasks: `idlergear task list`
- Create new task: `idlergear task create "Issue with hybrid embeddings..."`
- Document findings: `idlergear note create "Discovered that..."`
