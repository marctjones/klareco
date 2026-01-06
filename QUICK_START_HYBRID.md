# Quick Start: Hybrid Embeddings

## Prerequisites Check

```bash
# Verify models exist
ls -lh models/root_embeddings/best_model.pt
ls -lh models/topical_embeddings/best_model.pt
ls -lh data/corpus/unified_corpus.jsonl
```

## Option 1: Quick Test (1K sentences, ~1 minute)

```bash
# Test with small dataset first
./scripts/build_hybrid_indexes.sh --limit 1000

# Verify 128d embeddings
python -c "
import json
with open('data/indexes/slot_hybrid/slot_index.jsonl') as f:
    entry = json.loads(f.readline())
    print(f'✓ Embedding dimension: {len(entry[\"full_embedding\"])}d (expected 128d)')
"

# Test retrieval
python scripts/demo_slot_retrieval.py --index data/indexes/slot_hybrid -i
```

**Try these test queries:**
- `Kiu estis Napoleono?` (Who was Napoleon? - tests proper noun handling)
- `Kio estas algoritmo?` (What is an algorithm? - tests technical terms)
- `Kie loĝas tigroj?` (Where do tigers live? - tests content words)

## Option 2: Full Build (~8-11 hours)

```bash
# In tmux/screen session:
tmux new -s hybrid_build

# Run build script
./scripts/build_hybrid_indexes.sh

# Detach: Ctrl+b, d
# Reattach: tmux attach -t hybrid_build
```

**Monitor progress:**
```bash
# Watch log output
tail -f data/indexes/slot_hybrid/checkpoint.json

# Check index size
du -sh data/indexes/slot_hybrid/
```

## Option 3: Manual Step-by-Step

```bash
# Step 1: Build slot index with hybrid embeddings
python scripts/index_slot_based.py \
    --corpus data/corpus/unified_corpus.jsonl \
    --output data/indexes/slot_hybrid \
    --root-model models/root_embeddings/best_model.pt \
    --affix-model models/affix_transforms_v2/best_model.pt \
    --topical-model models/topical_embeddings/best_model.pt \
    --hybrid \
    --resume

# Step 2: Build HNSW index
./scripts/build_hnsw_index.sh data/indexes/slot_hybrid

# Step 3: Test retrieval (FAISS, ScaNN auto-build on first use)
python scripts/demo_slot_retrieval.py --index data/indexes/slot_hybrid -i
```

## Benchmarking

```bash
# Run benchmark suite (~30 minutes)
./scripts/benchmark_qa_all.sh

# Results saved to:
# - benchmark_results/combined_YYYYMMDD_HHMMSS.json
# - benchmark_results/report_YYYYMMDD_HHMMSS.html
```

## Expected Results

**Small test (1K sentences):**
- Indexing: ~1 second
- HNSW build: ~2 seconds
- Success rate: 100%

**Full corpus (4.2M sentences):**
- Slot indexing: ~6-8 hours
- HNSW build: ~2-3 hours
- Success rate: >95%

## Troubleshooting

**Error: "Topical model not found"**
```bash
# Train topical embeddings first
./scripts/train_topical_embeddings.sh
```

**Error: "Corpus not found"**
```bash
# Build corpus first
./scripts/parse_corpus.sh
```

**Want to restart from scratch?**
```bash
rm -rf data/indexes/slot_hybrid
./scripts/build_hybrid_indexes.sh
```

## Next Steps

1. ✅ Build hybrid indexes
2. ⏭️ Benchmark performance (Task #80)
3. ⏭️ Compare hybrid vs linguistic-only
4. ⏭️ Integrate into AST-aware retriever (Task #82)

## See Also

- **Full documentation**: `HYBRID_EMBEDDINGS_GUIDE.md`
- **Architecture details**: `CLAUDE.md`
- **Task list**: `idlergear task list`
