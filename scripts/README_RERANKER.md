# Reranker Testing Scripts

Quick reference for testing the trained reranker model.

## Scripts

### 1. `./scripts/test_reranker.sh`

**Purpose**: Unit test the reranker on synthetic query/document pairs

**Usage**:
```bash
./scripts/test_reranker.sh
```

**What it tests**:
- 3 hardcoded query/document pairs
- Verifies relevant docs score higher than irrelevant docs
- Fast test (~2 seconds)

**Example output**:
```
Test Case 1: "Kio estas hundo?"
  Relevant doc:   0.9927 ✓
  Irrelevant doc: 0.3875 ✓
  ✓ Correct ranking
```

---

### 2. `./scripts/demo_reranked_rag.sh`

**Purpose**: Test reranker integration with full RAG pipeline

**Usage**:
```bash
# Run default test queries
./scripts/demo_reranked_rag.sh

# Test specific query
./scripts/demo_reranked_rag.sh --query "Kio estas hundo?"

# Get more results
./scripts/demo_reranked_rag.sh --query "Kie vivas la homoj?" --top-k 10

# Use different index
./scripts/demo_reranked_rag.sh --index-dir data/indexes/my_index

# Help
./scripts/demo_reranked_rag.sh --help
```

**What it shows**:
- Results WITHOUT reranking (structural scoring only)
- Results WITH reranking (structural + neural)
- Demonstrates ranking improvements

**Default test queries**:
1. "Kio estas hundo?" (What is a dog?)
2. "Kie vivas la homoj?" (Where do people live?)
3. "Kiu inventis la telefon?" (Who invented the telephone?)

---

## Model Files

- **Compositional embedding**: `models/root_embeddings/best_model.pt`
- **Reranker**: `models/reranker/best_model.pt`
- **Training log**: `logs/training/reranker_YYYYMMDD_HHMMSS.log`

---

## Architecture

```
Query → Parser → AST
                  ↓
         ┌────────────────────┐
         │  Stage 1: Retrieval│  (Structural scoring, fast)
         │  - Root matching   │
         │  - Get 50 candidates│
         └────────────────────┘
                  ↓
         ┌────────────────────┐
         │  Stage 2: Reranking│  (Neural scoring, precise)
         │  - AST embeddings  │
         │  - Relevance score │
         └────────────────────┘
                  ↓
              Top 10 Results
```

---

## Score Combination

Current weights (in `demo_reranked_rag.py`):
```python
combined_score = 0.3 * structural_score + 0.7 * reranker_score
```

**Tuning tips**:
- More weight on structural (e.g., 0.5/0.5): Faster, respects root matching more
- More weight on reranker (e.g., 0.2/0.8): More semantic, may retrieve unexpected results
- Pure reranker (0.0/1.0): Maximum neural scoring

---

## Performance

**Reranker test** (~2 seconds):
- Loads model: ~1s
- Scores 6 examples: ~1s

**RAG demo** (~5-10 seconds):
- Loads retriever + reranker: ~3s
- Structural retrieval: ~1s
- Reranking 20 candidates: ~1s per query

---

## Monitoring Training

While testing, monitor training progress:
```bash
tail -f logs/training/reranker_YYYYMMDD_HHMMSS.log
```

The reranker trains for 20 epochs. You can test with any checkpoint - the model updates in place at `models/reranker/best_model.pt`.

---

## Next Steps

1. **Benchmark**: Run on evaluation set to measure ranking quality
2. **Tune weights**: Adjust structural vs neural score combination
3. **A/B test**: Compare retrieval quality with/without reranking
4. **Production**: Integrate `RerankedRetriever` into main RAG pipeline

---

## Troubleshooting

**"ModuleNotFoundError: No module named 'klareco'"**
- Scripts should auto-set PYTHONPATH, but if not:
  ```bash
  export PYTHONPATH=/home/marc/Projects/klareco:$PYTHONPATH
  ```

**"Index not found"**
- Ensure Kuzu index exists: `ls -lh data/indexes/kuzu_index/`
- Build index if missing: `./scripts/index_kuzu.sh`

**"Model not found"**
- Check reranker exists: `ls -lh models/reranker/best_model.pt`
- Training must complete at least 1 validation cycle to save checkpoint
