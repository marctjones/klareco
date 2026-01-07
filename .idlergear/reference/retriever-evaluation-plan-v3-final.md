---
id: 1
title: Retriever Evaluation Plan v3 - Final
created: '2026-01-06T21:34:37.418830Z'
updated: '2026-01-06T21:34:37.419730Z'
---
# Retriever Evaluation Plan v3 - Final

## Current Status

### Corpus: ✅ Ready
- **File**: `data/indexes/slot_hybrid/slot_index.jsonl`
- **Size**: 4.37M documents, 32GB
- **Embeddings**: 128d hybrid (64d linguistic + 64d topical)
- **Sources**: Fundamento, Wikipedia, books

### Benchmark Questions: ⚠️ Needs Refinement
- **File**: `data/benchmarks/datasets/qa_benchmark_v1.jsonl`
- **Total**: 50 questions
- **Require retrieval**: 17 questions
  - 11 from Fundamento
  - 6 from Wikipedia
- **Don't require retrieval**: 33 (grammar rules, reasoning, negative)

### Problem: Current Benchmark Conflates Tasks
The 50-question benchmark mixes:
1. **Retrieval** (can we find the right document?) - 17 questions
2. **Reasoning** (can we infer from grammar rules?) - 10 questions  
3. **Grammar** (does the system know Esperanto rules?) - 10 questions
4. **Definitions** (often answerable without retrieval) - 10 questions
5. **Negative** (unanswerable questions) - 10 questions

For retriever comparison, we only care about #1.

## Evaluation Strategy

### Phase 1: Retrieval-Only Benchmark (17 questions)

Use only the 17 questions marked `requires_retrieval: true`.

**Metrics:**
- **Recall@1**: Answer appears in top result
- **Recall@5**: Answer appears in top 5
- **Recall@10**: Answer appears in top 10
- **MRR**: Mean Reciprocal Rank
- **Latency**: ms per query
- **Memory**: Peak RSS during search

### Phase 2: Verify Corpus Coverage

Before running, verify each answer exists in corpus:
```bash
# For each question's acceptable_answers, grep corpus
grep -c "Zamenhof" data/corpus/unified_corpus.jsonl  # Should be > 0
```

**Known corpus content:**
- "Zamenhof" appears 10,198 times
- "fondis" appears 14,579 times  
- "naskiĝis" appears 55,307 times
- "1859" appears 2,694 times
- "Esperanto" appears 4,194,928 times

### Phase 3: Head-to-Head Comparison

**Retrievers to test (4 active):**
1. ASTAwareRetriever (+ HNSW prefilter) - Full AST analysis
2. HNSWSlotRetriever - Fastest
3. FAISSSlotRetriever - Popular baseline
4. HybridFAISSMmapRetriever - Best accuracy expected

**Index requirements:**
| Retriever | slot_index.jsonl | hnsw/ | mmap/ | faiss/ |
|-----------|-----------------|-------|-------|--------|
| ASTAwareRetriever | ✅ | ✅ | ❌ | ❌ |
| HNSWSlotRetriever | ✅ | ✅ | 🔄 Building | ❌ |
| FAISSSlotRetriever | ✅ | ❌ | ❌ | 🔄 Building |
| HybridFAISSMmapRetriever | ✅ | ❌ | 🔄 Building | 🔄 Building |

Run `./scripts/build_hybrid_mmap_faiss.sh` to build missing indexes.

## Evaluation Script Design

```bash
./scripts/evaluate_retrievers.sh
```

**Features:**
- Tests all 4 active retrievers
- Uses only 17 retrieval-requiring questions
- Saves detailed results for analysis
- Restartable with checkpoints
- Logs progress and ETA

**Output:**
```
data/benchmarks/results/hybrid_retriever_comparison_YYYYMMDD.json
```

## Questions to Answer

1. **Which retriever is best for Esperanto Q&A?**
   - Measure: Recall@10 on 17 factual questions
   
2. **Is AST analysis worth the latency cost?**
   - Compare: ASTAware (~400ms) vs HNSW (~3ms)
   
3. **Do hybrid embeddings improve over linguistic-only?**
   - Compare: slot_hybrid (128d) vs slot_full (64d) results

4. **Which question types work best?**
   - Break down by: factual, definition, grammar

## Next Steps

1. ✅ Build mmap/ and faiss/ indexes (running now)
2. Create evaluation script
3. Run head-to-head comparison
4. Analyze results and pick best retriever
