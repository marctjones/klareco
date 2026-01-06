---
id: 80
title: Benchmark hybrid vs linguistic-only embeddings
state: open
created: '2026-01-06T05:43:56.720174Z'
labels:
- evaluation
- embeddings
- benchmark
priority: high
---
Compare retrieval performance between linguistic-only (64d) and hybrid (128d) embeddings.

**Prerequisites:**
- Task #78 complete (retrievers updated)
- Task #79 complete (hybrid indexes built)
- Baseline results with linguistic-only embeddings

**Metrics to compare:**
- Recall@k (k=1,5,10,20)
- Precision@k
- MRR (Mean Reciprocal Rank)
- Query latency
- Memory usage

**Benchmark script:**
```bash
# Run M1 benchmark with hybrid embeddings
./scripts/benchmark_qa_enhanced.py \
  --retriever-types faiss hnsw mmap scann \
  --embedding-mode hybrid \
  --output benchmark_results/hybrid_comparison.json

# Compare with baseline
python scripts/compare_benchmarks.py \
  --baseline benchmark_results/linguistic_only.json \
  --experiment benchmark_results/hybrid_comparison.json \
  --output-report benchmark_results/hybrid_vs_linguistic.html
```

**Expected improvements:**
- Better recall on proper noun queries (e.g., "Kio estas Parizo?")
- Improved topical clustering (documents about same topic)
- Similar or better performance on linguistic queries

**Success criteria:**
- Benchmark runs without errors
- Results documented with statistical significance
- Clear recommendation on whether hybrid embeddings improve retrieval
