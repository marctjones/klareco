---
id: 1
title: Retriever Evaluation Plan
created: '2026-01-06T20:30:17.300672Z'
updated: '2026-01-06T20:30:17.300691Z'
---
# Retriever Evaluation Plan

## Current State Assessment

### What We Have ✅

**Corpus:**
- 4.38M sentences in `data/corpus/unified_corpus.jsonl`
- Sources: Fundamento, Wikipedia, books, etc.
- Each entry has: text, source metadata, full AST parse, parse_rate

**Indexes:**
- `slot_full/` - Full 4.38M corpus index with:
  - FAISS index (1.1GB)
  - HNSW index (1.7GB)
  - MultiFAISS per-slot indexes (2.2GB)
  - ScaNN index
  - SQLite database
  - Slot index JSONL (20GB)
- `slot_hybrid/` - Hybrid embeddings index (newer, with 128d embeddings)

**Q&A Benchmark:**
- `data/benchmarks/datasets/qa_benchmark_v1.jsonl` - 50 questions:
  - 10 factual (who founded Esperanto, etc.)
  - 10 definition (what is akuzativo, etc.)
  - 10 grammar (how to form past tense, etc.)
  - 10 reasoning (what is opposite of rapida, etc.)
  - 10 negative (unanswerable questions)
- Only ~17 questions require retrieval (others are reasoning/grammar/negative)

**Existing Benchmark Infrastructure:**
- `scripts/benchmark_qa_enhanced.py` - Full benchmark with checkpointing
- Previous results in `data/benchmarks/results/` (baseline 10-12%, klareco 41%)

**Benchmark Articles:**
- Wikipedia articles list for testing (Kamala Harris, Einstein, etc.)

### What's Missing ❌

1. **Retrieval-focused benchmark** - Current 50 Q&A questions conflate:
   - Retrieval accuracy (can we find the right document?)
   - Answer extraction (can we pull out the answer?)
   - Reasoning (can we infer from grammar rules?)
   
2. **Ground truth relevance judgments** - The `benchmark_queries.jsonl` files are auto-generated (query = document text) which is trivial to match

3. **Multi-retriever comparison** - Need consistent test across ALL 8 production retrievers

4. **AST-specific metrics** - Current metrics don't measure:
   - Slot matching accuracy
   - Entity recognition accuracy
   - Question classification accuracy

---

## Evaluation Plan

### Phase 1: Retrieval-Only Benchmark

**Goal:** Measure pure retrieval accuracy (can we find relevant documents?)

**Test Set Design:**
1. Take 50-100 questions that REQUIRE retrieval (exclude reasoning/negative)
2. For each question, annotate 1-5 "gold" relevant documents from corpus
3. Create relevance judgments: {question_id, [doc_ids], relevance_level}

**Questions to include:**
- Factual: "Kiu fondis Esperanton?" → docs mentioning Zamenhof founding
- Definition from corpus: "Kio estas UEA?" → UEA article
- Historical: "Kiam naskiĝis Zamenhof?" → Zamenhof biography

**Metrics:**
- Recall@1, Recall@5, Recall@10 (does gold doc appear in top-k?)
- MRR (Mean Reciprocal Rank)
- Latency (ms per query)
- Memory usage (peak RSS)

### Phase 2: Head-to-Head Retriever Comparison

**Retrievers to test (8 total):**
1. ASTAwareRetriever (full AST analysis + HNSW prefilter)
2. ASTAwareRetriever (full AST analysis + keyword prefilter)
3. HNSWSlotRetriever (HNSW + mmap slots)
4. FAISSSlotRetriever (FAISS + slot rerank)
5. HybridFAISSMmapRetriever (FAISS + mmap hybrid)
6. MultiFAISSSlotRetriever (per-slot FAISS indexes)
7. ScaNNSlotRetriever (if TensorFlow available)
8. SQLiteSlotRetriever (disk-only baseline)

**Test Protocol:**
1. Warm-up: 5 queries (discard)
2. Benchmark: 100 queries
3. Measure per-query: latency, memory delta, retrieval success
4. Report: mean/median/p95 latency, recall@k, memory

### Phase 3: AST-Aware Component Analysis

**Goal:** Understand which AST components help most

**Sub-experiments:**
1. **Question Classification Accuracy**
   - Ground truth: manually label 50 questions with type (WHO/WHAT/WHEN/WHERE/HOW)
   - Measure: classifier accuracy

2. **Entity Recognition Accuracy**
   - Ground truth: manually tag entities in questions
   - Measure: precision/recall of entity recognizer

3. **Slot Matching Value**
   - Compare: full slot matching vs full embedding only
   - Ablation: disable each slot (SUBJ/VERB/OBJ) to measure contribution

4. **Semantic Relations Impact**
   - Compare: with vs without synonym expansion
   - Measure: recall improvement from ReVo synonyms

### Phase 4: Corpus Quality Analysis

**Goal:** Ensure corpus is suitable for evaluation

**Checks:**
1. **Coverage test:** Do gold answers exist in corpus?
   - For each benchmark question, grep corpus for acceptable answers
   - Flag questions where answer not in corpus

2. **Source diversity:** Balance of Fundamento vs Wikipedia vs books
   - Count sentences per source
   - Ensure benchmark draws from multiple sources

3. **Parse quality:** Are parsed ASTs accurate?
   - Sample 100 sentences, manual review of AST accuracy
   - Especially for complex sentences with proper names

---

## Index Requirements

**Current indexes support most retrievers:**

| Retriever | Index Needed | Status |
|-----------|--------------|--------|
| ASTAwareRetriever | slot_index.jsonl + offsets + hnsw/ | ✅ Have |
| HNSWSlotRetriever | slot_index.jsonl + hnsw/ + mmap/ | ✅ Have |
| FAISSSlotRetriever | slot_index.jsonl + faiss/ | ✅ Have |
| HybridFAISSMmapRetriever | faiss/ + mmap/ | ✅ Have |
| MultiFAISSSlotRetriever | multifaiss/ | ✅ Have |
| ScaNNSlotRetriever | scann/ | ✅ Have |
| SQLiteSlotRetriever | slot_index.db | ✅ Have |
| MemoryMappedSlotRetriever | mmap/ | ✅ Have |

**Note:** `slot_hybrid/` uses 128d hybrid embeddings (linguistic + topical).
`slot_full/` uses 64d linguistic-only embeddings.

Decision needed: Run evaluation on which index?
- slot_full: More index types available
- slot_hybrid: Newer embeddings, may be more accurate

---

## Questions to Answer

1. **Which retriever is best for Q&A?**
   - Expected: ASTAwareRetriever (most AST analysis)
   - Measure: Recall@10 on factual questions

2. **Which retriever is fastest?**
   - Expected: HNSWSlotRetriever (~2-3ms)
   - Measure: p50/p95 latency

3. **Is AST analysis worth the cost?**
   - Compare: ASTAware (0.4s) vs HNSW-only (3ms)
   - Tradeoff: 130x slower for how much accuracy gain?

4. **Do hybrid embeddings help?**
   - Compare: slot_full (64d) vs slot_hybrid (128d)
   - Measure: Recall improvement

5. **What's the retrieval bottleneck?**
   - Profile: prefilter vs slot matching vs final ranking
   - Optimize: the slowest stage

---

## Immediate Next Steps

1. **Create retrieval-focused test set** (2-4 hours)
   - Select 50 retrieval-requiring questions
   - Manually find gold documents in corpus
   - Save as `data/benchmarks/datasets/retrieval_benchmark_v1.jsonl`

2. **Write unified benchmark script** (1-2 hours)
   - Input: benchmark file + list of retrievers
   - Output: comparison table with metrics
   - Save detailed results for analysis

3. **Run head-to-head comparison** (30 min - 2 hours depending on corpus)
   - All 8 retrievers on same 50 questions
   - Generate comparison report

4. **Analyze results** (1 hour)
   - Identify best retriever for Klareco's use case
   - Document findings in IdlerGear reference
