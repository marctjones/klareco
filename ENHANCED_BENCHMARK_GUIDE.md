# Enhanced Benchmark Guide

This guide explains the new enhanced benchmarking and indexing scripts for Klareco.

## Overview

Two new production-ready scripts have been created:

1. **`./scripts/build_verified_indexes.sh`** - Build indexes with full metadata verification
2. **`./scripts/benchmark_qa_all.sh`** - Enhanced Q&A benchmark with checkpointing and resource monitoring

---

## 1. Build Verified Indexes

### What It Does

Creates slot-based retriever indexes while verifying that:
- ✅ Corpus contains Wikipedia data (tier 6)
- ✅ Corpus has source citations (book/article metadata)
- ✅ Corpus has parse annotations (parse_status, success_rate)
- ✅ Corpus has proper noun detection (category: proper_name, proper_name_known)
- ✅ Index preserves all metadata
- ✅ All retriever types are built correctly

### Parser Improvements Included

The corpus already includes annotations from the latest parser:
- **Parse status**: `success`, `failed` for each word
- **Proper nouns**: `proper_name` (capitalized), `proper_name_known` (in dictionary), `proper_name_esperantized` (with Esperanto endings)
- **Parse statistics**: `success_rate`, `categories` breakdown
- **Citation info**: `source.name`, `source.tier`, `source.source_name`, `source.chapter`, `source.article_title`

### Usage

```bash
# Build verified indexes (uses unified_corpus.jsonl by default)
./scripts/build_verified_indexes.sh

# Use custom corpus and output directory
./scripts/build_verified_indexes.sh data/corpus/my_corpus.jsonl data/indexes/my_indexes
```

### What It Verifies

**Stage 1: Corpus Verification**
- Scans entire corpus
- Checks for Wikipedia content (must have some!)
- Checks for source citations (tier, name, article_title, etc.)
- Checks for parse statistics (parse_status on words)
- Checks for proper noun annotations
- Shows breakdown by source type and tier
- Displays sample entry structure

**Stage 2: Index Building**
- Builds base slot index with checkpointing
- Preserves all source metadata
- Embeddings for SUBJ/VERB/OBJ slots

**Stage 3: Index Verification**
- Verifies index has source metadata
- Verifies Wikipedia documents are included
- Checks slots and features are present

**Stage 4: Retriever Indexes**
Auto-detects index size and builds appropriate retrievers:
- **Small (<100K docs)**: All retrievers (mmap, faiss, multifaiss, hnsw, scann)
- **Medium (100K-1M)**: Optimized retrievers (faiss, multifaiss, hnsw, scann)
- **Large (>1M docs)**: Highly-optimized only (multifaiss, hnsw, scann)

### Output

```
data/indexes/slot_verified/
├── slot_index.jsonl           # Base index with full metadata
├── checkpoint.json             # Resume point if interrupted
├── mmap/                       # Memory-mapped retriever
├── faiss/                      # FAISS retriever
├── multifaiss/                 # Multi-FAISS (separate per slot)
├── hnsw/                       # HNSW graph retriever
└── scann/                      # Google ScaNN retriever

logs/indexing/
├── slot_index_TIMESTAMP.log    # Build log
├── faiss_TIMESTAMP.log         # FAISS build log
├── hnsw_TIMESTAMP.log          # HNSW build log
└── scann_TIMESTAMP.log         # ScaNN build log
```

### Verification Output Example

```
═══════════════════════════════════════════════════════════
VERIFICATION RESULTS
═══════════════════════════════════════════════════════════

Parse Statistics:
  Total entries:           4,381,608
  Has parse_statistics:    4,381,608 (100.0%)
  ✓ All entries have parse statistics

Proper Noun Detection:
  Sentences with proper nouns: 524,193 (12.0%)
  ✓ Proper nouns detected
  Categories:
    proper_name: 412,456
    proper_name_known: 98,231
    proper_name_esperantized: 13,506

Source Citations:
  Has source metadata:     4,381,608 (100.0%)
  ✓ All entries have source metadata

  Source breakdown:
    wikipedia                 4,187,532 (95.6%)
    gutenberg                   187,697 ( 4.3%)
    fundamenta_krestomatio        6,379 ( 0.1%)

  Tier breakdown:
    Tier 2:           6,379 ( 0.1%)
    Tier 5:         187,697 ( 4.3%)
    Tier 6:       4,187,532 (95.6%)

Wikipedia Content:
  Wikipedia sentences:     4,187,532 (95.6%)
  ✓ Wikipedia content present

═══════════════════════════════════════════════════════════
VERDICT
═══════════════════════════════════════════════════════════

✓ CORPUS VERIFICATION PASSED
  All required metadata present:
    ✓ Parse statistics
    ✓ Proper noun annotations
    ✓ Source citations
    ✓ Wikipedia content
```

---

## 2. Enhanced Q&A Benchmark

### What It Does

Tests all retriever implementations on 50 real Esperanto questions with:
- ✅ **Checkpointing** - Save progress every 10 questions, resume on restart
- ✅ **Resource monitoring** - Track memory (RSS) and CPU per retriever
- ✅ **Progress logging** - Show per-question progress with ETA
- ✅ **Detailed output** - Full retrieved document texts for Claude Code analysis
- ✅ **Restartable** - Resume from last checkpoint if interrupted/crashed

### Usage

```bash
# Run enhanced benchmark on all available retrievers
./scripts/benchmark_qa_all.sh

# Use custom index
./scripts/benchmark_qa_all.sh --index data/indexes/slot_verified

# Use custom benchmark questions
./scripts/benchmark_qa_all.sh \
    --benchmark data/benchmarks/datasets/my_questions.jsonl

# Run directly with Python for more control
python scripts/benchmark_qa_enhanced.py \
    --index data/indexes/slot_verified \
    --benchmark data/benchmarks/datasets/qa_benchmark_v1.jsonl \
    --top-k 10 \
    --output benchmark_results/qa/results.json \
    --fresh  # Start over, ignore checkpoints
```

### Progress Output Example

```
22:45:12 - INFO - Testing MultiFAISS on 50 questions...
22:45:13 - INFO -   [5/50] | Latency: 1.2ms | Accuracy: 80% | Memory: 945MB | CPU: 12% | ETA: 2m 15s
22:45:25 - INFO -   [10/50] | Latency: 1.1ms | Accuracy: 90% | Memory: 948MB | CPU: 11% | ETA: 1m 52s | 💾 checkpoint
22:45:37 - INFO -   [15/50] | Latency: 1.3ms | Accuracy: 87% | Memory: 951MB | CPU: 13% | ETA: 1m 28s
22:45:49 - INFO -   [20/50] | Latency: 1.2ms | Accuracy: 85% | Memory: 953MB | CPU: 12% | ETA: 1m 5s | 💾 checkpoint
...
22:47:15 - INFO -   [50/50] | Latency: 1.2ms | Accuracy: 88% | Memory: 962MB | CPU: 12% | ETA: 0s
22:47:15 - INFO -   ✓ MultiFAISS complete: 44/50 found
22:47:15 - INFO -   ✓ Peak memory: 962MB, Avg CPU: 12.3%
```

### Results Output Example

```
════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
Q&A RETRIEVAL BENCHMARK RESULTS (ENHANCED)
════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

Retriever         Top-1    Top-5   Top-10  Latency     Memory      CPU
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
MultiFAISS        64.0%    86.0%    88.0%    1.2ms      962MB    12.3%
HNSW              62.0%    84.0%    86.0%    1.5ms     1024MB    15.1%
ScaNN             58.0%    82.0%    84.0%    1.8ms      897MB    14.2%
Hybrid            56.0%    80.0%    82.0%    2.1ms     1156MB    16.5%
MemoryMapped      54.0%    78.0%    80.0%  145.3ms     2341MB     8.9%

Rankings:
  🎯 Best Accuracy: MultiFAISS (44/50 found in top-10)
  ⚡ Fastest:       MultiFAISS (1.2ms avg)
  💾 Lowest Memory: ScaNN (897MB peak)

Metrics explained:
  Top-N:   % of questions where answer was found in top-N retrieved documents
  Latency: Average query time in milliseconds
  Memory:  Peak memory usage in megabytes
  CPU:     Average CPU usage during queries
```

### Output Files

```
benchmark_results/qa/
├── qa_benchmark_TIMESTAMP.json      # Full results for Claude Code
├── qa_benchmark_TIMESTAMP.log       # Complete execution log
├── multifaiss_checkpoint.json       # Checkpoints (deleted on success)
├── hnsw_checkpoint.json
└── scann_checkpoint.json
```

### JSON Output Structure (for Claude Code)

```json
[
  {
    "name": "MultiFAISS",
    "answer_in_top_1": 32,
    "answer_in_top_5": 43,
    "answer_in_top_10": 44,
    "total_questions": 50,
    "avg_time_ms": 1.2,
    "peak_memory_mb": 962,
    "memory_delta_mb": 312,
    "avg_cpu_pct": 12.3,
    "questions": [
      {
        "id": "q001",
        "question": "Kiu fondis Esperanton?",
        "category": "factual",
        "gold_answer": "Ludoviko Lazaro Zamenhof fondis Esperanton.",
        "acceptable_answers": ["Zamenhof", "L.L. Zamenhof", ...],
        "found_at_rank": 1,
        "query_time_ms": 1.15,
        "retrieved_docs": [
          "Full text of top-1 result...",
          "Full text of top-2 result...",
          ...
        ],
        "top_result": "Full text of top-1 result..."
      },
      ...
    ]
  },
  ...
]
```

### Checkpointing and Restartability

**Automatic Resume:**
- Checkpoints saved every 10 questions
- If benchmark crashes or is interrupted, simply re-run:
  ```bash
  ./scripts/benchmark_qa_all.sh
  ```
- It will automatically resume from the last checkpoint

**Force Fresh Start:**
```bash
python scripts/benchmark_qa_enhanced.py \
    --index data/indexes/slot_verified \
    --fresh
```

### Resource Monitoring

The enhanced benchmark tracks:
- **Peak Memory (MB)**: Maximum RSS memory used by the process
- **Memory Delta (MB)**: Memory increase from baseline
- **Average CPU (%)**: Mean CPU usage across all queries
- **Query Latency (ms)**: Time per query

This helps identify:
- Memory-hungry retrievers (risk of OOM crashes)
- CPU-intensive retrievers (may slow down system)
- Performance vs resource tradeoffs

---

## Claude Code Analysis Workflow

### Step 1: Run Enhanced Benchmark

```bash
./scripts/benchmark_qa_all.sh --index data/indexes/slot_verified
```

### Step 2: Pass Results to Claude Code

The JSON file contains:
- All 50 questions with gold answers
- Top-10 retrieved documents for each question (full text!)
- Whether answer was found and at what rank
- Performance metrics (latency, memory, CPU)

### Step 3: Ask Claude to Analyze

```
I ran the enhanced Q&A benchmark. Here's the results file:
benchmark_results/qa/qa_benchmark_20260105_223000.json

Please analyze:
1. Which retriever is best overall?
2. Which retriever is best for each question category (factual, negative, etc.)?
3. Where did each retriever fail and why?
4. Are there patterns in failures (e.g., all fail on proper nouns)?
5. Latency vs accuracy tradeoffs
6. Memory usage concerns
7. Recommendations for production use
```

Claude Code will:
- Read the JSON file
- Examine retrieved documents for each failed question
- Assess relevance manually
- Identify patterns (proper noun failures, tier-specific issues, etc.)
- Provide concrete recommendations

---

## FAQ

### Q: Do the indexes include parser improvements?

**A:** The **corpus** includes all parser improvements (parse_status, proper_name annotations), but the **current indexes** do NOT preserve this information in the slot index.

**To use parser improvements in retrieval**, you would need to:
1. Modify `klareco/rag/slot_indexer.py` to store `parse_statistics` and word-level `parse_status`
2. Rebuild indexes with `./scripts/build_verified_indexes.sh`
3. Modify retrievers to use this information (e.g., boost results with known proper nouns)

**Current status**: The build script VERIFIES the corpus has these annotations but doesn't yet use them for retrieval ranking.

### Q: Why does verification pass but index doesn't use parse info?

**A:** The verification checks that the **source corpus** has all required metadata. The **indexer** currently only extracts:
- Slot embeddings (SUBJ, VERB, OBJ)
- Basic features (negita, tempo, fraztipo, modo)
- Source metadata (tier, name, article_title)

It does NOT yet extract:
- Parse status per word
- Proper noun categories
- Success rate

This is a **feature gap** - the data exists in the corpus but isn't used by the retriever yet.

### Q: How do I know if my system crashed due to memory?

**A:** The enhanced benchmark tracks peak memory. If you see:
- Process disappeared without error message
- System became unresponsive
- OOM (Out of Memory) in system logs: `sudo journalctl | grep -i oom`

Check the last logged memory value in the benchmark log. If it was approaching your system's total RAM, that's likely the cause.

### Q: Can I run benchmarks in parallel?

**A:** No. Each benchmark should run sequentially because:
- Memory tracking would be inaccurate
- Checkpoints would conflict
- System resource contention would skew latency results

Run one at a time in separate sessions.

### Q: What if I want to add my own questions?

Create a JSONL file with this format:
```json
{"id": "q051", "question": "Your question?", "category": "factual", "gold_answer": "Expected answer", "acceptable_answers": ["answer1", "answer2"], "source_tier": 1, "difficulty": "easy", "requires_retrieval": true, "expected_source": "wikipedia"}
```

Then run:
```bash
./scripts/benchmark_qa_all.sh --benchmark path/to/your_questions.jsonl
```

---

## Next Steps

1. **Build verified indexes**:
   ```bash
   ./scripts/build_verified_indexes.sh
   ```

2. **Run enhanced benchmark**:
   ```bash
   ./scripts/benchmark_qa_all.sh --index data/indexes/slot_verified
   ```

3. **Analyze with Claude Code**:
   Pass the JSON results file to Claude for detailed analysis

4. **Iterate**:
   - Identify weak retrievers
   - Check memory usage
   - Optimize based on Claude's recommendations
