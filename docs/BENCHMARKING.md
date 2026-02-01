# Benchmarking Guide

Quick reference for running Klareco benchmarks.

## Available Benchmarks

| Benchmark | Script | Purpose | Runtime |
|-----------|--------|---------|---------|
| **M1 Model** | `benchmark_m1.py` | Test M1 on 273K test set | ~2 min |
| **M1 Impact** | `benchmark_m1_impact.py` | Compare with/without M1 | ~1 min |
| **Reranker** | `benchmark_reranker.py` | Test reranker quality | ~2 min |
| **RAG E2E** | `evaluate_rag_test_set.py` | Test full pipeline (30Q) | ~5 min |
| **Progress Tracking** | `track_evaluation_progress.py` | Track metrics over time | instant |

## Quick Start

### 1. Test M1 Model

```bash
python scripts/benchmark_m1.py --show-examples
```

**What it measures**: M1 accuracy on held-out test set
**Expected**: 91%+ overall accuracy, 82%+ on role-swaps

### 2. Test M1 Impact

```bash
python scripts/benchmark_m1_impact.py --show-examples
```

**What it measures**: Precision improvement with M1 filtering
**Expected**: +50% precision, 3-4 implausible results filtered per query

### 3. Test RAG Pipeline

```bash
python scripts/evaluate_rag_test_set.py --expected works
```

**What it measures**: End-to-end accuracy on test questions
**Expected**: 67% correct on "should work" questions (factual, grammar, definitions)

### 4. Test Reranker

```bash
python scripts/benchmark_reranker.py --show-examples
```

**What it measures**: Reranker ranking quality
**Expected**: Positive improvement ratio, reasonable overhead

## Common Use Cases

### Compare Pipeline Configurations

```bash
# Retrieval only
python scripts/evaluate_rag_test_set.py --no-m1 --no-rerank

# Retrieval + M1
python scripts/evaluate_rag_test_set.py --no-rerank

# Retrieval + M1 + Reranker (full pipeline)
python scripts/evaluate_rag_test_set.py
```

### Test Specific Categories

```bash
# Test factual questions
python scripts/evaluate_rag_test_set.py --category factual_simple

# Test grammar questions
python scripts/evaluate_rag_test_set.py --category grammar

# Test multiple categories
python scripts/evaluate_rag_test_set.py --category factual_simple --category definition
```

### Save Results for Analysis

```bash
# M1 benchmark
python scripts/benchmark_m1.py \
  --output results/m1_$(date +%Y%m%d).json \
  --show-examples

# M1 impact
python scripts/benchmark_m1_impact.py \
  --output results/m1_impact_$(date +%Y%m%d).json \
  --show-examples

# RAG evaluation
python scripts/evaluate_rag_test_set.py \
  --output results/rag_eval_$(date +%Y%m%d).jsonl
```

## Expected Results

### M1 Model (Current Training)

- **Overall accuracy**: 91.17% (epoch 8)
- **Role-swap detection**: ~82% (hardest case)
- **Subject/object/verb**: ~89-90%

### M1 Impact

- **Precision improvement**: +50-60%
- **Filtering rate**: ~36% of candidates filtered
- **Speed overhead**: ~22% (12ms per query)

### RAG Pipeline (Expected)

| Question Type | Expected Performance |
|---------------|---------------------|
| Factual (simple) | ✅ 100% (4/4) |
| Grammar | ✅ 67% (2/3) |
| Definitions | ✅ 100% (2/2) |
| Negation | ⚠️ Partial |
| Comparison | ⚠️ Partial |
| Multi-hop | ❌ Fails (needs Stage 3) |
| Pronoun resolution | ❌ Fails (needs Stage 3) |
| Temporal reasoning | ❌ Fails (needs Stage 4) |

## Troubleshooting

### M1 model not found

```bash
# Train M1 first
./scripts/train_m1_semantic_tier_priority.sh

# Or test with current checkpoint
./scripts/test_m1_rag_now.sh
```

### Reranker not found

```bash
# Train reranker
./scripts/train_reranker.sh
```

### Kuzu index not found

```bash
# Build index
python scripts/index_kuzu.py \
  --corpus data/enhanced_corpus/corpus_with_metadata.jsonl \
  --output-dir data/indexes/kuzu_index
```

### Stage 1 embeddings not found

```bash
# Train Stage 1
./scripts/train_roots.sh
```

## Interpreting Results

### M1 Benchmark

**Good signs**:
- Overall accuracy ≥90%
- Role-swap accuracy ≥80%
- F1 score ≥0.90

**Warning signs**:
- Role-swap accuracy <75% (may mislabel symmetric relations)
- High false positive rate (filters too much)
- High false negative rate (lets nonsense through)

### M1 Impact

**Good signs**:
- Precision improvement ≥40%
- Avg M1 score improvement ≥0.2
- Overhead <30%

**Warning signs**:
- Precision improvement <20% (M1 not helping much)
- Overhead >50% (too slow)
- Filtering too aggressively (>80% filtered)

### RAG Evaluation

**Good signs**:
- ≥60% correct on "should work" questions
- High precision on factual/grammar/definition
- Partial credit on complex questions

**Warning signs**:
- <40% correct on "should work" questions
- Zero correct on factual questions (retrieval broken)
- High error rate (pipeline issues)

## Advanced Usage

### Custom Threshold

```bash
# Test M1 with higher threshold (stricter filtering)
python scripts/benchmark_m1_impact.py --m1-threshold 0.7

# Test with lower threshold (more lenient)
python scripts/benchmark_m1_impact.py --m1-threshold 0.3
```

### Custom Queries

```bash
# Create queries file
cat > my_queries.txt <<EOF
Kiu fondis Esperanton?
Kio estas la Fundamento?
Kie naskiĝis Zamenhof?
EOF

# Test M1 impact
python scripts/benchmark_m1_impact.py --queries my_queries.txt

# Test reranker
python scripts/benchmark_reranker.py --queries my_queries.txt
```

### Limit Test Size

```bash
# Quick M1 test (10K examples)
python scripts/benchmark_m1.py --max-examples 10000
```

## Regression Testing

Track performance over time:

```bash
# Baseline (before changes)
python scripts/evaluate_rag_test_set.py \
  --output results/baseline.jsonl

# After M1 integration
python scripts/evaluate_rag_test_set.py \
  --output results/with_m1.jsonl

# After Stage 3 features
python scripts/evaluate_rag_test_set.py \
  --output results/stage3.jsonl

# Compare
diff -u results/baseline.jsonl results/with_m1.jsonl
diff -u results/with_m1.jsonl results/stage3.jsonl
```

## Progress Tracking

Track evaluation metrics over time with granular scoring:

```bash
# Show current metrics
python scripts/track_evaluation_progress.py

# Save snapshot with descriptive name
python scripts/track_evaluation_progress.py --save --name "after_extraction_fix"

# Compare all snapshots
python scripts/track_evaluation_progress.py --compare
```

### Granular Scoring

Unlike binary pass/fail, granular scoring gives partial credit:

- **Retrieval (R)**: Where is answer in retrieved docs? (top-1=1.0, top-10=0.4)
- **Extraction (E)**: Was correct answer extracted? (exact=1.0, fuzzy=0.5)
- **Alignment (A)**: From which rank was answer extracted?
- **Robustness (B)**: How many top-5 docs have the answer?

**Formula**: `0.40×R + 0.30×E + 0.20×A + 0.10×B`

**Example**: System retrieves answer in top-3 (R=0.8) but extracts wrong entity (E=0.0):
- Binary: ✗ Incorrect (0%)
- Granular: 0.32 (32%) - shows retrieval is working!

### Current Baseline (2026-02-01)

After extraction fixes (commit 15316ce):
```
Granular:    0.487 / 1.000
  Retrieval: 0.967 (excellent)
  Extraction: 0.000 (bottleneck)  ← needs work
  Alignment:  0.503 (moderate)
  Robustness: 0.000 (no multi-doc)

Binary: 0% (0/30 correct)
```

**Key insight**: Retrieval works (97% in top-10), extraction fails (0% exact matches)

## See Also

- `/tmp/benchmarks_summary.md` - Comprehensive benchmark documentation
- `scripts/demo_full_rag.sh` - Interactive RAG demo
- `CLAUDE.md` - Development commands and architecture
