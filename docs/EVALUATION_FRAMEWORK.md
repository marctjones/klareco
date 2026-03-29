# Comprehensive Evaluation Framework

## Overview

The comprehensive evaluation framework tracks multi-stage metrics and performance across the entire extractive QA pipeline, enabling fine-grained analysis of where improvements/regressions occur.

## Quick Start

### Recommended: Full Evaluation Suite (Comprehensive)

```bash
# Run full suite: baseline + top-k sweep + ablations (~30 minutes)
python scripts/evaluate_full_suite.py \
  --output results/suite_$(date +%Y%m%d)/

# Quick suite: skip ablations (~20 minutes)
python scripts/evaluate_full_suite.py \
  --skip-ablations \
  --output results/suite_quick/

# Baseline only (fastest, ~5 minutes)
python scripts/evaluate_full_suite.py \
  --baseline-only \
  --output results/baseline_$(date +%Y%m%d)/
```

**Why use the full suite?**
- Provides context for interpreting metrics (is 36% accuracy good at top-k=20?)
- Identifies optimal configuration (should we use top-k=10 or 30?)
- Validates assumptions (is M1 helping or hurting?)
- Detects bottlenecks automatically (extraction vs ranking vs retrieval)

### Alternative: Single Evaluation (Quick Testing)

```bash
# Run single evaluation with specific configuration
python scripts/evaluate_pipeline_comprehensive.py \
  --output results/baseline.json \
  --export-csv

# Analyze results
python scripts/analyze_evaluation_results.py results/baseline.json

# Compare two runs
python scripts/analyze_evaluation_results.py results/baseline.json results/improved.json
```

## Metrics Tracked

### 1. Answer Quality
- **Keyword match accuracy**: Does the final answer contain expected keywords? (existing metric)
- **Answer coherence**: Length, sentence structure
- **Citation coverage**: How many facts have citations?

### 2. Retrieval Quality
- **Retrieval recall**: Did top-K sentences contain the answer?
  - Recall@5, Recall@10, Recall@20
- **Mean Reciprocal Rank (MRR)**: Average rank of first correct sentence
- **Retrieval failures**: Questions with 0 retrieved sentences

### 3. Extraction Quality
- **Facts extracted per question**: How many facts were extracted from ASTs?
- **Facts selected per question**: How many facts made it to the final answer?
- **M1 filtering rate**: What % of facts were removed by plausibility filter?
- **Question-type filtering rate**: What % removed by question-type filter?

### 4. Performance/Timing
- **Total time per question**: End-to-end latency
- **Time per pipeline stage**:
  - Parse time (query AST parsing)
  - Retrieval time (Whoosh + Kuzu)
  - Reranking time (neural reranker, if enabled)
  - Extraction time (fact extraction from ASTs)
  - Scoring time (importance scoring)
  - Generation time (discourse planning + linearization)
- **CPU time vs wall time**: Identify concurrency opportunities

### 5. Pipeline Health
- **Parse failures**: Questions that failed to parse
- **Retrieval failures**: Questions with 0 results
- **Extraction failures**: Questions with 0 facts extracted
- **Generation failures**: Questions with empty answers

## Scripts

### 1. `evaluate_pipeline_comprehensive.py`

Runs comprehensive evaluation with full metrics tracking.

**Features:**
- Multi-stage timing profiling
- Retrieval quality metrics (recall, MRR)
- Extraction statistics
- JSON output for detailed analysis
- CSV export for spreadsheet analysis

**Usage:**
```bash
# Basic evaluation
python scripts/evaluate_pipeline_comprehensive.py

# With options
python scripts/evaluate_pipeline_comprehensive.py \
  --test-set data/test_sets/qa_test_diverse_30.jsonl \
  --top-k 20 \
  --no-m1 \
  --no-rerank \
  --limit 10 \
  --output results/test_run.json \
  --export-csv \
  --verbose
```

**Options:**
- `--test-set PATH`: Path to test set (default: `qa_test_diverse_30.jsonl`)
- `--db PATH`: Path to Kuzu database (default: `v2.1_kuzu_index_full`)
- `--top-k N`: Number of sentences to retrieve (default: 20)
- `--no-m1`: Disable M1 plausibility filtering
- `--no-rerank`: Disable neural reranking
- `--limit N`: Limit to first N questions
- `--output PATH`: Save JSON results to file
- `--export-csv`: Export results to CSV
- `--verbose`: Show detailed per-question output

**Output:**

JSON file with:
```json
{
  "metadata": {
    "test_set": "...",
    "num_questions": 50,
    "top_k": 20,
    "use_m1": true,
    "use_rerank": true
  },
  "aggregates": {
    "overall": {"accuracy": 0.32, ...},
    "timing": {"total_time_mean": 2.5, ...},
    "retrieval": {"recall_at_20": 0.75, ...},
    "extraction": {"facts_extracted_mean": 8.5, ...},
    "answer": {"answer_length_mean": 250, ...},
    "by_question_type": {...}
  },
  "results": [
    {
      "question_id": "q1",
      "question_text": "...",
      "timing": {...},
      "retrieval": {...},
      "extraction": {...},
      "answer": {...},
      "success": true
    },
    ...
  ]
}
```

CSV file with one row per question and columns for all metrics.

### 2. `analyze_evaluation_results.py`

Analyzes evaluation JSON and generates insights.

**Features:**
- Failure point analysis (where does the pipeline fail?)
- Timing bottleneck identification
- Question type performance breakdown
- Comparison mode (compare two runs)

**Usage:**
```bash
# Analyze single run
python scripts/analyze_evaluation_results.py results/baseline.json

# Compare two runs (baseline vs improved)
python scripts/analyze_evaluation_results.py results/baseline.json results/improved.json
```

**Output:**

```
================================================================================
EVALUATION ANALYSIS
================================================================================

Test Set: data/test_sets/qa_test_diverse_30.jsonl
Questions: 50
Configuration: top_k=20, M1=True, Rerank=True

--------------------------------OVERALL ACCURACY--------------------------------
Correct: 16/50 (32.0%)

---------------------------------FAILURE ANALYSIS-------------------------------
✓ Success: 16 (32.0%)
✗ Retrieval empty: 2 (4.0%)
✗ Retrieval no answer: 15 (30.0%)
✗ Extraction no facts: 5 (10.0%)
✗ Extraction wrong facts: 8 (16.0%)
✗ Generation poor: 4 (8.0%)

-------------------------------RETRIEVAL PERFORMANCE----------------------------
Recall@5:  45.0%  (answer in top 5 sentences)
Recall@10: 60.0%  (answer in top 10 sentences)
Recall@20: 75.0%  (answer in top 20 sentences)
MRR: 0.325  (mean reciprocal rank)

-------------------------------EXTRACTION PERFORMANCE---------------------------
Facts extracted (avg): 8.5
Facts extracted (median): 7
Facts selected (avg): 3.2
Facts selected (median): 3

--------------------------------TIMING BREAKDOWN--------------------------------
Total time per question: 2.456s (±0.842s)
  Parse:      0.023s  (0.9%)
  Retrieval:  1.234s  (50.2%)
  Generation: 1.199s  (48.9%)

Bottlenecks:
  Retrieval: 50.2% of total time
  Generation: 48.9% of total time
  Parse: 0.9% of total time

--------------------------PERFORMANCE BY QUESTION TYPE--------------------------
Type         Acc      Retr     Extr     Time     MRR
--------------------------------------------------------------------------------
HOW          50.0%    50.0%    100.0%   1.234s   0.500
HOW_MANY     80.0%    100.0%   100.0%   1.567s   0.800
WHAT         40.0%    60.0%    80.0%    2.123s   0.350
WHEN         10.0%    30.0%    70.0%    2.456s   0.150
WHERE        40.0%    70.0%    90.0%    2.234s   0.400
WHO          10.0%    40.0%    60.0%    2.678s   0.100
WHY          50.0%    50.0%    100.0%   1.890s   0.500

-------------------------------ANSWER QUALITY----------------------------------
Answer length (avg): 247 chars
Answer length (median): 230 chars
Citations per answer (avg): 1.8

================================================================================
```

### 3. `translate_triviaqa_to_esperanto.py`

Converts TriviaQA questions to Esperanto QA format (requires manual translation).

**Usage:**
```bash
python scripts/translate_triviaqa_to_esperanto.py \
  --input data/external/triviaqa_sample_1000.jsonl \
  --output data/test_sets/triviaqa_esperanto_100.jsonl \
  --limit 100
```

**Output:**

JSONL file with:
```json
{
  "id": "triviaqa_1",
  "question": "[TRANSLATE: Who was President when the first Peanuts cartoon was published?]",
  "question_en": "Who was President when the first Peanuts cartoon was published?",
  "question_type": "WHO",
  "expected_keywords": ["truman", "harry"],
  "answer_variations_en": ["Harry S. Truman", "Harry Truman", ...]
}
```

**Note:** Questions are marked `[TRANSLATE: ...]` and need manual Esperanto translation.

## Workflow: Evaluating Changes

### Before Making Changes

1. **Run baseline evaluation:**
   ```bash
   python scripts/evaluate_pipeline_comprehensive.py \
     --output results/baseline_$(date +%Y%m%d).json \
     --export-csv
   ```

2. **Analyze baseline:**
   ```bash
   python scripts/analyze_evaluation_results.py results/baseline_*.json
   ```

3. **Identify bottlenecks:**
   - Check failure analysis (where are questions failing?)
   - Check timing breakdown (where is time spent?)
   - Check question type performance (which types are weak?)

### After Making Changes

1. **Run new evaluation:**
   ```bash
   python scripts/evaluate_pipeline_comprehensive.py \
     --output results/after_change_$(date +%Y%m%d).json \
     --export-csv
   ```

2. **Compare results:**
   ```bash
   python scripts/analyze_evaluation_results.py \
     results/baseline_*.json \
     results/after_change_*.json
   ```

3. **Analyze changes:**
   - Did overall accuracy improve/regress?
   - Did timing improve/regress?
   - Which question types were affected?
   - Did retrieval quality change?
   - Did extraction quality change?

### Example Workflow

```bash
# Day 1: Baseline evaluation
python scripts/evaluate_pipeline_comprehensive.py \
  --output results/baseline_20240115.json \
  --export-csv

# Analyze baseline
python scripts/analyze_evaluation_results.py results/baseline_20240115.json
# Output shows: Retrieval is the bottleneck (50% of time)

# Day 2: Optimize retrieval (implement change)
# ... make changes to retrieval code ...

# Re-evaluate
python scripts/evaluate_pipeline_comprehensive.py \
  --output results/optimized_retrieval_20240116.json \
  --export-csv

# Compare
python scripts/analyze_evaluation_results.py \
  results/baseline_20240115.json \
  results/optimized_retrieval_20240116.json

# Output shows:
# - Retrieval time: 1.234s -> 0.567s (-54% ✓)
# - Total time: 2.456s -> 1.789s (-27% ✓)
# - Accuracy: 32.0% -> 32.0% (unchanged ✓)
# Conclusion: Successful optimization (faster with no accuracy loss)
```

## Key Insights from Metrics

### Retrieval Quality Metrics

**Recall@K**: What percentage of questions have the answer in top K sentences?
- **High recall (>70%)**: Retrieval is working well, focus on extraction/generation
- **Low recall (<50%)**: Retrieval is the bottleneck, improve query expansion or ranking

**MRR (Mean Reciprocal Rank)**: Average of 1/rank for first correct sentence
- **High MRR (>0.5)**: Answer usually in top 2-3 sentences
- **Low MRR (<0.3)**: Answer buried deep, improve ranking

### Extraction Quality Metrics

**Facts extracted**: How many facts are extracted from retrieved sentences?
- **Too few (<3)**: Parser or extraction patterns missing cases
- **Too many (>20)**: Extraction too permissive, may include noise

**Facts selected**: How many facts make it to final answer?
- **Much lower than extracted**: Filtering/scoring is working
- **Same as extracted**: Filtering may be too permissive

### Timing Metrics

**Bottleneck identification**:
- **Retrieval >60%**: Optimize Whoosh queries, Kuzu queries, or caching
- **Generation >60%**: Optimize fact extraction, scoring, or discourse planning
- **Parse >10%**: Optimize parser (unlikely - parsing is usually fast)

## CSV Analysis in Spreadsheet

The exported CSV can be analyzed in Excel, Google Sheets, or pandas:

**Useful analyses:**
1. **Accuracy by retrieval quality**: Do questions with higher MRR have better accuracy?
2. **Timing vs question type**: Which question types are slowest?
3. **Extraction rate vs success**: Do more facts lead to better answers?
4. **Retrieval recall vs final accuracy**: If retrieval recall is high but accuracy is low, extraction/generation is the problem

**Example pandas analysis:**
```python
import pandas as pd

df = pd.read_csv('results/baseline_20240115.csv')

# Accuracy by retrieval quality
print(df.groupby('contains_answer')['keyword_match'].mean())

# Timing by question type
print(df.groupby('question_type')['total_time'].mean())

# Success rate by number of facts
df['fact_bins'] = pd.cut(df['facts_extracted'], bins=[0, 3, 7, 15, 100])
print(df.groupby('fact_bins')['success'].mean())
```

## Test Sets

### Current Test Sets

1. **qa_test_diverse_30.jsonl** (50 questions)
   - Hand-curated Esperanto questions
   - Covers 8 question types (WHO, WHAT, WHERE, WHEN, WHY, HOW, HOW_MANY, WHICH)
   - High-quality expected keywords

2. **triviaqa_sample_1000.jsonl** (1000 English questions)
   - General trivia questions from TriviaQA
   - Needs translation to Esperanto
   - Larger test set for statistical significance

### Creating New Test Sets

**Format:**
```json
{
  "id": "unique_id",
  "question": "Esperanto question text",
  "question_type": "WHO|WHAT|WHERE|WHEN|WHY|HOW|HOW_MANY|WHICH|OTHER",
  "expected_keywords": ["keyword1", "keyword2", ...]
}
```

**Guidelines:**
- Use short keyword stems (4-6 characters) that can be found in text
- Include multiple synonyms as keywords
- Lowercase all keywords
- Test keywords match actual corpus content

## Future Enhancements

### Potential New Metrics

1. **Semantic answer similarity**: Use embeddings to measure answer quality beyond keyword matching
2. **Answer fluency**: Use language model perplexity to measure coherence
3. **Fact relevance scoring**: Do extracted facts actually answer the question?
4. **Citation accuracy**: Do citations point to sentences that actually support the answer?

### Potential New Features

1. **Continuous monitoring**: Track metrics over time as code changes
2. **Regression detection**: Automatically flag when metrics regress
3. **A/B testing**: Compare two configurations side-by-side
4. **Drill-down analysis**: Click on failed question to see detailed trace

## See Also

- `scripts/evaluate_extractive_qa.py` - Original simpler evaluation (keyword matching only)
- `scripts/demo_extractive_qa.py` - Interactive demo for single questions
- `docs/EVALUATION_UNIFIED_EXTRACTOR.md` - Evaluation results for unified extractor

## Full Evaluation Suite

The full evaluation suite (`evaluate_full_suite.py`) runs a comprehensive set of tests to provide complete context for interpreting metrics and identifying bottlenecks.

### What It Runs

1. **Baseline Evaluation** (top_k=20, M1=True, Rerank=True)
   - Current default configuration
   - Provides reference point for comparisons

2. **Top-K Optimization Sweep** (top_k = 5, 10, 20, 30, 50)
   - Tests different numbers of sentences for answer generation
   - Identifies optimal configuration
   - Reveals whether extraction or ranking is the bottleneck
   - Analyzes noise patterns (M1 filtering rate vs top-k)

3. **M1 Ablation Test** (with/without M1 filtering)
   - Measures M1 filter impact
   - Tests if M1 is helping (+accuracy) or hurting (-accuracy)
   - Validates M1 threshold calibration

4. **Reranker Ablation Test** (with/without neural reranking)
   - Measures reranker impact
   - Tests if reranker improves over BM25 baseline
   - Validates reranker training quality

### Output

The suite generates:
- Individual JSON results for each configuration
- `SUITE_REPORT.txt` with comprehensive analysis including:
  - Baseline results summary
  - Top-K optimization curve analysis (plateau/linear/peak-decline patterns)
  - Noise analysis (M1 filtering rates)
  - Ablation test results (M1 and reranker impact)
  - Automatic bottleneck identification (extraction/ranking/retrieval)
  - Prioritized recommendations

### Example Report Output

```
================================================================================
FULL EVALUATION SUITE REPORT
================================================================================

BASELINE RESULTS (Current Configuration)
Accuracy: 36.0% (18/50)
Configuration: top_k=20, M1=True, Rerank=True

TOP-K OPTIMIZATION SWEEP
Top-K    Accuracy  Time     Facts Extr  Facts Sel  Filter %
------------------------------------------------------------------------
5        32.0%     1.2s     45.3        3.8        91.6%
10       34.0%     1.4s     89.7        4.2        95.3%
20       36.0%     1.7s     126.8       3.7        97.1%     ←
30       38.0%     1.9s     154.2       3.9        97.5%     ✓
50       38.0%     2.3s     203.5       4.1        98.0%
100      36.0%     3.1s     387.9       4.3        98.9%

Analysis:
  📊 PLATEAU PATTERN: Accuracy plateaus after initial increase
  → Diagnosis: Answer usually in first few sentences
  → Bottleneck: EXTRACTION (we have the answer but don't extract it correctly)
  → Recommendation: Fix extraction patterns

Optimal top-k: 30 (38.0% accuracy)

ABLATION TESTS
M1 Filter:
  With M1:    36.0%
  Without M1: 32.0%
  Difference: -4.0%
  ✓ M1 is HELPING accuracy (filtering noise effectively)
  → Recommendation: Keep M1 enabled

Neural Reranker:
  With reranker:    36.0%
  Without reranker: 34.0%
  Difference: -2.0%
  = Reranker has minimal impact on accuracy
  → Recommendation: Consider disabling for speed

RECOMMENDATIONS
1. CHANGE TOP-K: Use top_k=30 instead of 20
   Expected improvement: +2.0% accuracy

2. PRIMARY BOTTLENECK: EXTRACTION
   17 questions fail even with answer in retrieved set
   → Recommendation: Fix extraction patterns (object verification, definition patterns)

3. PERFORMANCE OPTIMIZATION: RETRIEVAL
   Retrieval consumes 88.9% of total time
   → Recommendation: Cache Kuzu queries or optimize indexes
================================================================================
```

### When to Use

**Use full suite:**
- Before major code changes (establish baseline with context)
- After major code changes (detect regressions and validate improvements)
- When investigating performance issues (automatic bottleneck detection)
- For regular benchmarking (weekly/monthly regression testing)

**Use single evaluation:**
- Quick iteration during development
- Testing specific configuration changes
- When you know what you're testing

### Integration with Development Workflow

```bash
# Week 1: Baseline with full context
python scripts/evaluate_full_suite.py --output results/week1_baseline/

# Week 2: After improving extraction patterns
python scripts/evaluate_full_suite.py --output results/week2_extraction_fix/

# Compare
python scripts/analyze_evaluation_results.py \
  results/week1_baseline/baseline.json \
  results/week2_extraction_fix/baseline.json
```

### Interpreting Top-K Patterns

The suite automatically detects these patterns:

**Plateau Pattern:**
```
Accuracy: ___/‾‾‾‾‾
```
→ Answer usually in top 5-10 sentences
→ **Bottleneck: EXTRACTION** (we have it but don't extract it)
→ Fix: Extraction patterns (object verification, definition patterns)

**Linear Growth:**
```
Accuracy:     /‾
         ___/
       /
```
→ Answer often ranked deep (20-50+)
→ **Bottleneck: RANKING** (answer buried too low)
→ Fix: Reranking or query expansion

**Peak Then Decline:**
```
Accuracy:  /\
         /  \___
```
→ Adding noise at high top-k
→ **Bottleneck: M1 FILTERING** (overwhelmed by noise)
→ Fix: Improve M1 filter or extraction patterns

**Flat Line:**
```
Accuracy: ‾‾‾‾‾‾‾‾‾
```
→ Retrieval broken (answer not in any top-k)
→ **Bottleneck: RETRIEVAL** (query expansion needed)
→ Fix: Drastically improve query expansion

See also: `docs/TOP_K_OPTIMIZATION_EXPERIMENT.md` for detailed explanation of top-k optimization theory.
