# RAG Evaluation Snapshots

This directory contains timestamped snapshots of RAG system evaluation metrics.

## Purpose

Track progress over time by comparing metrics across snapshots. Each snapshot captures:
- Binary accuracy (correct/incorrect)
- Granular scores (retrieval, extraction, alignment, robustness)
- Component-level breakdowns

## Snapshot Naming

Format: `YYYYMMDD_HHMMSS_[name].json`

Examples:
- `20260201_153450_baseline_v1.json` - Official baseline v1
- `20260201_160000_after_extraction_fix.json` - After fixing extraction bug
- `20260202_120000_new_reranker.json` - After retraining reranker

## Usage

### View Current Metrics
```bash
python scripts/track_evaluation_progress.py
```

### Save New Snapshot
```bash
# Auto-timestamped
python scripts/track_evaluation_progress.py --save

# With descriptive name
python scripts/track_evaluation_progress.py --save --name "after_my_fix"
```

### Compare All Snapshots
```bash
python scripts/track_evaluation_progress.py --compare
```

Shows side-by-side comparison of last 5 snapshots with deltas.

## Snapshot History

### Baseline v1 (2026-02-01 15:34)
**File**: `20260201_153450_baseline_v1.json`

**Metrics**:
- Granular: 0.487 (R=0.967, E=0.000, A=0.503, B=0.000)
- Binary: 0% (0/30 correct)

**State**:
- Retrieval excellent (96.7% in top-10)
- Extraction bottleneck (0% exact matches)
- After extraction person/agent improvements (commit 15316ce)

**Next priorities**: Fix extraction for definitions, WHAT questions, WHERE questions

---

## Adding New Snapshots

After making improvements:

1. Re-run evaluation:
   ```bash
   python scripts/evaluate_rag_test_set.py
   ```

2. Save snapshot with descriptive name:
   ```bash
   python scripts/track_evaluation_progress.py --save --name "after_fix_X"
   ```

3. Force add and commit (data/ is gitignored):
   ```bash
   git add -f data/evaluation/snapshots/*.json
   git commit -m "Benchmark: After [description]"
   ```

4. Update this README with new entry in "Snapshot History"

## Metrics Explained

### Granular Score (0-1.000)
Weighted combination: `0.40×R + 0.30×E + 0.20×A + 0.10×B`

- **R (Retrieval)**: Where does answer appear in retrieved docs?
  - Top-1=1.0, Top-2=0.9, Top-3=0.8, Top-5=0.6, Top-10=0.4

- **E (Extraction)**: Was correct answer extracted?
  - Exact=1.0, Fuzzy=0.5, Wrong/None=0.0

- **A (Alignment)**: From which rank was answer extracted?
  - Top-1=1.0, Top-2=0.9, Top-3=0.8, Top-5=0.6, Top-10=0.4

- **B (Robustness)**: How many top-5 docs have answer?
  - 3+=1.0, 2=0.6, 1=0.0

### Binary Accuracy
Simple pass/fail:
- **Correct**: Exact or fuzzy match to expected answer
- **Partial**: Contains some expected keywords
- **Incorrect**: Wrong or no extraction

## See Also

- `scripts/track_evaluation_progress.py` - Snapshot tool
- `scripts/evaluate_rag_test_set.py` - Run full evaluation
- `docs/BENCHMARKING.md` - Complete benchmarking guide
- `data/evaluation/rag_test_set.jsonl` - 30 test questions
