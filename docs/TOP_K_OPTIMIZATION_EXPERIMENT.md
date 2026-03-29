# Top-K Optimization Experiment

## Question

**What is the optimal number of sentences to use for answer generation?**

Currently we default to top-20, but is that better or worse than 5, 10, 30, 50, or 100?

## Hypothesis: There's a Sweet Spot

Too few sentences (5):
- ✅ Fast
- ✅ Less noise for M1 to filter
- ❌ Might miss relevant information

Too many sentences (100):
- ❌ Slow
- ❌ More noise (extraction overwhelmed)
- ❌ M1 filter has to work harder
- ✅ More likely to contain answer

Sweet spot (??):
- ✅ Fast enough
- ✅ Contains answer
- ✅ Not too much noise

## What This Experiment Will Reveal

### 1. Accuracy Curve

**If accuracy increases linearly with top-k:**
```
Accuracy
   ^
   |          /
   |        /
   |      /
   |    /
   |  /
   +----------> top-k
   5  10  20  30  50  100
```
→ **Diagnosis**: Retrieval/ranking is weak - answer often ranked low
→ **Fix**: Improve query expansion or reranking

**If accuracy plateaus early:**
```
Accuracy
   ^
   |  ___________
   | /
   |/
   |
   |
   +----------> top-k
   5  10  20  30  50  100
```
→ **Diagnosis**: Retrieval/ranking is good - answer usually in top 10
→ **Fix**: Focus on extraction (we have the answer but don't extract it)

**If accuracy peaks then decreases:**
```
Accuracy
   ^
   |    __
   |   /  \
   |  /    \
   | /      \
   |/        \_
   +----------> top-k
   5  10  20  30  50  100
```
→ **Diagnosis**: Adding noise at high top-k - extraction confused
→ **Fix**: M1 filter might be too weak, or extraction needs better patterns

### 2. Noise Analysis

Track: **Facts extracted vs facts selected** at each top-k

**If filter rate increases with top-k:**
- top-k=5: Extract 20 facts → Select 4 (80% filtered)
- top-k=20: Extract 100 facts → Select 4 (96% filtered)
- top-k=100: Extract 500 facts → Select 4 (99.2% filtered)

→ **Diagnosis**: M1 is doing its job (filtering more noise as top-k grows)

**If filter rate stays constant:**
- top-k=5: Extract 20 facts → Select 4 (80% filtered)
- top-k=20: Extract 100 facts → Select 20 (80% filtered)
- top-k=100: Extract 500 facts → Select 100 (80% filtered)

→ **Diagnosis**: M1 is scaling linearly (constant signal-to-noise ratio)

**If filter rate decreases:**
→ **Diagnosis**: M1 filter is overwhelmed or miscalibrated

### 3. Question Type Differences

**WHO questions:** Might benefit from MORE sentences (person names buried deeper)
**HOW_MANY questions:** Might work with FEWER sentences (numeric facts usually prominent)
**WHEN questions:** Might need MORE sentences (temporal info often in context)

Track optimal top-k per question type.

### 4. Timing vs Accuracy Tradeoff

Calculate "value score" = accuracy / time

Find the point where:
- Small increase in top-k → Large increase in time
- Small increase in top-k → Tiny increase in accuracy

This is the **diminishing returns threshold**.

### 5. Retrieval Quality Check

Compare:
- Recall@5 at top-k=5 (what % have answer in top 5?)
- Recall@10 at top-k=10
- Recall@20 at top-k=20

If recall@20 is much higher than recall@10:
→ **Diagnosis**: Answer often ranked 11-20, reranking needs improvement

If recall@5 ≈ recall@10 ≈ recall@20:
→ **Diagnosis**: Answer usually in top 5, extraction is the bottleneck

## Running the Experiment

### Quick Test (3 values)
```bash
python scripts/experiment_top_k_optimization.py \
  --top-k-values 5 20 50 \
  --output results/top_k_quick/
```

### Full Test (6 values, ~30 minutes)
```bash
python scripts/experiment_top_k_optimization.py \
  --top-k-values 5 10 20 30 50 100 \
  --output results/top_k_full/
```

### Custom Test
```bash
python scripts/experiment_top_k_optimization.py \
  --top-k-values 3 7 15 40 80 \
  --output results/top_k_custom/
```

## Expected Output

### Summary Table
```
Top-K  Accuracy  Correct  Total Time  Facts Extracted  Facts Selected
-----  --------  -------  ----------  ---------------  --------------
5      32.0%     16/50    1.2s        45.3             3.8
10     34.0%     17/50    1.4s        89.7             4.2
20     36.0%     18/50    1.7s        126.8            3.7
30     38.0%     19/50    1.9s        154.2            3.9
50     38.0%     19/50    2.3s        203.5            4.1
100    36.0%     18/50    3.1s        387.9            4.3
```

### Diminishing Returns Analysis
```
top_k 5→10:  +2.0% accuracy, +0.2s time (efficiency: 10.0%/s) ✓
top_k 10→20: +2.0% accuracy, +0.3s time (efficiency: 6.7%/s) ✓
top_k 20→30: +2.0% accuracy, +0.2s time (efficiency: 10.0%/s) ✓
top_k 30→50: +0.0% accuracy, +0.4s time (efficiency: 0.0%/s) ✗
top_k 50→100: -2.0% accuracy, +0.8s time (REGRESSION) ✗✗
```

**Interpretation**: Optimal top-k = 30 (last point before diminishing returns)

### Noise Analysis
```
top_k=5:   Extract 45.3 facts → Select 3.8 facts (91.6% filtered)
top_k=10:  Extract 89.7 facts → Select 4.2 facts (95.3% filtered)
top_k=20:  Extract 126.8 facts → Select 3.7 facts (97.1% filtered)
top_k=30:  Extract 154.2 facts → Select 3.9 facts (97.5% filtered)
top_k=50:  Extract 203.5 facts → Select 4.1 facts (98.0% filtered)
top_k=100: Extract 387.9 facts → Select 4.3 facts (98.9% filtered)
```

**Interpretation**: M1 filtering rate increases with top-k (good - filtering more noise)

### Optimal Top-K by Question Type
```
WHO:       top_k=30 (20.0%)
WHAT:      top_k=20 (40.0%)
WHERE:     top_k=20 (70.0%)
WHEN:      top_k=50 (20.0%)
WHY:       top_k=50 (50.0%)
HOW:       top_k=10 (100.0%)
HOW_MANY:  top_k=5 (100.0%)
```

**Interpretation**:
- HOW/HOW_MANY: Work with few sentences (direct answers)
- WHEN/WHY: Need more sentences (context-dependent)
- WHO/WHAT/WHERE: Medium (20-30 sentences optimal)

## Possible Outcomes

### Outcome 1: "Sweet Spot at 20-30"
```
Accuracy peaks at top_k=20-30, then plateaus or decreases
```

**Conclusion**: Current default (20) is optimal or close to optimal
**Action**: Keep top_k=20, focus on extraction improvements

### Outcome 2: "More is Better"
```
Accuracy keeps increasing up to top_k=100
```

**Conclusion**: Answer often ranked low (position 30-100)
**Action**: Improve reranking OR increase top_k default

### Outcome 3: "Less is Better"
```
Accuracy peaks at top_k=5-10
```

**Conclusion**: Adding sentences adds noise, extraction is confused
**Action**: Improve extraction patterns OR strengthen M1 filtering

### Outcome 4: "No Difference"
```
Accuracy same at all top-k values
```

**Conclusion**: Retrieval is broken (answer not in any top-k)
**Action**: Improve query expansion drastically

## Integration with Corrected Analysis

This experiment addresses your excellent questions:

1. **"Why cutting off at sentence 5?"**
   - We're NOT cutting off - but this experiment tests if we SHOULD
   - Maybe 5 is actually optimal!

2. **"Is rank 6 a problem?"**
   - This experiment will tell us
   - If accuracy same at top_k=5 vs top_k=20, then rank 6-20 doesn't matter

3. **"Are more sentences adding noise?"**
   - Noise analysis section directly tests this
   - Track facts extracted vs selected at each top-k

4. **"Test if 10 relevant sentences but answer at #11 is a problem"**
   - If top_k=10 has much lower accuracy than top_k=20, then YES
   - If same accuracy, then NO (extraction finds answer from any of top 10)

## After the Experiment

Based on results, we'll know:
1. **Optimal top-k value** (or per question type)
2. **Whether extraction or ranking is the bottleneck**
3. **Whether M1 filtering is handling noise well**
4. **Where to focus improvement efforts**

Then we can proceed with:
- Priority 1: Fix extraction patterns (if plateau shows extraction is bottleneck)
- Priority 2: Improve ranking (if linear growth shows ranking is bottleneck)
- Priority 3: Adjust M1 threshold (if noise analysis shows filter issues)

## See Also

- `docs/CORRECTED_ANALYSIS.md` - Root cause analysis showing extraction is broken
- `docs/EVALUATION_FRAMEWORK.md` - Comprehensive evaluation metrics
- `scripts/experiment_top_k_optimization.py` - Experiment script
