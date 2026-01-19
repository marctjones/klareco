# Tier0 Root Cause Found - 2026-01-19

**Issue**: #12 - Investigate why tier0 data is filtered out during M1 training data generation
**Epic**: #14 - RAG Quality Investigation
**Status**: ROOT CAUSE IDENTIFIED ✅

---

## Executive Summary

**ROOT CAUSE**: The training data generation script hit the `--max-triples 200000` limit after processing only ~350K sentences (8% of the 4.2M sentence corpus), then stopped. If tier0 sentences appear later in the corpus, they were never reached.

**EVIDENCE**: Production log shows script stopped at:
- 300K sentences processed → 155K triples found
- 200K triples reached → **STOPPED** (never processed remaining 92% of corpus)

**IMPACT**: Zero tier0 examples in M1 training data despite corpus containing 22,516 tier0 sentences.

---

## Timeline of Discovery

### 1. Initial Observation (Jan 19, 09:00)
```bash
# M1 training data has ZERO tier0
$ jq -r '.source.tier' data/training/m1_semantic_full/train.jsonl | sort | uniq -c
   6,316  tier 2
 160,825  tier 5
 152,859  tier 6
-------
 320,000  total (0 tier0!)
```

### 2. Corpus Verification (Jan 19, 09:30)
```bash
# But corpus DOES have tier0!
$ jq -r 'select(.source.tier == 0)' corpus_full_with_tier0.jsonl | wc -l
22,516  ← Tier0 IS in corpus!
```

### 3. Script Analysis (Jan 19, 10:00)
Examined `prepare_m1_training_data_semantic.py`:
- Three filter stages: parse_rate, AST existence, S-V-O extraction
- **Key parameter**: `--min-parse-rate 0.0` (should accept all)
- **Key parameter**: `--max-triples 200000` ⚠️

### 4. Metadata Confirmation (Jan 19, 10:15)
```json
{
  "corpus": "corpus_full_with_tier0.jsonl",  ✅ Correct corpus
  "min_parse_rate": 0.0,                     ✅ Accepts all
  "max_triples": 200000,                     ⚠️ Limit reached!
  "total_examples": 400000
}
```

### 5. **SMOKING GUN** - Production Log (Jan 19, 10:20)
```
logs/training/prepare_m1_semantic_20260119_011642.log:

01:16:47 - Processed 100,000 sentences, found 51,839 triples
01:16:51 - Processed 200,000 sentences, found 107,246 triples
01:16:56 - Processed 300,000 sentences, found 155,241 triples
01:17:05 - Loaded 200,000 positive triples  ← STOPPED HERE
```

**Key finding**: Script processed only ~350K sentences before hitting 200K triple limit.

---

## Root Cause Analysis

### Why This Happened

```python
# In prepare_m1_training_data_semantic.py (line 236):
if max_triples and len(triples) >= max_triples:
    break  # Stop processing corpus once limit reached
```

**Process**:
1. Script iterates through `corpus_full_with_tier0.jsonl` (4.2M sentences)
2. Extracts S-V-O triples from each sentence
3. Stops when `len(triples) >= max_triples` (200K)
4. Only processed 350K / 4,200K sentences (8.3%)
5. **If tier0 appears after sentence 350K → never seen!**

### Critical Question ✅ ANSWERED

**WHERE does tier0 appear in the corpus?**

**CONFIRMED by analysis** (`analyze_tier0_filtering.py` completed):
- **Tier0 starts at line 4,208,112 (99.5% through corpus)**
- **Tier0 ends at line 4,230,627**
- **Total corpus: 4,230,628 lines**
- **Script stopped at ~350K sentences (8.3% of corpus)**
- **Result: Tier0 was NEVER REACHED**

This definitively proves the root cause!

---

## Evidence Summary

| Evidence | Finding | Implication |
|----------|---------|-------------|
| Training data | 0 tier0 / 320K examples | Tier0 completely missing |
| Corpus | 22,516 tier0 sentences | Tier0 data exists |
| Script parameters | `--max-triples 200000` | Hard limit on processing |
| Production log | Stopped at 350K sentences | Only processed 8% of corpus |
| Metadata | Used correct corpus | Not a wrong-file issue |
| Filter settings | `min_parse_rate 0.0` | Not a parse rate issue |

**Conclusion**: Tier0 appears AFTER the 350K sentence cutoff point in the corpus.

---

## Proposed Solutions

### Solution 1: Stratified Sampling (RECOMMENDED)
Sample triples from ALL tiers, ensuring tier0 representation.

```python
def load_corpus_triples_stratified(
    corpus_path: Path,
    max_triples: int,
    tier_weights: Dict[int, float] = {0: 0.1, 2: 0.1, 5: 0.4, 6: 0.4}
) -> List[Dict]:
    """
    Load triples with stratified sampling across tiers.

    Ensures tier0 (10% target) is included even if it appears late.
    """
    # First pass: count triples per tier
    tier_triples = defaultdict(list)

    for line in open(corpus_path):
        entry = json.loads(line)
        tier = entry.get('source', {}).get('tier')
        triple = extract_svo_triple(entry.get('ast'))

        if triple:
            tier_triples[tier].append(triple)

    # Second pass: sample from each tier according to weights
    sampled = []
    for tier, weight in tier_weights.items():
        n_sample = int(max_triples * weight)
        sampled.extend(random.sample(tier_triples[tier], min(n_sample, len(tier_triples[tier]))))

    return sampled
```

**Pros**:
- Guarantees tier0 inclusion (e.g., 10% = 20K tier0 examples)
- Maintains tier diversity
- Fair representation of all data sources

**Cons**:
- Requires two passes through corpus (slower)
- More complex code

### Solution 2: Shuffle Corpus (SIMPLE)
Randomize corpus order before processing.

```bash
# Shuffle corpus before training
jq -c '.' corpus_full_with_tier0.jsonl | shuf > corpus_full_shuffled.jsonl

# Then train with shuffled corpus
./scripts/train_m1_semantic.sh --corpus corpus_full_shuffled.jsonl
```

**Pros**:
- Simple to implement
- Ensures tier0 has chance to appear in first 350K
- Works with existing code

**Cons**:
- Still no guarantee of tier0 (depends on random shuffle)
- Loses any meaningful corpus ordering
- `shuf` on 18GB file is slow and memory-intensive

### Solution 3: Remove Max Triples Limit
Process entire corpus without limit.

```bash
# Train without max_triples
python prepare_m1_training_data_semantic.py \
    --max-triples 0  # Process all 4.2M sentences
```

**Pros**:
- Guarantees ALL tier0 included
- Simple (just change parameter)

**Cons**:
- Would generate ~8.4M training examples (4.2M pos + 4.2M neg)
- Much longer training time
- May include low-quality examples

### Solution 4: Multi-Pass with Tier Priority
Process tier0 first, then fill remaining quota with other tiers.

```python
def load_corpus_triples_prioritized(
    corpus_path: Path,
    max_triples: int,
    priority_tiers: List[int] = [0, 2]
) -> List[Dict]:
    """Load tier0 and tier2 first, then fill remaining with tiers 5,6."""

    triples_by_tier = {tier: [] for tier in range(7)}

    # Collect all triples by tier
    for line in open(corpus_path):
        entry = json.loads(line)
        tier = entry.get('source', {}).get('tier')
        triple = extract_svo_triple(entry.get('ast'))

        if triple:
            triples_by_tier[tier].append(triple)

    # Prioritize tier0 and tier2
    result = []
    for tier in priority_tiers:
        result.extend(triples_by_tier[tier])

    # Fill remaining quota with other tiers
    remaining = max_triples - len(result)
    for tier in [5, 6]:
        n_sample = remaining // 2
        result.extend(random.sample(triples_by_tier[tier], n_sample))

    return result[:max_triples]
```

**Pros**:
- Guarantees ALL tier0 included
- Maintains quality focus (tier0 and tier2 are authoritative)
- Still respects max_triples limit

**Cons**:
- Requires full corpus scan (slower)
- More complex implementation

---

## Recommendation ✅ IMPLEMENTED

**CHOSEN SOLUTION: Tier Priority (Solution 4)**

Implementation complete:
- Created `scripts/prepare_m1_training_data_tier_priority.py` (two-pass processing)
- Created `scripts/train_m1_semantic_tier_priority.sh` (training pipeline)

**Target distribution**:
- 10% tier0 (~20K examples) - authoritative grammar (ALL tier0 included)
- 10% tier2 (~20K examples) - Fundamento (ALL tier2 included)
- 40% tier5 (~80K examples) - Wikipedia (sampled)
- 40% tier6 (~80K examples) - Gutenberg (sampled)

**Advantages of tier priority over other solutions**:
- Guarantees ALL tier0 included (not just sampled)
- Maintains quality focus on tiers 0 and 2
- Respects max_triples limit
- Single corpus scan (no shuffling required)

---

## Expected Impact

### Before (Current)
- M1 accuracy: 86.37%
- Tier0 in training: 0 / 320K (0%)
- Sources: Wikipedia (50%), Gutenberg (48%), Krestomatio (2%)

### After (With Tier0)
- M1 accuracy: 87-88% (estimated +1-2 points)
- Tier0 in training: 20K / 320K (6-10%)
- Sources: Balanced across all quality tiers

**Why improvement expected**:
- Tier0 has authoritative grammar patterns
- Tier0 includes grammatical edge cases (PMEG)
- Tier0 provides high-quality Q&A pairs (Lingvaj Respondoj)
- More diverse training examples improves generalization

---

## Action Items

### Immediate (This Week)
- [x] **Issue #12**: Identify root cause ✅ DONE
- [x] **Tier Priority Implementation**: Created scripts ✅ DONE
  - Created `prepare_m1_training_data_tier_priority.py`
  - Created `train_m1_semantic_tier_priority.sh`
- [ ] **Issue #10**: Test tier priority implementation
- [ ] Train M1 with tier0-included data
- [ ] Compare accuracy: current (86.37%) vs. with-tier0

### Short-term (Next 2 Weeks)
- [ ] Retrain M1 with tier-prioritized data
- [ ] Validate tier0 is included in training data
- [ ] Update demo with new model
- [ ] Document tier priority strategy

### Documentation Updates
- [ ] Update `CLAUDE.md` with tier sampling strategy
- [ ] Add stratified sampling to training best practices
- [ ] Document tier weights rationale

---

## Related Issues

- **Issue #12**: Investigate tier0 filtering ← THIS ISSUE (RESOLVED)
- **Issue #10**: Fix M1 training data generation to use tier0 (NEXT)
- **Epic #14**: RAG Quality Investigation (IN PROGRESS)
- **Issue #13**: Rename misnamed datasets
- **Issue #15**: Investigate M1 scoring issues
- **Issue #16**: Investigate retrieval ranking

---

**Report Date**: 2026-01-19
**Status**: ROOT CAUSE IDENTIFIED ✅ + SOLUTION IMPLEMENTED ✅
**Next Step**: Test tier priority implementation, then train M1 (Issue #10)
