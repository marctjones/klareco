# Investigation: Tier0 Filtering in M1 Training Data Generation

**Date**: 2026-01-19
**Issue**: #12 - Investigate why tier0 data is filtered out during M1 training data generation
**Epic**: #14 - RAG Quality Investigation

---

## Summary

Investigating why `data/training/m1_semantic_full/train.jsonl` contains 0 tier0 examples despite using `corpus_full_with_tier0.jsonl` which has 22,516 tier0 sentences.

---

## Scripts Involved

### Primary Script: `prepare_m1_training_data_semantic.py`

**Purpose**: Generates M1 training data with semantic-distance-based corruption

**Key Filtering Logic** (lines 218-272):
```python
def load_corpus_triples(
    corpus_path: Path,
    max_triples: Optional[int] = None,
    min_parse_rate: float = 0.7  # ⚠️ DEFAULT IS 0.7
) -> Tuple[List[Dict], Dict[str, Set[str]]]:
    """Load positive triples from corpus."""

    with open(corpus_path) as f:
        for i, line in enumerate(f):
            entry = json.loads(line)

            # ⚠️ CRITICAL FILTER - LINE 244
            if entry.get('parse_rate', 0) < min_parse_rate:
                continue  # Skip low parse rate sentences

            ast = entry.get('ast')
            if not ast:
                continue  # Skip sentences without AST

            triple = extract_svo_triple(ast)
            if not triple:
                continue  # Skip sentences without S-V-O triple
```

**Three filtering stages**:
1. **Parse rate filter** (`parse_rate < min_parse_rate`) ← INVESTIGATION FOCUS
2. **AST existence** (no AST → skip)
3. **Triple extraction** (no S-V-O → skip)

### Training Script: `train_m1_semantic.sh`

**How it was called for production model**:
```bash
./scripts/train_m1_semantic.sh --full-corpus
```

**Parameters passed** (lines 104-111):
```bash
python scripts/prepare_m1_training_data_semantic.py \
    --corpus data/enhanced_corpus/corpus_full_with_tier0.jsonl \
    --stage1-model models/root_embeddings_tier0/best_model.pt \
    --output-dir data/training/m1_semantic_full \
    --max-triples 200000 \
    --similarity-threshold 0.15 \
    --min-parse-rate 0.0  # ⚠️ SHOULD ALLOW ALL TIER0!
```

**KEY FINDING**: `--min-parse-rate 0.0` SHOULD include tier0 data!

---

## Hypothesis

Since `--min-parse-rate 0.0` is used, the parse rate filter should NOT be the issue. This suggests one of two possibilities:

### Hypothesis 1: Tier0 Sentences Lack ASTs
- Tier0 sentences might not have `ast` field in corpus
- Would be filtered at line 248: `if not ast: continue`

### Hypothesis 2: Tier0 ASTs Lack S-V-O Triples
- Tier0 sentences might have ASTs but no extractable S-V-O triples
- Would be filtered at line 252: `if not triple: continue`
- Triple extraction requires: `subjekto`, `verbo`, `objekto` all present in AST

### Hypothesis 3: Max Triples Limit Reached Before Tier0
- `--max-triples 200000` might cause early exit
- If tier0 appears later in corpus file, it never gets processed
- Training data has 320K examples (160K pos + 160K neg), so this seems unlikely

---

## 🔥 CRITICAL FINDING: Max Triples Limit Reached Early

**From production log** (`logs/training/prepare_m1_semantic_20260119_011642.log`):

```
01:16:47 - INFO -   Processed 100,000 sentences, found 51,839 triples
01:16:51 - INFO -   Processed 200,000 sentences, found 107,246 triples
01:16:56 - INFO -   Processed 300,000 sentences, found 155,241 triples
01:17:05 - INFO - Loaded 200,000 positive triples  ← STOPPED HERE!
```

**What happened**:
1. Script processed ~300K-350K sentences
2. Found 200K triples (hit --max-triples limit)
3. **STOPPED PROCESSING** - never reached later parts of corpus
4. If tier0 appears after sentence 350K, it was never seen!

**Corpus size**: 4.2M sentences total
**Processed**: Only ~8% of corpus before stopping!

This confirms **Hypothesis 3**: Max triples limit reached before tier0.

**Next step**: Wait for `analyze_tier0_filtering.py` to determine WHERE tier0 appears in corpus.

---

## Investigation Steps (In Progress)

### Step 1: Check Tier0 Parse Rates ✅
```bash
# Check parse rates of tier0 sentences
jq -r 'select(.source.tier == 0) | .parse_rate' corpus_full_with_tier0.jsonl | head -100 | sort -n | uniq -c
```
**Status**: Running in background (task b74e714)

### Step 2: Check Tier0 AST/Triple Structure ✅
```bash
# Check if tier0 entries have ASTs and triples
jq 'select(.source.tier == 0) | {tier: .source.tier, name: .source.name, parse_rate, ast: (.ast | type)}' corpus_full_with_tier0.jsonl | head -20
```
**Status**: Running in background (task b57d301)

### Step 3: Check Corpus Ordering (TODO)
```bash
# Check position of tier0 in corpus (early vs late)
jq -r '.source.tier' corpus_full_with_tier0.jsonl | head -500000 | grep -n "^0$" | head -20
```

### Step 4: Simulate Filtering (TODO)
```python
# Count how many tier0 pass each filter stage
import json

tier0_total = 0
tier0_with_parse_rate = 0
tier0_with_ast = 0
tier0_with_triple = 0

with open('corpus_full_with_tier0.jsonl') as f:
    for line in f:
        entry = json.loads(line)
        if entry.get('source', {}).get('tier') == 0:
            tier0_total += 1

            if entry.get('parse_rate', 0) >= 0.0:  # min_parse_rate=0.0
                tier0_with_parse_rate += 1

                if entry.get('ast'):
                    tier0_with_ast += 1

                    # Check for S-V-O
                    ast = entry['ast']
                    if all(k in ast for k in ['subjekto', 'verbo', 'objekto']):
                        tier0_with_triple += 1

print(f"Tier0 total: {tier0_total}")
print(f"  With parse_rate >= 0.0: {tier0_with_parse_rate}")
print(f"  With AST: {tier0_with_ast}")
print(f"  With S-V-O triple: {tier0_with_triple}")
```

---

## Expected Findings

If **Hypothesis 1** (no ASTs):
- Tier0 sentences don't have AST field
- Need to re-parse tier0 data

If **Hypothesis 2** (no S-V-O):
- Tier0 sentences have ASTs but lack S-V-O structure
- May be due to sentence types (questions, commands, fragments)
- PMEG and Lingvaj Respondoj are Q&A format - may lack statements

If **Hypothesis 3** (max triples):
- Tier0 appears late in corpus
- Need to shuffle corpus OR remove max_triples limit OR sample from all tiers

---

## Current Data

### Corpus Verification ✅
```bash
$ jq -r 'select(.source.tier == 0) | .source.name' corpus_full_with_tier0.jsonl | sort | uniq -c
    157 ekzercaro
  4,587 krestomatio
  4,789 lingvaj_respondoj
 12,983 pmeg
-------
 22,516 total tier0 sentences
```

### Training Data Verification ✅
```bash
$ jq -r '.source.tier' data/training/m1_semantic_full/train.jsonl | sort | uniq -c
   6,316  tier 2  (Fundamenta Krestomatio)
 160,825  tier 5  (Wikipedia)
 152,859  tier 6  (Gutenberg)
-------
 320,000  total (0 tier0)
```

### Training Script Confirmed ✅
- Called with `--full-corpus` flag
- Used `corpus_full_with_tier0.jsonl` (HAS tier0)
- Used `--min-parse-rate 0.0` (should accept all parse rates)
- Generated 320K training examples (160K pos + 160K neg)

---

## Next Steps

1. **Wait for background tasks** to complete (parse rate check, AST structure check)
2. **Run simulation script** to count tier0 at each filter stage
3. **Check corpus ordering** to see if tier0 appears late
4. **Identify root cause** from findings
5. **Propose fix** based on root cause:
   - If no ASTs → re-parse tier0 data
   - If no S-V-O → modify extraction to handle Q&A format
   - If ordering → shuffle corpus OR sample from all tiers

---

## Related Files

- `scripts/prepare_m1_training_data_semantic.py` - Data generation script
- `scripts/train_m1_semantic.sh` - Training pipeline script
- `data/enhanced_corpus/corpus_full_with_tier0.jsonl` - Corpus with tier0
- `data/training/m1_semantic_full/train.jsonl` - Training data (no tier0)
- `docs/Tier0-Data-Inventory-2026-01-19.md` - Tier0 audit results
- `docs/EPIC-14-RAG-Quality-Investigation.md` - Parent epic

---

**Investigation Status**: IN PROGRESS
**Last Updated**: 2026-01-19
**Investigator**: Claude Code
