# Semantic Query Expansion Findings

**Date:** 2026-03-31
**Context:** Step 2 of semantic annotation expansion roadmap

## Summary

Tested semantic query expansion using verb class taxonomy for extractive QA. **Result: No improvement** in accuracy (0/10 with or without semantic expansion).

## What We Built

### 1. Semantic Query Expander (`klareco/rag/semantic_query_expander.py`)
- Expands query roots using APARTENAS_AL_VERBA_KLASO relationships
- Finds semantic synonyms via verb class membership
- Example: `fond` (found) → all members of `kreado-26` (Creation) class

### 2. Integration into QA Pipeline
- Added `--use-semantic` flag to `evaluate_extractive_qa.py`
- Expands query before embedding-based expansion
- Limits expansion to 20 roots to prevent query explosion

## Annotation Quality Issues Discovered

### Problem: Frequency-Based Morphological Expansion Corrupted Database

**Initial State (After Full Pipeline):**
- 1,643 total annotations
- Method: Manual (23) + Gazetteer (161) + ReVo (143) + **Frequency-based (1,316)**

**Issue:** Frequency-based expansion used 3-character prefix matching, which created massive false positives:
- `fond` → `kreado-26` (correct)
- **But also:** `tri`, `trib`, `trinidad`, `trink`, `el`, `post`, `lok` → `kreado-26` (WRONG!)
- Result: Top-level class `kreado-26` had 215 members, most unrelated

**Root Cause:** Morphological similarity (shared prefix) ≠ semantic similarity

### Solution: Complete Reset to High-Quality Annotations Only

**Actions Taken:**
1. Created `remove_frequency_based_annotations.py` - Deleted 545 polluted top-level annotations
2. Created `reset_semantic_annotations.py` - Deleted ALL annotations (1,127 total)
3. Reloaded ONLY high-quality annotations:
   - Manual (23 roots) - Hand-curated from phase_0 files
   - Gazetteer (109 verbs + 52 entities) - From verb class examples + place/person lists
   - ReVo expansion (122 verbs) - Via synonym chains (depth 1-2)

**Final State:**
- 239 verb annotations (high-quality, clean)
- 0 entity annotations (persistence issue, need debugging)
- Down from 1,643 polluted annotations

## Evaluation Results

### Initial Test: 10 WHO Questions
- Questions 1-10 from 50-question test set
- All WHO questions about Zamenhof and Esperanto
- Example: "Kiu fondis Esperanton?" (Who founded Esperanto?)

| Configuration | Accuracy | Notes |
|---------------|----------|-------|
| Baseline (no semantic) | 0/10 (0.0%) | No semantic expansion |
| With semantic expansion | 0/10 (0.0%) | Using cleaned annotations |

### Follow-up Test: 5 WHAT Questions (General Trivia)
- Questions 11-15 from test set
- Definitional questions ("What is X?")
- Examples: "Kio estas hundo?" (What is a dog?), "Kio estas libro?" (What is a book?)

| Configuration | Accuracy | Notes |
|---------------|----------|-------|
| Baseline (no semantic) | **1/5 (20.0%)** | Correctly answered "hundo" → "besto" |
| With semantic expansion | **1/5 (20.0%)** | Same result, no improvement |

**Conclusion:** Semantic expansion provides **zero benefit** across both question types tested.

## Root Cause Analysis

Semantic expansion didn't help because the core retrieval/extraction pipeline has fundamental issues:

1. **Retrieval Problem**: Not finding relevant sentences about Zamenhof
   - Query: "Kiu fondis Esperanton?" (Who founded Esperanto?)
   - Expanded roots: `fond` → `[fond, kre, produk, far, skrib, kompoz, ...]`
   - But still retrieves 0 relevant sentences

2. **Extraction Problem**: Even when retrieving sentences, answer extraction fails
   - Returns empty answers for most questions
   - Doesn't identify "Zamenhof" as the answer span

3. **Query Expansion Too Noisy**: Even cleaned annotations expand to unrelated roots
   - `fond` expands to `el`, `lok`, `trink` (function words + unrelated verbs)
   - This is because even fine-grained classes have 10-100 members
   - Many members added via ReVo synonym expansion are not truly semantic synonyms

## Recommendations

### Option A: Fix Core Retrieval/Extraction (Highest Priority)
The 0/10 accuracy suggests fundamental problems:
- Whoosh FTS retrieval may not be finding relevant sentences
- AST role-based retrieval may be too restrictive
- Answer extraction may be failing to identify answer spans

**Recommended:** Debug retrieval first (check if relevant sentences are being retrieved at all)

### Option B: Improve Annotation Quality (Medium Priority)
Even cleaned annotations have noise:
- Fine-grained classes still too large (10-100 members each)
- ReVo synonym chains propagate semantic drift
- Need manual review of verb class membership

**Recommended:** Only use manual + gazetteer annotations (132 roots), skip ReVo expansion

### Option C: Abandon Semantic Expansion (Low Priority)
If retrieval can't find relevant documents:
- Semantic expansion won't help (can't expand to better roots if retrieval is broken)
- Focus on improving core retrieval algorithm instead
- Consider alternative approaches (e.g., dense retrieval, re-ranking)

## Next Steps

**BEFORE continuing with semantic expansion:**

1. **Debug retrieval**:
   ```bash
   python scripts/demo_extractive_qa.py "Kiu fondis Esperanton?" --verbose
   # Check: Are ANY relevant sentences retrieved?
   # Check: Do retrieved sentences mention "Zamenhof"?
   ```

2. **If retrieval works, debug extraction**:
   ```bash
   # Manually check if answer extraction identifies "Zamenhof" from retrieved sentences
   ```

3. **If both work, then revisit semantic expansion**:
   - But likely the issue is retrieval/extraction, not query expansion

## Files Created

| File | Purpose |
|------|---------|
| `klareco/rag/semantic_query_expander.py` | Semantic query expansion using verb class taxonomy |
| `scripts/remove_frequency_based_annotations.py` | Remove polluted top-level class annotations |
| `scripts/reset_semantic_annotations.py` | Complete reset to high-quality annotations only |
| `docs/SEMANTIC_EXPANSION_FINDINGS.md` | This document |

## Status

- ✅ **Step 1 COMPLETE:** Expand annotations to 1,000+ (achieved 1,643, but polluted)
- ⚠️ **Step 2 IN PROGRESS:** Test extractive QA with semantic expansion (0/10 accuracy, no benefit)
- 🔴 **BLOCKED:** Core retrieval/extraction broken, must fix before continuing

**Recommendation:** Pause semantic expansion work and focus on fixing core QA pipeline first.
