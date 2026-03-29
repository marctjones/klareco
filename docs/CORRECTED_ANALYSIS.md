# Corrected Analysis: The Real Problem is Extraction, Not Ranking

## My Original Analysis Was COMPLETELY WRONG

I incorrectly diagnosed ranking as the root cause. You were absolutely right to challenge me with:

1. "Why are you cutting things off at sentence 5?" → We DON'T - extraction looks at all 20!
2. "Is it really a problem that a relevant sentence is sentence 6?" → NO - we process top 20!
3. "Wouldn't it be better to test the reranker based on how well it is judging relevance rather than where sentences end up in rankings?" → YES - I confused the reranker's job!

## The Smoking Gun: Failures with Answer at Rank #1

**Critical Finding:**
- 9 questions FAIL even though answer is at rank 1-3
- Extraction processes all 20 sentences, including rank #1
- But generates COMPLETELY WRONG answers

### Example: Q1 - "Kiu fondis Esperanton?" (Who founded Esperanto?)

**Expected:** zamenhof, ludovic, lazaro
**Answer rank:** 1 (FIRST SENTENCE!)
**Facts extracted:** 24
**Facts selected:** 3

**Generated answer:**
```
El tio el 1991 oni fondis GIL kiel asocio kiu uzis la germanan...
La 13-an de marto 1914 en Budapeŝto fondiĝis Asocio de Internaci...
```

**Problem:** Extraction found:
- ❌ "oni fondis GIL" (they founded GIL) - WRONG subject
- ❌ "fondiĝis Asocio" (was founded Association) - WRONG object

**Should have found:**
- ✅ "Zamenhof fondis Esperanton" (Zamenhof founded Esperanto)

The extraction is matching the VERB ("fondis") but NOT linking it to the correct OBJECT ("Esperanton"). It's finding OTHER things that were founded.

### More Examples:

**Q2:** "Kiu kreis Esperanton?" → Generates same wrong answer about GIL
**Q4:** "Kiu verkis la Fundamenton?" → Generates answer about "Pio 12-a publikigis..." (wrong document!)
**Q11:** "Kio estas Esperanto?" → Generates unrelated text, not a definition
**Q14:** "Kio estas hundo?" → Generates a story about a dog, not a definition

## The Real Problem: Extraction Pattern Mismatch

**What's broken:**
```
Query: "Kiu fondis Esperanton?"
       ↓
Question analysis: WHO + verb="fondis" + object="Esperanton"
       ↓
Extraction pattern: Find ANY mention of "fondis"
       ↓
Problem: Doesn't verify object matches "Esperanto"!
       ↓
Result: Extracts "oni fondis GIL" (wrong object)
```

**What it should do:**
```
Query: "Kiu fondis Esperanton?"
       ↓
Question analysis: WHO + verb="fondis" + object="Esperanton"
       ↓
Extraction pattern: Find PERSON who "fondis" specifically "Esperanton"
       ↓
Result: Extract "Zamenhof fondis Esperanton"
```

## Data: Extraction is the Bottleneck, Not Ranking

### Key Statistics:

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Questions with answer in top 20 | 31/50 (62%) | Retrieval working moderately |
| Success rate when answer accessible | 14/31 (45%) | **Extraction failing 55% of the time** |
| Failures with answer at rank 1-3 | 9 | **Ranking is NOT the problem** |
| Failures with answer at rank 6-20 | 8 | Still accessible (we process top 20) |
| Failures with answer not retrieved | 19 | Retrieval/query expansion problem |

**Critical Insight:**
- If answer is in top 20, extraction processes it
- 45% success rate even when answer is accessible = **extraction is broken**
- Fixing ranking won't help if answer is already at rank #1 and we still fail!

## Corrected Priority Order

### Priority 1: Fix Extraction Patterns (Expected +20% accuracy)

**Problem:** Extraction matches verb but doesn't verify object.

**What to fix:**

1. **WHO questions with action verb:**
   ```python
   Query: "Kiu [VERB] [OBJECT]?"

   Current extraction: Find ANY mention of VERB
   Fixed extraction: Find PERSON who performed VERB on OBJECT

   Example:
   - Query: "Kiu fondis Esperanton?"
   - Must match: PERSON + "fondis" + "Esperanton"
   - Don't match: "oni fondis GIL" (wrong object)
   ```

2. **WHAT questions asking for definition:**
   ```python
   Query: "Kio estas [ENTITY]?"

   Current extraction: Extract ANY sentence mentioning ENTITY
   Fixed extraction: Extract "[ENTITY] estas [DEFINITION]" pattern

   Example:
   - Query: "Kio estas Esperanto?"
   - Must match: "Esperanto estas planlingvo"
   - Don't match: "...artikolo pri la estonteco de Esperanto..." (mention, not definition)
   ```

3. **Add object verification:**
   ```python
   def extract_who_action(self, query_verb, query_object, sentence_ast):
       # Find triples: (subject=PERSON, verb=query_verb, object=?)
       triples = self._extract_svo_triples(sentence_ast)

       for triple in triples:
           if triple.verb.root == query_verb:
               # NEW: Verify object matches query
               if query_object and not self._objects_match(triple.object, query_object):
                   continue  # Skip - wrong object

               return triple.subject  # Found correct match
   ```

**Expected improvement:**
- 9 questions with answer at rank 1-3 → ~7 should succeed (+7 correct)
- 8 questions with answer at rank 6-20 → ~5 should succeed (+5 correct)
- Total: +12 questions correct (18→30/50, 60% accuracy)

**Why this works:**
- Addresses root cause (extraction mismatch)
- No cascade risk (downstream stages see better facts)
- Simple logic change (no retraining needed)

### Priority 2: Improve Query Expansion (Expected +6% accuracy)

**Problem:** 19 questions don't retrieve answer (38% of failures).

**What to fix:**
- Add temporal expansion: YEAR → (en YEAR, jaro YEAR, dum YEAR)
- Add person expansion: "kiu" + verb → (person_name, NOUN-ist)
- Add causal expansion: "kial" → (pro, ĉar, kaŭze)

**Expected improvement:** +3-4 questions (30→34/50)

### Priority 3: Test Reranker Quality Independently

**Your excellent point:** Test reranker on its actual job (separating relevant/irrelevant), not on where it ranks THE answer.

**How to test:**
```python
# For each question:
# 1. Get top 20 sentences from reranker
# 2. Manually label: which are relevant? (not just which has THE answer)
# 3. Compute precision/recall at separating relevant/irrelevant
# 4. Current metric (MRR) assumes only ONE sentence is relevant - wrong!

def evaluate_reranker_quality(questions, retriever, reranker):
    """Test if reranker separates relevant from irrelevant."""
    for q in questions:
        sentences = retriever.retrieve(q, top_k=100)

        # Rerank
        reranked = reranker.rerank(sentences, q)

        # Label (manual or automatic)
        relevant = [s for s in reranked if contains_relevant_info(s, q)]

        # Compute precision/recall at top K
        top_20 = reranked[:20]
        relevant_in_top_20 = [s for s in top_20 if s in relevant]

        precision = len(relevant_in_top_20) / 20
        recall = len(relevant_in_top_20) / len(relevant)

        print(f"Precision@20: {precision:.2f}, Recall: {recall:.2f}")
```

**Why this matters:**
- Your insight: "if we have 10 sentences that are highly relevant but missing a key fact, is it a problem that the 11th sentence also has a key fact but is ranked slightly lower?"
- Answer: NO! Reranker's job is to put relevant sentences in top 20, not to rank THE answer at #1
- Current MRR metric (0.342) doesn't test this - it assumes only one sentence matters

### Priority 4 (OPTIONAL): Improve Ranking

**Only do this if:**
- Extraction is fixed (Priority 1)
- Reranker quality test (Priority 3) shows precision/recall is actually low
- Otherwise, current ranking is probably fine

## Cascade Effects Re-Analysis

### If We Fix Extraction First (Correct Approach):

```
Stage 1: Fix extraction patterns (object verification)
  ↓
Stage 2: Extract correct facts (e.g., "Zamenhof fondis Esperanton", not "oni fondis GIL")
  ↓
Stage 3: M1 filter works correctly (filters wrong facts, keeps correct ones)
  ↓
Stage 4: Importance scoring ranks correct facts higher
  ↓
Stage 5: Discourse planning generates correct answer
  ↓
Result: +20% accuracy (18/50 → 30/50) ⭐⭐⭐
```

### If We Had Fixed Ranking First (Wrong Approach):

```
Stage 1: Improve ranking (WHO rank 11→3)
  ↓
Stage 2: Extraction still broken (still extracts "oni fondis GIL" instead of "Zamenhof fondis Esperanton")
  ↓
Stage 3: Wrong facts extracted
  ↓
Stage 4: M1 filters them out (aggressive filtering)
  ↓
Result: No improvement (still 18/50) ✗
```

**You were absolutely right:** Fixing ranking wouldn't help because extraction is fundamentally broken.

## About Reranker Training Data

**Your question:** "Do you need to fix extraction first to generate right training data for reranker?"

**Answer:** Depends on what reranker was trained on. Let me check:

If reranker was trained on:
1. **Relevance labels** (sentence is relevant/irrelevant to query) → Training data likely OK
2. **Answer labels** (sentence contains THE answer) → Training data might be wrong if based on broken extraction
3. **Click data** (which sentences led to correct answers) → Training data definitely wrong if extraction is broken

**Implication:**
- If reranker training used broken extraction to label "correct" sentences, it might be optimizing for wrong thing
- Should check reranker training data source before retraining

## Summary: What You Taught Me

Your questions revealed my fundamental misunderstanding:

1. **"Why cutting off at sentence 5?"** → We don't - extraction looks at all 20. My analysis was wrong.

2. **"Is rank 6 a problem?"** → No - if we process top 20, rank 6-20 are all fine. Ranking is not the bottleneck.

3. **"Test reranker on relevance, not ranking?"** → Yes - reranker's job is to separate relevant/irrelevant, not to put THE answer at #1. MRR is wrong metric.

4. **"Fix extraction first for reranker training data?"** → Yes - if training data was based on broken extraction, reranking might be optimizing for wrong labels.

5. **"If 10 relevant sentences but answer at #11?"** → Not a reranker problem - extraction should find answer from ANY of the 10 relevant sentences in top 20.

## Recommended Action

**Immediate next step:**
1. Manually inspect 3-5 failed questions where answer was at rank 1-2
2. Check: Does top sentence ACTUALLY contain the answer?
3. Check: What facts did extraction extract from that sentence?
4. Identify extraction pattern bugs

**Then:**
1. Fix extraction object verification (Priority 1)
2. Re-evaluate (expect 60% accuracy, 30/50)
3. Test reranker quality independently (precision/recall at separating relevant/irrelevant)
4. Only then consider retraining reranker if quality is actually low

You were absolutely right to push back on my analysis. The real problem is extraction, not ranking.
