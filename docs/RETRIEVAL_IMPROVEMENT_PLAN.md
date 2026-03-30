# Retrieval Improvement Plan

**Date:** 2026-03-29
**Current Performance:** 38% retrieval recall @ k=30, 12% end-to-end accuracy
**Target:** 55-60% retrieval recall @ k=30, 30-35% end-to-end accuracy

---

## Executive Summary

**Current Bottleneck:** Retrieval is the primary failure point in the QA pipeline.

| Stage | Current Performance | Target |
|-------|---------------------|--------|
| Retrieval @ k=30 | 38% (19/50) | 55-60% |
| Extraction (given retrieval) | 100% | 100% (maintain) |
| Selection (given extraction) | 100% | 100% (maintain) |
| Generation (given selection) | 33% | 50% |
| **End-to-End** | **12%** | **30-35%** |

**Key Finding:** 22% of questions (11/50) have answers retrieved but ranked too low (rank 11-22).
These are **quick wins** - the retrieval finds the answer, but semantic ranking fails to prioritize it.

---

## Current Retrieval Architecture

```
Query → AST Parse → Role-Based Pattern Matching (Kuzu) → AST Semantic Ranking → Top-k Results
```

**Strengths:**
- ✅ Grammatically precise (high precision when it works)
- ✅ Explainable (clear why each document matches)
- ✅ Fast (sub-50ms queries)

**Weaknesses:**
- ❌ Brittle pattern matching (62% of failures - no matches found)
- ❌ Poor ranking (22% of questions - answer at rank 11-22)
- ❌ No fallback mechanism (when AST fails, returns 0 results)

---

## Failure Analysis

### Breakdown of 50-Question Test Set

| Failure Type | Count | % | Description |
|--------------|-------|---|-------------|
| **Success** | 8 | 16% | Answer in top 10 |
| **Ranking Failure** | 11 | 22% | Answer retrieved but rank 11-22 |
| **Pattern Mismatch** | 31 | 62% | AST pattern doesn't match corpus |

### Critical Examples of Ranking Failures

1. **"Kiu fondis Esperanton?"** → Answer at rank **16**
   - Query pattern: WHO (verb=fond, object=esperant)
   - Expected: Zamenhof
   - Issue: Generic facts about "fondis Esperanton" rank higher than agent-specific facts

2. **"Kio estas hundo?"** → Answer at rank **22**
   - Query pattern: WHAT IS (entity=hund)
   - Expected: besto (IS-A definition)
   - Issue: Narrative facts about specific dogs outrank IS-A definition

3. **"Kie okazas konversacio?"** → Answer at rank **16**
   - Query pattern: WHERE (verb=okaz, entity=konversaci)
   - Expected: loko
   - Issue: Specific event descriptions outrank location facts

**Pattern:** Deterministic features (IS-A relations, agent centrality) are being overwhelmed by high document scores or completeness metrics.

---

## Improvement Strategy

### Phase 1: Fix Ranking (Quick Wins - 1 Week)

**Target:** Improve from 38% → 45% retrieval recall @ k=30

**Actions:**

1. **Boost IS-A Facts for WHAT Questions**
   - Current: IS-A gets question_relevance=1.0 (35% weight = 0.35 final)
   - Proposed: Add definit

ional boost +0.2 for IS-A + WHAT combination
   - Expected impact: +5-7% on WHAT questions

2. **Boost Agent Facts for WHO Questions**
   - Current: Facts with `agent` argument get question_relevance=0.8
   - Proposed: Boost to 1.0 and add +0.15 for exact agent match
   - Expected impact: +10-15% on WHO questions

3. **Penalize Generic Facts**
   - Current: Generic facts (no entity match) still score 0.3+
   - Proposed: Reduce score to 0.1 for facts with no query entity match
   - Expected impact: Better discrimination

**Implementation:**
```python
# In importance_scorer.py
def _score_question_relevance(self, fact, question_type, query_entity):
    score = ... # existing logic

    # NEW: Boost for critical fact types
    if question_type == QuestionType.WHAT and fact.relation == RelationType.IS_A:
        if query_entity and self._entity_matches(query_entity, fact, exact=True):
            score = min(1.0, score + 0.2)  # Definitional boost

    if question_type == QuestionType.WHO:
        if 'agent' in fact.arguments:
            agent = fact.arguments['agent']
            if query_entity and self._entity_matches(query_entity, fact, exact=True):
                score = min(1.0, score + 0.15)  # Exact entity match boost

    return score
```

**Expected Result:** 11 ranking failures → 5 ranking failures (+12% recall improvement)

---

### Phase 2: Add BM25 Fallback (Medium Effort - 2 Weeks)

**Target:** Improve from 45% → 52% retrieval recall @ k=30

**Problem:** When AST pattern matching returns 0 results (62% of failures), system gives up.

**Solution:** Add hybrid retrieval with BM25 fallback.

```
Query → AST Pattern Matching
        ├─ IF results > 0 → AST Semantic Ranking → Return
        └─ IF results == 0 → BM25 Full-Text Search → Basic Ranking → Return
```

**Implementation:**
```python
# In whoosh_retriever.py
def retrieve(self, query_roots, top_k, ...):
    # Try AST-first retrieval
    ast_results = self.retrieve_with_ast_roles(...)

    if len(ast_results) >= top_k // 2:
        # AST retrieval successful
        return ast_results[:top_k]
    else:
        # AST failed or returned too few - add BM25 fallback
        bm25_results = self._bm25_fallback(query_roots, top_k)

        # Merge: AST results (higher quality) + BM25 results
        combined = ast_results + [r for r in bm25_results if r not in ast_results]
        return combined[:top_k]

def _bm25_fallback(self, query_roots, top_k):
    """Whoosh BM25 search when AST pattern matching fails."""
    with self.ix.searcher() as searcher:
        # Build OR query on roots
        query = Or([Term("roots", root) for root in query_roots])
        results = searcher.search(query, limit=top_k)

        # Reconstruct ASTs from corpus
        docs = []
        for hit in results:
            doc = {
                'text': hit['text'],
                'ast': self._reconstruct_ast(hit.docnum),
                'score': hit.score,
                'source': 'bm25_fallback'
            }
            docs.append(doc)
        return docs
```

**Expected Result:** 31 pattern mismatch failures → 15 failures (+32% recall improvement)

---

### Phase 3: Pattern Variant Expansion (Long-term - 3 Weeks)

**Target:** Improve from 52% → 60% retrieval recall @ k=30

**Problem:** AST patterns are too rigid.

**Examples of Pattern Variants:**

| Query Pattern | Corpus Variants |
|---------------|-----------------|
| "Kiu fondis X?" | "X estis fondita de Y", "Y estas la fondinto de X" |
| "Kio estas X?" | "X estas Y", "X, kiu estas Y", "X - Y" |
| "Kie okazas X?" | "X okazas en Y", "En Y okazas X", "La loko de X estas Y" |

**Solution:** Expand AST pattern matching to include common grammatical variants.

**Implementation:**
```python
# In whoosh_retriever.py
def _generate_pattern_variants(self, query_ast, question_type):
    """Generate grammatical variants of query pattern."""
    variants = [query_ast]  # Original

    if question_type == QuestionType.WHO:
        # Add passive voice variant: "X was created by WHO"
        variants.append(self._to_passive_voice(query_ast))
        # Add participial variant: "WHO is the creator of X"
        variants.append(self._to_participial(query_ast))

    if question_type == QuestionType.WHAT:
        # Add appositive variant: "X, which is Y"
        variants.append(self._to_appositive(query_ast))

    return variants

def retrieve_with_ast_roles(self, ...):
    # Generate pattern variants
    patterns = self._generate_pattern_variants(query_ast, question_type)

    # Try each pattern
    all_results = []
    for pattern in patterns:
        results = self._query_kuzu_for_pattern(pattern, ...)
        all_results.extend(results)

    # Deduplicate and rank
    return self._deduplicate_and_rank(all_results)
```

**Expected Result:** 15 remaining pattern failures → 5 failures (+20% recall improvement)

---

## Implementation Timeline

### Week 1: Fix Ranking (Phase 1)
- [ ] Day 1-2: Implement boost for IS-A + WHAT
- [ ] Day 3-4: Implement boost for agent + WHO
- [ ] Day 5: Test on 50-question set, measure improvement
- [ ] Expected: 38% → 45% recall

### Week 2-3: BM25 Fallback (Phase 2)
- [ ] Week 2: Implement BM25 fallback in retriever
- [ ] Week 3: Test and tune merge strategy
- [ ] Expected: 45% → 52% recall

### Week 4-6: Pattern Variants (Phase 3)
- [ ] Week 4: Design pattern variant grammar
- [ ] Week 5: Implement variant generation
- [ ] Week 6: Test and optimize
- [ ] Expected: 52% → 60% recall

---

## Expected End-to-End Impact

| Metric | Current | After Phase 1 | After Phase 2 | After Phase 3 |
|--------|---------|---------------|---------------|---------------|
| Retrieval @ k=30 | 38% | 45% | 52% | 60% |
| Extraction | 100% | 100% | 100% | 100% |
| Selection | 100% | 100% | 100% | 100% |
| Generation | 33% | 40% | 45% | 50% |
| **End-to-End** | **12%** | **18%** | **23%** | **30%** |

**Timeline:** 6 weeks to reach 30% end-to-end accuracy (2.5x improvement)

---

## Validation Criteria

After each phase, measure:
1. **Retrieval @ k=10, 30, 100** (primary metric)
2. **MRR (Mean Reciprocal Rank)** - should improve from 0.07 to 0.15+
3. **Ranking failure rate** - should decrease from 22% to <10%
4. **Pattern mismatch rate** - should decrease from 62% to <20%
5. **End-to-end accuracy** - ultimate goal

---

## Next Steps

**Immediate Action (Today):**
1. Implement Phase 1 ranking improvements
2. Run evaluation on 50-question test set
3. Document results

**This Week:**
1. Complete Phase 1 implementation
2. Begin Phase 2 design (BM25 fallback)
3. Create unit tests for ranking improvements

**This Month:**
1. Complete Phase 1 and Phase 2
2. Begin Phase 3 design
3. Target: 52% retrieval recall, 23% end-to-end accuracy
