# Retrieval Architecture Analysis: Are We Using BM25 or AST-Aware?

**Date:** 2026-03-29
**Question:** "Why are we using BM25 for ranking? Shouldn't we use AST-aware approach?"

---

## TL;DR: We ARE Using AST-Aware Retrieval (But With No Ranking)

**Surprise finding:** We're NOT using BM25! We're using pure AST-aware graph queries.

**The problem:** We have AST-aware RETRIEVAL but NO intelligent RANKING.

---

## Current Architecture (From Code Analysis)

### What We Actually Use

Looking at `klareco/rag/whoosh_retriever.py`:

```python
def retrieve(self, query_ast, top_k=20):
    """
    Uses Kuzu graph queries to match grammatical structure (verb + object roles)
    Returns ONLY sentences with correct grammatical patterns
    NO fallback to BM25 text matching (pure AST-first approach)
    """
    return self.retrieve_with_ast_roles(query_ast, top_k)
```

### How AST Retrieval Works

For WHO questions (e.g., "Kiu fondis Esperanton?"):

```cypher
# Kuzu graph query - matches AST structure
MATCH (ft:Frazoteksto)-[:FRAZOTEKSTO_HAVAS_AST]->(a:AST)-[:AST_HAVAS_FRAZON]->(frazo:Frazo)
MATCH (frazo)-[:HAVAS_VERBON]->(verb:Vorto)
WHERE verb.radiko IN ['fond', 'kre', 'establ']  # Verb + synonyms

MATCH (frazo)-[:HAVAS_OBJEKTON_VORTGRUPO]->(obj_vg:Vortgrupo)-[:HAVAS_KERNON]->(obj_kerno:Vorto)
WHERE obj_kerno.radiko = 'esperant' AND obj_kerno.kazo = 'akuzativo'  # Object + case

RETURN ft.id, ft.teksto
LIMIT 100
```

**This matches:**
1. ✅ Grammatical roles (verb, object)
2. ✅ Morphological features (accusative case)
3. ✅ Root-level semantics (verb synonyms)
4. ✅ AST structure (not just keyword matching)

---

## The Ranking Problem (Why U-Shaped Curve Exists)

### Current "Ranking" Implementation

```python
def _execute_kuzu_query(kuzu_query, top_k):
    result = kuzu_conn.execute(kuzu_query)

    documents = []
    while result.has_next():
        row = result.get_next()
        documents.append({
            'text': text,
            'score': 100.0 - len(documents),  # ← JUST RANK ORDER
        })

    return documents[:top_k]
```

**Problem:** The "score" is just `100 - rank`, not an actual relevance score!

**Result ordering:** Whatever order Kuzu graph database returns results (likely insertion order or index scan order).

---

## Why AST Retrieval Is Correct But Ranking Is Missing

### What AST Retrieval Does Well

✅ **Precision:** Only retrieves grammatically valid matches
- Query: "Kiu fondis Esperanton?"
- Won't retrieve: "Esperanto estas lingvo" (wrong grammatical pattern)
- Will retrieve: "Zamenhof fondis Esperanton" (matches verb + object structure)

✅ **Linguistic correctness:** Respects Esperanto grammar
- Checks accusative case: `obj.kazo = 'akuzativo'`
- Checks verb forms: `verb.radiko IN [synonyms]`

✅ **Compositional:** Uses morpheme-level roots, handles agglutination

### What's Missing: Ranking Quality

❌ **No relevance scoring:** Results returned in arbitrary order
❌ **No semantic ranking:** Can't distinguish "better" matches
❌ **No proximity scoring:** Doesn't prefer closer synonym matches

**Example problem:**
```
Query: "Kiu fondis Esperanton?"
Results (arbitrary order):
  1. "La junularo fondis GIL." (wrong entity)
  2. "Zamenhof fondis Esperanton en 1887." (correct answer)
  3. "Oni fondis societojn." (vague)
```

Even though result #2 is the correct answer, Kuzu might return it at position 2 or 10 or 50, depending on database internals.

---

## BM25 vs AST-Aware: False Dichotomy

### The Real Options

| Approach | Retrieval | Ranking | Pros | Cons |
|----------|-----------|---------|------|------|
| **Current** | AST graph queries | None (insertion order) | Fast, grammatically precise | No relevance ranking |
| **Pure BM25** | Keyword matching | TF-IDF scores | Good ranking, fast | Ignores grammar, keyword soup |
| **Hybrid** | AST graph queries | BM25 on matched set | Both precision AND ranking | More complex |

### What BM25 Actually Does

**BM25 Formula:**
```
score(D,Q) = Σ IDF(qi) · (f(qi,D) · (k₁ + 1)) / (f(qi,D) + k₁ · (1 - b + b · |D|/avgdl))
```

**Translation:**
- `f(qi,D)`: How many times query word appears in document (term frequency)
- `IDF(qi)`: How rare the word is across corpus (inverse document frequency)
- Length normalization: Penalize long documents

**BM25 strengths:**
- ✅ Ranks by keyword relevance
- ✅ Penalizes common words (IDF)
- ✅ Fast and well-tested

**BM25 weaknesses for Esperanto:**
- ❌ Doesn't understand morphology: "fondis" ≠ "fond" ≠ "fondinto"
- ❌ Doesn't understand grammar: Can't distinguish subject from object
- ❌ Doesn't understand structure: Treats sentence as bag-of-words

---

## The Best Solution: Hybrid AST + Semantic Ranking

### Proposed Architecture

```
Query AST
    ↓
[1] AST Graph Query (Kuzu)
    → Retrieve sentences with correct grammatical structure (precision)
    → Returns ~50-100 candidates
    ↓
[2] Semantic Ranking (NEW)
    → Score by:
      a) Synonym distance (fond → kre: high, fond → iniciat: lower)
      b) Structural match quality (query AST vs candidate AST similarity)
      c) Entity prominence (is query entity the subject or buried in modifier?)
    → Returns top-k ranked by relevance
```

### Why Not Pure BM25?

**BM25 would be a DOWNGRADE from current AST retrieval.**

Example where BM25 fails but AST succeeds:

**Query:** "Kiu fondis Esperanton?" (Who founded Esperanto?)

**Sentence 1:** "La fundamentoj de Esperanto estas simplaj."
- BM25: HIGH score (has "Esperanto", "fundamentoj" ~ "fond")
- AST: REJECT (no verb "fond", "fundamentoj" is noun not verb)
- ✅ AST is correct - this is about Esperanto's foundations, not founding

**Sentence 2:** "Zamenhof fondis Esperanton en 1887."
- BM25: HIGH score (has "Esperanto", "fondis")
- AST: ACCEPT (matches verb+object pattern)
- ✅ Both work, but AST gives structural confidence

**Sentence 3:** "Oni uzas Esperanton en multaj landoj."
- BM25: MEDIUM score (has "Esperanto")
- AST: REJECT (verb is "uz" not "fond", wrong pattern)
- ✅ AST is correct - not about founding

---

## Recommendations

### Option A: Add Semantic Ranking (Best for Pure Esperanto AI)

**Keep AST retrieval, add learned/deterministic ranking:**

```python
def rank_ast_matches(query_ast, candidate_asts):
    """Rank AST matches by structural and semantic similarity."""
    scores = []
    for cand in candidate_asts:
        score = 0.0

        # 1. Synonym distance (deterministic)
        verb_sim = synonym_similarity(query_ast.verb, cand.verb)
        score += verb_sim * 0.4

        # 2. Structural match (deterministic)
        if query_ast.object == cand.object:  # Exact object match
            score += 0.3

        # 3. Entity prominence (deterministic)
        if cand.subject_is_entity():  # Answer entity is subject (good)
            score += 0.2

        # 4. Root embedding similarity (learned - 64D only)
        root_sim = embedding_similarity(query_roots, cand_roots)
        score += root_sim * 0.1

        scores.append(score)

    return ranked_by_score(candidates, scores)
```

**Advantages:**
- ✅ Preserves grammatical precision of AST retrieval
- ✅ Adds relevance ranking
- ✅ Aligns with "Pure Esperanto AI" thesis (structure + minimal learned)
- ✅ Explainable (can show why one result ranked higher)

**Effort:** 2-3 days

### Option B: Hybrid AST + BM25 (Pragmatic)

**Use AST for recall, BM25 for ranking:**

```python
# Step 1: AST retrieval (get grammatically valid candidates)
candidates = kuzu_query_ast_structure(query_ast, limit=100)

# Step 2: BM25 re-rank
for cand in candidates:
    cand['score'] = bm25_score(query_text, cand['text'])

return sorted(candidates, key=lambda x: x['score'], reverse=True)[:top_k]
```

**Advantages:**
- ✅ Fast to implement (BM25 already exists)
- ✅ Proven ranking algorithm
- ✅ Still benefits from AST precision

**Disadvantages:**
- ❌ BM25 doesn't understand morphology (will rank "fundamentoj" high for "fond")
- ❌ Loses explainability

**Effort:** 1 day

### Option C: Keep Current, Fix Extraction (Pragmatic Short-Term)

**Don't change retrieval/ranking yet, just fix extraction:**

The 60% of failures are extraction bugs (object verification missing), not ranking problems.

After fixing extraction, re-evaluate whether ranking is still the bottleneck.

**Effort:** 2 hours

---

## My Recommendation: Option A (Semantic AST Ranking)

**Reasoning:**

1. **AST retrieval is correct** - Don't replace it with BM25
2. **Missing component is ranking** - Not BM25 specifically, but relevance scoring
3. **Best approach: AST + semantic ranking** - Aligns with project thesis
4. **BM25 is the wrong tool** - Doesn't understand Esperanto morphology/grammar

**Implementation priority:**
1. Fix extraction bugs first (#710 - 2 hours)
2. Add semantic ranking (#711 modified - 2-3 days)
3. Re-evaluate if BM25 still needed

---

## What The Evaluation Actually Reveals

**U-shaped curve is NOT a BM25 problem** - it's a "no ranking" problem.

- k=5: Get first 5 results (lucky if answer is early in arbitrary order)
- k=20: Get 20 results (more noise, answer might be at position 15)
- k=50: Get 50 results (eventually include answer, extraction has more chances)

**The fix:** Rank results by relevance so answer appears in top 5 consistently.

---

## Action Items

1. ✅ **Clarify architecture** - We ARE using AST retrieval (this doc)
2. 🔄 **Fix extraction** - Issue #710 (object verification)
3. ⚙️ **Add semantic ranking** - Modify Issue #711 to focus on AST ranking, not query expansion
4. 📊 **Re-evaluate** - After fixes, measure if ranking improves top-k curve

---

## Related Issues

- #709: BM25 ranking analysis (CLOSED - revealed "no ranking" problem)
- #710: Object verification (extraction bug)
- #711: Query expansion model (SHOULD BE: Semantic ranking model)
- #713: QA Improvement Epic
