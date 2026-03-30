# AST-First Retrieval: Design Analysis and Ideal Architecture

**Date:** 2026-03-29
**Goal:** Design ideal AST-first retrieval that leverages grammatical structure without falling back to BM25

---

## Executive Summary

**Current Performance:** 38% retrieval recall @ k=30, with 62% pattern mismatch failures

**Key Insight:** BM25's robustness comes from **under-specification** (any keyword match), while AST's precision comes from **over-specification** (exact grammatical pattern). The ideal system uses **structured relaxation** - starting with tight grammatical constraints and progressively relaxing them in principled ways.

**Proposed Solution:** Multi-stage AST-first retrieval with grammatical variant expansion, not keyword fallback.

---

## Part 1: Current System Analysis

### Architecture

```
Query → AST Parse → Pattern Detection → Kuzu Graph Query → AST Semantic Ranking → Top-k
                    (WHO/WHAT/etc)      (verb + object)    (40% verb, 30% obj, 20% subj, 10% emb)
```

### Pattern Matching (Kuzu Graph Queries)

**WHO Questions:** "Kiu fondis Esperanton?"
```cypher
MATCH (frazo)-[:HAVAS_VERBON]->(verb) WHERE verb.radiko IN ['fond', 'kre', ...]
MATCH (frazo)-[:HAVAS_OBJEKTON_VORTGRUPO]->(obj_vg)-[:HAVAS_KERNON]->(obj)
WHERE obj.radiko = 'esperant' AND obj.kazo = 'akuzativo'
```

**WHAT Questions:** "Kio estas hundo?"
```cypher
MATCH (frazo)-[:HAVAS_ALIAJN]->(alia:Vorto)
WHERE alia.radiko = 'hund'
```

**Problem:** WHAT questions use **weakest possible constraint** (any mention in "aliaj"). This is why they fail most often.

### Semantic Ranking

After retrieval, documents ranked by:
- **Verb similarity (40%):** Synonym distance (1.0 exact, 0.8 direct, 0.5 indirect, 0.0 unrelated)
- **Object match (30%):** Exact root match
- **Subject prominence (20%):** Proper noun subject bonus
- **Embedding similarity (10%):** Cosine similarity of root vectors (currently disabled due to poor discrimination)

### Failure Analysis

**62% Pattern Mismatches - Why?**

1. **Grammatical Variants Not Handled**
   - Query: "Kiu fondis Esperanton?" (active voice, verb=fond)
   - Corpus: "Esperanto estis fondita de Zamenhof" (passive voice, verb=est)
   - Result: No match (verb mismatch)

2. **Overly Strict Role Constraints**
   - Query: "Kio estas hundo?" needs IS-A relation
   - Corpus: "Hundo estas besto" → should match
   - But query pattern only matches if "hund" is in "aliaj" (modifiers)
   - Doesn't check if "hund" is the subject of an IS-A relation

3. **Missing IS-A Relation Detection**
   - Current WHAT pattern: Match ANY mention of entity
   - Should prioritize: Subject + verb="est" + predicate nominative pattern
   - Example: "X estas Y" → X IS-A Y

4. **Verb Synonym Expansion Too Conservative**
   - Only 3 synonyms per verb (max_count=3)
   - Misses many valid paraphrases (e.g., "publikigis" ≈ "eldonis" ≈ "aperigis")

---

## Part 2: BM25 Comparison

### How BM25 Works

```
Query: "Kiu fondis Esperanton?"
→ Terms: ["kiu", "fondis", "esperanton"]
→ BM25 Score = Σ IDF(term) × TF(term, doc) × boost_factor
→ Returns documents containing ANY of these terms, ranked by frequency
```

### BM25 Strengths

1. **Robust to Paraphrasing**
   - Matches partial keyword overlap
   - Doesn't require specific grammatical structure
   - Works even if query and corpus use different constructions

2. **Graceful Degradation**
   - If exact phrase not found, falls back to individual keywords
   - Always returns something (unless no keywords match at all)

3. **Well-Understood Ranking**
   - TF-IDF based
   - Rare terms weighted more heavily (good for entities)
   - Document length normalization

4. **Fast**
   - Inverted index lookups are O(1)
   - Efficiently handles millions of documents

### BM25 Weaknesses

1. **Ignores Grammar**
   - Can't distinguish "Hundo mordis katon" from "Kato mordis hundon"
   - No understanding of subject/object roles

2. **Keyword Ambiguity**
   - "banka" could be adjective (bank-related) or noun form (banks)
   - Can't use grammatical role to disambiguate

3. **No Compositional Understanding**
   - "mal-bona" (bad) matches "bona" (good) because it contains the root
   - Can't reason about morphological composition

4. **Weak for Definitional Questions**
   - "Kio estas hundo?" matches narratives about dogs, not definitions
   - No special handling of IS-A relations

### Architectural Differences

| Aspect | BM25 | AST-First |
|--------|------|-----------|
| **Query Representation** | Bag of keywords | Structured tree |
| **Match Criterion** | Keyword presence | Grammatical pattern |
| **Ranking** | Term frequency + IDF | Structural similarity |
| **Robustness** | High (under-specified) | Low (over-specified) |
| **Precision** | Low (many false positives) | High (when it works) |
| **Recall** | High | Low (62% failures) |
| **Grammatical Awareness** | None | Full |
| **Compositionality** | None | Native support |

### The Tension

- **BM25:** "Match anything containing these words" → High recall, low precision
- **AST:** "Match this exact grammatical pattern" → High precision, low recall

**Question:** Can we get BM25's recall with AST's precision?

**Answer:** Yes, through **structured relaxation** - progressive weakening of grammatical constraints.

---

## Part 3: Ideal AST-First Retrieval Design

### Core Insight: Grammatical Variant Expansion

Instead of falling back to keywords (BM25), **expand to grammatical variants** of the query pattern.

**Example:** "Kiu fondis Esperanton?"

**Stage 1: Exact Pattern**
- Match: verb IN ['fond', 'kre'] + object='esperant' + subject=?
- Result: Direct statements like "Zamenhof fondis Esperanton"

**Stage 2: Grammatical Variants**
- Passive voice: "Esperanto estis fondita de Zamenhof"
- Participial: "Zamenhof, la fondinto de Esperanto, ..."
- Nominalization: "La fondado de Esperanto fare de Zamenhof"

**Stage 3: Relaxed Constraints**
- Co-occurrence: verb + object in same sentence (any roles)
- Pragmatic reasoning: "La patro de Esperanto" → likely refers to creator

### Proposed Architecture

```
Query AST
   ↓
[1] Pattern Classification
    ├─ Question type detection (WHO/WHAT/WHERE/WHEN/WHY/HOW)
    ├─ Semantic intent (IS-A, ACTION, LOCATION, TIME, CAUSE, MANNER)
    └─ Key entities and relations
   ↓
[2] Multi-Stage Retrieval (cascading)
    ├─ Stage 1: Exact Pattern (tight grammatical constraints)
    ├─ Stage 2: Grammatical Variants (expanded constructions)
    ├─ Stage 3: Role-Relaxed (same roots, any roles)
    └─ Stage 4: Co-occurrence (same sentence, loose constraints)
   ↓
[3] Unified Ranking (importance-aware)
    ├─ Grammatical match quality (how close to ideal pattern?)
    ├─ Semantic role centrality (is query entity the subject?)
    ├─ Fact importance scoring (IS-A > narrative)
    └─ Source quality (Wikipedia > Gutenberg)
   ↓
Top-k Results
```

### Component 1: Enhanced Pattern Classification

**Current:** Detect question word (KIU/KIO/KIE/etc.) → simple pattern

**Proposed:** Deep semantic intent analysis

```python
class QueryIntent:
    question_type: QuestionType  # WHO, WHAT, etc.
    semantic_intent: SemanticIntent  # IS-A, HAS-PROPERTY, ACTED-ON, LOCATED-AT, etc.
    query_entity: str  # Main entity being asked about
    query_relation: RelationType  # Expected relation type
    answer_type: AnswerType  # PERSON, THING, PLACE, TIME, REASON, MANNER

def classify_query(query_ast: Dict) -> QueryIntent:
    """
    Enhanced query classification using AST structure.

    Examples:
    - "Kio estas hundo?" → WHAT + IS-A + entity="hund" + answer_type=CATEGORY
    - "Kiu fondis Esperanton?" → WHO + ACTED-ON + entity="esperant" + answer_type=PERSON
    - "Kie naskiĝis Zamenhof?" → WHERE + BORN + entity="zamenhof" + answer_type=PLACE
    """
```

**Why Better:** Explicit semantic intent guides retrieval strategy

### Component 2: Grammatical Variant Generator

**Key Innovation:** Generate grammatical variants of query pattern using Esperanto morphology

```python
class GrammaticalVariant:
    pattern_type: str  # "active", "passive", "participial", "nominalization", "relative_clause"
    cypher_query: str  # Kuzu graph pattern
    confidence: float  # How likely this variant matches the intent (1.0 = perfect)

def generate_variants(query_intent: QueryIntent) -> List[GrammaticalVariant]:
    """
    Generate grammatical variants for a query.

    For "Kiu fondis Esperanton?" (WHO + ACTED-ON):

    Variant 1: Active voice (1.0 confidence)
    - Pattern: [agent=?] VERB [patient=esperant]
    - Cypher: MATCH (frazo)-[:HAVAS_VERBON]->(verb) WHERE verb.radiko IN ['fond', 'kre']
              MATCH (frazo)-[:HAVAS_OBJEKTON]->(obj) WHERE obj.radiko = 'esperant'
              MATCH (frazo)-[:HAVAS_SUBJEKTON]->(subj) RETURN subj

    Variant 2: Passive voice (0.9 confidence)
    - Pattern: [patient=esperant] VERB-passive de [agent=?]
    - Cypher: MATCH (frazo)-[:HAVAS_VERBON]->(verb) WHERE verb.radiko = 'est'
              MATCH (frazo)-[:HAVAS_SUBJEKTON]->(subj) WHERE subj.radiko = 'esperant'
              MATCH (frazo)-[:HAVAS_ALIAJN]->(participle) WHERE participle.radiko IN ['fond', 'kre'] AND participle.vortspeco = 'participo'
              MATCH (frazo)-[:HAVAS_ALIAJN]->(agent) WHERE agent IN preposition('de')
              RETURN agent

    Variant 3: Participial construction (0.8 confidence)
    - Pattern: [agent=?], la VERB-into de [patient=esperant]
    - Cypher: MATCH (frazo)-[:HAVAS_SUBJEKTON]->(subj)
              MATCH (frazo)-[:HAVAS_ALIAJN]->(apposition) WHERE apposition contains ['fond', 'kre'] + 'into'
              MATCH (apposition)-[:HAVAS_ALIAJN]->(obj) WHERE obj.radiko = 'esperant'
              RETURN subj

    Variant 4: Nominalization (0.7 confidence)
    - Pattern: La VERB-ado de [patient=esperant] fare de [agent=?]
    - Cypher: Complex pattern for nominalized constructions
    """
```

**Why This Works:**

1. **Esperanto Grammar is Regular:** Passive = "est" + past participle, Nominalization = root + "ado", etc.
2. **AST Enables Detection:** Can identify these patterns in corpus
3. **Confidence Ranking:** More common constructions ranked higher
4. **Exhaustive Coverage:** Handles all major grammatical transformations

### Component 3: IS-A Relation Detection (Critical for WHAT Questions)

**Current Problem:** "Kio estas hundo?" matches any sentence mentioning "hundo"

**Proposed Solution:** Explicit IS-A pattern matching

```python
def retrieve_is_a_facts(entity_root: str, top_k: int) -> List[Dict]:
    """
    Retrieve IS-A facts for entity.

    Patterns to match (in priority order):

    1. Direct IS-A statement (1.0 confidence)
       - "Hundo estas besto" (entity=hund IS-A besto)
       - Pattern: [subject=entity] VERB("est") [predicate_nominative]
       - Cypher: MATCH (frazo)-[:HAVAS_SUBJEKTON]->(subj) WHERE subj.radiko = 'hund'
                 MATCH (frazo)-[:HAVAS_VERBON]->(verb) WHERE verb.radiko = 'est'
                 MATCH (frazo)-[:HAVAS_OBJEKTON]->(pred_nom) RETURN pred_nom

    2. Appositive construction (0.9 confidence)
       - "Hundo, besto kiu..." (entity, category ...)
       - Pattern: [entity], [category] [relative_clause]
       - Cypher: Complex pattern for appositive detection

    3. Definition introduction (0.8 confidence)
       - "Hundo estas vorto kiu signifas..."
       - Pattern: [entity] estas [definition_marker] ...
       - Look for "difino", "signifo", "kategorio" markers

    4. Reverse IS-A (0.7 confidence)
       - "Besto kiel hundo..." (category such_as entity)
       - Pattern: [category] [such_as_marker] [entity]
       - Look for "kiel", "ekzemple", "nome" markers
    """
```

**Implementation in Kuzu:**

```cypher
# IS-A Priority Query
# Returns documents ordered by pattern confidence

# Priority 1: Direct IS-A (entity as subject)
MATCH (ft:Frazoteksto)-[:FRAZOTEKSTO_HAVAS_AST]->(a:AST)-[:AST_HAVAS_FRAZON]->(frazo:Frazo)
MATCH (frazo)-[:HAVAS_SUBJEKTON]->(subj_vg:Vortgrupo)-[:HAVAS_KERNON]->(subj:Vorto)
WHERE subj.radiko = 'hund'
MATCH (frazo)-[:HAVAS_VERBON]->(verb:Vorto)
WHERE verb.radiko = 'est'
MATCH (frazo)-[:HAVAS_OBJEKTON]->(obj_vg:Vortgrupo)-[:HAVAS_KERNON]->(obj:Vorto)
WHERE obj.kazo = 'nominativo'  # Predicate nominative
RETURN ft.id AS id, ft.teksto AS text, 1.0 AS confidence

UNION

# Priority 2: Reverse IS-A (entity as predicate nominative)
MATCH (ft:Frazoteksto)-[:FRAZOTEKSTO_HAVAS_AST]->(a:AST)-[:AST_HAVAS_FRAZON]->(frazo:Frazo)
MATCH (frazo)-[:HAVAS_VERBON]->(verb:Vorto)
WHERE verb.radiko = 'est'
MATCH (frazo)-[:HAVAS_OBJEKTON]->(obj_vg:Vortgrupo)-[:HAVAS_KERNON]->(obj:Vorto)
WHERE obj.radiko = 'hund' AND obj.kazo = 'nominativo'
MATCH (frazo)-[:HAVAS_SUBJEKTON]->(subj_vg:Vortgrupo)-[:HAVAS_KERNON]->(subj:Vorto)
RETURN ft.id AS id, ft.teksto AS text, 0.9 AS confidence

ORDER BY confidence DESC
LIMIT 50
```

### Component 4: Unified Importance-Aware Ranking

**Current:** AST semantic ranking happens AFTER retrieval (post-filter)

**Proposed:** Integrate importance scoring INTO retrieval stage

```python
def compute_retrieval_score(
    candidate_ast: Dict,
    query_intent: QueryIntent,
    pattern_variant: GrammaticalVariant,
    fact_importance: FactImportanceScorer
) -> float:
    """
    Unified scoring function combining:
    1. Grammatical match quality (how well does this match the pattern variant?)
    2. Semantic role centrality (is the answer in the right grammatical position?)
    3. Fact importance (IS-A > HAS-PROPERTY > narrative)
    4. Source quality (Wikipedia > Gutenberg)

    Returns score in [0, 1] where:
    - 1.0 = Perfect match (exact pattern, definiti definitional fact, Wikipedia)
    - 0.5 = Partial match (variant pattern, relevant fact, medium source)
    - 0.0 = Poor match (loose co-occurrence, narrative fact, low-quality source)
    """

    # Component 1: Grammatical match quality (40%)
    # How close is this to the ideal grammatical pattern?
    grammatical_score = pattern_variant.confidence  # 1.0 for exact, 0.9 for passive, etc.

    # Component 2: Semantic role centrality (20%)
    # Is the query entity in the expected grammatical role?
    role_score = score_role_centrality(candidate_ast, query_intent)
    # Example: For WHO questions, answer should be subject (1.0), not modifier (0.3)

    # Component 3: Fact importance (30%)
    # Extract fact from candidate AST and score it
    fact = extract_fact_from_ast(candidate_ast)
    importance_score = fact_importance.score(
        fact, query_intent.question_type, query_intent.query_entity
    ).final_score

    # Component 4: Source quality (10%)
    source_score = get_source_quality(candidate_ast.metadata)

    # Weighted combination
    total_score = (
        grammatical_score * 0.40 +
        role_score * 0.20 +
        importance_score * 0.30 +
        source_score * 0.10
    )

    return total_score
```

**Key Innovation:** Importance scoring is used during retrieval, not just post-retrieval selection. This ensures definitional facts rank higher even if they use variant grammatical patterns.

---

## Part 4: Addressing the 62% Pattern Mismatch Problem

### Current Failures by Question Type

**WHAT Questions (0% accuracy):**
- Problem: Match any mention, no IS-A priority
- Solution: Explicit IS-A pattern matching (Component 3)
- Expected improvement: 0% → 60% (most WHAT questions have IS-A answers)

**WHO Questions (20% accuracy):**
- Problem: Only match active voice
- Solution: Passive voice + participial variants (Component 2)
- Expected improvement: 20% → 70% (many corpus sentences use passive)

**WHERE Questions (20% accuracy):**
- Problem: Loose location matching (any "aliaj" mention)
- Solution: Explicit location patterns (en + place, subject + verb("loĝi"/etc.))
- Expected improvement: 20% → 60%

**WHEN Questions (10% accuracy):**
- Problem: No temporal pattern detection
- Solution: Temporal expression patterns (dates, "en + year", verb.tempo)
- Expected improvement: 10% → 50%

### Estimated Impact

| Question Type | Current | With Grammatical Variants | With IS-A Detection | Combined |
|---------------|---------|--------------------------|---------------------|----------|
| WHAT | 0% | 0% → 20% | 20% → 60% | 60% |
| WHO | 20% | 20% → 50% | - | 50% |
| WHERE | 20% | 20% → 50% | - | 50% |
| WHEN | 10% | 10% → 40% | - | 40% |
| WHY | 0% | 0% → 20% | - | 20% |
| **Overall** | **12%** | **25%** | **35%** | **50%** |

**Timeline:**
- Component 3 (IS-A detection): 3 days implementation, +23% recall
- Component 2 (grammatical variants): 1 week implementation, +13% recall
- Component 4 (unified ranking): 2 days integration, quality improvement
- **Total: 2 weeks to 50% retrieval recall**

---

## Part 5: Advantages Over BM25 Fallback

### Why AST-First with Variants > BM25 Fallback

**1. Maintains Grammatical Precision**
- AST variants preserve subject/object distinction
- BM25 treats "Hundo mordis katon" same as "Kato mordis hundon"

**2. Handles Esperanto Compositionality**
- AST understands "mal-bona" ≠ "bona"
- BM25 would match both because keyword overlap

**3. Explainable**
- AST: "Matched passive voice variant with confidence 0.9"
- BM25: "Keywords appeared X times" (no grammatical insight)

**4. Better for Definitional Questions**
- AST: Can explicitly prioritize IS-A relations
- BM25: No concept of definitional vs narrative facts

**5. Leverages Linguistic Structure**
- Esperanto's regularity makes grammatical variants predictable
- BM25 ignores this structure entirely

### When BM25 Still Helps

**Edge Cases Where BM25 is Superior:**
- Very long questions with many keywords (more robust to noise)
- Completely ungrammatical queries (typos, fragments)
- Cross-lingual queries (mixing languages)

**Proposed Hybrid (if needed):**
- Stage 5 (optional): BM25 fallback only when Stages 1-4 return < 5 results
- Use BM25 as last resort, not first choice

---

## Part 6: Implementation Roadmap

### Phase 1: IS-A Detection (3 days) ← **HIGHEST IMPACT**

**Why First:** WHAT questions are 0% accurate, IS-A detection would fix most of them

**Implementation:**
1. Add `retrieve_is_a_pattern()` to WhooshRetriever
2. Update `_retrieve_what_pattern()` to use IS-A detection
3. Test on "Kio estas X?" questions

**Expected Impact:** 0% → 60% on WHAT questions, +20% overall recall

**Files to Modify:**
- `klareco/rag/whoosh_retriever.py` - Add IS-A pattern
- Test on 50-question set

### Phase 2: Passive Voice Variants (3 days)

**Why Second:** WHO questions are 20% accurate, many use passive voice in corpus

**Implementation:**
1. Add passive voice detection to `_retrieve_who_pattern()`
2. Generate passive Cypher query variant
3. Merge active + passive results

**Expected Impact:** 20% → 50% on WHO questions, +10% overall recall

### Phase 3: Unified Ranking (2 days)

**Why Third:** Integrate importance scoring into retrieval ranking

**Implementation:**
1. Refactor ranking to use `FactImportanceScorer` during retrieval
2. Add confidence weighting for pattern variants
3. Test ranking quality

**Expected Impact:** +5-10% precision improvement

### Phase 4: Grammatical Variant Framework (1 week)

**Why Last:** Generalizes Phase 2 to all question types

**Implementation:**
1. Create `GrammaticalVariantGenerator` class
2. Implement variant generation for all question types
3. Integrate into retrieval pipeline

**Expected Impact:** Handles remaining edge cases, +5% recall

### Total Timeline: 2-3 weeks

---

## Part 7: Success Metrics

### Target Performance

| Metric | Current | After IS-A | After Passive | After Ranking | After Variants | Final Goal |
|--------|---------|-----------|---------------|---------------|----------------|------------|
| **Retrieval @ k=30** | 38% | 48% | 53% | 55% | 58% | 55-60% |
| **WHAT accuracy** | 0% | 60% | 60% | 65% | 65% | 60%+ |
| **WHO accuracy** | 20% | 20% | 50% | 55% | 60% | 60%+ |
| **WHERE accuracy** | 20% | 20% | 30% | 40% | 50% | 50%+ |
| **Overall accuracy** | 12% | 22% | 28% | 32% | 38% | 35-40% |

### Validation Tests

**Test 1: IS-A Retrieval**
- Query: "Kio estas hundo?"
- Expected: "Hundo estas besto" ranked in top 3
- Measure: IS-A fact rank position

**Test 2: Passive Voice**
- Query: "Kiu fondis Esperanton?"
- Expected: Both "Zamenhof fondis" AND "Esperanto estis fondita de Zamenhof" retrieved
- Measure: Variant coverage (% of passive constructions found)

**Test 3: Importance-Aware Ranking**
- Query: "Kio estas Esperanto?"
- Expected: Definitional facts rank higher than narrative facts
- Measure: IS-A facts in top 10 vs top 30

---

## Conclusion

**Key Insight:** The solution to AST retrieval's brittleness is not abandoning structure (BM25 fallback), but **embracing structure more deeply** through grammatical variant expansion.

**Why This Approach Wins:**
1. Preserves grammatical precision of AST approach
2. Achieves robustness through principled structural relaxation
3. Leverages Esperanto's regular grammar for predictable variants
4. Maintains explainability (can say WHY each result matched)
5. Integrates importance scoring directly into retrieval

**Expected Result:** 55-60% retrieval recall (from 38%), 35-40% end-to-end accuracy (from 12%), **WITHOUT using BM25**.

**Next Action:** Implement Phase 1 (IS-A detection) first - 3 days work for +20% recall improvement.
