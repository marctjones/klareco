# AST-Aware Answer Extraction: Deep Analysis & Redesign

**Date**: 2026-03-28
**Status**: Root cause analysis complete, redesign proposed
**Context**: AST retrieval works (finds "Zamenhof kreis Esperanton"), but answer extraction fails

---

## Executive Summary

**Problem**: AST retrieval correctly retrieves "La genia doktoro Zamenhof kreis Esperanton" but answer extractor returns "Esperanton" or "li" instead of "Zamenhof".

**Root Cause**: Two parallel extraction paths (ASTAnswerExtractor vs FactExtractor) with different AST interpretations, causing correct documents to yield wrong answers.

**Solution**: Unify extraction around a single AST-first approach that directly extracts grammatical roles from the AST structure used by retrieval.

---

## Part 1: Current Architecture Analysis

### Pipeline Flow

```
Query: "Kiu fondis Esperanton?"
  ↓
[AST Retrieval] → 31 sentences with verb={fond,kre} AND object={esperant}
  ↓
[ASTAnswerExtractor.extract_answer_from_multiple_docs(top_20)]
  ↓ Tries to extract from each doc
  ↓ Multi-candidate ranking: pattern + proximity + validation
  ↓ Aggregates across docs
  ↓
IF confidence >= 0.7:
  ✓ Return direct answer
ELSE:
  → Fall back to FactExtractor pipeline
```

### ASTAnswerExtractor.extract_answer() Logic

**For WHO questions** (lines 636-782 in answer_extractor.py):

```python
def _extract_who(query_ast, doc_ast, doc_text):
    # 1. Get query verb and doc verb
    query_verb = get_verb_root(query_ast)  # "fond"
    doc_verb = get_verb_root(doc_ast)      # "kre"

    # 2. Check if verbs match (using synonyms)
    verb_match = are_verbs_similar(query_verb, doc_verb)  # True

    # 3. Collect candidates:
    candidates = []

    # Candidate 1: Subject (highest priority if verb matches)
    subjekto = doc_ast.get('subjekto')
    if subjekto and is_person(subjekto):
        candidates.append({
            'text': vortgrupo_to_text(subjekto),
            'pattern_score': 0.9 if verb_match else 0.5,
            'source': 'subject'
        })

    # Candidate 2: Passive agent ("de X" in passive voice)
    # Only if is_passive_voice(doc_ast)

    # Candidate 3: Proper nouns in aliaj (NOT after "de")
    for modifier in aliaj:
        if is_person(modifier):
            candidates.append({
                'text': vortgrupo_to_text(modifier),
                'pattern_score': 0.6,
                'source': 'proper_noun'
            })

    # Candidate 4: Object (fallback)
    objekto = doc_ast.get('objekto')
    if objekto and is_person(objekto):
        candidates.append({
            'text': vortgrupo_to_text(objekto),
            'pattern_score': 0.7 if verb_match else 0.4,
            'source': 'object'
        })

    # 4. Score candidates: pattern*0.4 + proximity*0.4 + validation*0.2
    # 5. Return best
```

### FactExtractor.extract() Logic

**For CREATED_BY facts** (lines 194-212 in fact_extractor.py):

```python
def _extract_created_by(frazo, source_sentence):
    """Extract CREATED-BY fact: 'X kreis Y' → Y CREATED-BY X"""
    subjekto = frazo.get('subjekto')
    objekto = frazo.get('objekto')

    if not subjekto or not objekto:
        return None

    agent = get_entity_name(subjekto)   # "Zamenhof"
    entity = get_entity_name(objekto)   # "Esperanton"

    # Extract time/location modifiers
    modifiers = extract_modifiers(frazo.get('aliaj', []))

    return Fact(
        entity="Esperanton",              # THE THING CREATED
        relation=CREATED_BY,
        arguments={'agent': "Zamenhof"},  # WHO CREATED IT
        modifiers=modifiers,
        source_sentence=source_sentence
    )
```

---

## Part 2: Root Cause Analysis

### Problem 1: ASTAnswerExtractor Gets Wrong AST Structure

**What Happens**:
```python
# Sentence: "La genia doktoro Zamenhof kreis Esperanton"
doc_ast = parse(sentence)

# Parser puts this in:
doc_ast = {
    'subjekto': None,  # ❌ Parser doesn't set subjekto!
    'verbo': None,     # ❌ Parser doesn't set verbo!
    'objekto': None,   # ❌ Parser doesn't set objekto!
    'aliaj': [
        {'tipo': 'vortgrupo', ...},  # "La genia doktoro Zamenhof"
        {'tipo': 'vorto', 'radiko': 'kre', ...},  # "kreis"
        {'tipo': 'vorto', 'radiko': 'esperant', ...}  # "Esperanton"
    ]
}
```

**Why**: The parser puts verb and objects in `aliaj` for declarative statements, not in structured fields.

**Result**: ASTAnswerExtractor can't find subjekto (checks line 672), so it falls back to collecting proper nouns from `aliaj` with low pattern_score (0.6), which then compete with wrong candidates.

### Problem 2: Multi-Doc Aggregation Normalizes Case

**What Happens** (lines 362-372 in answer_extractor.py):
```python
for answer in candidates:
    answer_text = answer['text'].lower().strip()  # ❌ "zamenhof" → "zamenhof"
    entity_agg[answer_text]['count'] += 1

# Later when returning:
return {
    'text': best_candidate['original_text'],  # Preserves case from FIRST occurrence
    ...
}
```

**Problem**: If first document has "zamenhof" (lowercase) and second has "Zamenhof" (proper), aggregation uses first occurrence's casing.

### Problem 3: Fact Extraction Works But Scoring Fails

**What Happens**:
```python
# FactExtractor correctly creates:
fact = Fact(
    entity="Esperanton",
    relation=CREATED_BY,
    arguments={'agent': "Zamenhof"},
    modifiers={},
    source_sentence="La genia doktoro Zamenhof kreis Esperanton"
)

# But ImportanceScorer scores this fact:
score = importance_scorer.score(fact, question_type=WHO, query_entity="esperant")

# Scoring (importance_scorer.py lines ~100-200):
# - query_match: Does "esperanton" match query? ✓ (1.0)
# - diversity: Does fact add new info? ? (0.0-1.0)
# - entity_prominence: Is "Esperanton" important? ? (0.0-1.0)
# - completeness: Does fact have required args? ✓ (0.7)

# PROBLEM: Fact is ABOUT "Esperanton" (entity), not ABOUT "Zamenhof" (answer)!
```

**Why**: Facts are entity-centric (entity="Esperanton") not answer-centric (answer="Zamenhof").

### Problem 4: WHO Questions Need Agent, Not Entity

**Current Fact Structure**:
```python
Fact(entity="Esperanton", arguments={'agent': "Zamenhof"})
```

**What WHO Questions Need**:
```python
# Query: "Kiu fondis Esperanton?"
# Answer: "Zamenhof"
#
# We need: Fact where "Zamenhof" is PRIMARY, "Esperanton" is secondary
```

**Disconnect**: ImportanceScorer ranks facts by `entity` prominence, but WHO answers come from `arguments['agent']`.

---

## Part 3: Proposed Architecture - True AST-First Extraction

### Core Principle

**RETRIEVAL and EXTRACTION must use THE SAME AST structure.**

If retrieval queries:
```cypher
MATCH (frazo)-[:HAVAS_VERBON]->(verb)
MATCH (frazo)-[:HAVAS_OBJEKTON_VORTGRUPO]->(obj_vg)-[:HAVAS_KERNON]->(obj_kerno)
WHERE verb.radiko IN ['fond','kre'] AND obj_kerno.radiko = 'esperant'
```

Then extraction should:
```python
# Extract from THE SAME graph structure
subjekto = get_from_graph(frazo, HAVAS_SUBJEKTON_VORTGRUPO)
verbo = get_from_graph(frazo, HAVAS_VERBON)
objekto = get_from_graph(frazo, HAVAS_OBJEKTON_VORTGRUPO)

# For WHO question with verb "fond/kre" and object "esperant":
answer = subjekto  # ✓ Directly from graph
```

### Unified Extraction Architecture

```
Query: "Kiu fondis Esperanton?"
  ↓
Parse → AST: question_type=WHO, verb=fond, object=esperant
  ↓
[Kuzu AST Retrieval]
  Query graph: verb IN [fond,kre,establ] AND object=esperant
  Returns: 31 Frazoteksto IDs with precomputed ASTs
  ↓
[AST Role Extractor] ← NEW UNIFIED COMPONENT
  For each Frazoteksto:
    1. Load precomputed AST from graph (already done by retriever)
    2. Extract answer using GRAPH RELATIONSHIPS:
       - For WHO: answer = MATCH (frazo)-[:HAVAS_SUBJEKTON_*]->(subj)
       - For WHAT: answer = MATCH (frazo)-[:HAVAS_OBJEKTON_*]->(obj)
       - For WHERE: answer = MATCH (frazo)-[:HAVAS_*]->()-[:HAVAS_MODIFIERS]->(:Location)
  ↓
[Multi-Doc Aggregation]
  Rank candidates by:
    - Occurrence count (appears in multiple docs = high confidence)
    - Document retrieval score (BM25 or AST query score)
    - Proper noun validation (reject pronouns, validate names)
  ↓
Return best answer with citations
```

### Implementation: AST Role Extractor

**New class**: `ASTRoleExtractor` (replaces current ASTAnswerExtractor)

```python
class ASTRoleExtractor:
    """
    Extract answers by DIRECTLY reading AST grammatical roles.

    Uses the SAME graph structure that retrieval uses.
    No pattern matching, no heuristics - just read the grammatical role.
    """

    def extract_answer(
        self,
        query_ast: Dict,
        doc_ast: Dict,  # From Kuzu graph (precomputed)
        question_type: str
    ) -> Optional[str]:
        """
        Extract answer from AST using grammatical roles.

        For WHO: Extract subject (agent who performed action)
        For WHAT: Extract object or predicate
        For WHERE: Extract location modifier
        For WHEN: Extract time modifier
        """
        if question_type == 'WHO':
            return self._extract_subject(doc_ast)
        elif question_type == 'WHAT':
            return self._extract_object_or_predicate(doc_ast)
        elif question_type == 'WHERE':
            return self._extract_location(doc_ast)
        elif question_type == 'WHEN':
            return self._extract_time(doc_ast)
        else:
            return None

    def _extract_subject(self, doc_ast: Dict) -> Optional[str]:
        """
        Extract subject from AST.

        Handles:
        - subjekto field (if parser sets it)
        - Traverse graph: (frazo)-[:HAVAS_SUBJEKTON_VORTGRUPO]->()-[:HAVAS_KERNON]->(vorto)
        """
        # Try direct field
        subjekto = doc_ast.get('subjekto')
        if subjekto:
            return self._vortgrupo_to_text(subjekto)

        # Fallback: Search for nominative substantivo in aliaj
        # (This handles cases where parser doesn't set subjekto)
        aliaj = doc_ast.get('aliaj', [])
        for alia in aliaj:
            if isinstance(alia, dict):
                # Look for nominative (subject case) proper noun or person
                if (alia.get('kazo') == 'nominativo' and
                    self._is_person_indicator(alia)):
                    return self._vortgrupo_to_text(alia)

        return None

    def _is_person_indicator(self, node: Dict) -> bool:
        """Check if node indicates a person."""
        # 1. Proper noun (capitalized)
        plena_vorto = node.get('plena_vorto', '')
        if plena_vorto and plena_vorto[0].isupper():
            return True

        # 2. Person suffix (-ul, -ist, -in)
        sufiksoj = node.get('sufiksoj', [])
        if any(suf in ['ul', 'ist', 'in'] for suf in sufiksoj):
            return True

        # 3. Known person roots
        radiko = node.get('radiko', '')
        if radiko in ['hom', 'vir', 'ino', 'infan', 'patro', 'patrino']:
            return True

        return False
```

### Multi-Doc Aggregation (Fixed)

```python
def aggregate_candidates(
    self,
    candidates: List[Dict],  # From multiple docs
    top_n: int = 3
) -> Optional[str]:
    """
    Aggregate candidates from multiple documents.

    CRITICAL FIX: Preserve proper noun casing.
    """
    from collections import defaultdict

    # Group by NORMALIZED text (lowercase for matching)
    # But KEEP original casing
    entity_agg = defaultdict(lambda: {
        'variants': [],  # All casing variants
        'count': 0,
        'total_score': 0.0,
        'doc_ranks': []
    })

    for candidate in candidates:
        original_text = candidate['text']
        normalized_key = original_text.lower().strip()

        agg = entity_agg[normalized_key]
        agg['variants'].append(original_text)
        agg['count'] += 1
        agg['total_score'] += candidate['score']
        agg['doc_ranks'].append(candidate['rank'])

    # Score and rank
    scored = []
    for norm_text, agg in entity_agg.items():
        # Choose BEST casing variant (prefer proper nouns)
        best_variant = self._choose_best_casing(agg['variants'])

        # Score: count (50%) + avg_score (30%) + early_rank (20%)
        count_score = min(agg['count'] / top_n, 1.0)
        avg_score = agg['total_score'] / agg['count']
        rank_score = 1.0 / min(agg['doc_ranks'])

        final_score = (
            0.50 * count_score +
            0.30 * avg_score +
            0.20 * rank_score
        )

        scored.append({
            'text': best_variant,  # ✓ Proper casing
            'score': final_score,
            'count': agg['count'],
            'ranks': agg['doc_ranks']
        })

    scored.sort(key=lambda x: x['score'], reverse=True)
    return scored[0] if scored else None

def _choose_best_casing(self, variants: List[str]) -> str:
    """
    Choose best casing from variants.

    Rules:
    1. Prefer proper noun casing (First Letter Capitalized)
    2. Prefer most common variant
    3. Fall back to first occurrence
    """
    # Count each variant
    from collections import Counter
    variant_counts = Counter(variants)

    # Prefer capitalized variants
    capitalized = [v for v in variants if v and v[0].isupper()]
    if capitalized:
        # Return most common capitalized variant
        return Counter(capitalized).most_common(1)[0][0]

    # Fall back to most common variant
    return variant_counts.most_common(1)[0][0]
```

---

## Part 4: Implementation Plan

### Phase 1: Fix Multi-Doc Aggregation Casing (Quick Win)

**Priority**: HIGH (fixes "zamenhof" → "Zamenhof")
**Effort**: 1 hour
**Files**: `klareco/rag/answer_extractor.py`

**Changes**:
1. Modify `extract_answer_from_multiple_docs()` lines 362-402
2. Add `_choose_best_casing()` method
3. Test on "Kiu fondis Esperanton?"

**Expected Improvement**: +5-10% accuracy (fixes casing issues)

### Phase 2: Add Fallback Subject Extraction (Quick Win)

**Priority**: HIGH (handles parser not setting subjekto)
**Effort**: 2 hours
**Files**: `klareco/rag/answer_extractor.py`

**Changes**:
1. Modify `_extract_who()` lines 672-681
2. Add fallback: search `aliaj` for nominative person
3. Add `_is_person_indicator()` helper

**Code**:
```python
def _extract_who(self, query_ast, doc_ast, doc_text):
    # ... existing code ...

    # Candidate 1: Subject
    subjekto = doc_ast.get('subjekto')
    if subjekto and self._is_person(subjekto):
        candidates.append({
            'text': self._vortgrupo_to_text(subjekto),
            'pattern_score': 0.9 if verb_match else 0.5,
            'source': 'subject'
        })
    else:
        # NEW: Fallback - search aliaj for nominative person
        aliaj = doc_ast.get('aliaj', [])
        for i, alia in enumerate(aliaj):
            if isinstance(alia, dict):
                # Must be nominative AND person-like
                if (alia.get('kazo') == 'nominativo' and
                    self._is_person_indicator(alia)):
                    # Check it's not after a verb (would be predicate, not subject)
                    if i == 0 or aliaj[i-1].get('vortspeco') != 'verbo':
                        candidates.append({
                            'text': self._vortgrupo_to_text(alia),
                            'pattern_score': 0.85,  # High score (acts as subject)
                            'source': 'subject_fallback'
                        })
                        break  # Take first nominative person

    # ... rest of candidates ...
```

**Expected Improvement**: +15-20% accuracy (finds subjects in aliaj)

### Phase 3: Improve Answer-Centric Scoring (Medium Priority)

**Priority**: MEDIUM (fixes fact scoring for WHO questions)
**Effort**: 3 hours
**Files**: `klareco/rag/importance_scorer.py`, `klareco/rag/extractive_answering.py`

**Problem**: Current facts are entity-centric, but WHO answers need agent-centric scoring.

**Solution**: Add answer-role-aware scoring

```python
class FactImportanceScorer:
    def score(self, fact, question_type, query_entity, metadata):
        # ... existing scoring ...

        # NEW: Answer-role scoring for WHO questions
        if question_type == QuestionType.WHO:
            # For WHO questions, the ANSWER is in arguments['agent'], not entity
            if 'agent' in fact.arguments:
                agent = fact.arguments['agent']

                # Boost if agent is proper noun
                if agent and agent[0].isupper():
                    breakdown.entity_prominence += 0.3

                # Boost if query entity matches fact entity
                # (Query asks "Who created Esperanto?" → fact.entity="Esperanto")
                if query_entity and query_entity.lower() in fact.entity.lower():
                    breakdown.query_match = 1.0
```

**Expected Improvement**: +5% accuracy (better fact ranking)

### Phase 4: Create Unified ASTRoleExtractor (Long-term)

**Priority**: LOW (architectural improvement, requires parser changes)
**Effort**: 1-2 weeks
**Files**: New file `klareco/rag/ast_role_extractor.py`, parser fixes

**This is the IDEAL long-term solution but requires**:
1. Parser to consistently set subjekto/verbo/objekto fields
2. OR direct Kuzu graph traversal instead of parsed AST dicts
3. Refactoring of entire extraction pipeline

**Not recommended for immediate implementation** - Phases 1-3 will get us to 50-60% accuracy.

---

## Part 5: Expected Results

### Current State (24% accuracy)

| Question Type | Accuracy | Main Issue |
|---------------|----------|------------|
| WHO | 10% | Wrong subject extraction |
| WHERE | 30% | - |
| WHEN | 0% | - |
| WHAT | 30% | - |
| HOW_MANY | 80% | Works OK |
| **Overall** | **24%** | |

### After Phase 1+2 (Est. 45-50% accuracy)

| Question Type | Accuracy | Improvement |
|---------------|----------|-------------|
| WHO | 50-60% | +40-50% (subject fallback + casing) |
| WHERE | 35% | +5% (casing fixes) |
| WHEN | 10% | +10% (general improvements) |
| WHAT | 40% | +10% (casing + fallbacks) |
| HOW_MANY | 80% | - |
| **Overall** | **45-50%** | **+21-26%** |

### After Phase 3 (Est. 50-55% accuracy)

| Question Type | Accuracy | Improvement |
|---------------|----------|-------------|
| WHO | 60-70% | +10% (answer-centric scoring) |
| WHERE | 40% | +5% (better fact ranking) |
| WHEN | 15% | +5% |
| WHAT | 45% | +5% |
| **Overall** | **50-55%** | **+5%** |

---

## Part 6: Testing Strategy

### Unit Tests (Prevent Regressions)

```python
# tests/test_ast_answer_extraction.py

def test_who_question_with_nominative_subject():
    """Test WHO extraction when subject is in aliaj (nominative)."""
    query = "Kiu fondis Esperanton?"
    doc = "La genia doktoro Zamenhof kreis Esperanton."

    query_ast = parse(query)
    doc_ast = parse(doc)

    extractor = ASTAnswerExtractor()
    answer = extractor.extract_answer(query_ast, doc_ast, doc)

    assert answer is not None
    assert "Zamenhof" in answer['text']
    assert answer['confidence'] >= 0.7

def test_multi_doc_aggregation_preserves_casing():
    """Test that proper noun casing is preserved in aggregation."""
    candidates = [
        {'text': 'zamenhof', 'score': 0.9, 'rank': 1},
        {'text': 'Zamenhof', 'score': 0.9, 'rank': 2},
        {'text': 'ZAMENHOF', 'score': 0.8, 'rank': 3},
    ]

    extractor = ASTAnswerExtractor()
    result = extractor.aggregate_candidates(candidates, top_n=3)

    # Should prefer proper noun casing
    assert result['text'] == 'Zamenhof'
    assert result['text'] != 'zamenhof'

def test_who_question_fact_scoring():
    """Test that WHO questions score facts by agent, not entity."""
    fact = Fact(
        entity="Esperanton",
        relation=RelationType.CREATED_BY,
        arguments={'agent': "Zamenhof"},
        modifiers={}
    )

    scorer = FactImportanceScorer()
    score = scorer.score(fact, QuestionType.WHO, query_entity="esperant", metadata={})

    # Should have high score (query_entity matches fact.entity, agent is proper noun)
    assert score.final_score >= 0.7
```

### Integration Test

```python
def test_full_pipeline_who_question():
    """Test full pipeline: retrieval → extraction → answer."""
    query = "Kiu fondis Esperanton?"

    # Retrieve with AST role constraints
    retriever = WhooshRetriever(...)
    docs = retriever.retrieve(query_roots=['fond', 'esperant'], query_ast=parse(query))

    # Extract answer
    generator = ExtractiveAnswerGenerator(use_ast_extraction=True)
    answer = generator.generate(docs, query, QuestionType.WHO, query_entity="esperant")

    # Verify
    assert "Zamenhof" in answer.text
    assert len(answer.citations) > 0
    assert answer.citations[0].sentence_text contains "Zamenhof"
```

---

## Part 7: Prioritized Action Items

### Immediate (This Week)

1. ✅ **Commit AST-first retrieval** (Done - Phase 1-3 complete)
2. **Implement Phase 1**: Fix multi-doc aggregation casing (1 hour)
3. **Implement Phase 2**: Add subject fallback extraction (2 hours)
4. **Test on "Kiu fondis Esperanton?"**: Verify "Zamenhof" is returned
5. **Run 50-question evaluation**: Measure actual improvement

### Short-term (Next Week)

6. **Implement Phase 3**: Answer-centric fact scoring (3 hours)
7. **Add unit tests**: Prevent regressions (2 hours)
8. **Run full evaluation again**: Target 50-55% accuracy

### Long-term (Future Iterations)

9. **Phase 4**: Build unified ASTRoleExtractor (optional, 1-2 weeks)
10. **Parser improvements**: Consistently set subjekto/objekto fields
11. **Direct graph extraction**: Read from Kuzu graph instead of parsed dicts

---

## Conclusion

The AST-first retrieval is working correctly - we're retrieving the right documents. The problem is in answer extraction, which has two issues:

1. **Multi-doc aggregation loses proper noun casing** → Phase 1 fixes this
2. **Subject extraction fails when parser puts subject in aliaj** → Phase 2 fixes this

Implementing Phases 1-2 (3 hours total) should improve WHO question accuracy from 10% to 50-60%, bringing overall accuracy from 24% to 45-50%.

The key insight: **Don't build a new extraction architecture yet**. Fix the two bugs in the existing ASTAnswerExtractor, and it will work.
