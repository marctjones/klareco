# Deep Analysis: Query Expansion and Extraction Pattern Improvements
## Opus-Level Thinking on System Bottlenecks

**Date:** March 29, 2026
**Scope:** Query expansion (30% failures) and extraction patterns (27% failures)
**Goal:** Design improvements to reach 85% accuracy (from 53.3%)

---

## Executive Summary

The current system has two critical bottlenecks:

1. **Query Expansion (30% failures):** Answers not retrieved because queries don't match document vocabulary
2. **Extraction Patterns (27% failures):** Answers retrieved but extraction patterns fail to identify them

**Key Insight:** These are NOT independent problems - they interact through the M1 plausibility filter and reranker, creating cascade effects that amplify failures.

---

## Part 1: Query Expansion - Root Cause Analysis

### Current Approach (What We Have)

```python
# From scripts/demo_extractive_qa.py

Step 1: Extract roots from query AST
  "Kiu estis Lincoln?" → ['lincoln']

Step 2: Expand with manual synonyms (ULTRA-CONSERVATIVE)
  MANUAL_SYNONYMS = {
    'kre': ['iniciati'],   # ONLY iniciati
    'fond': ['iniciati'],  # Too conservative!
    'ling': ['lingv'],
  }

Step 3: Expand with embeddings (threshold=0.70)
  - Find cosine similar roots
  - Skip entity roots (proper names)
  - Top-5 similar terms only

Step 4: Expand with morphology
  - Reflexive ↔ transitive
  - Verb forms

Problem: Entity extraction → ['lincoln'] → Not expanded → No morphological variants
```

### Why This Fails (30% of Questions)

**Failure Mode 1: Proper Name Variations**
```
Query: "Kiu estis Lincoln?"
Expansion: ['lincoln'] (no expansion - it's an entity)
Corpus has: "Abraham Lincoln", "Prezidanto Lincoln", "Lincoln estis..."
Problem: "Lincoln" in isolation has low BM25 score
Result: Document not retrieved

What we need: ["lincoln", "abraham", "prezidento", "usona"]
```

**Failure Mode 2: Compound Terms**
```
Query: "Kiu estis Thomas Jefferson?"
Expansion: ['thomas', 'jefferson'] (both entities, no expansion)
Corpus has: "Thomas Jefferson estis tria usona prezidento..."
Problem: "Thomas" + "Jefferson" separately don't score well
Result: Need phrase matching, not just word matching

What we need: ["thomas jefferson" (phrase), "prezidento", "usona", "tria"]
```

**Failure Mode 3: Title/Role Without Name**
```
Query: "Kiu malkovris radioaktivecon?"
Expansion: ['malkover', 'radioaktiv'] + synonyms
Corpus has: "Marie Curie esploris radioaktivecon"
Problem: Different verb ("esploris" vs "malkovris")
Result: Synonym list too conservative, missing "esplor"

What we need: ["malkover", "esplor", "stud", "trov", "radioaktiv", "curie", "marie"]
```

**Failure Mode 4: Definitional WHAT Questions**
```
Query: "Kio estas basketbalo?"
Expansion: ['basketbal'] (no expansion - it's an entity/sport name)
Corpus has: "Basketbalo estas sporto..."
Problem: Simple match works IF document starts with definition
Result: Fails if definition is buried in longer article

What we need: ["basketbal", "sport", "lud", "team", "pilk"] + definitional patterns
```

---

## Part 2: Proposed Query Expansion Improvements

### Strategy: Multi-Level Expansion with Question-Type Awareness

```python
def expand_query_intelligently(query_ast, question_type, entity):
    """
    Question-type-aware expansion with cascade levels.

    Level 1: Core terms (always include)
    Level 2: Contextual expansion (question-type specific)
    Level 3: Associative expansion (semantic field)
    """

    expanded = set()

    # LEVEL 1: Core terms (no filtering)
    core_roots = extract_roots_from_ast(query_ast)
    expanded.update(core_roots)

    # LEVEL 2: Question-type-specific expansion
    if question_type == QuestionType.WHO:
        # WHO questions need title/role terms
        if entity:  # "Kiu estis Lincoln?" → entity="lincoln"
            expanded.update(expand_person_query(entity))
            # Returns: ["prezidento", "usona", "historio", "naskiĝ", "mort"]

        # Always add person indicators
        expanded.update(["person", "homo", "viro", "virino"])

    elif question_type == QuestionType.WHAT:
        # WHAT questions need definition terms
        expanded.update(["signif", "difin", "estas", "nomiĝ", "konsist"])

        # If entity is concrete noun, add category terms
        if entity and is_concrete_noun(entity):
            category = get_category(entity)  # "basketbalo" → "sporto"
            if category:
                expanded.add(category)

    elif question_type == QuestionType.WHEN:
        # WHEN questions need temporal terms
        expanded.update(["jar", "dat", "temp", "period", "epok"])

    elif question_type == QuestionType.WHERE:
        # WHERE questions need location terms
        expanded.update(["lok", "urb", "land", "region", "situ"])

    # LEVEL 3: Semantic field expansion
    for root in core_roots:
        if not is_entity_root(root):
            # Get semantic field (verb class, noun category)
            field = get_semantic_field(root)
            expanded.update(field[:3])  # Top-3 from field

    # LEVEL 4: Morphological expansion (all forms)
    morphological = expand_morphology_aggressive(expanded)
    expanded.update(morphological)

    return expanded


def expand_person_query(person_root):
    """
    Expand person name with contextual terms.

    Strategy:
    1. Check person gazetteer (if exists)
    2. Add title/role terms for that person
    3. Add nationality/affiliation terms
    4. Add biographical event terms
    """
    from klareco.knowledge import person_gazetteer, person_roles

    context_terms = set()

    # Check if we know this person
    if person_root in person_gazetteer:
        info = person_gazetteer[person_root]
        context_terms.update(info.get('roles', []))      # ["prezidento"]
        context_terms.update(info.get('affiliations', []))  # ["usona"]
        context_terms.update(info.get('known_for', []))  # ["unua", "fondinto"]

    # Generic person query terms
    context_terms.update([
        "nask", "mort", "viv",      # birth, death, life
        "fond", "kre", "establ",    # founding, creating
        "prezident", "reĝ", "direktor",  # titles
    ])

    return context_terms
```

### Specific Improvements

**Improvement 1: Proper Name Context Enrichment**
```python
# BEFORE:
"Kiu estis Lincoln?" → ['lincoln']  # 30% recall

# AFTER:
"Kiu estis Lincoln?" → [
    'lincoln',           # Core
    'abraham',           # First name (from gazetteer)
    'prezidento',        # Role (from gazetteer)
    'usona',             # Nationality (from gazetteer)
    'nask', 'mort',      # Biography terms
    'historio'           # Domain term
]
# Expected: 70% → 95% recall
```

**Improvement 2: Phrase Matching for Compound Names**
```python
# BEFORE:
"Thomas Jefferson" → ['thomas'] OR ['jefferson']  # Weak scoring

# AFTER:
def build_query_with_phrases(roots, entities):
    """
    Use Whoosh phrase queries for multi-word entities.
    """
    terms = []
    phrases = []

    # Detect multi-word entities
    if len(entities) > 1:
        # "thomas jefferson" as exact phrase
        phrase = " ".join(entities)
        phrases.append(phrase)

        # Also individual words (fallback)
        terms.extend(entities)

    # Build Whoosh query
    query_str = ""
    for phrase in phrases:
        query_str += f'text:"{phrase}"^3.0 '  # Boost phrases 3x
    for term in terms:
        query_str += f'text:{term} '

    return query_str

# "Thomas Jefferson" → phrase match + word match + role terms
# Expected: 40% → 85% recall
```

**Improvement 3: Aggressive Verb Synonym Expansion**
```python
# BEFORE:
MANUAL_SYNONYMS = {
    'kre': ['iniciati'],  # ONLY one synonym
}

# AFTER:
from klareco.knowledge import get_verb_class_synonyms

VERB_CLASS_SYNONYMS = {
    # Creation verbs (HIGH priority for WHO questions)
    'creation': ['kre', 'fond', 'establ', 'iniciati', 'komenc', 'origin', 'invent', 'desegn'],

    # Discovery verbs (HIGH priority for WHO questions)
    'discovery': ['malkover', 'trov', 'esplor', 'detekt', 'konstant', 'observ'],

    # Writing/authoring verbs
    'authoring': ['verk', 'skrib', 'publikig', 'autor', 'kompil'],

    # Leadership verbs
    'leadership': ['direkt', 'gvid', 'kondukt', 'reg', 'administr'],
}

def expand_verb_aggressively(verb_root, question_type):
    """
    Question-type-aware verb expansion.

    WHO questions: Expand aggressively (need to find actor)
    WHAT questions: Expand conservatively (avoid noise)
    """
    if question_type == QuestionType.WHO:
        # WHO questions: aggressive expansion
        verb_class = classify_verb(verb_root)
        if verb_class:
            return VERB_CLASS_SYNONYMS[verb_class]

    # Other question types: conservative
    return get_synonyms(verb_root)[:2]  # Top-2 only
```

**Expected Impact:**
- Retrieval recall: 70% → 90% (+20%)
- Answer in top-20: 21/30 → 27/30 (+6 questions fixed)
- **Overall accuracy: 53% → 73% (+20%)**

---

## Part 3: Extraction Patterns - Root Cause Analysis

### Current Approach (What We Have)

```python
# From klareco/rag/extractive_answering.py (simplified)

Step 1: Extract facts from all retrieved sentences
  for sentence in sentences:
    facts = extract_facts(sentence)  # Get all (subj, verb, obj) triples

Step 2: Filter by question type
  facts = filter_by_question_type(facts, question_type)

Step 3: Apply M1 plausibility filter
  facts = m1_filter(facts, query)  # Remove implausible facts

Step 4: Rank by importance
  facts = rank_facts(facts)

Problem: extract_facts() returns ALL facts, doesn't verify query match
```

### Why This Fails (27% of Questions)

**From CORRECTED_ANALYSIS.md:**

**Failure Example 1: Object Mismatch**
```
Query: "Kiu fondis Esperanton?"
       ↓
Sentence: "En 1991 oni fondis GIL kiel asocio..."
       ↓
Extraction: (oni, fondis, GIL)  ✓ Extracted
       ↓
Problem: Extracts "fondis GIL" but query asks about "Esperanton"
       ↓
M1 Filter: PASSES (plausible that "someone founded GIL")
       ↓
Result: WRONG answer included

What's missing: Object verification
Should check: Does extracted object match query object?
```

**Failure Example 2: Definition Pattern Missing**
```
Query: "Kio estas basketbalo?"
       ↓
Sentence: "Basketbalo estas usona sporto en kiu..."
       ↓
Extraction: Tries to find (subject, verb, object) triple
       ↓
Problem: Definition pattern not recognized ("X estas Y" = definition)
       ↓
Result: Extracts something else or nothing

What's missing: Definition pattern matcher
Should recognize: "X estas [DEFINITION]" pattern
```

**Failure Example 3: Temporal Extraction**
```
Query: "Kiam okazis la Vendo de Luiziano?"
       ↓
Sentence: "La Vendo de Luiziano okazis en 1803..."
       ↓
Extraction: (Vendo, okazis, ?)  ← Missing temporal argument
       ↓
Problem: Date "en 1803" not extracted as fact argument
       ↓
Result: Can't generate answer with date

What's missing: Temporal argument extraction
Should extract: (event, temporal_marker, date)
```

---

## Part 4: Proposed Extraction Pattern Improvements

### Strategy: Multi-Pattern Extraction with Query-Aware Matching

```python
class ImprovedExtractor:
    """
    Question-type-aware extraction with pattern library.
    """

    def extract_with_verification(self, sentence, query_ast, question_type):
        """
        Extract facts using question-type-specific patterns,
        with query constraint verification.
        """
        # Get query constraints
        query_verb = extract_verb(query_ast)  # "fondis"
        query_object = extract_object(query_ast)  # "esperanton"
        query_subject = extract_subject(query_ast)  # usually interrogative

        # Route to pattern matcher based on question type
        if question_type == QuestionType.WHO:
            facts = self.extract_who_pattern(
                sentence, query_verb, query_object
            )

        elif question_type == QuestionType.WHAT:
            facts = self.extract_what_pattern(
                sentence, query_subject
            )

        elif question_type == QuestionType.WHEN:
            facts = self.extract_when_pattern(
                sentence, query_verb, query_object
            )

        elif question_type == QuestionType.WHERE:
            facts = self.extract_where_pattern(
                sentence, query_verb, query_object
            )

        return facts


    def extract_who_pattern(self, sentence, query_verb, query_object):
        """
        WHO questions: Find PERSON who performed action on object.

        Pattern: [PERSON] [VERB] [OBJECT]
        Constraint: OBJECT must match query object (or synonym)
        """
        facts = []

        # Extract (subject, verb, object) triples from AST
        triples = get_svo_triples(sentence['ast'])

        for (subj, verb, obj) in triples:
            # Verify verb matches query verb (or synonym)
            if not verbs_match(verb, query_verb):
                continue

            # **CRITICAL: Verify object matches query object**
            if query_object and not objects_match(obj, query_object):
                continue  # FILTER OUT mismatches!

            # Check if subject is a person
            if is_person(subj):
                facts.append({
                    'type': 'who_answer',
                    'person': subj,
                    'action': verb,
                    'object': obj,
                    'confidence': 0.9  # High confidence (verified match)
                })

        return facts


    def extract_what_pattern(self, sentence, query_subject):
        """
        WHAT questions: Find definitions or descriptions.

        Patterns:
        1. "X estas Y" (X is Y) - definition
        2. "X konsistas el Y" (X consists of Y) - composition
        3. "X signifas Y" (X means Y) - meaning
        """
        facts = []
        ast = sentence['ast']

        # Pattern 1: "X estas Y" (definitional copula)
        if self.is_definition_pattern(ast, query_subject):
            definition = extract_definition(ast)
            facts.append({
                'type': 'definition',
                'term': query_subject,
                'definition': definition,
                'confidence': 0.95  # Very high (explicit definition)
            })

        # Pattern 2: "X konsistas el Y"
        elif self.is_composition_pattern(ast, query_subject):
            composition = extract_composition(ast)
            facts.append({
                'type': 'composition',
                'term': query_subject,
                'parts': composition,
                'confidence': 0.85
            })

        # Pattern 3: Descriptive sentence containing subject
        elif contains_subject(ast, query_subject):
            description = extract_description(ast, query_subject)
            facts.append({
                'type': 'description',
                'term': query_subject,
                'description': description,
                'confidence': 0.70  # Lower (not explicit definition)
            })

        return facts


    def is_definition_pattern(self, ast, query_subject):
        """
        Check if AST matches: "[query_subject] estas [DEFINITION]"

        Example:
        AST: {tipo: 'frazo', subjekto: {radiko: 'basketbal'},
              verbo: {radiko: 'est'}, objekto: {radiko: 'sport'}}
        Query: "Kio estas basketbalo?" → query_subject="basketbal"
        Result: True (matches definition pattern)
        """
        # Check subject matches query
        subj = ast.get('subjekto', {})
        if not subject_matches(subj, query_subject):
            return False

        # Check verb is copula ("esti")
        verb = ast.get('verbo', {})
        if verb.get('radiko') != 'est':
            return False

        # Check there's a predicate (definition)
        obj = ast.get('objekto')
        if not obj:
            return False

        return True


    def extract_when_pattern(self, sentence, query_verb, query_object):
        """
        WHEN questions: Find temporal expressions.

        Patterns:
        1. "en [YEAR]" (in [year])
        2. "je [DATE]" (at [date])
        3. "[MONTH] [YEAR]"
        4. Relative time: "antaŭ [NUMBER] jaroj"
        """
        facts = []
        ast = sentence['ast']

        # Find temporal expressions in 'aliaj' (modifiers)
        temporal_expressions = extract_temporal_from_aliaj(ast)

        # Verify the sentence is about the query event
        if query_verb and query_object:
            if not mentions_event(ast, query_verb, query_object):
                return []  # Not about the right event

        for temp_expr in temporal_expressions:
            facts.append({
                'type': 'temporal',
                'event': {
                    'verb': query_verb,
                    'object': query_object
                },
                'time': temp_expr,
                'confidence': 0.90
            })

        return facts
```

### Specific Pattern Improvements

**Pattern 1: Object Verification for WHO Questions**
```python
# BEFORE (from CORRECTED_ANALYSIS.md example):
Query: "Kiu fondis Esperanton?"
Sentence: "En 1991 oni fondis GIL..."
Extraction: (oni, fondis, GIL)  ✓ Extracted
Result: WRONG ANSWER (extracts "fondis GIL")

# AFTER:
Query: "Kiu fondis Esperanton?"
Sentence: "En 1991 oni fondis GIL..."
Extraction Check:
  - Verb matches? fondis == fondis ✓
  - Object matches? GIL == Esperanton ✗
  - FILTER OUT (object mismatch)
Result: Fact rejected, look for correct match

Query: "Kiu fondis Esperanton?"
Sentence: "Zamenhof fondis Esperanton en 1887..."
Extraction Check:
  - Verb matches? fondis == fondis ✓
  - Object matches? Esperanton == Esperanton ✓
  - Subject is person? Zamenhof = person ✓
  - ACCEPT
Result: CORRECT ANSWER

Expected: Fixes 9 questions with object mismatch (~18% of failures)
```

**Pattern 2: Definition Recognition for WHAT Questions**
```python
# BEFORE:
Query: "Kio estas basketbalo?"
Sentence: "Basketbalo estas usona sporto en kiu..."
Extraction: Tries generic (subj, verb, obj) extraction
Result: Fails or extracts unrelated fact

# AFTER:
Query: "Kio estas basketbalo?"
Sentence: "Basketbalo estas usona sporto en kiu..."
Pattern Match:
  - Subject == query? basketbalo == basketbalo ✓
  - Verb == "estas"? ✓
  - Has predicate? "usona sporto..." ✓
  - DEFINITION PATTERN MATCHED
Extraction: {
    'type': 'definition',
    'term': 'basketbalo',
    'definition': 'usona sporto en kiu...',
    'confidence': 0.95
}
Result: CORRECT ANSWER

Expected: Fixes 4-5 definition questions (~15% of failures)
```

**Pattern 3: Temporal Extraction for WHEN Questions**
```python
# BEFORE:
Query: "Kiam okazis la Vendo de Luiziano?"
Sentence: "La Vendo okazis en 1803..."
Extraction: (Vendo, okazis, ?) ← Missing date
Result: Can't generate complete answer

# AFTER:
Query: "Kiam okazis la Vendo de Luiziano?"
Sentence: "La Vendo okazis en 1803..."
Pattern Match:
  - Find temporal markers in 'aliaj': "en 1803" ✓
  - Verify sentence about query event: "Vendo" ✓
  - Extract temporal expression: {year: 1803, marker: "en"}
Extraction: {
    'type': 'temporal',
    'event': 'Vendo de Luiziano',
    'time': '1803',
    'confidence': 0.90
}
Result: CORRECT ANSWER "en 1803"

Expected: Fixes 2-3 temporal questions (~10% of failures)
```

---

## Part 5: Pipeline Cascade Effects

### How Fixes Interact

**Cascade 1: Query Expansion → Retrieval → Extraction**
```
Improved query expansion
  ↓
More relevant documents retrieved (70% → 90% recall)
  ↓
Better match quality in retrieved set
  ↓
Extraction patterns have better input
  ↓
Higher extraction success rate (45% → 70%)
  ↓
MULTIPLIER EFFECT: 0.90 * 0.70 = 63% → 73% (+10%)
```

**Cascade 2: Extraction Patterns → M1 Filter → Ranking**
```
Improved extraction with object verification
  ↓
Fewer incorrect facts extracted (noise reduction)
  ↓
M1 filter has less noise to filter
  ↓
M1 filter rate drops (96% → 85%)
  ↓
More correct facts survive filtering
  ↓
Ranking/discourse planning has better input
  ↓
QUALITY IMPROVEMENT: Cleaner, more accurate answers
```

**Cascade 3: Both Fixes → Reranker**
```
Better retrieval (more relevant docs)
  +
Better extraction (verified facts)
  ↓
Reranker sees higher-quality candidates
  ↓
Current reranker performance improves
  ↓
May eliminate need for reranker retraining
```

### Anti-Patterns to Avoid

**Anti-Pattern 1: Over-Expansion**
```
Problem: Expanding query too aggressively
  "Kiu fondis Esperanton?" → 50 synonym terms
  ↓
Result: Retrieves too many irrelevant documents
  ↓
Extraction overwhelmed with noise
  ↓
M1 filter rate → 99% (filtering almost everything)
  ↓
NET EFFECT: WORSE performance

Mitigation: Question-type-aware expansion limits
  WHO: Expand aggressively (need to find person)
  WHAT: Expand conservatively (avoid definitions of other things)
```

**Anti-Pattern 2: Pattern Over-Specificity**
```
Problem: Extraction patterns too rigid
  "X estas Y" requires exact AST structure
  ↓
Result: Misses valid definitions in other forms
  "X, kiu estas Y..." (relative clause definition)
  "Y nomata X" (named Y called X)
  ↓
NET EFFECT: Lower recall

Mitigation: Pattern variants + confidence scores
  Exact pattern: confidence 0.95
  Variant pattern: confidence 0.75
  Generic pattern: confidence 0.50
```

---

## Part 6: Implementation Strategy

### Phase 1: Query Expansion (Expected +20%, 2 days)

**Priority Order:**
1. **Proper name context enrichment** (1 day)
   - Build person_gazetteer with roles/affiliations
   - Add expand_person_query() function
   - Expected: +10% (5 questions fixed)

2. **Phrase matching for compounds** (0.5 days)
   - Add phrase query support to Whoosh retriever
   - Boost phrase matches 3x over word matches
   - Expected: +5% (2-3 questions fixed)

3. **Aggressive verb synonym expansion** (0.5 days)
   - Add verb class synonyms (creation, discovery, authoring)
   - Make expansion question-type-aware
   - Expected: +5% (2 questions fixed)

**Files to Modify:**
- `scripts/demo_extractive_qa.py` - expand_query functions
- `klareco/knowledge/gazetteers.py` - add person_gazetteer
- `klareco/knowledge/synonyms.py` - add verb_class_synonyms
- `klareco/rag/whoosh_retriever.py` - add phrase query support

**Testing:**
```bash
# Test on failed retrieval cases
python scripts/demo_extractive_qa.py "Kiu estis Lincoln?" --verbose
python scripts/demo_extractive_qa.py "Kiu malkovris radioaktivecon?" --verbose

# Run evaluation
python scripts/evaluate_pipeline_comprehensive.py --output results/after_expansion_fix.json
```

### Phase 2: Extraction Patterns (Expected +15%, 2 days)

**Priority Order:**
1. **Object verification for WHO questions** (1 day)
   - Add objects_match() function
   - Add verification to extract_who_pattern()
   - Expected: +10% (5 questions fixed)

2. **Definition pattern for WHAT questions** (0.5 days)
   - Add is_definition_pattern() function
   - Add extract_definition() function
   - Expected: +3% (2 questions fixed)

3. **Temporal extraction for WHEN questions** (0.5 days)
   - Add extract_temporal_from_aliaj() function
   - Add temporal pattern matching
   - Expected: +2% (1 question fixed)

**Files to Modify:**
- `klareco/rag/extractive_answering.py` - extraction patterns
- `klareco/knowledge/temporal.py` - temporal patterns
- Add new file: `klareco/rag/pattern_matchers.py` - pattern library

**Testing:**
```bash
# Test on failed extraction cases
python scripts/demo_extractive_qa.py "Kiu fondis Esperanton?" --verbose
python scripts/demo_extractive_qa.py "Kio estas basketbalo?" --verbose

# Run evaluation
python scripts/evaluate_pipeline_comprehensive.py --output results/after_extraction_fix.json
```

### Phase 3: Integration and Optimization (1 day)

**Tasks:**
1. Test cascade effects
2. Tune confidence thresholds
3. Adjust M1 filter parameters
4. Re-evaluate on full suite

**Expected Final Results:**
```
Baseline: 53.3% (16/30)
After Query Expansion: 73.3% (22/30)  [+20%]
After Extraction Fix: 88.3% (26.5/30) [+15%]
Target: 85%+ accuracy ✓ ACHIEVED
```

---

## Part 7: Risk Analysis

### Risk 1: Over-Expansion Causing Noise

**Likelihood:** Medium
**Impact:** High (could reduce accuracy)

**Symptoms:**
- Retrieval recall increases but precision drops
- M1 filter rate increases dramatically (>98%)
- More facts extracted but fewer correct ones

**Mitigation:**
- Start conservative, measure retrieval precision
- Monitor M1 filter rate (should stay <95%)
- Use question-type-aware expansion limits
- A/B test: conservative vs aggressive expansion

### Risk 2: Pattern Matching Too Rigid

**Likelihood:** Medium
**Impact:** Medium (might not improve much)

**Symptoms:**
- Extraction patterns match few sentences
- No improvement in extraction success rate
- Same questions still failing

**Mitigation:**
- Implement pattern variants with different confidences
- Log pattern match failures for debugging
- Add fallback to generic extraction
- Build pattern test suite

### Risk 3: Cascade Effects Unpredictable

**Likelihood:** Low
**Impact:** Medium (unexpected behavior)

**Symptoms:**
- Accuracy changes in unexpected ways
- Different questions start failing
- M1 filter behavior changes drastically

**Mitigation:**
- Implement changes incrementally
- Run evaluation after each change
- Keep baseline configuration for comparison
- Document all parameter changes

### Risk 4: Implementation Takes Longer

**Likelihood:** High
**Impact:** Low (just timing)

**Mitigation:**
- Break into small testable pieces
- Test each piece independently
- Have rollback plan (git branches)
- Focus on high-impact changes first

---

## Part 8: Expected Accuracy Trajectory

### Conservative Estimate (Lower Bound)

```
Baseline: 53.3% (16/30)

After Query Expansion (Phase 1):
  - Proper names: +6% (2 questions)
  - Phrases: +3% (1 question)
  - Verb synonyms: +3% (1 question)
  Subtotal: 65.3% (19.6/30) → Round to 66.7% (20/30)

After Extraction Patterns (Phase 2):
  - Object verification: +7% (2 questions)
  - Definition patterns: +3% (1 question)
  - Temporal extraction: +3% (1 question)
  Subtotal: 79.7% (23.9/30) → Round to 80% (24/30)

Conservative target: 80% accuracy
```

### Optimistic Estimate (Upper Bound)

```
Baseline: 53.3% (16/30)

After Query Expansion (Phase 1):
  - Proper names: +10% (3 questions)
  - Phrases: +7% (2 questions)
  - Verb synonyms: +7% (2 questions)
  Cascade effects: +3% (1 question)
  Subtotal: 80% (24/30)

After Extraction Patterns (Phase 2):
  - Object verification: +10% (3 questions)
  - Definition patterns: +3% (1 question)
  - Temporal extraction: +3% (1 question)
  Cascade effects: +3% (1 question)
  Subtotal: 93% (28/30)

Optimistic target: 90%+ accuracy
```

### Realistic Estimate (Most Likely)

```
Baseline: 53.3% (16/30)

Phase 1: Query Expansion
  Expected: +16.7% (5 questions fixed)
  Result: 70% (21/30)

Phase 2: Extraction Patterns
  Expected: +16.7% (5 questions fixed)
  Result: 86.7% (26/30)

Realistic target: 85-87% accuracy
```

---

## Conclusion

**Recommendation:** Implement both fixes in sequence, Phase 1 then Phase 2.

**Rationale:**
1. Query expansion improves retrieval (prerequisite for extraction)
2. Extraction patterns benefit from better retrieval quality
3. Both fixes have positive cascade effects
4. Minimal risk of negative interactions
5. Clear testing strategy at each phase

**Timeline:**
- Phase 1 (Query Expansion): 2 days
- Phase 2 (Extraction Patterns): 2 days
- Phase 3 (Integration): 1 day
- **Total: 5 days to 85%+ accuracy**

**Next Step:** User approval to proceed with implementation.
