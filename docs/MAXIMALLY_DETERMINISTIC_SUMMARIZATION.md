# Maximally Deterministic Summarization Analysis

## Critical Question: Do We Actually Need These Models?

Let's analyze each proposed model to see if Esperanto's regularity + AST annotations can replace it with deterministic processing.

---

## Model 1: Semantic Importance Adjuster (2M params)

### Original Purpose
Adjust deterministic importance scores using semantic understanding.

### Can We Replace With Deterministic Processing?

**YES! We already have root embeddings (320K params, existing).**

```python
# Instead of training a 2M model to adjust scores...
# Use existing embeddings + deterministic formula!

def adjust_importance_deterministically(info_unit, query_ast, base_score):
    """Adjust using existing embeddings (0 new params)"""

    # Extract roots
    unit_roots = extract_roots(info_unit['ast'])
    query_roots = extract_roots(query_ast)

    # Compute semantic similarity using EXISTING embeddings
    similarities = []
    for unit_root in unit_roots:
        for query_root in query_roots:
            unit_emb = get_root_embedding(unit_root)  # Existing 320K model
            query_emb = get_root_embedding(query_root)
            sim = cosine_similarity(unit_emb, query_emb)
            similarities.append(sim)

    if not similarities:
        return base_score

    avg_similarity = mean(similarities)

    # Deterministic adjustment formula
    if avg_similarity > 0.7:  # High semantic overlap
        adjustment = +0.15
    elif avg_similarity > 0.5:  # Medium overlap
        adjustment = +0.10
    elif avg_similarity > 0.3:  # Low overlap
        adjustment = +0.05
    else:
        adjustment = 0.0

    return min(base_score + adjustment, 1.0)
```

**Result**: ✅ **0 new learned parameters** (reuse existing root embeddings)

---

## Model 2: Topic Assignment Classifier (3M params)

### Original Purpose
Assign information units to topics.

### Can We Replace With Deterministic Processing?

**YES! Use Kuzu graph queries + root overlap.**

```python
def assign_topic_deterministically(info_unit, topic_candidates, kuzu_db):
    """Assign to topic using graph queries (0 new params)"""

    unit_entities = extract_entities(info_unit['ast'])
    unit_roots = extract_roots(info_unit['ast'])

    topic_scores = {}

    for topic in topic_candidates:
        score = 0.0

        # Method 1: Entity co-occurrence in Kuzu (deterministic query)
        for entity in unit_entities:
            co_occurrence = kuzu_db.query("""
                MATCH (s:Frazoteksto)-[:MENTIONS_ENTITY]->(e1:Entity {name: $entity})
                MATCH (s)-[:MENTIONS_ENTITY]->(e2:Entity)
                WHERE e2.name IN $topic_entities
                RETURN count(s) as co_count
            """, {'entity': entity, 'topic_entities': topic['entities']})

            score += co_occurrence / 100.0  # Normalize

        # Method 2: Root overlap (deterministic)
        topic_roots = set(topic['characteristic_roots'])
        overlap = len(unit_roots & topic_roots) / len(unit_roots)
        score += overlap * 0.5

        # Method 3: Semantic similarity using EXISTING embeddings
        for unit_root in unit_roots:
            for topic_root in topic_roots:
                sim = cosine_similarity(
                    get_root_embedding(unit_root),
                    get_root_embedding(topic_root)
                )
                score += sim * 0.3

        topic_scores[topic['id']] = score

    # Assign to highest-scoring topic
    best_topic = max(topic_scores, key=topic_scores.get)
    return best_topic
```

**Result**: ✅ **0 new learned parameters** (Kuzu queries + existing embeddings)

---

## Model 3: Sentence Construction Planner (2M params)

### Original Purpose
Decide which information units go in the same sentence.

### Can We Replace With Deterministic Processing?

**YES! AST structure explicitly encodes syntactic relationships.**

```python
def group_units_into_sentences_deterministically(selected_units):
    """Use AST structure to group (0 new params)"""

    sentences = []
    used_units = set()

    for unit in selected_units:
        if unit['id'] in used_units:
            continue

        # Start new sentence with this unit
        sentence_units = [unit]
        used_units.add(unit['id'])

        # RULE 1: Add syntactically-related units (deterministic from AST)
        for other_unit in selected_units:
            if other_unit['id'] in used_units:
                continue

            # Check AST relationships
            if has_syntactic_relation(unit, other_unit):
                # Examples:
                # - unit is subject, other_unit is verb of same clause
                # - unit is noun, other_unit is modifier of that noun
                # - unit is verb, other_unit is object of that verb
                sentence_units.append(other_unit)
                used_units.add(other_unit['id'])

        # RULE 2: Add appositives (deterministic from AST)
        for other_unit in selected_units:
            if other_unit['id'] in used_units:
                continue

            if other_unit['type'] == 'appositive' and other_unit['modifies'] == unit['id']:
                sentence_units.append(other_unit)
                used_units.add(other_unit['id'])

        # RULE 3: Add satellites to nucleus (deterministic from RST)
        for other_unit in selected_units:
            if other_unit['id'] in used_units:
                continue

            if (unit['rst_role'] == 'nucleus' and
                other_unit['rst_role'] == 'satellite' and
                other_unit['rst_parent'] == unit['id']):
                sentence_units.append(other_unit)
                used_units.add(other_unit['id'])

        # RULE 4: Don't make sentences too long (deterministic threshold)
        if len(sentence_units) > 5:  # Max 5 units per sentence
            # Keep only highest importance units
            sentence_units = sorted(sentence_units,
                                   key=lambda u: u['importance'],
                                   reverse=True)[:5]

        sentences.append(sentence_units)

    return sentences

def has_syntactic_relation(unit1, unit2):
    """Check if units have syntactic relation in AST (deterministic)"""

    # Same sentence?
    if unit1['sentence_id'] != unit2['sentence_id']:
        return False

    # Subject-verb relation?
    if (unit1['ast_role'] == 'subjekto' and unit2['ast_role'] == 'verbo'):
        return True

    # Verb-object relation?
    if (unit1['ast_role'] == 'verbo' and unit2['ast_role'] == 'objekto'):
        return True

    # Noun-modifier relation?
    if unit2['path'].startswith(unit1['path']):
        return True  # unit2 is child of unit1 in AST

    return False
```

**Result**: ✅ **0 new learned parameters** (AST structure is explicit)

---

## Model 4: Discourse Ordering Model (2M params)

### Original Purpose
Order information units within sentences.

### Can We Replace With Deterministic Processing?

**YES! Esperanto grammar rules + RST + information structure.**

```python
def order_units_in_sentence_deterministically(units):
    """Order using linguistic principles (0 new params)"""

    # RULE 1: Esperanto word order (deterministic)
    # Default: SVO (Subject-Verb-Object)

    ordered = []

    # Step 1: Add subject first
    subjects = [u for u in units if u['ast_role'] == 'subjekto']
    ordered.extend(subjects)

    # Step 2: Add appositives after subject (deterministic from AST)
    for subject in subjects:
        appositives = [u for u in units if u['type'] == 'appositive' and u['modifies'] == subject['id']]
        ordered.extend(appositives)

    # Step 3: Add verb
    verbs = [u for u in units if u['ast_role'] == 'verbo']
    ordered.extend(verbs)

    # Step 4: Add object
    objects = [u for u in units if u['ast_role'] == 'objekto']
    ordered.extend(objects)

    # Step 5: Add modifiers (deterministic ordering by importance)
    modifiers = [u for u in units if u['type'] in ['temporal', 'locative', 'manner']]

    # RST principle: Nucleus before satellite (deterministic)
    nucleus_modifiers = [u for u in modifiers if u['rst_role'] == 'nucleus']
    satellite_modifiers = [u for u in modifiers if u['rst_role'] == 'satellite']

    ordered.extend(nucleus_modifiers)
    ordered.extend(satellite_modifiers)

    # RULE 2: Information structure (deterministic from query)
    # Topic (given from query) should come first
    # This is already handled by SVO order above

    # RULE 3: If temporal modifier, can go first for emphasis (optional topicalization)
    # Only if explicitly marked as topicalized in AST
    temporal = [u for u in units if u['type'] == 'temporal' and u.get('topicalized')]
    if temporal:
        ordered = temporal + [u for u in ordered if u not in temporal]

    return ordered
```

**Result**: ✅ **0 new learned parameters** (grammar rules + RST)

---

## Model 5: Paragraph Break Predictor (1M params)

### Original Purpose
Decide where to insert paragraph breaks.

### Can We Replace With Deterministic Processing?

**YES! Topic changes + discourse boundaries.**

```python
def insert_paragraph_breaks_deterministically(sentences):
    """Insert breaks using deterministic rules (0 new params)"""

    paragraphs = []
    current_paragraph = []

    for i, sentence in enumerate(sentences):
        current_paragraph.append(sentence)

        # Check if should break after this sentence
        should_break = False

        # RULE 1: Topic change (deterministic)
        if i < len(sentences) - 1:
            current_topic = get_majority_topic(sentence)
            next_topic = get_majority_topic(sentences[i + 1])

            if current_topic != next_topic:
                should_break = True

        # RULE 2: RST major discourse boundary (deterministic)
        if sentence_has_major_discourse_boundary(sentence, sentences[i + 1] if i < len(sentences) - 1 else None):
            should_break = True

        # RULE 3: Paragraph length threshold (deterministic)
        if len(current_paragraph) >= 4:  # Max 4 sentences per paragraph
            should_break = True

        # RULE 4: Last sentence (deterministic)
        if i == len(sentences) - 1:
            should_break = True

        if should_break:
            paragraphs.append(current_paragraph)
            current_paragraph = []

    return paragraphs

def sentence_has_major_discourse_boundary(sent1, sent2):
    """Detect major discourse boundary (deterministic from connectives)"""

    if sent2 is None:
        return False

    # Check if sent2 starts with major discourse marker
    sent2_roots = extract_roots(sent2['ast'])

    major_markers = {
        'sed',      # but (contrast)
        'tamen',    # however (contrast)
        'cetere',   # moreover (addition of major point)
        'finfine',  # finally (conclusion)
        'rezulte',  # as a result (major consequence)
    }

    # If sentence starts with major marker → major boundary
    if sent2_roots and sent2_roots[0] in major_markers:
        return True

    return False

def get_majority_topic(sentence):
    """Get dominant topic of sentence (deterministic)"""

    # Count units per topic in this sentence
    topic_counts = {}
    for unit in sentence['units']:
        topic = unit['topic']
        topic_counts[topic] = topic_counts.get(topic, 0) + 1

    # Return most common topic
    return max(topic_counts, key=topic_counts.get)
```

**Result**: ✅ **0 new learned parameters** (rules from discourse theory)

---

## Summary: Learned vs Deterministic

| Component | Original | After Analysis | New Params | Rationale |
|-----------|----------|----------------|------------|-----------|
| **Model 1: Importance** | 2M learned | ✅ Deterministic | **0** | Reuse existing root embeddings (320K) |
| **Model 2: Topics** | 3M learned | ✅ Deterministic | **0** | Kuzu queries + root overlap |
| **Model 3: Grouping** | 2M learned | ✅ Deterministic | **0** | AST structure is explicit |
| **Model 4: Ordering** | 2M learned | ✅ Deterministic | **0** | Grammar rules + RST |
| **Model 5: Paragraphs** | 1M learned | ✅ Deterministic | **0** | Topic changes + discourse markers |
| **Total** | 10M | **0M** | **0** | 🎉 Fully deterministic! |

## Why This Works for Esperanto (But Not English)

### Esperanto Advantages

1. **Explicit case marking (-n)**
   - Subject vs object clear from morphology
   - Enables flexible word order analysis
   - No ambiguity in syntactic roles

2. **Regular grammar (zero exceptions)**
   - SVO is default, variations are marked
   - Appositives have clear position rules
   - Relative clauses use "kiu" consistently

3. **Compositional morphology**
   - "fundinto" = "fund-int-o" = one who founded
   - Enables deterministic coreference
   - Derivational affixes have clear semantics

4. **Explicit discourse markers**
   - "ĉar" = because (cause)
   - "por" = for (purpose)
   - "sed" = but (contrast)
   - All unambiguous!

5. **AST structure captures everything**
   - Syntactic relations explicit
   - RST relations identifiable from connectives
   - Information structure (topic/focus) from position

### English Would Need Learning

In English, all 5 models would be necessary:

1. **Importance**: Unclear which parts of sentence are important (no case marking)
2. **Topics**: Word order ambiguous, need to learn patterns
3. **Grouping**: Syntax ambiguous (e.g., "I saw the man with the telescope")
4. **Ordering**: Many valid orderings, need to learn preferences
5. **Paragraphs**: Discourse markers less explicit, need to learn

**Esperanto makes the implicit explicit!**

---

## Revised Architecture: 100% Deterministic

```
┌─────────────────────────────────────────────────────────────┐
│ Input: Query + Retrieved Sentences                         │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Parse to ASTs (Deterministic)                              │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Extract Information Units (Deterministic)                  │
│   - Use AST boundaries                                      │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Build Information Graph (Deterministic + Existing Embeddings)│
│   - Syntactic edges (AST)                                  │
│   - Coreference edges (Esperanto rules)                    │
│   - Entity edges (Kuzu queries)                            │
│   - Semantic edges (cosine similarity, existing embeddings) │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Compute Base Importance (Deterministic)                    │
│   - PageRank                                               │
│   - Entity salience (Kuzu)                                │
│   - Entropy                                                │
│   - Information structure                                  │
│   - RST nucleus/satellite                                  │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Adjust Importance (Deterministic Formula)                  │
│   - Use existing root embeddings (320K, already trained)   │
│   - Cosine similarity with query                           │
│   - Deterministic adjustment thresholds                    │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Identify Topics (Deterministic)                            │
│   - Root overlap clustering                                │
│   - Kuzu co-occurrence queries                            │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Assign Units to Topics (Deterministic)                     │
│   - Entity co-occurrence in Kuzu                           │
│   - Root overlap                                           │
│   - Semantic similarity (existing embeddings)              │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Select Units (Deterministic Threshold)                     │
│   - importance > threshold for target length               │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Group into Sentences (Deterministic from AST)              │
│   - Syntactic constraints (subject-verb-object)            │
│   - Appositives follow nouns                               │
│   - RST satellites follow nucleus                          │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Fuse Units (Deterministic AST Operations)                  │
│   - Same subject → appositive or coordinate                │
│   - Relative clause insertion                              │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Order Units in Sentences (Deterministic from Grammar)      │
│   - SVO word order                                         │
│   - Nucleus before satellite (RST)                         │
│   - Topic before comment                                   │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Deparse to Text (Deterministic)                            │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Insert Paragraph Breaks (Deterministic)                    │
│   - Topic changes                                          │
│   - Discourse boundaries                                   │
│   - Length thresholds                                      │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Output: Summary                                             │
└─────────────────────────────────────────────────────────────┘
```

**Total new learned parameters**: **0** 🎉

**Reused existing parameters**: 320K (root embeddings, already trained)

---

## Optional Learning (If Deterministic Insufficient)

After implementing and testing the deterministic system, we might discover specific cases where learning helps:

### Potential Model 1: Importance Adjustment Residual (1M params)
**Only if**: Deterministic adjustment formula has systematic errors

```python
# Instead of replacing deterministic with learned...
# Add small residual model on top!

deterministic_score = compute_base_importance(unit)
deterministic_adjustment = adjust_with_embeddings(unit, query)  # Existing embeddings

# Optional: Add learned residual (1M params)
residual = tiny_learned_model(unit_features)
final_score = deterministic_score + deterministic_adjustment + residual
```

**Training data**: Only cases where deterministic is systematically wrong

### Potential Model 2: Ambiguous Topic Resolver (0.5M params)
**Only if**: Some units genuinely ambiguous (rare in practice)

```python
# Use deterministic for 95% of cases
topic_scores = compute_deterministic_topic_scores(unit)

# If ambiguous (top 2 topics close)
if topic_scores[0] - topic_scores[1] < 0.1:
    # Use tiny learned model as tiebreaker (0.5M params)
    refined_scores = tiny_topic_model(unit_features)
    topic = max(refined_scores)
else:
    topic = max(topic_scores)  # Deterministic sufficient
```

**Training data**: Only ambiguous cases

### Total Optional Models: ~1.5M params (vs 10M originally!)

**Philosophy**: Deterministic first, learning only for proven gaps.

---

## Recommended Implementation Plan

### Phase 1: Pure Deterministic (Weeks 1-3)
Implement complete pipeline with 0 new learned parameters:
1. Information unit extraction (AST boundaries)
2. Information graph (deterministic edges + existing embeddings)
3. Importance scoring (PageRank + Kuzu + entropy + info structure + RST)
4. Importance adjustment (existing embeddings + formula)
5. Topic identification (root overlap + Kuzu)
6. Topic assignment (Kuzu co-occurrence + existing embeddings)
7. Unit selection (thresholds)
8. Sentence grouping (AST structure)
9. AST fusion (grammar rules)
10. Ordering (SVO + RST + info structure)
11. Deparsing (existing)
12. Paragraph breaks (topic changes + discourse markers)

**Test on 50-100 queries**: Measure quality

**Hypothesis**: Will achieve 85-90% quality with 0 new parameters!

### Phase 2: Identify Gaps (Week 4)
Run comprehensive evaluation:
- Where does deterministic fail?
- Are failures systematic or random?
- Can we fix with better heuristics?

### Phase 3: Optional Learning (Week 5+, if needed)
**Only if** deterministic has systematic failures:
- Train tiny residual models (~1.5M total)
- Only for proven gaps
- Deterministic remains primary

---

## Why This Is Unique to Klareco

**No other system can do this** because:

1. **English/other languages**: Need 100M+ param models for summarization (BART, T5, etc.)
2. **Esperanto's regularity**: Enables deterministic processing
3. **AST structure**: Makes implicit explicit
4. **Kuzu knowledge graph**: Provides deterministic entity salience
5. **Linguistic theory**: RST, information structure work cleanly in Esperanto

**This is Klareco's core thesis**: Maximize deterministic processing, minimize learned parameters.

**Summarization proves the thesis**: We can do complex NLP (multi-sentence summarization) with potentially ZERO new learned parameters!

---

## Conclusion

**You were absolutely right to question the models!**

By leveraging:
- ✅ Esperanto's regular grammar
- ✅ AST structure (explicit syntactic relations)
- ✅ Existing root embeddings (320K params, already trained)
- ✅ Kuzu knowledge graph (entity salience, co-occurrence)
- ✅ Linguistic theory (RST, information structure)

We can build a **fully deterministic summarization system** with:
- **0 new learned parameters**
- **High explainability** (every decision traceable)
- **No training data needed** (besides existing root embeddings)
- **No hallucination risk** (only rearranges retrieved facts)

This is **impossible in English** and showcases Esperanto's unique advantages!

Ready to implement the pure deterministic system?
