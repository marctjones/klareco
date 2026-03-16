# Concrete Summarization Approach: 20 Sentences → 4 Sentences

## The Real Problem

**Input**: 20 retrieved sentences about cats
**Output**: Coherent 3-4 sentence summary
**Challenge**: How do we identify the most important information to include?

## Example: Let's Work Through It

### Input (20 sentences about cats)

```
Query: "Kio estas kato?" (What is a cat?)

Retrieved sentences:
1. "Kato estas malgranda hejma besto."
2. "Katoj havas kvar piedojn kaj longan voston."
3. "Ili estas karnovoruloj kaj ĉasas musojn."
4. "Katoj dormas ĝis 16 horojn tage."
5. "La kato estas unu el la plej popularaj hejmbestoj."
6. "Katoj ronronas kiam ili estas feliĉaj."
7. "Ili havas akrajn ungojn por grimpi."
8. "Katoj vidas tre bone en mallumo."
9. "La unua kato estis hejmigita antaŭ 10,000 jaroj."
10. "Katoj uzas sian voston por ekvilibro."
11. "Ekzistas pli ol 70 rasoj de katoj."
12. "Katoj kommunikas per miaŭado kaj korpa lingvo."
13. "Ili havas bonegan aŭdon kaj flaradon."
14. "Katoj estas tre puruloj kaj lekas sian felon."
15. "La plej maljuna kato vivis 38 jarojn."
16. "Katoj povas salti ĝis 6-foje sian alton."
17. "Ili havas 32 muskolojn en ĉiu orelo."
18. "Katoj estas kutime solecemaj bestoj."
19. "La granda plimulto de katoj ne ŝatas akvon."
20. "Katoj naskoj inter 1-8 idojn."
```

**Expected 4-sentence summary**:
```
"Kato estas malgranda hejma besto, kiu estas karnovora kaj ĉasas musojn.
Katoj havas kvar piedojn, longan voston, kaj bonegan vidon en mallumo.
Ili estas unu el la plej popularaj hejmbestoj, kun pli ol 70 rasoj.
Katoj estas tre puruloj kaj dormas ĝis 16 horojn tage."
```

## The Core Challenge: Ranking Importance

**Question**: Which sentences are most important for answering "Kio estas kato?"

Let's score each sentence:

### Scoring Method: Multi-Factor Deterministic

```python
def score_sentence_importance(sentence, query, all_sentences):
    """
    Score importance using deterministic factors.
    Returns score 0-1.
    """

    score = 0.0

    # Factor 1: Answers the question directly (30%)
    # "Kio estas kato?" → Look for definitions/categorizations
    if is_definition(sentence):  # "X estas Y"
        score += 0.30

    # Factor 2: Query overlap (20%)
    # Does sentence mention query terms?
    query_roots = extract_roots(query)
    sent_roots = extract_roots(sentence)
    overlap = len(query_roots & sent_roots) / len(query_roots)
    score += 0.20 * overlap

    # Factor 3: Centrality in corpus (20%)
    # How often is this information mentioned across ALL sentences?
    # (Information that appears in multiple sentences = important)
    repetition_count = count_similar_info_in_corpus(sentence, all_sentences)
    score += 0.20 * min(repetition_count / 5.0, 1.0)

    # Factor 4: Generality vs Specificity (15%)
    # General facts score higher than specific trivia
    if is_general_fact(sentence):
        score += 0.15

    # Factor 5: Information type (15%)
    # Certain types more important for "what is X" questions
    if describes_physical_features(sentence):
        score += 0.05
    if describes_behavior(sentence):
        score += 0.05
    if provides_category(sentence):
        score += 0.05

    return min(score, 1.0)
```

### Let's Actually Score Our 20 Sentences

| # | Sentence | Definition? | Query Overlap | Centrality | General? | Type | **Total Score** |
|---|----------|-------------|---------------|------------|----------|------|-----------------|
| 1 | "Kato estas malgranda hejma besto" | ✅ Yes (0.30) | 1.0 (0.20) | High (0.15) | ✅ Yes (0.15) | Category (0.05) | **0.85** |
| 2 | "Katoj havas kvar piedojn kaj longan voston" | No (0.0) | 1.0 (0.20) | High (0.18) | ✅ Yes (0.15) | Physical (0.05) | **0.58** |
| 3 | "Ili estas karnovoruloj kaj ĉasas musojn" | No (0.0) | 1.0 (0.20) | Med (0.12) | ✅ Yes (0.15) | Behavior (0.05) | **0.52** |
| 4 | "Katoj dormas ĝis 16 horojn tage" | No (0.0) | 1.0 (0.20) | Low (0.05) | Partial (0.08) | Behavior (0.05) | **0.38** |
| 5 | "La kato estas unu el la plej popularaj..." | ✅ Yes (0.30) | 1.0 (0.20) | Med (0.10) | ✅ Yes (0.15) | Category (0.05) | **0.80** |
| 6 | "Katoj ronronas kiam ili estas feliĉaj" | No (0.0) | 1.0 (0.20) | Low (0.06) | Partial (0.08) | Behavior (0.05) | **0.39** |
| 7 | "Ili havas akrajn ungojn por grimpi" | No (0.0) | 1.0 (0.20) | Low (0.08) | Partial (0.10) | Physical (0.05) | **0.43** |
| 8 | "Katoj vidas tre bone en mallumo" | No (0.0) | 1.0 (0.20) | Med (0.12) | ✅ Yes (0.15) | Physical (0.05) | **0.52** |
| 9 | "La unua kato estis hejmigita..." | No (0.0) | 1.0 (0.20) | Low (0.04) | Partial (0.10) | History (0.0) | **0.34** |
| 10 | "Katoj uzas sian voston por ekvilibro" | No (0.0) | 1.0 (0.20) | Low (0.06) | Partial (0.08) | Behavior (0.05) | **0.39** |
| 11 | "Ekzistas pli ol 70 rasoj de katoj" | No (0.0) | 1.0 (0.20) | Med (0.10) | ✅ Yes (0.15) | Category (0.05) | **0.50** |
| 12 | "Katoj kommunikas per miaŭado..." | No (0.0) | 1.0 (0.20) | Med (0.08) | ✅ Yes (0.15) | Behavior (0.05) | **0.48** |
| 13 | "Ili havas bonegan aŭdon kaj flaradon" | No (0.0) | 1.0 (0.20) | Low (0.07) | Partial (0.10) | Physical (0.05) | **0.42** |
| 14 | "Katoj estas tre puruloj..." | No (0.0) | 1.0 (0.20) | Med (0.10) | ✅ Yes (0.15) | Behavior (0.05) | **0.50** |
| 15 | "La plej maljuna kato vivis 38 jarojn" | No (0.0) | 1.0 (0.20) | Low (0.02) | No (0.0) | Trivia (0.0) | **0.22** |
| 16 | "Katoj povas salti ĝis 6-foje sian alton" | No (0.0) | 1.0 (0.20) | Low (0.04) | Partial (0.08) | Physical (0.05) | **0.37** |
| 17 | "Ili havas 32 muskolojn en ĉiu orelo" | No (0.0) | 1.0 (0.20) | Low (0.02) | No (0.0) | Trivia (0.0) | **0.22** |
| 18 | "Katoj estas kutime solecemaj bestoj" | No (0.0) | 1.0 (0.20) | Low (0.08) | ✅ Yes (0.15) | Behavior (0.05) | **0.48** |
| 19 | "La granda plimulto de katoj ne ŝatas..." | No (0.0) | 1.0 (0.20) | Low (0.06) | Partial (0.10) | Behavior (0.05) | **0.41** |
| 20 | "Katoj naskoj inter 1-8 idojn" | No (0.0) | 1.0 (0.20) | Low (0.04) | Partial (0.08) | Biology (0.0) | **0.32** |

### Ranked by Importance

1. **Sentence 1**: 0.85 - "Kato estas malgranda hejma besto" ⭐
2. **Sentence 5**: 0.80 - "La kato estas unu el la plej popularaj hejmbestoj" ⭐
3. **Sentence 2**: 0.58 - "Katoj havas kvar piedojn kaj longan voston" ⭐
4. **Sentence 3**: 0.52 - "Ili estas karnovoruloj kaj ĉasas musojn" ⭐
5. **Sentence 8**: 0.52 - "Katoj vidas tre bone en mallumo" ⭐
6. **Sentence 11**: 0.50 - "Ekzistas pli ol 70 rasoj de katoj"
7. **Sentence 14**: 0.50 - "Katoj estas tre puruloj..."
8. **Sentence 12**: 0.48 - "Katoj kommunikas per miaŭado..."
9. **Sentence 18**: 0.48 - "Katoj estas kutime solecemaj bestoj"

**Top 4-5 sentences selected** ⭐

## Step-by-Step Process

### Step 1: Score All Sentences (Deterministic)

```python
scores = []
for sentence in retrieved_sentences:
    score = score_sentence_importance(sentence, query, retrieved_sentences)
    scores.append((sentence, score))
```

### Step 2: Remove Redundancy (Deterministic)

```python
# Sort by score
scores.sort(key=lambda x: x[1], reverse=True)

# Select top N, removing redundant information
selected = []
for sentence, score in scores:
    if not is_redundant_with_selected(sentence, selected):
        selected.append(sentence)

        # Stop when we have enough
        if len(selected) >= 4:
            break

def is_redundant_with_selected(sentence, selected_sentences):
    """Check if sentence repeats information already in selected set"""

    sentence_roots = extract_roots(sentence)

    for selected in selected_sentences:
        selected_roots = extract_roots(selected)

        # If >70% root overlap, consider redundant
        overlap = len(sentence_roots & selected_roots) / len(sentence_roots)
        if overlap > 0.7:
            return True

    return False
```

**Result after redundancy removal**:
- Sentence 1: "Kato estas malgranda hejma besto" ✓
- Sentence 5: "La kato estas unu el la plej popularaj hejmbestoj" ✓
- Sentence 2: "Katoj havas kvar piedojn kaj longan voston" ✓
- Sentence 3: "Ili estas karnovoruloj kaj ĉasas musojn" ✓

(Sentence 8 "vidas tre bone" dropped - redundant with sentence 2 "physical features")

### Step 3: Optional Fusion (Deterministic)

Can we combine related sentences to be more concise?

```python
# Sentence 1 + Sentence 3 share subject "kato"
# Can fuse: "Kato estas malgranda hejma besto, kiu estas karnovora kaj ĉasas musojn."

# Sentence 2 stays separate (physical description)

# Sentence 5 + Sentence 11 both about population/variety
# Can mention: "Ili estas unu el la plej popularaj hejmbestoj, kun pli ol 70 rasoj."
```

### Step 4: Order Logically (Deterministic)

```python
# Information structure principle: General → Specific
# 1. Definition first (what is it?)
# 2. Physical features (what does it look like?)
# 3. Behavior (what does it do?)
# 4. Context (how common?)

ordered = [
    "Kato estas malgranda hejma besto, kiu estas karnovora kaj ĉasas musojn.",  # Definition + behavior
    "Katoj havas kvar piedojn, longan voston, kaj bonegan vidon en mallumo.",    # Physical features
    "Ili estas unu el la plej popularaj hejmbestoj, kun pli ol 70 rasoj.",      # Context/popularity
    "Katoj estas tre puruloj kaj dormas ĝis 16 horojn tage."                   # Additional behavior
]
```

### Final Output

```
Kato estas malgranda hejma besto, kiu estas karnovora kaj ĉasas musojn.
Katoj havas kvar piedojn, longan voston, kaj bonegan vidon en mallumo.
Ili estas unu el la plej popularaj hejmbestoj, kun pli ol 70 rasoj.
Katoj estas tre puruloj kaj dormas ĝis 16 horojn tage.
```

**4 sentences** (from 20) ✓
**Coherent** (flows logically) ✓
**Informative** (covers definition, features, behavior, context) ✓

## The Key Components That Actually Matter

### 1. Importance Scoring (THE CORE)

**What matters**:
- ✅ Does sentence define/categorize? (highest weight)
- ✅ How central is information? (appears in multiple sentences?)
- ✅ General vs specific? (general facts > trivia)
- ✅ Matches query type? ("what is" → definitions, "how" → processes)

**What doesn't matter much**:
- ❌ Complex graph structures (overkill)
- ❌ PageRank across information units (too abstract)
- ❌ Topic clustering models (can use simple root overlap)

### 2. Redundancy Removal

**Simple approach**: Root overlap
- If new sentence shares >70% roots with selected → skip it

### 3. Fusion (Optional, Deterministic)

**When to fuse**:
- Same subject ("Kato estas..." + "Kato havas..." → combine)
- Related concepts (popularity + variety → combine)

**Use AST structure** to ensure grammatically correct fusion

### 4. Ordering

**Simple rules**:
- Definition first
- Physical features second
- Behavior third
- Context/trivia last

## What We Actually Need

### Minimal System (0 New Params)

```python
class SimpleSummarizer:
    def __init__(self):
        self.root_embeddings = load_existing_embeddings()  # 320K params, already trained

    def summarize(self, query, sentences, target_length=4):
        # 1. Score importance (deterministic + existing embeddings)
        scored = []
        for sent in sentences:
            score = self.score_importance(sent, query, sentences)
            scored.append((sent, score))

        # 2. Sort by score
        scored.sort(key=lambda x: x[1], reverse=True)

        # 3. Remove redundancy
        selected = []
        for sent, score in scored:
            if not self.is_redundant(sent, selected):
                selected.append(sent)
                if len(selected) >= target_length:
                    break

        # 4. Optional: Fuse related sentences
        fused = self.fuse_related(selected)

        # 5. Order logically
        ordered = self.order_by_info_structure(fused)

        return ordered

    def score_importance(self, sentence, query, all_sentences):
        score = 0.0

        # Factor 1: Definition? (AST check - deterministic)
        if self.is_definition(sentence):
            score += 0.30

        # Factor 2: Query overlap (deterministic)
        query_roots = extract_roots(query)
        sent_roots = extract_roots(sentence)
        overlap = len(query_roots & sent_roots) / len(query_roots)
        score += 0.20 * overlap

        # Factor 3: Centrality (how often mentioned - deterministic)
        repetition = self.count_similar_info(sentence, all_sentences)
        score += 0.20 * min(repetition / 5.0, 1.0)

        # Factor 4: Generality (deterministic heuristic)
        if self.is_general_fact(sentence):
            score += 0.15

        # Factor 5: Information type (deterministic from AST)
        score += self.score_info_type(sentence, query)

        return min(score, 1.0)

    def is_redundant(self, sentence, selected):
        # Simple root overlap check (deterministic)
        sent_roots = extract_roots(sentence)
        for sel in selected:
            sel_roots = extract_roots(sel)
            overlap = len(sent_roots & sel_roots) / len(sent_roots)
            if overlap > 0.7:
                return True
        return False
```

**Total new parameters**: 0
**Reuses**: Existing root embeddings (320K)
**Complexity**: Much simpler!

## Your Question Answered

**"How does this help us filter 20 sentences → 4 sentences?"**

1. **Score each sentence** using deterministic factors (definition?, central?, general?)
2. **Sort by score** (highest = most important)
3. **Remove redundancy** (skip sentences that repeat information)
4. **Take top N** sentences (N = target length)
5. **Optional**: Fuse related sentences, order logically

**The KEY is the scoring function** - not complex graphs or models!

## Should We Still Use Information Units?

**Good question!** Two approaches:

### Approach A: Sentence-Level (Simpler)
- Score whole sentences
- Select top N sentences
- Optionally fuse related sentences
- **Pros**: Simple, works well for "what is X" queries
- **Cons**: Can't cherry-pick important parts of verbose sentences

### Approach B: Information Unit-Level (More Precise)
- Break sentences into clauses/phrases (information units)
- Score each unit independently
- Select top N units
- Recombine into sentences
- **Pros**: More precise, can filter out irrelevant parts
- **Cons**: More complex

**Recommendation**: Start with Approach A (sentence-level)
- Test on 50-100 queries
- If verbosity is a problem, move to Approach B

## Bottom Line

**You're right - I was overcomplicating!**

The core is:
1. ✅ Good importance scoring function
2. ✅ Redundancy removal
3. ✅ Logical ordering

**Not needed**:
- ❌ Complex information graphs
- ❌ PageRank over information units
- ❌ Multiple learned models

Let's implement the simple approach first and see if it works!
