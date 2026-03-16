# Summarization Approaches: Extractive vs Abstractive vs AST Fusion

## The Question

When summarizing N retrieved sentences, do we:
1. **Keep original sentences** as-is (extractive)?
2. **Construct new sentences** from key points (abstractive)?
3. **Hybrid approach** using AST manipulation (fusion)?

## Option A: Extractive Summarization (Simplest)

**Process**: Look at sentences as a group, pick the best ones

```python
def extractive_summarize(sentences, query, max_sentences=3):
    """Select top N most important sentences, keep them unchanged"""

    # Step 1: Score each sentence independently
    scores = []
    for sent in sentences:
        score = compute_importance(sent, query)
        scores.append((sent, score))

    # Step 2: Sort by score
    scores.sort(key=lambda x: x[1], reverse=True)

    # Step 3: Take top N
    summary = [sent for sent, score in scores[:max_sentences]]

    return summary
```

**Example**:
```
Query: "Kiu fondis Esperanton?"

Input sentences (10):
1. "Zamenhof fondis Esperanton en 1887." (score: 0.95)
2. "Ludoviko Lazaro Zamenhof estis pola kuracisto." (0.90)
3. "La fundinto vivis en Bjalistoko." (0.85)
4. "Li kreis la lingvon por paco." (0.80)
5. "Esperanto estas planlingvo." (0.70)
...

Output (top 3):
1. "Zamenhof fondis Esperanton en 1887."
2. "Ludoviko Lazaro Zamenhof estis pola kuracisto."
3. "La fundinto vivis en Bjalistoko."
```

**Pros**:
- ✅ Simple to implement
- ✅ No risk of changing meaning
- ✅ No hallucination
- ✅ Fully explainable (show scores)

**Cons**:
- ❌ Verbose (3 separate sentences when could be 1-2)
- ❌ Redundant (repeats "Zamenhof" 3 times)
- ❌ Not concise (doesn't combine related info)

## Option B: Abstractive Summarization (Hardest)

**Process**: Identify key facts, construct NEW sentences

```python
def abstractive_summarize(sentences, query, max_sentences=3):
    """Extract key facts and construct new sentences"""

    # Step 1: Extract all facts from sentences (as a GROUP)
    facts = []
    for sent in sentences:
        facts.extend(extract_facts(sent))

    # Example facts extracted:
    # - Person: Zamenhof
    # - Action: fondis
    # - Object: Esperanton
    # - Time: 1887
    # - Attribute: pola kuracisto
    # - Location: Bjalistoko

    # Step 2: Identify most important facts
    important_facts = rank_facts(facts, query)

    # Step 3: Construct NEW sentences from facts
    summary = construct_sentences(important_facts)

    return summary
```

**Example**:
```
Query: "Kiu fondis Esperanton?"

Input sentences (10):
1. "Zamenhof fondis Esperanton en 1887."
2. "Ludoviko Lazaro Zamenhof estis pola kuracisto."
3. "La fundinto vivis en Bjalistoko."
...

Extracted facts:
- [Person: Zamenhof, full_name: Ludoviko Lazaro Zamenhof]
- [Action: fondis, object: Esperanton, time: 1887]
- [Attribute: pola kuracisto]
- [Location: vivis, place: Bjalistoko]

Construct NEW sentences:
1. "Ludoviko Lazaro Zamenhof, pola kuracisto, fondis Esperanton en 1887."
   (combines facts from sentences 1, 2)
2. "Li vivis en Bjalistoko."
   (keeps sentence 3)

Output (2 sentences instead of 3):
1. "Ludoviko Lazaro Zamenhof, pola kuracisto, fondis Esperanton en 1887."
2. "Li vivis en Bjalistoko."
```

**Pros**:
- ✅ Very concise (2 sentences vs 3)
- ✅ Natural flow (combines related info)
- ✅ Reduces redundancy

**Cons**:
- ❌ HARD to implement (how to construct grammatical sentences?)
- ❌ Risk of changing meaning (did we preserve all important details?)
- ❌ Risk of errors (grammar mistakes, wrong word order)
- ❌ Hard to explain (how did we construct this?)

**Traditional NLP**: This requires a large language model (100M+ params) to generate fluent text. Very hard!

**BUT IN ESPERANTO**: We can do this deterministically with AST manipulation! 🎉

## Option C: AST Fusion (Hybrid - Best for Esperanto)

**Key insight**: Esperanto's regular grammar means we can construct new sentences BY MANIPULATING ASTs, not generating text!

**Process**: Look at sentences as a group, identify patterns, FUSE ASTs

```python
def ast_fusion_summarize(sentences, query, max_sentences=3):
    """Combine sentences using AST manipulation"""

    # Step 1: Parse all sentences to ASTs
    asts = [parse(sent) for sent in sentences]

    # Step 2: Look at ASTs as a GROUP, find patterns
    patterns = find_fusion_opportunities(asts)
    # Patterns found:
    # - AST[0] and AST[1] share subject "Zamenhof" → can fuse
    # - AST[2] has subject "fundinto" (refers to Zamenhof) → can keep separate

    # Step 3: Score each AST for importance
    scores = [compute_importance(ast, query) for ast in asts]

    # Step 4: Select top ASTs to include
    selected = select_top_asts(asts, scores, patterns, max_sentences)

    # Step 5: FUSE selected ASTs using deterministic rules
    fused_asts = []
    for pattern in patterns:
        if pattern['type'] == 'same_subject':
            # Combine two sentences with same subject
            fused = fuse_same_subject(pattern['ast1'], pattern['ast2'])
            fused_asts.append(fused)
        elif pattern['type'] == 'keep':
            fused_asts.append(pattern['ast'])

    # Step 6: Deparse ASTs back to text
    summary = [deparse(ast) for ast in fused_asts]

    return summary
```

**Example with AST operations**:

```
Query: "Kiu fondis Esperanton?"

Input sentences (10):
1. "Zamenhof fondis Esperanton en 1887."
2. "Ludoviko Lazaro Zamenhof estis pola kuracisto."
3. "La fundinto vivis en Bjalistoko."

Step 1: Parse to ASTs
AST[0]:
{
  "subjekto": {"radiko": "zamenhofo", "vortspeco": "substantivo"},
  "verbo": {"radiko": "fond", "tempo": "pasinto"},
  "objekto": {"radiko": "esperanto"},
  "aliaj": [{"radiko": "en", "tipo": "tempo", "valoro": "1887"}]
}

AST[1]:
{
  "subjekto": {"radiko": "zamenhofo", "plena_nomo": "Ludoviko Lazaro Zamenhof"},
  "verbo": {"radiko": "est", "tempo": "pasinto"},
  "objekto": {"radiko": "kuracisto", "priskriboj": [{"radiko": "pola"}]}
}

AST[2]:
{
  "subjekto": {"radiko": "fundinto"},  # Refers to Zamenhof
  "verbo": {"radiko": "viv", "tempo": "pasinto"},
  "aliaj": [{"radiko": "en", "tipo": "loko", "valoro": "Bjalistoko"}]
}

Step 2: Find fusion opportunities
Pattern 1: AST[0] and AST[1] have SAME SUBJECT "zamenhofo"
  → Can fuse with "kaj" (and)
  → Can also add full name from AST[1]

Pattern 2: AST[2] has different subject "fundinto"
  → Keep separate (don't fuse)

Step 3: Score importance
AST[0]: 0.95 (directly answers "kiu")
AST[1]: 0.90 (provides detail about answer)
AST[2]: 0.85 (biographical detail)

Step 4: Select top ASTs
All 3 are important, but fuse AST[0] and AST[1]

Step 5: FUSE ASTs
Fuse AST[0] + AST[1]:
{
  "subjekto": {
    "radiko": "zamenhofo",
    "plena_nomo": "Ludoviko Lazaro Zamenhof",  # From AST[1]
    "apozicio": {  # Appositive phrase
      "radiko": "kuracisto",
      "priskriboj": [{"radiko": "pola"}]
    }
  },
  "verbo": {
    "tipo": "kunmetita",  # Compound verb phrase
    "verboj": [
      {"radiko": "fond", "tempo": "pasinto", "objekto": {"radiko": "esperanto"}},
      {"radiko": "est", "tempo": "pasinto", "objekto": {"radiko": "kuracisto"}}
    ],
    "ligilo": "kaj"  # Connected with "kaj"
  },
  "aliaj": [{"radiko": "en", "tipo": "tempo", "valoro": "1887"}]
}

Step 6: Deparse fused AST
"Ludoviko Lazaro Zamenhof, pola kuracisto, fondis Esperanton en 1887 kaj estis pola kuracisto."

Wait, that's redundant! Better fusion:

Fused AST (better):
{
  "subjekto": {
    "radiko": "zamenhofo",
    "plena_nomo": "Ludoviko Lazaro Zamenhof",
    "apozicio": "pola kuracisto"  # Appositive, not predicate
  },
  "verbo": {"radiko": "fond", "tempo": "pasinto"},
  "objekto": {"radiko": "esperanto"},
  "aliaj": [{"radiko": "en", "tipo": "tempo", "valoro": "1887"}]
}

Deparsed:
"Ludoviko Lazaro Zamenhof, pola kuracisto, fondis Esperanton en 1887."

Output (2 sentences):
1. "Ludoviko Lazaro Zamenhof, pola kuracisto, fondis Esperanton en 1887."
2. "La fundinto vivis en Bjalistoko."
```

**Pros**:
- ✅ Concise (2 sentences vs 3)
- ✅ Natural (combines related info)
- ✅ **SAFE** (AST operations preserve grammar)
- ✅ **Explainable** (show exactly what we fused and why)
- ✅ **Deterministic** (no learned model needed for fusion)
- ✅ **No hallucination** (only rearrange existing facts)

**Cons**:
- ⚠️ More complex than extractive (but much simpler than full abstractive)
- ⚠️ Need to implement fusion rules (but they're deterministic!)

## Why AST Fusion Works for Esperanto

**In English**: Constructing new sentences is HARD
- Irregular grammar (I am, you are, he is)
- Word order matters (subject-verb-object strict)
- Agreement rules complex (singular/plural, gender)
- Idioms, phrasal verbs (can't decompose)

**In Esperanto**: Constructing new sentences is EASY via ASTs
- Regular grammar (mi estas, vi estas, li estas)
- Flexible word order (case markers clarify roles)
- Simple agreement (adjectives match noun number/case)
- Compositional (every word decomposes to roots)

**Example of why AST manipulation is safe**:

```
Original ASTs (parsed):
AST1: {"subjekto": "hundo", "verbo": "kuras"}
AST2: {"subjekto": "hundo", "verbo": "manĝas"}

Fusion operation:
fuse_same_subject(AST1, AST2, ligilo="kaj")

Result AST:
{
  "subjekto": "hundo",
  "verbo": {
    "tipo": "kunmetita",
    "verboj": ["kuras", "manĝas"],
    "ligilo": "kaj"
  }
}

Deparse:
"Hundo kuras kaj manĝas."  # Guaranteed grammatical!

Why guaranteed?
1. Subject case preserved (nominative, no -n)
2. Verbs keep their form (present tense -as)
3. Word order follows Esperanto rules
4. Deparser applies deterministic rules
```

## Detailed Fusion Process (Step by Step)

Let me show how we'd process a group of sentences:

### Input
```
Query: "Kiu fondis Esperanton?"

Retrieved sentences (5):
1. "Zamenhof fondis Esperanton en 1887."
2. "Ludoviko Lazaro Zamenhof estis pola kuracisto."
3. "La fundinto vivis en Bjalistoko."
4. "Li kreis la lingvon por internacia komunikado."
5. "Esperanto estas planlingvo."
```

### Step 1: Parse all to ASTs (as a group)
```python
asts = [parse(sent) for sent in sentences]
# Now we have 5 ASTs to work with
```

### Step 2: Build entity graph (identify what refers to what)
```python
entities = {
    "zamenhofo": [AST[0], AST[1]],  # Sentences 0 and 1 mention Zamenhof
    "fundinto": [AST[2]],            # Sentence 2 uses "fundinto" (founder)
    "li": [AST[3]],                  # Sentence 3 uses "li" (he)
    "esperanto": [AST[0], AST[4]]    # Sentences 0 and 4 mention Esperanto
}

# Resolve coreferences (deterministic in Esperanto!)
# "fundinto" = founder = Zamenhof (from context)
# "li" = he = most recent person = Zamenhof
```

### Step 3: Find fusion opportunities
```python
opportunities = []

# Pattern 1: Same subject
if AST[0].subjekto == AST[1].subjekto:  # Both about Zamenhof
    opportunities.append({
        "type": "same_subject",
        "asts": [AST[0], AST[1]],
        "method": "appositive",  # Can add "pola kuracisto" as appositive
        "priority": 0.9  # High priority (reduces redundancy)
    })

# Pattern 2: Subject-object relation
if AST[0].subjekto == "zamenhofo" and AST[0].objekto == "esperanto":
    if AST[4].subjekto == "esperanto":  # Sentence 4 is about Esperanto
        opportunities.append({
            "type": "subject_object_chain",
            "asts": [AST[0], AST[4]],
            "method": "relative_clause",  # "Esperanton, kiu estas planlingvo"
            "priority": 0.7
        })
```

### Step 4: Score importance (individual + group)
```python
# Individual scores (query overlap)
scores = [0.95, 0.90, 0.85, 0.80, 0.70]

# Group adjustments
# - AST[0] + AST[1] can fuse → boost both (reduce redundancy)
# - AST[2] uses "fundinto" (coreferent with Zamenhof) → boost (adds info)
# - AST[4] is generic (not specific to founder) → lower

adjusted_scores = [0.95, 0.92, 0.87, 0.82, 0.65]
```

### Step 5: Select top ASTs and apply fusions
```python
# Select top 3: AST[0], AST[1], AST[2]

# Apply fusion 1: AST[0] + AST[1] (same subject)
fused_ast_01 = fuse_appositive(AST[0], AST[1])
# Result: "Ludoviko Lazaro Zamenhof, pola kuracisto, fondis Esperanton en 1887."

# Keep AST[2]: "La fundinto vivis en Bjalistoko."

# Final ASTs: [fused_ast_01, AST[2]]
```

### Step 6: Deparse to text
```python
summary = [deparse(ast) for ast in [fused_ast_01, AST[2]]]

# Output:
# "Ludoviko Lazaro Zamenhof, pola kuracisto, fondis Esperanton en 1887.
#  La fundinto vivis en Bjalistoko."
```

### Output
```
Summary (2 sentences, down from 5):
"Ludoviko Lazaro Zamenhof, pola kuracisto, fondis Esperanton en 1887. La fundinto vivis en Bjalistoko."

Operations performed (explainable):
1. Fused sentences 1 and 2 (same subject "Zamenhof")
   - Method: Added "pola kuracisto" as appositive
   - Reason: Reduces redundancy, more concise
2. Kept sentence 3 (biographical detail about founder)
   - Reason: High importance (0.87), adds context
3. Removed sentences 4 and 5
   - Reason: Lower importance, not directly about founder
```

## Implementation: Fusion Rules

Here are the deterministic fusion rules we'd implement:

### Rule 1: Same Subject Fusion
```python
def fuse_same_subject(ast1, ast2):
    """Combine sentences with same subject using 'kaj'"""

    if ast1['subjekto']['radiko'] != ast2['subjekto']['radiko']:
        return None  # Can't fuse

    # Check what verbs say
    if ast2['verbo']['radiko'] == 'est' and 'objekto' in ast2:
        # Second sentence is "X estas Y" → can be appositive
        return {
            "subjekto": {
                **ast1['subjekto'],
                "apozicio": ast2['objekto']  # Add as appositive
            },
            "verbo": ast1['verbo'],
            "objekto": ast1.get('objekto'),
            "aliaj": ast1.get('aliaj', [])
        }
    else:
        # Both have action verbs → combine with 'kaj'
        return {
            "subjekto": ast1['subjekto'],
            "verbo": {
                "tipo": "kunmetita",
                "verboj": [ast1['verbo'], ast2['verbo']],
                "ligilo": "kaj"
            },
            "objekto": None,  # Combined at verb level
            "aliaj": ast1.get('aliaj', []) + ast2.get('aliaj', [])
        }
```

### Rule 2: Relative Clause Insertion
```python
def add_relative_clause(main_ast, relative_ast):
    """Add relative clause to object"""

    if main_ast.get('objekto') is None:
        return None

    # Check if relative clause subject matches main clause object
    if relative_ast['subjekto']['radiko'] != main_ast['objekto']['radiko']:
        return None

    return {
        **main_ast,
        "objekto": {
            **main_ast['objekto'],
            "rilata_frazo": {  # Relative clause
                "pronomo": "kiu",
                "verbo": relative_ast['verbo'],
                "objekto": relative_ast.get('objekto')
            }
        }
    }

# Example:
# "Zamenhof fondis Esperanton." + "Esperanto estas planlingvo."
# → "Zamenhof fondis Esperanton, kiu estas planlingvo."
```

### Rule 3: Appositive Insertion
```python
def add_appositive(ast, detail_ast):
    """Add descriptive detail as appositive"""

    # If detail sentence is "X estas Y", Y can be appositive
    if detail_ast['verbo']['radiko'] == 'est':
        return {
            **ast,
            "subjekto": {
                **ast['subjekto'],
                "apozicio": detail_ast['objekto']
            }
        }

    return None

# Example:
# "Zamenhof fondis Esperanton." + "Zamenhof estis kuracisto."
# → "Zamenhof, kuracisto, fondis Esperanton."
```

## Comparison Summary

| Approach | Looks at | Key Points | Construction | Complexity | Best for |
|----------|----------|------------|--------------|------------|----------|
| **Extractive** | Individually | Score each | Keep original | Low | Quick summaries, safety-critical |
| **Abstractive** | Group | Extract facts | Generate new text | Very High | Fluent, concise (needs LLM) |
| **AST Fusion** | Group | AST patterns | Manipulate ASTs | Medium | **Esperanto** (regular grammar) |

## Recommendation for Klareco

**Start with**: AST Fusion (Option C)

**Why**:
1. Takes advantage of Esperanto's regular structure
2. More concise than extractive
3. Safer than full abstractive (no text generation)
4. **Fully explainable** (show AST operations)
5. **Deterministic** (no learned parameters needed)
6. Unique to Klareco (other systems can't do this!)

**Implementation order**:
1. Week 1: Implement extractive (baseline)
2. Week 2: Implement AST deduplicator (remove redundancy)
3. Week 3: Implement same-subject fusion rule
4. Week 4: Implement relative clause and appositive rules
5. Week 5: Test and compare vs extractive baseline

**If AST fusion proves insufficient**: Can always fall back to extractive or add minimal learned ranking model (5M params).

## Next Steps

Would you like me to:
1. Implement the AST fusion summarizer?
2. Start with extractive baseline first?
3. Create issues for summarization implementation?
