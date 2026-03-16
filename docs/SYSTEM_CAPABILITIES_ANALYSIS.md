# Klareco System Capabilities Analysis

**Critical Question**: "If we implement everything in our current design, what would our AI system be able to do?"

**Answer**: 🚨 **We're building a retrieval system, NOT a full Q&A system!**

## What We Can Do (Current Design)

### Input → Output Pipeline

**Input**: Esperanto question
**Output**: List of N relevant sentences from corpus

**Example**:
```
Query: "Kiu fondis Esperanton?"
Output:
  1. "Zamenhof fondis Esperanton en 1887." (score: 0.92)
  2. "La fundinto de Esperanto estis kuracisto." (score: 0.85)
  3. "Zamenhof estis pola kuracisto." (score: 0.78)
```

### Capabilities (Fully Implemented Design)

| Component | What It Does | Query Types Supported |
|-----------|--------------|----------------------|
| **M0 Parser** | Parse query → AST | All Esperanto queries |
| **Root Embeddings** | Semantic similarity | Synonym matching |
| **M1 Selectional** | Score (S,V,O) plausibility | Semantic coherence |
| **Entity Classifier** | Identify entity types | WHO (person), WHERE (place) |
| **M2.1 Taxonomy** | Classify entities | WHO (PERSONO), WHAT (OBJEKTO) |
| **M2.2 Coreference** | Resolve pronouns | Multi-sentence context |
| **Reranker** | Rank by relevance | Final ranking |

### Query Types We Handle Well

✅ **Factoid Questions** (answer in corpus):
```
"Kiu fondis Esperanton?" → Retrieve: "Zamenhof fondis Esperanton"
"Kiam estis fondita Esperanto?" → Retrieve: "...en 1887"
"Kie naskiĝis Zamenhof?" → Retrieve: "...en Bjalistoko"
```

✅ **Definition Questions**:
```
"Kio estas Esperanto?" → Retrieve: "Esperanto estas planlingvo..."
```

✅ **Simple WHO/WHAT/WHEN/WHERE**:
```
"Kiu estis Zamenhof?" → Find entity description
"Kio estas vorto?" → Find definition
```

## What We CANNOT Do (Current Design)

### ❌ Answer Extraction

**Problem**: We return full sentences, not extracted answers

**Current**:
```
Query: "Kiu fondis Esperanton?"
Output: "Zamenhof fondis Esperanton en 1887." (whole sentence)
```

**Should be**:
```
Query: "Kiu fondis Esperanton?"
Answer: "Zamenhof" (extracted)
```

**Missing**: Answer extraction module (deterministic for Esperanto!)

### ❌ Multi-Hop Reasoning

**Problem**: Can't connect facts across sentences

**Example**:
```
Query: "Kie naskiĝis la fondinto de Esperanto?"

Requires:
  Sentence 1: "Zamenhof fondis Esperanton"
  Sentence 2: "Zamenhof naskiĝis en Bjalistoko"
  Reasoning: fondinto = Zamenhof (hop 1) → kie = Bjalistoko (hop 2)

Current system: Returns both sentences separately
Should: Extract "Bjalistoko" as answer
```

**Missing**: Multi-hop reasoning chain

### ❌ Why/How Questions

**Problem**: Require causal reasoning

**Example**:
```
Query: "Kial Zamenhof fondis Esperanton?"
Current: Returns sentences mentioning "Zamenhof", "fondis", "Esperanto"
Should: Extract REASON ("...por internacia komunikado")
```

**Missing**: Causal reasoning (requires learned reasoning core?)

### ❌ Comparison Questions

**Problem**: Require comparing entities

**Example**:
```
Query: "Kiu estas pli alta: Zamenhof aŭ Einstein?"
Current: Cannot compare
Should: Extract heights, compare, return answer
```

**Missing**: Comparison reasoning + numerical reasoning

### ❌ Arithmetic/Counting Questions

**Problem**: Require mathematical reasoning

**Example**:
```
Query: "Kiom da jaroj vivis Zamenhof?"
Current: Might retrieve birth/death dates
Should: Calculate: 1917 - 1859 = 58 years
```

**Missing**: Mathematical reasoning module

### ❌ Summarization/Synthesis

**Problem**: Require generating new text

**Example**:
```
Query: "Resumo pri Zamenhof"
Current: Returns list of sentences
Should: Generate coherent summary synthesizing multiple facts
```

**Missing**: Text generation (linearizer + reasoning core)

### ❌ Common Sense Reasoning

**Problem**: Require world knowledge not in corpus

**Example**:
```
Query: "Ĉu Zamenhof povis paroli Esperanton kiam li estis bebo?"
Current: Might not find explicit statement
Should: Infer "No" from common sense (beboj ne povas paroli)
```

**Missing**: Common sense knowledge base + reasoning

## What's Missing for Full Q&A System

### Priority 1: Answer Extraction (Deterministic!)

**What**: Extract answer span from retrieved sentence

**Example**:
```python
query_ast = parse("Kiu fondis Esperanton?")
# query_type: 'kiu' (WHO question)
# query_focus: 'fond' (verb)

sentence = "Zamenhof fondis Esperanton en 1887."
sentence_ast = parse(sentence)

# Deterministic extraction:
# 1. Match verb: 'fond' (query) == 'fond' (sentence) ✓
# 2. Extract subject (WHO answer): 'Zamenhof'
answer = "Zamenhof"
```

**Why deterministic**: Esperanto grammar makes this rule-based!
- Kiu (who) → Extract subject
- Kion (what) → Extract object (accusative -n)
- Kiam (when) → Extract temporal phrase
- Kie (where) → Extract locative phrase

**Implementation**: ~500 lines of deterministic rules (NOT a model!)

**Status**: ❌ Not designed yet!

### Priority 2: Multi-Hop Reasoning Chain

**What**: Connect facts across sentences

**Example**:
```python
query = "Kie naskiĝis la fondinto de Esperanto?"

# Hop 1: Identify "fondinto" = Zamenhof
fact1 = find("kiu fondis Esperanton?") → "Zamenhof"

# Hop 2: Find birthplace of Zamenhof
fact2 = find("kie naskiĝis Zamenhof?") → "Bjalistoko"

answer = "Bjalistoko"
```

**Why needed**: ~20-30% of questions require multi-hop

**Implementation**: Reasoning graph (deterministic chain + learned verification?)

**Status**: ❌ Not designed yet!

### Priority 3: Reasoning Core (20M Params)

**What**: The learned reasoning component from VISION.md

**Quote from VISION.md**:
> "Add minimal 20M param reasoning core, measure improvement"

**What it does**:
- Verify reasoning chains
- Handle ambiguous cases
- Provide confidence scores
- Learn reasoning patterns (not grammar!)

**Why needed**: For questions that can't be solved deterministically

**Status**: ❌ Not designed yet (but planned in VISION.md!)

### Priority 4: Answer Generation/Linearization

**What**: Generate natural language answer from AST

**Example**:
```python
answer_ast = {
    'tipo': 'respondo',
    'enteco': 'Zamenhof',
    'confidence': 0.95,
    'sources': [sent1, sent2]
}

# Linearize → Esperanto
answer_text = linearize(answer_ast)
# Output: "Zamenhof" or "Zamenhof fundis Esperanton en 1887."
```

**Status**: Basic deparser exists, but not answer-focused

## Comparison: Current vs Full Q&A

| Feature | Current Design | Full Q&A System | Gap |
|---------|---------------|-----------------|-----|
| **Input** | Esperanto question | Esperanto question | ✅ |
| **Parsing** | AST with deterministic features | AST | ✅ |
| **Retrieval** | Top-N relevant sentences | Top-N relevant sentences | ✅ |
| **Ranking** | M1 + Reranker | M1 + Reranker | ✅ |
| **Coreference** | 80% deterministic | 80% deterministic | ✅ |
| **Taxonomy** | 90% deterministic | 90% deterministic | ✅ |
| **Answer Extraction** | ❌ None | ✅ Deterministic extraction | ❌ MISSING |
| **Multi-Hop** | ❌ None | ✅ Reasoning chain | ❌ MISSING |
| **Reasoning Core** | ❌ None | ✅ 20M param core | ❌ MISSING |
| **Generation** | ❌ None | ✅ Linearizer | ⚠️ PARTIAL |
| **Output** | List of sentences | Extracted answer | ❌ GAP |

## Example Queries: Current vs Full System

### Query 1: "Kiu fondis Esperanton?"

**Current Design**:
```
Input: "Kiu fondis Esperanton?"
Process:
  1. Parse query → AST
  2. Retrieve sentences with 'fond' + 'Esperanto'
  3. Rank by M1 plausibility
  4. Rerank by relevance

Output:
  1. "Zamenhof fondis Esperanton en 1887." (0.92)
  2. "La fundinto estis pola kuracisto." (0.85)
  3. "Zamenhof penis krei internacian lingvon." (0.78)
```

**Full Q&A System**:
```
Input: "Kiu fondis Esperanton?"
Process:
  1-4. Same as current
  5. Extract answer: query_type='kiu' → extract subject
  6. Verify: Is "Zamenhof" a PERSONO? → YES (M2.1)
  7. Generate: "Zamenhof"

Output: "Zamenhof" (or full sentence: "Zamenhof fondis Esperanton en 1887.")
```

### Query 2: "Kie naskiĝis la fondinto de Esperanto?" (Multi-Hop)

**Current Design**:
```
Input: "Kie naskiĝis la fondinto de Esperanto?"
Process:
  1. Parse → AST
  2. Retrieve sentences with 'nask' + 'fondinto'
  3. Rank/Rerank

Output:
  1. "Zamenhof naskiĝis en Bjalistoko." (0.75)
  2. "La fondinto de Esperanto estis pola kuracisto." (0.68)
  3. "Zamenhof fondis Esperanton." (0.62)

Problem: Sentences don't directly answer the question!
User must infer: "fondinto" = Zamenhof, then find where Zamenhof was born
```

**Full Q&A System**:
```
Input: "Kie naskiĝis la fondinto de Esperanto?"
Process:
  1-4. Same retrieval
  5. Multi-hop reasoning:
     Hop 1: "fondinto de Esperanto" = ?
       Find: "Zamenhof fondis Esperanton" → Zamenhof
     Hop 2: "Kie naskiĝis Zamenhof?" = ?
       Find: "Zamenhof naskiĝis en Bjalistoko" → Bjalistoko
  6. Extract: Bjalistoko (LOKO entity)
  7. Verify: confidence = 0.88

Output: "Bjalistoko" (or "en Bjalistoko")
```

### Query 3: "Kial Zamenhof fondis Esperanton?" (Why/Causal)

**Current Design**:
```
Input: "Kial Zamenhof fondis Esperanton?"
Output:
  1. "Zamenhof fondis Esperanton en 1887." (0.80)
  2. "Li pensis krei internacian lingvon." (0.75)
  3. "Multaj lingvoj kaŭzas komunikadajn problemojn." (0.65)

Problem: Returns relevant sentences, but doesn't extract REASON
```

**Full Q&A System**:
```
Input: "Kial Zamenhof fondis Esperanton?"
Process:
  1-4. Retrieval
  5. Causal reasoning:
     Find causal connector: "por", "ĉar", "tial"
     Extract reason: "por internacia komunikado"
  6. Verify with reasoning core: Is this a valid reason? → 0.85
  7. Generate: "Por internacia komunikado" or full explanation

Output: "Por krei internacian lingvon."
```

## Klareco's Unique Advantage: Deterministic Answer Extraction!

**Key Insight**: Traditional Q&A systems need learned models for answer extraction. **Klareco can do it deterministically!**

### Why Deterministic?

**Esperanto Grammar**:
- Kiu (who) → Subject (nominative, no -n)
- Kion (what-accusative) → Object (-n ending)
- Kiam (when) → Temporal phrase
- Kie (where) → Locative phrase
- Kiom (how much) → Quantity

**Example Rules**:
```python
def extract_answer(query_ast, sentence_ast):
    question_word = query_ast['question_word']

    if question_word == 'kiu':  # WHO
        # Extract subject (nominative)
        return sentence_ast['subjekto']['radiko']

    elif question_word == 'kion':  # WHAT (accusative)
        # Extract object (has -n)
        return sentence_ast['objekto']['radiko']

    elif question_word == 'kiam':  # WHEN
        # Extract temporal phrase (no case marking)
        return find_temporal_phrase(sentence_ast)

    # ... ~10 more rules
```

**This is 100% deterministic and 100% explainable!**

Traditional systems need learned models because English/other languages don't have:
- Clear case marking (-n for accusative)
- Predictable word order
- Consistent temporal marking

**Esperanto gives us this for free!**

## What We Should Add

### Tier 1: Answer Extraction (CRITICAL) 🔴

**Why critical**: Without this, we're just a search engine, not a Q&A system

**Implementation**:
- Module: `klareco/answer_extraction.py` (~500 lines, deterministic)
- Uses: M0 Parser output (AST with case/number/tense)
- Rules: ~15 rules for kiu/kion/kiam/kie/kiel/kial/kiom/etc.
- Explainability: 100% (show which rule fired)

**Effort**: 1-2 weeks (including tests)

**Impact**: ⭐⭐⭐⭐⭐ (transforms system from retrieval → Q&A!)

### Tier 2: Multi-Hop Reasoning (HIGH) 🟡

**Why important**: ~20-30% of questions need it

**Implementation**:
- Module: `klareco/reasoning_chain.py`
- Method: Build reasoning graph, follow hops
- Hybrid: Deterministic chain building + learned verification (reasoning core)

**Effort**: 3-4 weeks

**Impact**: ⭐⭐⭐⭐ (handles complex questions)

### Tier 3: Reasoning Core (PLANNED) 🟢

**Why planned**: This IS the Klareco thesis! (from VISION.md)

**Quote from VISION.md**:
> "Month 3-4: Add 20M param reasoning core, measure improvement"

**Implementation**:
- Model: ~20M params (vs 100M+ in traditional systems)
- Focus: Learn reasoning patterns, NOT grammar
- Input: Annotated ASTs (grammar deterministic!)
- Output: Reasoning chain verification, confidence scores

**Effort**: 1-2 months (design + train + evaluate)

**Impact**: ⭐⭐⭐⭐⭐ (core Klareco thesis validation!)

### Tier 4: Better Answer Generation (NICE TO HAVE) 🟢

**Current**: Basic deparser exists
**Improvement**: Answer-focused generation (concise vs full sentence)

**Effort**: 1 week

**Impact**: ⭐⭐ (nice UX improvement)

## Recommended Roadmap

### Month 1-2: Current Design (Foundation)

✅ **Goal**: Build retrieval system
- Complete Phase 0-1 (AST infrastructure, inspection tools)
- Complete Epic #641 (Data pipeline)
- Complete Epic #616 Phase 1-2 (Root + M1)
- **Output**: Top-N relevant sentences for queries

### Month 3: Answer Extraction (Transform to Q&A!)

🔴 **Goal**: Extract answers from sentences
- Design deterministic extraction rules
- Implement ~15 rules for question types
- Test on Esperanto Q&A dataset
- **Output**: Extracted answers, not just sentences!

**This is where we become a Q&A system!**

### Month 4: Multi-Hop Reasoning

🟡 **Goal**: Handle complex questions
- Design reasoning chain algorithm
- Implement hop traversal
- Integrate with answer extraction
- **Output**: Answers requiring multiple facts

### Month 5-6: Reasoning Core (Thesis Validation!)

🟢 **Goal**: Add learned reasoning (20M params)
- Design architecture (AST-aware transformer?)
- Train on reasoning patterns
- Evaluate vs traditional models
- **Output**: Prove Klareco thesis!

## Current Design Status

### What We Have ✅

- ✅ Complete retrieval pipeline design
- ✅ Deterministic foundation (M0, M2.1, M2.2)
- ✅ Minimal learned components (Root, M1)
- ✅ AST-based architecture (explainable)

### What's Missing ❌

- ❌ Answer extraction (deterministic rules)
- ❌ Multi-hop reasoning (reasoning chain)
- ❌ Reasoning core (20M param learned component)
- ❌ Evaluation on Q&A dataset (need benchmark!)

### Confidence in Design

**Retrieval System**: ✅ HIGH - Well designed, ready to implement
**Answer Extraction**: ✅ HIGH - Esperanto makes it deterministic!
**Multi-Hop**: ⚠️ MEDIUM - Need to design reasoning chain
**Reasoning Core**: ⚠️ MEDIUM - Planned but not designed yet

## Final Answer

### "What would our AI system be able to do?"

**Current Design (Fully Implemented)**:
- ✅ Parse Esperanto questions
- ✅ Retrieve relevant sentences
- ✅ Rank by relevance (M1 + Reranker)
- ✅ Resolve pronouns (M2.2, 80% deterministic)
- ✅ Classify entities (M2.1, 90% deterministic)
- ❌ Extract answers (returns full sentences)
- ❌ Multi-hop reasoning
- ❌ Generate natural language answers

**It's a retrieval system, NOT a full Q&A system yet!**

### "What kind of queries?"

**Handles well**:
- Simple factoid (WHO/WHAT/WHEN/WHERE)
- Definition questions
- Single-hop questions

**Cannot handle**:
- Multi-hop questions (20-30% of questions)
- Why/How questions (causal reasoning)
- Comparison questions
- Arithmetic questions
- Summarization requests

### "What's missing to respond to all kinds of queries?"

**Tier 1 (CRITICAL)**: Answer Extraction (~2 weeks, deterministic!)
**Tier 2 (HIGH)**: Multi-Hop Reasoning (~4 weeks)
**Tier 3 (PLANNED)**: Reasoning Core (~2 months, 20M params)

**Good news**: Answer extraction can be 100% deterministic in Esperanto!

**Recommendation**: Add answer extraction to Epic #616 as Phase 5!
