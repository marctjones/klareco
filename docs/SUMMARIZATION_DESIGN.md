# AST-Based Summarization Design

## The Problem

**Current design flaw**: Listed "AST Reasoning Core (20-100M params)" but we don't need general reasoning - we need **summarization** after retrieval.

**Goal**: Given N retrieved sentences about a query, produce a concise summary (1-3 sentences).

## Key Insight: Summarization Can Be Mostly Deterministic!

Esperanto's regular grammar + AST structure enables deterministic summarization rules:

### 1. Sentence Fusion (100% Deterministic)
Combine sentences with shared subjects/objects using AST manipulation:

```
Input ASTs:
  {"subjekto": "hundo", "verbo": "kuras"}
  {"subjekto": "hundo", "verbo": "manĝas"}

Fusion Rule: Same subject → combine with "kaj"
Output AST:
  {"subjekto": "hundo", "verbo": {"tipo": "kunmetita", "verboj": ["kuras", "manĝas"], "ligilo": "kaj"}}

Text: "Hundo kuras kaj manĝas."
```

### 2. Redundancy Elimination (100% Deterministic)
Remove duplicate information using AST comparison:

```python
def are_semantically_identical(ast1, ast2):
    """Compare ASTs for semantic equivalence"""
    # Same subject root + verb root + object root → duplicate
    if (ast1['subjekto']['radiko'] == ast2['subjekto']['radiko'] and
        ast1['verbo']['radiko'] == ast2['verbo']['radiko'] and
        ast1['objekto']['radiko'] == ast2['objekto']['radiko']):
        return True
    return False

# Keep only unique information
unique_asts = deduplicate(retrieved_asts)
```

### 3. Important Information Extraction (Mostly Deterministic)
Use AST hierarchy to identify core information:

```python
def get_importance_score(ast, query_ast):
    """Score sentence importance using AST structure"""
    score = 0

    # Main clause > subordinate clause
    if ast['tipo'] == 'frazo' and not ast.get('subordinata'):
        score += 3

    # Subject+Verb+Object > modifiers
    if ast.get('subjekto'):
        score += 2
    if ast.get('objekto'):
        score += 2

    # Overlap with query (deterministic)
    query_roots = extract_roots(query_ast)
    sentence_roots = extract_roots(ast)
    overlap = len(query_roots & sentence_roots)
    score += overlap * 5

    return score
```

## Proposed Pipeline

```
┌─────────────────────────────────────────────────────────┐
│ Step 1: Retrieve (Existing)                            │
│   Query AST → Retriever → Top N sentences              │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ Step 2: Parse (Existing)                               │
│   Sentences → Parser → ASTs with annotations           │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ Step 3: Preprocess (NEW - 100% Deterministic)         │
│   • Deduplicate: Remove semantically identical ASTs    │
│   • Filter: Remove incomplete parses, tier5 words      │
│   • Sort: Order by relevance to query                  │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ Step 4: Score Importance (NEW - Mostly Deterministic)  │
│   • Query overlap (deterministic)                      │
│   • AST depth/complexity (deterministic)               │
│   • Optional: Tiny learned model (5M params max)       │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ Step 5: Select & Fuse (NEW - 100% Deterministic)      │
│   • Select top 3-5 most important ASTs                 │
│   • Fuse ASTs with shared subjects/objects             │
│   • Compress: Remove redundant modifiers               │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ Step 6: Deparse (Existing)                            │
│   ASTs → Deparser → Summary text                       │
└─────────────────────────────────────────────────────────┘
```

## Models Needed

### Option A: 100% Deterministic (No Learned Models)
```
1. Retriever (existing: root embeddings 320K + M1 + reranker)
2. Parser (existing: 16 rules)
3. AST Deduplicator (new: deterministic AST comparison)
4. Importance Scorer (new: deterministic heuristics)
5. AST Fuser (new: deterministic AST manipulation)
6. Deparser (existing: AST → text)
```

**Total new learned parameters**: 0 (reuse existing models!)

### Option B: Hybrid (Minimal Learning)
```
1-2. Same as Option A
3. AST Deduplicator (deterministic)
4. Importance Scorer (tiny 5M param model for ranking)
   - Input: AST features (query overlap, depth, etc.)
   - Output: Importance score 0-1
   - Training: Supervised on human-labeled important sentences
5-6. Same as Option A
```

**Total new learned parameters**: 5M (optional, for better ranking)

## Preprocessing Requirements

### Input to Summarizer
```python
{
    "query": "Kiu fondis Esperanton?",
    "query_ast": {...},  # Parsed query
    "retrieved_sentences": [
        {
            "text": "Zamenhof fondis Esperanton en 1887.",
            "ast": {...},
            "score": 0.92
        },
        {
            "text": "La fundinto estis pola kuracisto.",
            "ast": {...},
            "score": 0.85
        },
        # ... more sentences
    ]
}
```

### Preprocessing Steps

1. **Validate ASTs** - Ensure all retrieved sentences parsed successfully
   ```python
   valid_asts = [s for s in retrieved if s['ast']['parse_statistics']['success_rate'] > 0.8]
   ```

2. **Extract Core Information** - Identify subject/verb/object from each AST
   ```python
   def extract_core_info(ast):
       return {
           'subjekto': get_root(ast['subjekto']),
           'verbo': get_root(ast['verbo']),
           'objekto': get_root(ast.get('objekto'))
       }
   ```

3. **Build Dependency Graph** - Find relationships between sentences
   ```python
   # Sentence 1: "Zamenhof fondis Esperanton."
   # Sentence 2: "Zamenhof estis kuracisto."
   # → Same subject "Zamenhof" → potential fusion
   ```

4. **Compute Query Overlap** - Measure relevance to query
   ```python
   query_roots = {'kiu'}  # Question word
   sentence_roots = {'zamenhof', 'fond', 'esperant'}
   # High overlap → likely important
   ```

## Example: End-to-End Summarization

### Input
```
Query: "Kiu fondis Esperanton?"

Retrieved (5 sentences):
1. "Zamenhof fondis Esperanton en 1887." (0.95)
2. "Ludoviko Lazaro Zamenhof estis pola kuracisto." (0.90)
3. "La fundinto vivis en Bjalistoko." (0.85)
4. "Esperanto estas planlingvo." (0.70)
5. "La lingvo havas regulan gramatikon." (0.65)
```

### Step 3: Preprocess
```python
# Deduplicate: No exact duplicates
# Filter: All parsed successfully
# Sort: Already sorted by score
valid_sentences = [1, 2, 3, 4, 5]
```

### Step 4: Score Importance
```python
Sentence 1: score = 10 (answer to "kiu", has subject+verb+object, high query overlap)
Sentence 2: score = 8  (subject "Zamenhof" matches answer, biographical detail)
Sentence 3: score = 6  (subject "fundinto" relates to query)
Sentence 4: score = 3  (about Esperanto but not founder)
Sentence 5: score = 2  (generic information)
```

### Step 5: Select & Fuse
```python
# Select top 3: [1, 2, 3]

# Fuse sentences 1 and 2 (same subject "Zamenhof"):
Input ASTs:
  {"subjekto": "Zamenhof", "verbo": "fondis", "objekto": "Esperanton", "aliaj": ["en 1887"]}
  {"subjekto": "Zamenhof", "verbo": "estis", "objekto": "kuracisto", "priskriboj": ["pola"]}

Fused AST:
  {"subjekto": "Zamenhof", "verbo": "kunmetita", "verboj": [
      {"verbo": "fondis", "objekto": "Esperanton", "aliaj": ["en 1887"]},
      {"verbo": "estis", "objekto": "kuracisto", "priskriboj": ["pola"]}
  ]}

# Keep sentence 3 separate (different subject)
```

### Step 6: Deparse
```
Summary (2 sentences):
1. "Zamenhof fondis Esperanton en 1887 kaj estis pola kuracisto."
2. "La fundinto vivis en Bjalistoko."
```

## Why This Works for Esperanto

1. **Regular Grammar** → Fusion rules are deterministic
2. **Explicit Case Marking** → Easy to identify subjects/objects
3. **Compositional Morphology** → Root-level deduplication works
4. **AST Structure** → Makes fusion/compression explicit operations

## Advantages Over "Reasoning Core"

| Reasoning Core (20-100M params) | AST Summarizer (0-5M params) |
|--------------------------------|------------------------------|
| Black box, hard to debug | Fully explainable (show AST operations) |
| Needs large training data | Mostly deterministic rules |
| May hallucinate | Cannot hallucinate (only rearranges retrieved info) |
| Learns grammar rules | Grammar is explicit in ASTs |

## Implementation Plan

### Phase 1: Core Deterministic Components (Week 1)
1. AST deduplicator (compare ASTs for semantic equivalence)
2. Deterministic importance scorer (query overlap + AST features)
3. AST fuser (combine sentences with shared subjects/objects)

### Phase 2: Testing & Evaluation (Week 2)
4. Test on sample queries
5. Measure: Relevance, Coherence, Conciseness
6. Iterate on fusion rules

### Phase 3: Optional Learned Component (Week 3)
7. If deterministic scorer insufficient, train tiny 5M param model
8. Input: AST features (extracted deterministically)
9. Output: Importance score
10. Training: Supervised on human-labeled important sentences

## Open Questions

1. **How many sentences to include in summary?**
   - Option A: Fixed (e.g., top 3)
   - Option B: Dynamic based on information density

2. **How to handle contradictions?**
   - If two sentences contradict, which to keep?
   - Could use recency, source reliability, or show both

3. **Should we paraphrase or keep original sentences?**
   - Paraphrasing: More concise but risks changing meaning
   - Original: Safer but may be verbose

4. **Training data for optional learned scorer?**
   - Need human-labeled "important vs unimportant" sentences
   - Could use active learning: show model predictions, ask for labels

## Next Steps

1. ✅ Complete Epic #641 (data pipeline) - DONE
2. ⬜ Implement deterministic AST deduplicator
3. ⬜ Implement deterministic importance scorer
4. ⬜ Implement deterministic AST fuser
5. ⬜ Test end-to-end on sample queries
6. ⬜ Evaluate: need learned scorer or deterministic sufficient?
