# Quick Wins Analysis: Semantic Retrieval vs AST Reasoning

## TL;DR

**Semantic retrieval is MUCH easier and gives MORE impressive results faster!**

**Recommendation**: Do semantic retrieval first. Here's why:

---

## Complexity Comparison

### Semantic Retrieval: ⭐ LOW COMPLEXITY

**What we already have:**
- ✅ AST with `subjekto`, `verbo`, `objekto` fields
- ✅ Case marking (`kazo: "nominativo"` vs `"akuzativo"`)
- ✅ Parser extracts these automatically

**What we need to add:**
1. Simple mapping: nominative subject → aganto, accusative object → paciento
2. Extract (aganto, verbo, paciento) tuples from corpus
3. Build index: tuple → list of sentence IDs
4. Query matching with wildcards

**Code complexity**: ~200 lines
**Time to implement**: 2-3 hours
**Dependencies**: None (works with current parser!)

### AST Reasoning: ⭐⭐⭐ HIGH COMPLEXITY

**What we need:**
1. Reasoning pattern framework
2. Pattern library (extractive, comparative, transitive, causal, etc.)
3. Pattern selection logic
4. Response generation from patterns
5. Confidence scoring
6. Error handling for edge cases

**Code complexity**: ~1000+ lines
**Time to implement**: 1-2 weeks
**Dependencies**: Semantic retrieval helps reasoning!

---

## Impact Comparison

### Semantic Retrieval Impact: 🚀 IMMEDIATE & IMPRESSIVE

**Before** (keyword-only retrieval):
```
Query: "Kiu vidas la katon?" (Who sees the cat?)

Returns ANY sentence with "vid" and "kat":
  ❌ "La kato vidas la hundon." (cat is AGENT - wrong!)
  ❌ "La kato estas bela." (no "vidas" - wrong!)
  ✅ "La hundo vidas la katon." (cat is PATIENT - correct!)
```

**After** (semantic retrieval):
```
Query: "Kiu vidas la katon?"
  → Semantic signature: (aganto=?, verbo=vid, paciento=kat)

Returns ONLY sentences where cat is PATIENT:
  ✅ "La hundo vidas la katon." (correct!)
  ✅ "Frodo vidas la katon." (correct!)
  ❌ "La kato vidas..." (filtered out - cat is agent!)
```

**This is MAGIC to users!** The system understands who-does-what-to-whom!

### AST Reasoning Impact: 🎯 POWERFUL BUT GRADUAL

Reasoning builds on retrieval:
- First you need to retrieve relevant sentences
- Then you reason over them

Without good retrieval, reasoning has bad inputs!

**Example**:
```
Query: "Kial Frodo iris al Mordoro?"

Needs:
1. Retrieval: Find sentences about Frodo going to Mordor
2. Reasoning: Extract causal relationships from those sentences

If retrieval is bad (returns wrong sentences), reasoning fails!
```

---

## Implementation Difficulty

### Semantic Retrieval: EASY ✅

**Step 1: Extract semantic signatures** (1 hour)
```python
def extract_signature(ast: dict) -> tuple:
    """Extract (agent, action, patient) from AST."""

    # Get subject (usually agent in active voice)
    subjekto = ast.get('subjekto', {})
    aganto = None
    if subjekto:
        # Handle vortgrupo or vorto
        if subjekto.get('tipo') == 'vortgrupo':
            aganto = subjekto['kerno']['radiko']
        elif subjekto.get('tipo') == 'vorto':
            aganto = subjekto['radiko']

    # Get verb (action)
    verbo = ast.get('verbo', {})
    ago = verbo.get('radiko') if verbo else None

    # Get object (usually patient)
    objekto = ast.get('objekto', {})
    paciento = None
    if objekto:
        if objekto.get('tipo') == 'vortgrupo':
            paciento = objekto['kerno']['radiko']
        elif objekto.get('tipo') == 'vorto':
            paciento = objekto['radiko']

    return (aganto, ago, paciento)

# That's it! ~30 lines of code.
```

**Step 2: Build index** (30 minutes)
```python
def build_semantic_index(corpus_path: Path) -> dict:
    """Build signature → [sentence_ids] index."""

    index = {}

    with open(corpus_path, 'r') as f:
        for i, line in enumerate(f):
            entry = json.loads(line)
            ast = entry.get('ast')

            if not ast:
                continue

            sig = extract_signature(ast)

            if sig not in index:
                index[sig] = []
            index[sig].append(i)

    return index

# Another ~20 lines!
```

**Step 3: Query with wildcards** (30 minutes)
```python
def search_semantic(query_sig: tuple, index: dict, metadata: list, k=10):
    """Search with wildcards (None = match anything)."""

    matches = []

    for sig, sent_ids in index.items():
        score = 0.0

        # Match each component (None = wildcard)
        if query_sig[0] is None or query_sig[0] == sig[0]:
            score += 0.3  # Agent matches
        if query_sig[1] is None or query_sig[1] == sig[1]:
            score += 0.4  # Action matches (most important)
        if query_sig[2] is None or query_sig[2] == sig[2]:
            score += 0.3  # Patient matches

        if score > 0:
            for sid in sent_ids:
                matches.append({
                    'sentence_id': sid,
                    'score': score,
                    'text': metadata[sid]['text']
                })

    # Sort by score
    matches.sort(key=lambda x: x['score'], reverse=True)
    return matches[:k]

# ~30 lines!
```

**Total: ~80 lines of clean, simple code!**

### AST Reasoning: COMPLEX ❌

**Need to handle:**
- Pattern matching (which reasoning type?)
- Template filling (how to construct response?)
- Multiple evidence synthesis (combine info from multiple sentences)
- Confidence scoring (how sure are we?)
- Failure recovery (what if no pattern matches?)
- Edge cases (malformed questions, ambiguous queries)

**Minimum 500+ lines for basic reasoning, 1000+ for robust system**

---

## Demo Value

### Semantic Retrieval Demo: WOW FACTOR! 🤩

**Show user:**
```
You: "Kiu vidas la katon?"

OLD SYSTEM:
  → Returns: "La kato vidas la hundon." (WRONG - cat is seeing!)

NEW SYSTEM:
  → Returns: "La hundo vidas la katon." (RIGHT - cat is being seen!)

Explanation: "I understand that you're asking WHO does the seeing,
             and the cat is what's BEING SEEN. I filtered results
             to only sentences where 'kat' is in the patient role."
```

**Impressive because:**
- Solves a hard problem (role ambiguity)
- Clearly better than keyword search
- Shows understanding of linguistic structure
- Easy to explain what it's doing

### AST Reasoning Demo: LESS OBVIOUS 🤔

**Show user:**
```
You: "Kial Frodo iris al Mordoro?"

SYSTEM:
  → Returns: "Frodo iris al Mordoro por detrui la Ringon."

Explanation: "I found a sentence explaining the reason..."
```

**Less impressive because:**
- Hard to show it's doing "reasoning" vs "good retrieval"
- User can't easily see the difference from before
- Benefit is subtle (better answer selection)
- Harder to explain what changed

---

## Progressive Enhancement Strategy

**Optimal path:**

### Phase 5A: Semantic Retrieval (Week 1) ⭐⭐⭐

**Day 1: Proper Noun Dictionary** (3 hours)
- Extract from corpus
- Integrate into parser
- **Result**: Parse rates jump to 95%+

**Day 2: Semantic Signatures** (3 hours)
- Implement signature extraction
- Build semantic index from corpus
- **Result**: Can query by roles!

**Day 3: Integration** (4 hours)
- Add semantic search to retriever
- Make it optional (fallback to structural)
- Add to demo script
- **Result**: Working demo!

**Day 4: Polish** (4 hours)
- Add tests
- Handle edge cases
- Documentation
- **Result**: Production ready!

**Total time**: 1 week
**Wow factor**: HIGH
**Risk**: LOW (simple code, few dependencies)

### Phase 5B: AST Reasoning (Weeks 2-4) ⭐⭐

**Now reasoning builds on semantic retrieval!**

**Week 2: Basic Patterns**
- Extractive (just return best match)
- Simple templates ("X estas Y")
- **Result**: Structured responses

**Week 3: Advanced Patterns**
- Comparison (X vs Y)
- Causation (X because Y)
- **Result**: Multi-fact reasoning

**Week 4: Polish & Integrate**
- Confidence scoring
- Error handling
- Full integration
- **Result**: Robust reasoning system

**Total time**: 3 weeks additional
**Wow factor**: HIGH (but builds on retrieval!)
**Risk**: MEDIUM (more complex logic)

---

## Concrete Example: What Changes

### Current System (Structural + Neural Retrieval)

```python
# In retriever.py - two-stage retrieval

def retrieve(self, query: str, k: int = 5):
    # Stage 1: Structural filtering (fast)
    query_ast = parse(query)
    canon_sig = canonicalize_sentence(query_ast)

    # Filter by slot signatures (SUBJ/VERB/OBJ)
    candidates = filter_by_slots(canon_sig, self.metadata)

    # Stage 2: Neural reranking (slow but accurate)
    query_embedding = self.encoder.encode(query_ast)
    results = faiss_search(query_embedding, candidates)

    return results
```

**Problem**: Structural filtering uses slots (SUBJ/VERB/OBJ) but doesn't understand ROLES!

### With Semantic Retrieval (NEW Stage 1.5)

```python
def retrieve(self, query: str, k: int = 5):
    # Stage 1: Structural filtering (slots)
    query_ast = parse(query)
    canon_sig = canonicalize_sentence(query_ast)
    candidates_structural = filter_by_slots(canon_sig, self.metadata)

    # Stage 1.5: Semantic filtering (roles) ✨ NEW!
    semantic_sig = extract_signature(query_ast)
    candidates_semantic = filter_by_semantics(semantic_sig, candidates_structural)

    # Stage 2: Neural reranking
    query_embedding = self.encoder.encode(query_ast)
    results = faiss_search(query_embedding, candidates_semantic)

    return results
```

**Benefit**: Fewer candidates to neural stage → faster + more accurate!

---

## Why Semantic First?

### 1. **Foundation for Reasoning**

Reasoning needs good retrieval:
```
Reasoning Pattern: "Extract causal relation"

Step 1: Retrieve sentences about X
        ↓
Step 2: Find causal markers ("ĉar", "pro", "tial")
        ↓
Step 3: Extract cause and effect
```

If Step 1 returns bad sentences, Steps 2-3 fail!

### 2. **Immediate Validation**

You can test semantic retrieval instantly:
```python
# Test: Does it filter correctly?
results = search_semantic(
    (None, 'vid', 'kat'),  # "Who sees cat?"
    index, metadata
)

for r in results:
    ast = parse(r['text'])
    objekto = ast.get('objekto', {})

    # Verify cat is object (patient), not subject (agent)
    assert 'kat' in str(objekto)
```

Reasoning is harder to validate (subjective quality).

### 3. **Independent Value**

Semantic retrieval is useful even without reasoning!

Current workflow:
```
Query → Retrieval → Return best match
```

Still works! Just returns better matches.

Reasoning requires the full pipeline:
```
Query → Retrieval → Reasoning → Response generation
```

If any step fails, whole thing fails.

### 4. **User-Facing Improvement**

Users immediately see better search results.

Reasoning improvements are subtle (better answer selection).

---

## Recommended Implementation Order

### ⭐ PHASE 5A: Semantic Retrieval (DO THIS FIRST!)

**Week 1, Day 1-2: Proper Nouns** (Already planned)
- Extract dictionary
- Integrate into parser

**Week 1, Day 3-4: Semantic Index**
```python
# scripts/build_semantic_index.py (NEW)

def main():
    corpus = load_corpus('data/corpus_with_sources_v2.jsonl')

    # Extract signatures
    signatures = {}
    for i, entry in enumerate(corpus):
        sig = extract_signature(entry['ast'])
        if sig not in signatures:
            signatures[sig] = []
        signatures[sig].append(i)

    # Save index
    with open('data/semantic_index.json', 'w') as f:
        json.dump(signatures, f)

    print(f"Built semantic index with {len(signatures)} unique signatures")
```

**Week 1, Day 5: Integration**
```python
# In retriever.py

class KlarecoRetriever:
    def __init__(self, ..., semantic_index_path=None):
        # ... existing code ...

        if semantic_index_path:
            self.semantic_index = load_semantic_index(semantic_index_path)
        else:
            self.semantic_index = None

    def retrieve(self, query, k=5, use_semantic=True):
        # ... existing structural filtering ...

        # NEW: Add semantic filtering
        if use_semantic and self.semantic_index:
            query_ast = parse(query)
            semantic_sig = extract_signature(query_ast)
            candidates = self._filter_semantic(semantic_sig, candidates)

        # ... existing neural reranking ...
```

**Week 2: Testing & Demo**
- Write comprehensive tests
- Create demo script showing semantic vs keyword search
- Document the feature

**Total: 2 weeks including polish**

### ⭐⭐ PHASE 5B: AST Reasoning (DO AFTER SEMANTIC)

**Week 3-4: Pattern Framework**
- Design pattern interface
- Implement pattern selection
- Add 2-3 simple patterns

**Week 5: Advanced Patterns**
- Comparison, causation, etc.
- Multi-evidence synthesis

**Week 6: Integration & Polish**
- Full pipeline integration
- Error handling
- Documentation

**Total: 4 weeks**

---

## Cost-Benefit Analysis

### Semantic Retrieval

**Cost**: 2 weeks (with testing)
**Benefit**:
- Immediately visible improvement ✅
- Foundation for reasoning ✅
- Easy to test & validate ✅
- Low risk (simple code) ✅
- Independent value ✅

**ROI**: ⭐⭐⭐⭐⭐ (Excellent!)

### AST Reasoning (Without Semantic First)

**Cost**: 4 weeks
**Benefit**:
- Hard to demo (subtle improvements)
- Depends on retrieval quality
- Complex to test
- Higher risk (many edge cases)
- No value without full pipeline

**ROI**: ⭐⭐ (Good, but risky)

### AST Reasoning (With Semantic First)

**Cost**: 4 weeks (same)
**Benefit**:
- Builds on proven retrieval ✅
- Better inputs → better outputs ✅
- Can show clear improvements ✅
- Lower risk (retrieval already works) ✅

**ROI**: ⭐⭐⭐⭐ (Very good!)

---

## Decision Matrix

| Criterion | Semantic Retrieval | AST Reasoning |
|-----------|-------------------|---------------|
| **Complexity** | ⭐ Low (~80 lines) | ⭐⭐⭐ High (~1000 lines) |
| **Time to implement** | 1 week | 3-4 weeks |
| **Demo impact** | 🚀 HIGH (obvious improvement) | 🤔 MEDIUM (subtle) |
| **Risk** | ✅ LOW (simple) | ⚠️ MEDIUM (complex) |
| **Dependencies** | None | Needs good retrieval! |
| **Independent value** | ✅ YES | ❌ NO (needs pipeline) |
| **Foundation for future** | ✅ YES (enables reasoning) | ⭐ YES (core capability) |
| **Testability** | ✅ Easy | ⚠️ Hard |

## Final Recommendation

### DO THIS ORDER:

1. **Week 1**: Proper Noun Dictionary (3 days) + Semantic Signatures (2 days)
2. **Week 2**: Semantic Index + Integration + Demo
3. **Week 3-6**: AST Reasoning (builds on semantic retrieval)

### Why This Works:

**Week 2 Demo**:
```
"Look at this! The system now understands WHO does WHAT to WHOM!

Query: 'Kiu vidas la katon?'

Before: Returns any sentence with 'vid' and 'kat' (including wrong ones)
After: Returns ONLY sentences where cat is being seen (correct role!)

This is using pure structural reasoning from Esperanto grammar -
no neural network needed for this part!"
```

**Week 6 Demo**:
```
"Now the system can REASON over what it finds!

Query: 'Kial Frodo iris al Mordoro?'

Step 1: Semantic retrieval finds sentences about Frodo going to Mordor
Step 2: AST reasoning extracts causal relationship
Step 3: Constructs answer: 'Frodo iris... por detrui la Ringon.'

Both retrieval and reasoning are deterministic AST operations!"
```

---

## Bottom Line

**Semantic retrieval is:**
- ✅ Easier to implement (2 weeks vs 4 weeks)
- ✅ More impressive results (users see it immediately)
- ✅ Better foundation (reasoning needs good retrieval)
- ✅ Lower risk (simple, testable code)
- ✅ Independent value (works alone)

**DO SEMANTIC RETRIEVAL FIRST!** 🎯

It's the highest ROI change you can make to the system right now.
