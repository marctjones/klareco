# Klareco Retrieval Guide

**Best practices for querying the Klareco RAG system**

---

## Quick Start

### Basic Retrieval
```python
from klareco.rag.retriever import create_retriever

retriever = create_retriever(
    'data/corpus_index_v3',
    'models/tree_lstm/best_model.pt'
)

results = retriever.retrieve("La ringo de potenco", k=5)
for r in results:
    print(f"[{r['score']:.2f}] {r['text']}")
```

### Enhanced Retrieval (Recommended)
```python
from scripts.enhanced_retrieval import enhanced_retrieve

results = enhanced_retrieve(
    retriever,
    "Kiu estas Frodo?",
    k=5,
    expand_queries=True  # Query expansion + entity boosting
)
```

---

## Query Types & Best Practices

### 1. **Entity Queries** (Kiu/Who)

**Query:** "Kiu estas Frodo?" (Who is Frodo?)

**Standard retrieval:** May not find direct biographical info
```python
# Score: 1.59 - Generic sentence about languages
```

**Enhanced retrieval:** Expands query + boosts entity matches
```python
# Searches: "Kiu estas Frodo?", "Frodo", "Frodo estas", "pri Frodo"
# Score: 1.80 - "(Bilbo kaj Frodo Baginzoj, estante fraŭlaj...)"
```

**Best practice:**
- Use `enhanced_retrieve()` for "Kiu estas X?" queries
- Or rephrase as content query: "Frodo Baginzo" instead of "Kiu estas Frodo?"

---

### 2. **Content Queries** (What/About)

**Query:** "La ringo de potenco" (The ring of power)

**Works well** with standard retrieval:
```python
# Score: 2.00 - Excellent relevance
# "Sarumano, malsukcesinte ekposedi la Ringon..."
```

**Best practice:**
- Use descriptive phrases instead of questions
- Include key terms from what you're looking for
- Standard retrieval works great

---

### 3. **Location Queries** (Kie/Where)

**Query:** "Kie loĝas la hobitoj?" (Where do hobbits live?)

**Standard:** Score 1.74
**Enhanced:** Score 1.81 (expands to "hobitoj loĝas", "hobitoj troviĝas")

**Best practice:**
- Enhanced retrieval adds location verbs (loĝas, troviĝas)
- Or use noun phrases: "la hejmo de hobitoj"

---

### 4. **Temporal Queries** (Kiam/When)

**Query:** "Kiam okazis la batalo?" (When did the battle happen?)

**Best practice:**
- Include event name: "la batalo de Helmo Gorĝo"
- Enhanced retrieval helps but may need specific dates in corpus

---

## Performance Tips

### Two-Stage Retrieval

Klareco uses **two-stage hybrid retrieval**:

1. **Stage 1: Structural filtering** (~2ms, deterministic)
   - Matches slot roots (SUBJ/VERB/OBJ)
   - Reduces 26K corpus → ~500 candidates
   - Zero learned parameters

2. **Stage 2: Neural reranking** (~15ms, Tree-LSTM)
   - Semantic similarity on candidates only
   - 15M parameters (7x smaller than traditional LLMs)

**Total latency:** ~15-18ms (30-40% faster than neural-only)

### Tuning Parameters

```python
# More candidates for structural filter
results = retriever.retrieve(
    query,
    k=5,
    structural_candidates=1000  # Default: 500
)

# Adjust top-k for responder
from klareco.experts.extractive import create_extractive_responder

responder = create_extractive_responder(
    retriever,
    top_k=5  # Use more sources for context
)
```

---

## Query Expansion Strategies

### Automatic Expansion

The `enhanced_retrieve()` function automatically:

| Query Type | Expansions |
|------------|------------|
| **Kiu estas X?** | X, "X estas", "pri X" |
| **Kio estas X?** | X, "X estas", "la X" |
| **Kie ...?** | "X loĝas", "X troviĝas" |

### Manual Expansion

For complex queries, expand manually:
```python
# Instead of: "Kiu venkis en la batalo?"
queries = [
    "la venkinto de la batalo",
    "batalo venko",
    "post la batalo"
]

all_results = []
for q in queries:
    all_results.extend(retriever.retrieve(q, k=3))

# Sort and deduplicate
```

---

## Entity Boosting

Enhanced retrieval boosts scores when entity appears in result:

```python
# If searching for "Frodo" and result contains "Frodo"
score *= 1.2  # 20% boost
```

This helps surface results that mention the entity, even if semantic similarity is moderate.

---

## Common Patterns

### Pattern 1: Biographical Info
```python
# ❌ Less effective
"Kiu estas Gandalfo?"

# ✅ More effective (enhanced retrieval)
enhanced_retrieve(retriever, "Kiu estas Gandalfo?", expand_queries=True)

# ✅ Most effective (direct phrase)
retriever.retrieve("Gandalfo sorĉisto", k=5)
```

### Pattern 2: Relationships
```python
# ✅ Good
"Frodo kaj Samsaĝo"
"la amikeco de Frodo"
"la nevo de Bilbo"
```

### Pattern 3: Events
```python
# ✅ Good
"la batalo de Helmo Gorĝo"
"la detruo de la Ringo"
"la vojaĝo al Mordoro"
```

### Pattern 4: Descriptions
```python
# ✅ Excellent
"la potenco de la Unu Ringo"
"la karaktero de hobitoj"
"la historio de Gondoro"
```

---

## Corpus Quality Impact

### Corpus V2 Improvements

The current corpus (v2) has:
- ✅ **Complete sentences** (not fragments)
- ✅ **High parse quality** (88-94%)
- ✅ **Structural metadata** (slot signatures)

**Before (v1 - fragments):**
```
Query: "La ringo de potenco"
Score: 1.46
Text: "Sed la Grandaj Ringoj, la Ringoj de Potenco, ili estis efektive..."
```

**After (v2 - complete):**
```
Query: "La ringo de potenco"
Score: 2.00 (+37%)
Text: "Sarumano, malsukcesinte ekposedi la Ringon, dum la konfuzo kaj
       perfidoj de tiu tempo trovus en Mordoro la mankantajn pensojn..."
```

---

## Debugging Retrieval

### Check Structural Filtering

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Will show: "Structural filter kept N candidates"
results = retriever.retrieve(query, k=5)
```

### Inspect Slot Signatures

```python
from klareco.parser import parse
from klareco.canonicalizer import canonicalize_sentence

ast = parse("Frodo portas la ringon")
slots = canonicalize_sentence(ast)

for role, slot in slots.items():
    if slot and slot.root:
        print(f"{role}: {slot.root}")

# Output:
# verbo: port
# objekto: ring
```

### Check Parse Quality

```python
ast = parse("Your sentence here")
stats = ast.get('parse_statistics', {})
print(f"Parse rate: {stats.get('success_rate', 0):.2f}")
```

---

## Advanced: Custom Retrieval

### Hybrid Retrieval

Combine structural + semantic manually:

```python
from klareco.structural_index import rank_candidates_by_slot_overlap
from klareco.canonicalizer import canonicalize_sentence

# Get structural candidates
query_ast = parse(query)
query_slots = canonicalize_sentence(query_ast)
query_roots = {role: slot.root for role, slot in query_slots.items() if slot}

# Get all metadata
import json
metadata = []
with open('data/corpus_index_v3/metadata.jsonl', 'r') as f:
    for line in f:
        metadata.append(json.loads(line))

# Rank by overlap
candidate_indices = rank_candidates_by_slot_overlap(
    query_roots,
    metadata,
    limit=100
)

# Then rerank with neural model...
```

---

## Performance Benchmarks

### Latency (26K corpus)

| Mode | Mean | P95 | P99 |
|------|------|-----|-----|
| Structural-only | 2-3ms | 4ms | 5ms |
| Hybrid (default) | 15-18ms | 22ms | 28ms |
| Neural-only | 20-25ms | 30ms | 40ms |

**Recommendation:** Use hybrid (default) for best quality/speed balance

### Accuracy

| Query Type | Standard | Enhanced | Improvement |
|------------|----------|----------|-------------|
| Entity (Kiu) | 1.59 | 1.80 | +13% |
| Content | 2.00 | 2.16 | +8% |
| Location (Kie) | 1.74 | 1.81 | +4% |

---

## Examples

### Full Pipeline Example

```python
from klareco.rag.retriever import create_retriever
from klareco.experts.extractive import create_extractive_responder
from klareco.parser import parse
from scripts.enhanced_retrieval import enhanced_retrieve

# Setup
retriever = create_retriever('data/corpus_index_v3', 'models/tree_lstm/best_model.pt')
responder = create_extractive_responder(retriever, top_k=3)

# Query with enhancement
query = "Kiu estas Frodo?"
results = enhanced_retrieve(retriever, query, k=5, expand_queries=True)

# Get answer
query_ast = parse(query)
answer = responder.execute(query_ast, query)

print(f"Answer: {answer['answer']}")
print(f"Confidence: {answer['confidence']:.3f}")
print(f"Sources: {len(answer['sources'])}")
```

### Batch Queries

```python
queries = [
    "La ringo de potenco",
    "Gandalfo sorĉisto",
    "La batalo de Helmo Gorĝo"
]

for query in queries:
    results = enhanced_retrieve(retriever, query, k=3)
    print(f"\n{query}:")
    for i, r in enumerate(results, 1):
        print(f"  {i}. [{r['score']:.2f}] {r['text'][:80]}...")
```

---

## Troubleshooting

### Low Scores

**Symptom:** All scores < 1.0

**Causes:**
- Query doesn't match corpus content
- Spelling errors (Esperanto is phonetic, check x-system)
- Very specific query with no direct matches

**Solutions:**
- Use enhanced retrieval
- Broaden query terms
- Check if content exists in corpus

### No Structural Filtering

**Symptom:** Debug shows "Structural filter skipped"

**Causes:**
- Query has no parseable roots
- Index doesn't have structural metadata

**Solutions:**
- Use Index V3 (has structural metadata)
- Check query parses correctly: `parse(query)`

### Wrong Results

**Symptom:** Results don't match query intent

**Causes:**
- Semantic similarity favors wrong context
- Entity not prominent in correct results

**Solutions:**
- Use enhanced retrieval (entity boosting)
- Rephrase as content query
- Try multiple query variations

---

## See Also

- `RAG_STATUS.md` - System status and architecture
- `CORPUS_V2_RESULTS.md` - Corpus quality improvements
- `docs/TWO_STAGE_RETRIEVAL.md` - Technical architecture
- `scripts/enhanced_retrieval.py` - Enhanced retrieval implementation
- `scripts/demo_rag.py` - Interactive demo

---

**Status:** Production Ready
**Last Updated:** November 2025
