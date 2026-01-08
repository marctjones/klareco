---
id: 128
title: Add keyword-based inverted index for semantic role fallback
state: closed
created: '2026-01-07T03:52:52.883904Z'
labels:
- enhancement
- retrieval
- 'priority: high'
priority: high
---
## Problem

HNSW embedding search has low recall (~35%) for queries where the answer uses semantically equivalent but differently-embedded terms. For example:

- Query: "Kiu fondis Esperanton?" (embedding dominated by "fond")
- Target: "ZAMENHOF, Aŭtoro de la lingvo Esperanto" (embedding dominated by "aŭtor")
- Similarity: 0.1277 (too low to appear in top-1000 HNSW results)

The keyword prefilter times out because it greps a 32GB JSONL file.

## Proposed Solution

Build a lightweight **inverted index** mapping keywords → document IDs for fast keyword-based fallback.

### Architecture

```
data/indexes/slot_hybrid/
├── slot_index.jsonl       # Full documents (32GB)
├── keyword_index.json     # keyword → [doc_ids] mapping (~500MB)
└── slot_index.offsets.npy # Byte offsets for O(1) doc lookup
```

### Implementation

```python
class KeywordIndex:
    def __init__(self, index_path: Path):
        self.keyword_to_docs: Dict[str, List[int]] = {}
        self._load_index(index_path / "keyword_index.json")
    
    def search(self, keywords: List[str], max_results: int = 1000) -> List[int]:
        """Find documents containing ANY of the keywords."""
        doc_ids = set()
        for kw in keywords:
            if kw in self.keyword_to_docs:
                doc_ids.update(self.keyword_to_docs[kw][:max_results])
        return list(doc_ids)[:max_results]
```

### Build Script

```python
def build_keyword_index(index_path: Path):
    """Build inverted index from slot_index.jsonl."""
    keyword_to_docs = defaultdict(list)
    
    with open(index_path / "slot_index.jsonl") as f:
        for doc_id, line in enumerate(f):
            doc = json.loads(line)
            # Extract roots from slots
            for slot_name in ['SUBJ', 'VERB', 'OBJ']:
                slot = doc.get('slots', {}).get(slot_name, {})
                if slot and slot.get('root'):
                    keyword_to_docs[slot['root'].lower()].append(doc_id)
            # Also index full text keywords (top content words)
            text = doc.get('text', '').lower()
            for word in extract_content_words(text):
                keyword_to_docs[word].append(doc_id)
    
    # Save index
    with open(index_path / "keyword_index.json", 'w') as f:
        json.dump(keyword_to_docs, f)
```

### Integration with ASTAwareRetriever

```python
def _keyword_prefilter(self, query_ast, max_results=1000):
    # Extract keywords + semantic expansions
    keywords = self._extract_keywords(query_ast)
    expanded = self._expand_with_semantics(keywords)  # fond → [fond, aŭtor, kreint, ...]
    
    # Fast lookup via inverted index
    doc_ids = self.keyword_index.search(expanded, max_results)
    
    # Load documents by ID (O(1) via offsets)
    return [(0.0, self._get_document(doc_id)) for doc_id in doc_ids]
```

## Expected Impact

- Query "Kiu fondis Esperanton?" will find documents containing "aŭtor" via keyword expansion
- Target document "ZAMENHOF, Aŭtoro..." will appear in candidates
- Combined with Stage 2 pattern matching, should score highly

## Effort

- Build script: 2 hours
- Integration: 2 hours
- Testing: 1 hour
- **Total: ~5 hours**

## Related

- Task #222: Q&A Retrieval Evaluation (35% recall)
- Task #53: Multi-strategy retriever (completed - architecture ready for this)
- Task #232: Re-run benchmark after fixes
