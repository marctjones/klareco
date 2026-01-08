---
id: 147
title: Add stopword filtering for high-frequency roots in retrieval
state: closed
created: '2026-01-08T15:35:17.295859Z'
labels:
- bug
- performance
priority: high
---
## Problem
The `_extract_roots()` method in `kuzu_inverted_index.py` includes ultra-common roots like `est` (the verb "to be") which appears in 1.7 million documents. This causes retrieval to process millions of candidates, taking 50+ seconds per query.

## Evidence
Query "Kio estas elefanto?" extracts:
- `kio` (korelativo) - correctly skipped
- `est` (verbo) - **NOT skipped** → 1,698,910 documents!
- `elef` (substantivo) - 1,206 documents

Result: 1.9 million candidates, 52 seconds to score.

## Root Frequencies (from corpus)
```
est: 1,698,910 docs
hav: ~500,000 docs (estimate)
far: ~200,000 docs (estimate)
```

## Current Skip List
```python
skip_vortspeco = {'korelativo', 'pronomo', 'artikolo', 'prepozicio', 'konjunkcio'}
```

## Proposed Fix Options

### Option A: Stopword list for common roots
```python
STOPWORD_ROOTS = {'est', 'hav', 'far', 'pov', 'dev', 'vol', 'ir', 'ven', 'don', 'pren'}
```

### Option B: Skip roots with doc_freq > threshold
```python
if self.get_doc_frequency(root) > 100_000:
    continue  # Too common to be useful
```

### Option C: Use IDF weighting to effectively ignore
Already using BM25, but high-freq roots still generate candidates. Need to filter BEFORE candidate generation.

## Recommendation
Option A (stopword list) is fastest to implement and most predictable.

## Files
- `klareco/rag/kuzu_inverted_index.py` - `_extract_roots()` method

## Impact
Should reduce query time from 50+ seconds to < 1 second for typical queries.
