---
id: 103
title: Re-parse corpus and rebuild indexes with fundamento_roots.json
state: open
created: '2026-01-06T23:32:48.989140Z'
labels:
- tech-debt
- corpus
priority: low
---
The parser now uses `data/vocabularies/fundamento_roots.json` for proper morphological disambiguation (commit pending).

Pre-computed ASTs in the corpus and slot indexes were built with the old parser (without fundamento_roots), causing minor inconsistencies in morphological breakdown for words like:
- resanigos → was `resan+ig`, now correctly `re+san+ig`
- bonege → was `bo+neg`, now correctly `bon+eg`
- belulino → was `ulin`, now correctly `bel+ul+in`

**To rebuild:**
```bash
./scripts/parse_corpus.sh --fresh    # Re-parse corpus (~500K sentences)
./scripts/build_hybrid_indexes.sh    # Rebuild slot indexes
```

**Priority:** Low - queries are parsed with the new parser at runtime, so functionality is not affected. This is for full consistency between corpus ASTs and query ASTs.
