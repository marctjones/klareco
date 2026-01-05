---
id: 2
title: Rebuild corpus and index with expanded books extraction (194K sentences)
state: closed
created: '2026-01-02T04:16:42.094305Z'
labels:
- enhancement
- pipeline
priority: medium
---
The books extraction pipeline was updated to include all 111 cleaned texts (up from 7).

**Before:** 7 sources, 27K sentences, 8 MB
**After:** 111 sources, 194K sentences, 53 MB

To complete the pipeline:

1. **Rebuild corpus** (2-4 hours):
   ```bash
   ./scripts/parse_corpus.sh
   ```

2. **Rebuild index** (after corpus completes):
   ```bash
   python scripts/index_corpus.py --corpus data/enhanced_corpus/corpus_with_metadata.jsonl
   ```

3. **Optional: Retrain models** with expanded data

The extraction script now dynamically discovers all .txt files in data/cleaned/eo/ instead of using a hardcoded list.
