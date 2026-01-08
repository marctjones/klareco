---
id: 121
title: 'Bug: ASTAwareRetriever ignores predicate nominatives in aliaj for definition
  questions'
state: closed
created: '2026-01-07T02:28:21.724953Z'
labels:
- bug
- retrieval
priority: high
---
Implemented two fixes for definition question handling:

1. **Weighted query embedding** (`_embed_query_ast`):
   - For "Kio estas X?" questions, use weighted averaging
   - Predicate nominative (X) gets 2x weight
   - Correlatives (kio, kiu) get 0.1x weight
   - Copula verb (estas) gets 0.2x weight
   - This ensures HNSW prefilter finds documents about X, not random sentences

2. **Raw slot score for definition questions** (`_compute_slot_similarity`):
   - Don't normalize by matched_slots for definition questions
   - Having both PRED_TO_SUBJ AND VERB match should score higher than just PRED_TO_SUBJ
   - This rewards actual definitions "X estas Y" over mere mentions

Results improved from 66.7% top-5 to 75% top-5 accuracy on diagnostic benchmark.

Files modified:
- klareco/rag/ast_aware_retriever.py (lines 398-490, 392-404)
