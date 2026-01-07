---
id: 96
title: Fix ASTAware to use slot-based retrieval instead of averaged embeddings
state: closed
created: '2026-01-06T22:15:22.435478Z'
labels:
- bug
- priority-high
- architecture
priority: high
---
## Critical Architectural Bug

ASTAware retriever averages all root embeddings into a single vector for HNSW prefiltering, which completely defeats the purpose of our slot-based AST approach.

### FIXED (2026-01-06)

Added slot-based reranking to both HNSW and keyword prefilters:
- `_extract_query_slots()`: Extracts SUBJ/VERB/OBJ embeddings from query AST
- `_compute_slot_similarity()`: Compares slots with proper weighting
- HNSW prefilter now reranks using 30% HNSW + 70% slot similarity
- Keyword prefilter uses 40% keyword + 60% slot similarity

### What Changed
- Line 218-345: New `_hnsw_prefilter()` with slot reranking
- Line 287-345: New `_compute_slot_similarity()` method
- Line 320-480: New `_extract_query_slots()` method
- Line 482-629: Updated `_keyword_prefilter()` with slot reranking

### Still Needed
The semantic gap issue remains - keyword prefilter can't bridge "fondis" ↔ "aŭtoro". See note #86 for options.
