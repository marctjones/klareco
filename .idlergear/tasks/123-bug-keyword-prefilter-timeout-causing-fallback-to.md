---
id: 123
title: 'Bug: Keyword prefilter timeout causing fallback to incomplete search'
state: closed
created: '2026-01-07T02:28:22.196901Z'
labels:
- bug
- performance
priority: medium
---
## Problem
Keyword prefilter was timing out (30s) during searches, causing incomplete results.

## Root Cause
In `_search_entity_focused` and `_search_hybrid`, supplementary keyword search was being run EVEN when HNSW prefilter was available. This was intended to catch documents HNSW might miss, but:
1. HNSW with weighted embeddings already finds relevant documents
2. Keyword prefilter times out on 30GB corpus 
3. Benchmark showed NO accuracy improvement from supplementary search

## Fix
Disabled supplementary keyword prefilter when HNSW is available (lines 1107-1124 and 1228-1243 in ast_aware_retriever.py).

The keyword prefilter is still used as the PRIMARY prefilter when HNSW is unavailable (fallback mode).

## Results
- No more timeout messages
- Same accuracy: 33.3% top-1, 75% top-5
- Faster search (no 30s timeout wait)
