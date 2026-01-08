---
id: 143
title: Fix critical latency issue - 44s per query in Kuzu retriever
state: closed
created: '2026-01-08T15:25:34.414218Z'
labels:
- bug
- performance
priority: high
---
## Problem
Q&A benchmark shows 27.4s average latency per query. Target is < 1s.

## Root Cause Analysis (COMPLETED)

### Issue 1: Ultra-high frequency roots not filtered
The `_extract_roots()` method was including `est` (verb "to be") which appears in 1.7M documents. Query "Kio estas elefanto?" generated 1.9M candidates.

**Fix**: Added `STOPWORD_ROOTS` constant with ~20 ultra-common roots that are filtered during retrieval. Implemented in #147.

### Issue 2: Parser over-splitting words
"elefanto" was being parsed as `elef` + `-ant` suffix instead of `elefant` root.

**Fix**: Added -ant look-alikes (elefant, gigant, infant, etc.) to `PROTECTED_SUFFIX_ROOTS`. Implemented in #148.

## Results After Fix

| Query | Before | After |
|-------|--------|-------|
| Kio estas elefanto? | 44s | 0.09s |
| Priskribu hundon | ~30s | 1.3s |
| Kio estas rivero? | ~30s | 1.6s |
| Kiu kuris? | ~30s | 0.23s |

**Average latency reduced from 27s to ~0.8s** ✓

## Status: RESOLVED
