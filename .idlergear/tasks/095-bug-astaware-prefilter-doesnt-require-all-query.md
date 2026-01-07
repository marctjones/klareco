---
id: 95
title: 'Bug: ASTAware prefilter doesn''t require ALL query terms to match'
state: closed
created: '2026-01-06T22:11:16.283310Z'
labels:
- bug
- retrieval
- critical
priority: high
---
## Bug: Keyword prefilter only uses one keyword

The keyword prefilter in ASTAware only searches for the PRIMARY (longest) keyword, not ALL keywords. This causes poor recall.

### FIXED (2026-01-06)

Changed grep strategy from single keyword to chained grep:
- Before: `grep -i primary_keyword`
- After: `grep -i keyword1 | grep -i keyword2 | grep -i keyword3`

Takes top 3 keywords by length (most specific) and requires ALL to match.

### What Changed
- Line 482-629: Updated `_keyword_prefilter()` 
- Added `require_all_keywords` parameter (default: True)
- Uses chained subprocess.Popen for intersection search
- Also adds slot-based reranking (40% keyword, 60% slot)

### Trade-off
Requiring ALL keywords may reduce recall for semantic matches. See note #86 for the semantic gap issue.
