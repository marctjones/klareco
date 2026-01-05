---
id: 31
title: Filter Wikipedia extraction to exclude meta/discussion pages
state: open
created: '2026-01-05T01:09:46.871868Z'
labels:
- data-quality
- enhancement
priority: medium
---
## Problem

Wikipedia extraction includes ALL pages (556K total):
- Actual articles: ~258K
- Discussion pages (Vikipedio:Diskutejo/...)
- Meta pages (Vikipedio:Eble alinomendaj artikoloj/...)
- Archive pages (Vikipedio:Diskutejo/Arkivo/...)

**Top extracted pages by sentence count**:
1. Vikipedio:Diskutejo/Arkivo/2012/7 (1,872 sentences)
2. Historio de Unuiĝinta Reĝlando (1,726 sentences)
3. Vikipedio:Diskutejo/Teknikejo (1,591 sentences)

Discussion/meta pages pollute the index with non-encyclopedic content.

## Solution

Modify `extract_wikipedia.py` to filter by namespace:

```python
def should_extract_article(title: str) -> bool:
    """Only extract main namespace articles."""
    # Skip if contains namespace prefix
    if ':' in title:
        namespace = title.split(':')[0]
        # Allow only main namespace (no prefix) or specific namespaces
        if namespace != title:  # Has a prefix
            return False
    
    # Skip redirects, stubs, disambiguation pages
    # (already handled in existing code)
    
    return True
```

Apply filter before extraction to reduce from 556K → ~260K actual articles.

## Impact

- Cleaner index with only encyclopedic content
- Better retrieval quality (no discussion page noise)
- Smaller corpus/index size
- More relevant Q&A results

## Acceptance Criteria

- [ ] Only main namespace articles extracted
- [ ] Discussion pages excluded (Vikipedio:Diskutejo/...)
- [ ] Meta pages excluded (Vikipedio:Eble alinomendaj...)
- [ ] Archive pages excluded (Vikipedio:Diskutejo/Arkivo/...)
- [ ] Extraction count matches expected ~260K articles
