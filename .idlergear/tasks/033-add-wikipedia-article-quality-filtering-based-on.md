---
id: 33
title: Add Wikipedia article quality filtering based on length/structure
state: open
created: '2026-01-05T01:09:47.793694Z'
labels:
- data-quality
- enhancement
priority: low
---
## Problem

Current extraction includes ALL articles regardless of quality:
- Stub articles (1-2 sentences)
- Incomplete articles
- Low-quality articles

**Distribution**:
- Average: 7.6 sentences/article
- Many articles have <3 sentences (stubs)

Low-quality articles add noise to retrieval without adding value.

## Solution

Add quality filters to extraction:

```python
def is_quality_article(sentences: List[str], word_count: int) -> bool:
    """Filter low-quality articles."""
    
    # Minimum content thresholds
    if len(sentences) < 3:
        return False  # Stubs
    
    if word_count < 50:
        return False  # Too short
    
    # Check for typical stub markers
    stub_markers = ['{{stub}}', '{{Ŝtumpo}}', 'Tiu ĉi artikolo estas ŝtumpo']
    text = ' '.join(sentences).lower()
    if any(marker.lower() in text for marker in stub_markers):
        return False
    
    return True
```

## Configuration

Make thresholds configurable:
```bash
python scripts/extract_wikipedia.py \
  --min-sentences 3 \
  --min-words 50 \
  --exclude-stubs
```

## Impact

- Higher quality corpus
- Better retrieval precision
- Reduced index size
- More reliable Q&A answers

## Acceptance Criteria

- [ ] Configurable quality thresholds
- [ ] Stub detection and filtering
- [ ] Extraction statistics show quality metrics
- [ ] Document filtering criteria in extraction log
