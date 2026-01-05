---
id: 30
title: Download complete Wikipedia dump (current dump is outdated)
state: open
created: '2026-01-05T01:09:46.348651Z'
labels:
- data-quality
- enhancement
priority: medium
---
## Problem

Current Wikipedia dump has ~258K articles (from unknown date). Esperanto Wikipedia now has ~380K articles (as of Jan 2025).

**Current state**:
- Raw dump: `eo_wikipedia.xml.bz2` (348 MB, compressed)
- Estimated articles in dump: ~258K (78% of pages, rest are redirects)
- Current EO Wikipedia: ~380K articles

**Gap**: Missing ~122K articles (32% of current Wikipedia)

## Solution

Download latest Esperanto Wikipedia dump:

```bash
# Get latest dump from Wikimedia
wget https://dumps.wikimedia.org/eowiki/latest/eowiki-latest-pages-articles.xml.bz2 \
  -O data/raw/eo/wikipedia/eo_wikipedia_latest.xml.bz2
```

Then re-run extraction:
```bash
python scripts/extract_wikipedia.py \
  --input data/raw/eo/wikipedia/eo_wikipedia_latest.xml.bz2 \
  --output data/extracted/wikipedia_sentences.jsonl
```

## Impact

- More comprehensive Q&A coverage
- More up-to-date information
- Better representation of modern Esperanto usage

## References

- Esperanto Wikipedia stats: https://en.wikipedia.org/wiki/Esperanto_Wikipedia
- Wikimedia dumps: https://dumps.wikimedia.org/eowiki/
