---
id: 25
title: Rebuild corpus with Wikipedia data using build_enhanced_corpus.py
state: open
created: '2026-01-05T00:15:12.864602Z'
labels:
- P0-CRITICAL
- data-quality
- blocker
priority: high
---
**CRITICAL**: Current corpus/index is missing ALL Wikipedia data (2747 articles).

## Problem

1. **Bug #1 fix not applied**: Corpus built before parser improvements
2. **Wikipedia data missing**: Index has 0 Wikipedia articles despite having extracted data

## Impact

Queries like "Kiu fondis Esperanton?" fail because:
- ❌ No "L. L. Zamenhof" article (exists in extraction, not in corpus)
- ❌ No "Esperanto" article (exists in extraction, not in corpus)
- ❌ No factual Wikipedia content at all

## Current State

- ✅ Wikipedia extraction: `data/extracted/wikipedia_sentences.jsonl` (2747 articles, 386 sentences in "Esperanto" article alone)
- ✅ Books extraction: `data/extracted/books_sentences.jsonl`
- ❌ Corpus: `data/corpus/unified_corpus.jsonl` (4.2M sentences, 0 Wikipedia)
- ❌ Index: `data/indexes/slot_full` (4.2M docs, 0 Wikipedia)

## Correct Command

```bash
python scripts/build_enhanced_corpus.py --stage all
```

**Why this script?**
- Reads from `wikipedia_sentences.jsonl` (preserves article metadata)
- Reads from `books_sentences.jsonl`
- Assigns proper tiers (Wikipedia = tier 6)
- Checkpointed (restartable)
- Uses Bug #1-fixed parser

**NOT** `parse_corpus.py` (reads from broken `cleaned_wikipedia.txt` - 225MB single line)

## Expected Results

**Before**:
- 4.2M sentences (books only)
- 0 Wikipedia articles
- Lower parse rates (Bug #1 not applied)

**After**:
- ~4.5M+ sentences (books + Wikipedia)
- ~2747 Wikipedia articles
- Higher parse rates (Bug #1 applied)
- Queries like "Kiu fondis Esperanton?" will work

## Runtime

- 4-5 hours
- Logs to `logs/corpus_building.log`
- Checkpointed every 10K sentences

## Dependencies

- ✅ Bug #1 parser fix (issue #220)
- ✅ Extractions exist

## Next Steps

After rebuild:
1. Rebuild index (may be automatic)
2. Re-run Q&A benchmark (issue #232)
3. Verify "Kiu fondis Esperanton?" returns correct answer

**Priority**: P0 - Blocks all Q&A functionality
