---
id: 1
title: Include all cleaned Gutenberg texts in extraction pipeline
state: open
created: '2026-01-02T03:48:31.322416Z'
labels:
- data-quality
- training
priority: high
---
## Problem
Only 7 sources (8MB) from `data/cleaned/eo/` are in `books_sentences.jsonl`, but we have 113 cleaned files (24MB total).

**108 files / 16MB of high-quality Esperanto text is NOT being used**, including:
- Fundamenta Krestomatio (718K) - Zamenhof's authoritative anthology
- Dua Libro de l' Lingvo Internacia (29K) - Zamenhof's second book  
- Classic translations: Alice in Wonderland, Robinson Crusoe, Bible excerpts
- Ibsen plays, Mark Twain stories, Goethe, Andersen fairy tales

## Impact
- ~3x more training data available
- Missing authoritative Zamenhof texts that should be highest priority
- Current data skewed heavily toward Lord of the Rings (76% of books)

## Solution
1. Update extraction script to process ALL cleaned .txt files
2. Rebuild books_sentences.jsonl
3. Rebuild unified corpus
4. Rebuild index
5. Retrain models with expanded data
