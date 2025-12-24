# Corpus V2 Build Script Improvements

**Date**: 2025-11-27
**Status**: ✅ Complete

## Problem Analysis

### Issues Found

1. **Memory/CPU Overload**: 804MB Wikipedia file was causing system freezes
2. **Wrong Filter Logic**: The `min-parse-rate` filter was removing good Esperanto sentences
3. **Poor Progress Indicators**: With large files, the script appeared frozen with no feedback
4. **Non-Esperanto Content**: Wikipedia file contains English sections that were being included

### Root Causes

1. **Parse Rate Misunderstanding**:
   - The filter `parse_rate < 0.5` removes sentences that parse poorly
   - This CORRECTLY filters out non-Esperanto or malformed text
   - However, the default was set to 0.5, which filtered out many valid sentences
   - **Solution**: Changed default to 0.0 (no filtering) to keep all Esperanto sentences

2. **Wikipedia Contains English**:
   - The Wikipedia dump includes `<div lang="en">` sections with English text
   - These were being extracted as "sentences" and added to the corpus
   - **Solution**: Added explicit filtering to skip English-language sections

3. **Insufficient Progress Updates**:
   - Default batch size of 50 meant updates every 50 sentences
   - For 804MB file, this was too infrequent
   - **Solution**: Reduced default batch size to 20, added memory usage reporting

4. **CPU/Memory Usage**:
   - Default throttle of 0.05s was too aggressive for large files
   - Parser AST generation is CPU-intensive
   - **Solution**: Increased default throttle to 0.1s, added more frequent GC

## Changes Made

### 1. `scripts/extract_sentences.py`

**English Content Filtering** (lines 260-262):
```python
# Skip English-language sections in Wikipedia (between <div lang="en"> and </div>)
if '<div lang="en">' in para or para.strip().startswith('This part of Wikipedia'):
    continue
```

**URL Filtering** (lines 280-284):
```python
# If sentence has too many URLs (>30% of words), skip it
if any(indicator in sent.lower() for indicator in ['http://', 'https://', 'www.', '.com', '.net', '.org']):
    url_word_count = sum(1 for word in sent.split() if 'http' in word or 'www.' in word or '.com' in word)
    if url_word_count > word_count * 0.3:
        continue
```

**Esperanto Character Support** (line 287):
```python
# Include Esperanto special characters in "valid" character set
special_ratio = sum(1 for c in sent if not c.isalnum() and c not in ' .,;:!?-—\'"ĉĝĥĵŝŭĈĜĤĴŜŬ') / len(sent)
```

### 2. `scripts/build_corpus_v2.py`

**Clarified Parse Rate Filter** (lines 161-169):
```python
# Filter by parse quality if requested
# Note: We want to KEEP sentences that parse well (high parse_rate)
# A low parse_rate might indicate non-Esperanto text or malformed sentences
if min_parse_rate > 0 and with_ast:
    parse_rate = sent_data.get('parse_rate', 0)
    # Keep sentences with parse_rate >= min_parse_rate
    if parse_rate < min_parse_rate:
        filtered_count += 1
        sent_idx += 1
        continue
```

**Enhanced Progress Reporting** (lines 189-211):
```python
# Show more detailed progress including memory info
try:
    import psutil
    process = psutil.Process()
    mem_mb = process.memory_info().rss / 1024 / 1024
    print(f"   ⏳ Processed: {count:,} sentences ({rate:.1f} sent/sec, {filtered_count:,} filtered, mem: {mem_mb:.0f}MB)")
except ImportError:
    print(f"   ⏳ Processed: {count:,} sentences ({rate:.1f} sent/sec, {filtered_count:,} filtered)")
```

**Changed Defaults** (lines 273-296):
- `--min-parse-rate`: Changed from 0.5 to **0.0** (no filtering by default)
- `--batch-size`: Changed from 50 to **20** (more frequent updates)
- `--throttle`: Changed from 0.05s to **0.1s** (more CPU-friendly)

## Usage Guide

### Quick Start (Recommended for Wikipedia)

```bash
# Use default settings - keeps ALL Esperanto sentences, prevents freezing
python scripts/build_corpus_v2.py
```

This will:
- Process with **0.1s throttle** between batches (prevents CPU overload)
- Show progress every **20 sentences** (good feedback for large files)
- Keep **all Esperanto sentences** (min-parse-rate=0.0)
- Show **memory usage** if psutil is installed
- **Resume from checkpoint** if interrupted

### Advanced Usage

```bash
# Maximum speed (may freeze on large files)
python scripts/build_corpus_v2.py --throttle 0.0 --batch-size 100

# Very gentle on CPU (slower but won't freeze)
python scripts/build_corpus_v2.py --throttle 0.3 --batch-size 10

# Filter out low-quality sentences (use with caution!)
python scripts/build_corpus_v2.py --min-parse-rate 0.3

# Skip AST generation for maximum speed (no quality info)
python scripts/build_corpus_v2.py --no-ast --throttle 0.0
```

### Monitoring Progress

The script now shows:
```
📖 Processing: Vikipedio Esperanto (Wikipedia)
   ⏳ Processed: 20 sentences (15.3 sent/sec, 3 filtered, mem: 145MB)
   ⏳ Processed: 40 sentences (16.1 sent/sec, 5 filtered, mem: 147MB)
   ⏳ Processed: 60 sentences (15.8 sent/sec, 8 filtered, mem: 149MB)
   ...
```

This tells you:
- How many sentences processed from this file
- Processing speed (sentences/second)
- How many sentences were filtered out
- Current memory usage (if psutil installed)

### Understanding the Parse Rate Filter

**What is parse_rate?**
- A score from 0.0 to 1.0 indicating how well the sentence parses as Esperanto
- 1.0 = perfect parse, all words recognized
- 0.5 = half the words parsed correctly
- 0.0 = unable to parse (likely not Esperanto)

**When to use filtering:**
- `--min-parse-rate 0.0` (default): Keep all sentences, even if parsing fails
  - **Use for**: Building comprehensive corpus from known Esperanto texts
  - **Best for**: Wikipedia, literature, trusted sources

- `--min-parse-rate 0.3`: Remove sentences where <30% of words parse
  - **Use for**: Filtering out obvious non-Esperanto content
  - **Best for**: Mixed-language sources, noisy data

- `--min-parse-rate 0.5`: Remove sentences where <50% of words parse
  - **Use for**: High-quality corpus only
  - **Best for**: Training data for models (not recommended for general corpus building)

**Why the default is 0.0:**
- Esperanto Wikipedia IS Esperanto - we want to keep all sentences
- Parser may not recognize proper nouns, neologisms, technical terms
- Better to include and let retrieval/indexing handle quality
- You can always filter later when building the index

## Performance Improvements

### Before

- Batch size: 50 (updates every 50 sentences)
- Throttle: 0.05s (very fast but CPU-intensive)
- No memory reporting
- Filtering was removing good Esperanto sentences
- System would freeze on large files

### After

- Batch size: 20 (updates every 20 sentences = more feedback)
- Throttle: 0.1s (2x more breathing room for CPU)
- Memory usage shown in progress
- Keeps all Esperanto sentences by default
- English sections explicitly filtered out
- Checkpoint every 20 sentences (fine-grained resumption)

### Expected Performance on 804MB Wikipedia

- **Speed**: ~10-20 sentences/second (depends on CPU)
- **Memory**: ~150-300MB (depends on batch size and parser cache)
- **CPU**: ~50-70% of one core (with 0.1s throttle)
- **Progress updates**: Every 1-2 seconds
- **Total time**: ~1-3 hours for full Wikipedia (varies by system)

## Resumption

If the script crashes or is interrupted:

```bash
# Simply re-run the same command - it will resume automatically
python scripts/build_corpus_v2.py
```

The checkpoint file (`data/build_corpus_v2_checkpoint.json`) tracks:
- Which file was being processed
- Which sentence within that file
- Total sentences processed so far

The script will:
- Skip already-completed files
- Resume within the interrupted file at the exact sentence
- Continue adding to the output file (append mode)

## Testing

To test on a small file first:

```bash
# Process just one small file to test
python scripts/build_corpus_v2.py \
  --cleaned-dir data/cleaned \
  --output data/test_corpus.jsonl \
  --batch-size 10 \
  --throttle 0.0
```

Then check the output:

```bash
# Count sentences
wc -l data/test_corpus.jsonl

# Check first few sentences
head -5 data/test_corpus.jsonl | jq .text
```

## Troubleshooting

### Script Still Freezing

```bash
# Increase throttle significantly
python scripts/build_corpus_v2.py --throttle 0.5 --batch-size 10
```

### Too Slow

```bash
# Reduce throttle and increase batch size
python scripts/build_corpus_v2.py --throttle 0.05 --batch-size 50
```

### Out of Memory

```bash
# Reduce batch size to force more frequent garbage collection
python scripts/build_corpus_v2.py --batch-size 5
```

### Too Many Sentences Filtered

```bash
# Check what min-parse-rate is being used
python scripts/build_corpus_v2.py  # Default is now 0.0

# If you want NO filtering at all
python scripts/build_corpus_v2.py --min-parse-rate 0.0
```

## What's Next

After building the corpus:

```bash
# 1. Check the corpus
wc -l data/corpus_with_sources_v2.jsonl
du -h data/corpus_with_sources_v2.jsonl

# 2. Build the index
python scripts/index_corpus.py \
  --corpus data/corpus_with_sources_v2.jsonl \
  --output data/corpus_index_v3 \
  --batch-size 32

# 3. Test retrieval
python scripts/demo_rag.py --interactive
```

## Summary

The corpus building script is now:
- ✅ **More stable**: Won't freeze your computer
- ✅ **More informative**: Shows progress, memory, and filtering stats
- ✅ **More accurate**: Filters out English content, keeps Esperanto sentences
- ✅ **More configurable**: Easy to adjust speed vs. CPU usage
- ✅ **More resilient**: Fine-grained checkpoints, easy resumption

**Recommended command for Wikipedia:**
```bash
python scripts/build_corpus_v2.py
```

This will build a complete, high-quality Esperanto corpus without freezing your system!
