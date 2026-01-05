# Task #25 Quick Start Guide

**Goal**: Build unified corpus including Wikipedia data (P0-CRITICAL blocker)

## TL;DR - Run This

```bash
# In a separate terminal (not in Claude Code)
cd /home/marc/Projects/klareco
./scripts/build_corpus_with_wikipedia.sh
```

**Estimated time**: 1-2 hours
**Output**: `data/corpus/unified_corpus.jsonl`

---

## What This Does

1. ✅ Reads Gutenberg books extraction
2. ✅ Reads Wikipedia articles extraction
3. ✅ Parses all sentences to AST
4. ✅ Filters by quality (parse rate > 50%)
5. ✅ Assigns tiers (books=tier 5, Wikipedia=tier 6)
6. ✅ Merges into unified corpus
7. ✅ Validates Wikipedia data included

## Features

- **Live Progress**: Shows sentences/sec, percentage complete
- **Checkpointed**: Resume from interruptions (saves every 30s)
- **Validated**: Confirms Wikipedia articles present
- **Logged**: Full log saved to `logs/corpus_build_*.log`

## Usage Options

### Normal Mode (Recommended)
```bash
./scripts/build_corpus_with_wikipedia.sh
```

### Verbose Mode (Detailed Progress)
```bash
./scripts/build_corpus_with_wikipedia.sh --verbose
```

### Fresh Start (Ignore Checkpoints)
```bash
./scripts/build_corpus_with_wikipedia.sh --fresh
```

### Resume After Interruption
```bash
# Just run normally - automatically resumes from checkpoint
./scripts/build_corpus_with_wikipedia.sh
```

### Help
```bash
./scripts/build_corpus_with_wikipedia.sh --help
```

---

## Expected Output

### Progress Updates
```
[INFO] Processing books_sentences.jsonl: 8,234 sentences
Progress: 1,000/8,234 (12.1%) - 120 sent/sec
Progress: 2,000/8,234 (24.3%) - 125 sent/sec
...

[INFO] Processing wikipedia_sentences.jsonl: 4,204,488 sentences
Progress: 100,000/4,204,488 (2.4%) - 450 sent/sec
Progress: 200,000/4,204,488 (4.8%) - 460 sent/sec
...
```

### Final Summary
```
=== Summary ===

✓ Task #25 Complete!

Output:
  File: data/corpus/unified_corpus.jsonl
  Size: 21.3GB
  Lines: 4,512,722

Wikipedia sentences found: 4,204,488

Key articles found:
  ✓ L. L. Zamenhof: 127 sentences
  ✓ Esperanto: 386 sentences
  ✓ La Espero: 24 sentences

Next Steps:
  1. Rebuild index:
     ./scripts/index_slot.sh --fresh

  2. Test Q&A queries:
     python scripts/demo_slot_retrieval.py --query "Kiu kreis Esperanton?"

  3. Run benchmarks:
     ./scripts/benchmark_slot_retrievers.py
```

---

## Monitoring Progress

### Check Log File
```bash
# In another terminal
tail -f logs/corpus_build_$(date +%Y%m%d)*.log
```

### Count Output Lines
```bash
wc -l data/corpus/unified_corpus.jsonl
```

### Check for Wikipedia
```bash
# Quick check
python3 -c "
import json
wiki = 0
with open('data/corpus/unified_corpus.jsonl') as f:
    for line in f:
        if json.loads(line).get('source', {}).get('tier') == 6:
            wiki += 1
            if wiki >= 10: break
print(f'Found Wikipedia sentences: {wiki}+')
"
```

---

## Troubleshooting

### If Script Fails

1. **Check log file**:
   ```bash
   cat logs/corpus_build_*.log | tail -50
   ```

2. **Resume from checkpoint**:
   ```bash
   ./scripts/build_corpus_with_wikipedia.sh  # Automatically resumes
   ```

3. **Start fresh**:
   ```bash
   ./scripts/build_corpus_with_wikipedia.sh --fresh
   ```

### If No Wikipedia Found

This should NOT happen if script completes successfully, but if it does:

```bash
# Verify Wikipedia extraction exists
ls -lh data/extracted/wikipedia_sentences.jsonl

# Count sentences
wc -l data/extracted/wikipedia_sentences.jsonl

# Check a sample
head -1 data/extracted/wikipedia_sentences.jsonl | python3 -m json.tool
```

### If Out of Disk Space

Corpus will be ~20-25GB. Ensure you have at least 30GB free:

```bash
df -h .
```

---

## What Gets Created

```
data/corpus/
├── unified_corpus.jsonl           # FINAL OUTPUT
├── unified_corpus.jsonl.backup_*  # Backup of previous corpus
└── build_YYYYMMDD_HHMMSS/         # Build artifacts
    ├── books_corpus.jsonl         # Books only
    └── wikipedia_corpus.jsonl     # Wikipedia only

logs/
├── corpus_build_YYYYMMDD_HHMMSS.log  # Full log
└── corpus_build_checkpoint.json      # Resume state (deleted on success)
```

---

## After Completion

### Verify Wikipedia Inclusion

```bash
python3 << 'EOF'
import json
from collections import Counter

wiki_count = 0
key_articles = Counter()

with open('data/corpus/unified_corpus.jsonl') as f:
    for line in f:
        doc = json.loads(line)
        source = doc.get('source', {})

        if source.get('tier') == 6:
            wiki_count += 1
            article = source.get('article_title', '')
            if article in ['L. L. Zamenhof', 'Esperanto', 'La Espero']:
                key_articles[article] += 1

print(f"✓ Wikipedia sentences: {wiki_count:,}")
print(f"\nKey articles:")
for article, count in key_articles.items():
    print(f"  {article}: {count} sentences")
EOF
```

### Next: Rebuild Index

```bash
./scripts/index_slot.sh --fresh
```

This will create a new index including the Wikipedia data.

---

## Performance Notes

**Typical Speeds** (on modern laptop):
- Books: ~120-150 sentences/sec
- Wikipedia: ~400-500 sentences/sec

**Bottlenecks**:
- AST parsing (CPU-bound)
- Disk I/O for checkpoint writes

**Optimization**:
- Uses latest parser with Bug #1 fix
- Atomic checkpoint saves
- Batched progress updates

---

## Script Details

**Location**: `scripts/build_corpus_with_wikipedia.sh`

**Dependencies**:
- Python 3.8+
- Klareco parser
- Input files:
  - `data/extracted/books_sentences.jsonl`
  - `data/extracted/wikipedia_sentences.jsonl`

**Safety Features**:
- Backs up existing corpus
- Atomic checkpoint writes
- Validates output
- Color-coded messages
- Detailed error reporting

---

## Questions?

Check the script's help:
```bash
./scripts/build_corpus_with_wikipedia.sh --help
```

Or review the full log after completion.

---

**Status**: Ready to run
**Priority**: P0-CRITICAL
**Blocks**: ALL M1/M2 work

**RUN THIS FIRST!**
