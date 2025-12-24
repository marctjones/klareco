# Enhanced Corpus Building Guide

This guide explains how to build the enhanced Klareco corpus with full citation metadata.

## What's Enhanced?

The enhanced corpus includes rich metadata for proper citations:

### Wikipedia Articles
- **Article title**: "AIM", "Esperanto", "Komputi lo"
- **Article ID**: Unique Wikipedia article ID
- **Section**: Which section within the article
- **Timestamp**: When article was last edited

**Citation example**: `"Wikipedia: 'AIM', section 'Vidu ankaŭ'"`

### Books
- **Chapter name**: "ENKONDUKO", "Longe Atendita Festo"
- **Chapter number**: I, II, 1, 2 (if numbered)
- **Sentence position**: Which sentence within chapter
- **Paragraph**: Which paragraph in chapter

**Citation example**: `"La Mastro de l' Ringoj, Chapter 'Longe Atendita Festo', sentence 15"`

## Prerequisites

1. **Wikipedia dump**: `data/corpora/eo_wikipedia.xml.bz2`
2. **Cleaned books**: `data/cleaned/cleaned_*.txt`
3. **Python environment**: `.venv` with klareco installed
4. **Disk space**: ~50GB free (Wikipedia is large!)
5. **Time**: 4-6 hours total processing time

## Quick Start

### Option 1: Run All Extraction Steps Automatically (Recommended)

```bash
# Runs Wikipedia + Books extraction with automatic archiving
./scripts/run_full_extraction.sh
```

This script:
- Archives any existing extracted data to `data/archive/extraction_YYYYMMDD_HHMMSS/`
- Runs Wikipedia extraction (2-3 hours)
- Runs book extraction (5-10 minutes)
- Shows summary and next steps

**To skip archiving** (if starting fresh):
```bash
./scripts/run_full_extraction.sh --no-archive
```

### Option 2: Run Steps Manually

```bash
# Step 1: Extract Wikipedia (2-3 hours)
./scripts/run_wikipedia_extraction.sh

# Step 2: Extract Books (5-10 minutes)
./scripts/run_books_extraction.sh

# Step 3: Build Enhanced Corpus (1-2 hours)
./scripts/run_corpus_builder.sh
```

## Detailed Instructions

### Step 1: Wikipedia Extraction

Extracts Wikipedia articles with full metadata.

```bash
./scripts/run_wikipedia_extraction.sh
```

**What it does:**
- Parses MediaWiki XML dump
- Extracts article title, ID, sections
- Cleans MediaWiki markup (`{{templates}}`, `[[links]]`)
- Splits into sentences
- Saves to `data/extracted/wikipedia_sentences.jsonl`

**Progress indicators:**
- Shows current article being processed
- Updates every 100 articles
- Logs to `logs/wikipedia_extraction.log`

**Checkpointing:**
- Saves checkpoint every 1000 articles
- If interrupted, run again to resume
- Checkpoint file: `data/extracted/wikipedia_checkpoint.json`

**Output format:**
```json
{
  "text": "AIM estas tujmesaĝilo por babili kun iu alia per la Interreto",
  "source": "wikipedia",
  "source_name": "Vikipedio Esperanto",
  "article_title": "AIM",
  "article_id": 1,
  "section": "Priskribo",
  "section_level": 2,
  "timestamp": "2025-01-16T22:16:22Z"
}
```

**Expected output:**
- ~3.8 million sentences
- ~2-4 GB JSONL file
- Processing rate: ~50-100 articles/sec

### Step 2: Books Extraction

Extracts book sentences with chapter detection.

```bash
./scripts/run_books_extraction.sh
```

**What it does:**
- Detects chapter markers (ALL CAPS, numbered chapters)
- Tracks sentence position within chapters
- Extracts from all cleaned books
- Saves to `data/extracted/books_sentences.jsonl`

**Chapter detection patterns:**
- ALL CAPS lines: `ENKONDUKO`, `PROLOGO`
- Numbered: `CHAPTER 5`, `Ĉapitro 12`
- Roman numerals: `I.`, `XII.`

**Progress indicators:**
- Shows current book being processed
- Reports chapters detected
- Updates every 1000 sentences
- Logs to `logs/books_extraction.log`

**Output format:**
```json
{
  "text": "John Ronald Reuel Tolkien komencis sian eposon La Mastro de l'Ringoj",
  "source": "la_mastro_de_l_ringoj",
  "source_name": "La Mastro de l' Ringoj",
  "chapter": "ENKONDUKO",
  "chapter_number": null,
  "sentence_in_chapter": 15,
  "paragraph": 3,
  "line_number": 42
}
```

**Expected output:**
- ~1.1 million sentences from LOTR
- ~200K sentences from other books
- ~100-300 MB JSONL file
- Processing rate: ~500-1000 sentences/sec

### Step 3: Build Enhanced Corpus

Parses all sentences to ASTs and combines sources.

```bash
./scripts/run_corpus_builder.sh [--min-parse-rate 0.5]
```

**What it does:**
1. Parses Wikipedia sentences to ASTs
2. Parses book sentences to ASTs
3. Filters by parse quality (default: ≥50% parse rate)
4. Combines into single corpus
5. Adds parse statistics and word counts

**Arguments:**
- `--min-parse-rate`: Minimum parse rate to include (0.0-1.0)
  - `0.5` (default): Medium quality - includes most sentences
  - `0.7`: High quality - stricter filtering
  - `0.9`: Very high quality - only near-perfect parses

**Progress indicators:**
- Shows parsing progress every 100 sentences
- Reports include rate (how many pass filter)
- Updates processing speed
- Logs to `logs/corpus_building.log`

**Checkpointing:**
- Saves checkpoint every 1000 sentences
- Separate checkpoints for Wikipedia and books
- Resume from where it left off if interrupted

**Output format:**
```json
{
  "text": "AIM estas tujmesaĝilo por babili kun iu alia per la Interreto",
  "source": "wikipedia",
  "source_name": "Vikipedio Esperanto",
  "article_title": "AIM",
  "article_id": 1,
  "section": null,
  "section_level": 0,
  "timestamp": "2025-01-16T22:16:22Z",
  "ast": {
    "tipo": "frazo",
    "subjekto": {...},
    "verbo": {...},
    "objekto": {...},
    "parse_statistics": {
      "total_words": 12,
      "esperanto_words": 11,
      "success_rate": 0.9166
    }
  },
  "parse_rate": 0.9166,
  "word_count": 12
}
```

**Expected output:**
- Final corpus: `data/enhanced_corpus/corpus_with_metadata.jsonl`
- ~3-4 million sentences (after filtering)
- ~10-20 GB file (with full ASTs)
- Average parse rate: ~91%

## Monitoring Progress

### View Logs in Real-Time

```bash
# Wikipedia extraction
tail -f logs/wikipedia_extraction.log

# Books extraction
tail -f logs/books_extraction.log

# Corpus building
tail -f logs/corpus_building.log
```

### Check Progress

```bash
# Count extracted sentences
wc -l data/extracted/wikipedia_sentences.jsonl
wc -l data/extracted/books_sentences.jsonl

# Check checkpoint status
cat data/extracted/wikipedia_checkpoint.json | python3 -m json.tool

# Monitor corpus building
wc -l data/enhanced_corpus/wikipedia_parsed.jsonl
wc -l data/enhanced_corpus/books_parsed.jsonl
```

## Troubleshooting

### Script Fails Immediately

**Problem**: Permission denied or command not found

**Solution**:
```bash
chmod +x scripts/run_wikipedia_extraction.sh
chmod +x scripts/run_books_extraction.sh
chmod +x scripts/run_corpus_builder.sh
```

### Wikipedia Dump Not Found

**Problem**: `✗ Wikipedia dump not found`

**Solution**:
```bash
# Check if file exists
ls -lh data/corpora/eo_wikipedia.xml.bz2

# If missing, you need to download it
# Wikipedia dumps: https://dumps.wikimedia.org/eowiki/
```

### Out of Memory

**Problem**: Process killed or "MemoryError"

**Solution**: The scripts are designed to be memory-efficient (streaming). If you still have issues:
- Close other applications
- Reduce checkpoint interval (process fewer items before saving)
- Split Wikipedia extraction into batches

### Parsing Too Slow

**Problem**: Processing rate < 10 sentences/sec

**Solution**:
- This is normal for parsing (parser is deterministic, not optimized for speed)
- Let it run overnight
- Or increase `--min-parse-rate` to filter more aggressively
- Consider running on a faster machine

### Want to Resume

**Problem**: Interrupted in the middle

**Solution**: Just run the same command again!
- Scripts automatically detect checkpoints
- Resume from where they left off
- No data is lost

### Want to Start Fresh

**Problem**: Want to rebuild from scratch

**Solution**:
```bash
# Delete checkpoints and outputs
rm data/extracted/wikipedia_checkpoint.json
rm data/extracted/wikipedia_sentences.jsonl
rm data/extracted/books_sentences.jsonl
rm -rf data/enhanced_corpus/

# Run scripts again
./scripts/run_wikipedia_extraction.sh
./scripts/run_books_extraction.sh
./scripts/run_corpus_builder.sh
```

## Output Files

After completion, you'll have:

```
data/
├── extracted/
│   ├── wikipedia_sentences.jsonl       # Raw Wikipedia sentences
│   ├── wikipedia_checkpoint.json       # Wikipedia checkpoint
│   └── books_sentences.jsonl          # Raw book sentences
├── enhanced_corpus/
│   ├── wikipedia_parsed.jsonl         # Wikipedia with ASTs
│   ├── books_parsed.jsonl            # Books with ASTs
│   ├── corpus_with_metadata.jsonl    # ✓ FINAL CORPUS
│   ├── wiki_parse_checkpoint.json    # Checkpoint
│   └── books_parse_checkpoint.json   # Checkpoint
└── logs/
    ├── wikipedia_extraction.log       # Extraction logs
    ├── books_extraction.log          # Extraction logs
    └── corpus_building.log           # Building logs
```

**Main output**: `data/enhanced_corpus/corpus_with_metadata.jsonl`

## Next Steps

After building the enhanced corpus:

1. **Index for retrieval**:
   ```bash
   python scripts/index_corpus.py \
     --corpus data/enhanced_corpus/corpus_with_metadata.jsonl \
     --output data/enhanced_index
   ```

2. **Validate quality**:
   ```bash
   # Check statistics
   python scripts/analyze_corpus.py data/enhanced_corpus/corpus_with_metadata.jsonl

   # Sample entries
   head -10 data/enhanced_corpus/corpus_with_metadata.jsonl | jq .
   ```

3. **Create benchmark**:
   - Use the enhanced corpus to create a 50-question Q&A benchmark
   - Citations will now include article titles and chapter names!

## Corpus Statistics

Expected statistics for the final corpus:

| Source | Sentences | Parse Rate | Example Citation |
|--------|-----------|------------|------------------|
| Wikipedia | ~3.8M | 91-93% | "Wikipedia: 'AIM', section 'Uzo'" |
| LOTR | ~1.1M | 88-91% | "La Mastro de l' Ringoj, 'PROLOGO', sent. 42" |
| Hobbit | ~200K | 89-92% | "La Hobito, chapter 3, sent. 8" |
| Poe | ~50K | 87-90% | "Kadavrejo Strato, 'ENKONDUKO'" |
| **Total** | **~5M** | **~91%** | - |

## Performance Tips

### Running in Background

To run extraction in background and check later:

```bash
# Start in background
nohup ./scripts/run_wikipedia_extraction.sh > wikipedia.out 2>&1 &

# Check progress
tail -f wikipedia.out

# Or check logs
tail -f logs/wikipedia_extraction.log

# Check if still running
ps aux | grep extract_wikipedia
```

### Parallel Processing

You can run Wikipedia and Books extraction in parallel (different terminals):

```bash
# Terminal 1
./scripts/run_wikipedia_extraction.sh

# Terminal 2 (simultaneously)
./scripts/run_books_extraction.sh

# Then combine (after both finish)
./scripts/run_corpus_builder.sh
```

### Reduce Disk Usage

If disk space is limited:

```bash
# Delete extracted files after building corpus
rm data/extracted/wikipedia_sentences.jsonl
rm data/extracted/books_sentences.jsonl

# Keep only final corpus
# This saves ~2-4 GB
```

## Technical Details

### Wikipedia XML Parsing

Uses `xml.etree.ElementTree` with `iterparse()` for memory-efficient streaming:
- Processes one `<page>` element at a time
- Clears processed elements from memory
- Never loads full XML into RAM

### MediaWiki Markup Cleaning

Removes:
- Templates: `{{Informkesto}}` → (removed)
- Internal links: `[[Esperanto|Eo]]` → "Eo"
- Bold/italic: `'''text'''` → "text"
- HTML comments: `<!-- note -->` → (removed)
- External links: `[http://... text]` → "text"

### Chapter Detection Algorithm

Patterns detected (in order):
1. ALL CAPS lines (≥4 letters): `ENKONDUKO`
2. Numbered chapters: `CHAPTER 5`, `Ĉapitro 3`
3. Roman numerals: `I.`, `XII.`

Lines are marked as chapter boundaries, and all subsequent sentences are attributed to that chapter.

### AST Parsing

Uses Klareco's deterministic parser:
- 16 Esperanto grammar rules
- No learned parameters
- Morpheme-level decomposition
- Outputs complete grammatical AST

### Quality Filtering

Parse rate calculation:
```
parse_rate = successfully_parsed_words / total_words
```

Sentences with `parse_rate >= min_parse_rate` are included in final corpus.

Lower threshold = more sentences, lower quality
Higher threshold = fewer sentences, higher quality

Recommended: 0.5 (includes ~90% of sentences)

## License and Attribution

The enhanced corpus includes:
- **Wikipedia**: CC-BY-SA 3.0 (must attribute Vikipedio)
- **LOTR/Hobbit**: Copyright estate of J.R.R. Tolkien (Esperanto translations)
- **Edgar Allan Poe**: Public domain

When using the corpus, maintain source attribution in metadata.
