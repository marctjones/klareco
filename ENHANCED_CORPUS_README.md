# Enhanced Corpus with Full Citation Metadata

## Quick Start (TL;DR)

Run these three commands in separate terminals to rebuild the corpus with proper citations:

```bash
# Terminal 1: Extract Wikipedia articles with titles (~2-3 hours)
./scripts/run_wikipedia_extraction.sh

# Terminal 2: Extract books with chapter names (~5-10 minutes)
./scripts/run_books_extraction.sh

# Terminal 3: Combine and parse to ASTs (~1-2 hours)
# Wait for terminals 1 & 2 to finish first!
./scripts/run_corpus_builder.sh
```

**Result**: `data/enhanced_corpus/corpus_with_metadata.jsonl` (~5M sentences with full citations)

---

## What's New?

### Before (Old Corpus)
```json
{
  "text": "AIM estas tujmesaĝilo...",
  "source": "wikipedia",
  "source_name": "Vikipedio Esperanto (Wikipedia)",
  "paragraph": 1
}
```
**Citation**: "Wikipedia" ❌ (which article?)

### After (Enhanced Corpus)
```json
{
  "text": "AIM estas tujmesaĝilo...",
  "source": "wikipedia",
  "source_name": "Vikipedio Esperanto",
  "article_title": "AIM",
  "article_id": 1,
  "section": "Priskribo",
  "timestamp": "2025-01-16T22:16:22Z",
  "ast": {...},
  "parse_rate": 0.92
}
```
**Citation**: "Wikipedia: 'AIM', section 'Priskribo'" ✅

### For Books
```json
{
  "text": "John Ronald Reuel Tolkien komencis...",
  "source": "la_mastro_de_l_ringoj",
  "source_name": "La Mastro de l' Ringoj",
  "chapter": "ENKONDUKO",
  "sentence_in_chapter": 15,
  "paragraph": 3,
  "ast": {...},
  "parse_rate": 0.89
}
```
**Citation**: "La Mastro de l' Ringoj, 'ENKONDUKO', sentence 15" ✅

---

## Scripts Created

### 1. Wikipedia Extractor
**File**: `scripts/extract_wikipedia_with_metadata.py`
**Runner**: `scripts/run_wikipedia_extraction.sh`

**Features**:
- Parses MediaWiki XML dump (5.3GB compressed)
- Extracts article titles, IDs, sections
- Cleans MediaWiki markup
- Streams processing (low memory)
- Checkpoints every 1000 articles
- Progress updates every 100 articles
- Resumable if interrupted

**Output**: `data/extracted/wikipedia_sentences.jsonl` (~3.8M sentences)

### 2. Books Extractor
**File**: `scripts/extract_books_with_metadata.py`
**Runner**: `scripts/run_books_extraction.sh`

**Features**:
- Detects chapter markers (ALL CAPS, numbered chapters)
- Tracks sentence position within chapters
- Processes all cleaned books
- Progress updates every 1000 sentences
- Fast (~5-10 minutes total)

**Chapter detection**:
- ALL CAPS: `ENKONDUKO`, `PROLOGO`
- Numbered: `CHAPTER 5`, `Ĉapitro 12`
- Roman numerals: `I.`, `XII.`

**Output**: `data/extracted/books_sentences.jsonl` (~1.3M sentences)

### 3. Enhanced Corpus Builder
**File**: `scripts/build_enhanced_corpus.py`
**Runner**: `scripts/run_corpus_builder.sh`

**Features**:
- Parses all sentences to ASTs using Klareco parser
- Filters by parse quality (default: ≥50%)
- Combines Wikipedia + Books
- Checkpoints every 1000 sentences
- Resumable if interrupted
- Progress updates every 100 sentences

**Output**: `data/enhanced_corpus/corpus_with_metadata.jsonl` (~5M sentences)

---

## Usage

### Basic Usage (Default Settings)

```bash
# Step 1: Extract Wikipedia
./scripts/run_wikipedia_extraction.sh

# Step 2: Extract Books
./scripts/run_books_extraction.sh

# Step 3: Build Corpus
./scripts/run_corpus_builder.sh
```

### Advanced Usage

```bash
# Build with stricter quality filter (≥70% parse rate)
./scripts/run_corpus_builder.sh --min-parse-rate 0.7

# Run Wikipedia extraction in background
nohup ./scripts/run_wikipedia_extraction.sh > wiki.out 2>&1 &
tail -f logs/wikipedia_extraction.log
```

### Monitor Progress

```bash
# Watch logs in real-time
tail -f logs/wikipedia_extraction.log
tail -f logs/books_extraction.log
tail -f logs/corpus_building.log

# Check progress
wc -l data/extracted/wikipedia_sentences.jsonl
wc -l data/enhanced_corpus/corpus_with_metadata.jsonl
```

### Resume After Interruption

Scripts automatically resume from checkpoints. Just run them again:

```bash
# Will resume from checkpoint if exists
./scripts/run_wikipedia_extraction.sh
```

### Start Fresh

```bash
# Delete all outputs and checkpoints
rm -rf data/extracted/
rm -rf data/enhanced_corpus/
rm logs/*_extraction.log logs/corpus_building.log

# Run from scratch
./scripts/run_wikipedia_extraction.sh
```

---

## Expected Timeline

| Step | Time | Output Size |
|------|------|-------------|
| Wikipedia extraction | 2-3 hours | ~2-4 GB |
| Books extraction | 5-10 min | ~100-300 MB |
| Corpus building (parsing) | 1-2 hours | ~10-20 GB |
| **Total** | **~4-6 hours** | **~15-25 GB** |

**Note**: Times assume decent CPU. Parsing is the slowest step (deterministic parser, not optimized for speed).

---

## Output Files

After completion:

```
data/
├── extracted/
│   ├── wikipedia_sentences.jsonl       # 3.8M raw sentences
│   ├── wikipedia_checkpoint.json       # Resume point
│   └── books_sentences.jsonl          # 1.3M raw sentences
├── enhanced_corpus/
│   ├── wikipedia_parsed.jsonl         # Wikipedia with ASTs
│   ├── books_parsed.jsonl            # Books with ASTs
│   └── corpus_with_metadata.jsonl    # ✓ FINAL (5M sentences)
└── logs/
    ├── wikipedia_extraction.log
    ├── books_extraction.log
    └── corpus_building.log
```

**Main output**: `data/enhanced_corpus/corpus_with_metadata.jsonl`

---

## Citation Examples

### Wikipedia Article Citation
```python
entry = {
    "text": "AIM estas tujmesaĝilo por babili...",
    "source": "wikipedia",
    "article_title": "AIM",
    "article_id": 1,
    "section": "Uzo"
}

citation = f"Wikipedia: '{entry['article_title']}', section '{entry['section']}'"
# Output: "Wikipedia: 'AIM', section 'Uzo'"
```

### Book Chapter Citation
```python
entry = {
    "text": "La hobitoj estas malgranda raso...",
    "source": "la_hobito",
    "source_name": "La Hobito",
    "chapter": "NEATENDITA FESTO",
    "sentence_in_chapter": 42
}

citation = f"{entry['source_name']}, '{entry['chapter']}', sentence {entry['sentence_in_chapter']}"
# Output: "La Hobito, 'NEATENDITA FESTO', sentence 42"
```

### Full Citation with Parse Quality
```python
entry = {
    "text": "...",
    "source_name": "La Mastro de l' Ringoj",
    "chapter": "PROLOGO",
    "parse_rate": 0.92
}

citation = f"{entry['source_name']}, '{entry['chapter']}' (parse quality: {entry['parse_rate']:.0%})"
# Output: "La Mastro de l' Ringoj, 'PROLOGO' (parse quality: 92%)"
```

---

## Features

### Automatic Environment Setup
- Activates `.venv` automatically
- Installs klareco package if needed
- Creates necessary directories
- Checks prerequisites

### Progress Indicators
- Real-time progress updates
- Processing speed (sentences/sec)
- Estimated completion
- Current item being processed

### Error Handling
- Logs all errors with context
- Continues processing on errors
- Periodic error count summaries
- Detailed error logs for debugging

### Checkpointing
- Saves progress every 1000 items
- Resumes automatically on restart
- No data loss on interruption
- Separate checkpoints per stage

### Logging
- Timestamped logs
- Both file and console output
- Progress milestones
- Error details with context

---

## Quality Control

### Parse Rate Filtering

The `--min-parse-rate` parameter controls quality:

| Rate | Quality | Included | Use Case |
|------|---------|----------|----------|
| 0.3 | Low | ~98% | Maximum data |
| 0.5 | Medium | ~90% | **Default - balanced** |
| 0.7 | High | ~75% | Strict quality |
| 0.9 | Very High | ~50% | Only near-perfect |

**Recommendation**: Use default (0.5) for balanced quality/quantity.

### Statistics

Expected final corpus statistics:

```
Corpus Statistics:
  Wikipedia: 3,800,000 sentences
  Books: 1,300,000 sentences
  Total words: 60,000,000
  Average parse rate: 91%
```

---

## Troubleshooting

### "Wikipedia dump not found"
**Solution**: Check if `data/corpora/eo_wikipedia.xml.bz2` exists

### "Out of memory"
**Solution**: Scripts use streaming - close other apps if needed

### "Parsing too slow"
**Solution**: Normal! Parser is deterministic (not optimized). Run overnight.

### "Want to resume"
**Solution**: Just run the script again - auto-resumes from checkpoint

### "Want fresh start"
**Solution**: Delete `data/extracted/` and `data/enhanced_corpus/` directories

---

## Next Steps

After building enhanced corpus:

1. **Index for retrieval**:
   ```bash
   python scripts/index_corpus.py \
     --corpus data/enhanced_corpus/corpus_with_metadata.jsonl \
     --output data/enhanced_index
   ```

2. **Create 50-question benchmark**:
   - Now you can include proper citations!
   - Example: "Question source: Wikipedia article 'Esperanto'"

3. **Test RAG with citations**:
   ```bash
   python scripts/demo_rag.py "Kio estas AIM?"
   # Answer: "AIM estas tujmesaĝilo..."
   # Source: Wikipedia: 'AIM', section 'Priskribo'
   ```

---

## Documentation

Full documentation: `docs/CORPUS_BUILDING.md`

Includes:
- Detailed step-by-step guide
- Troubleshooting section
- Performance tips
- Technical details
- Advanced usage

---

## Files Created

| File | Purpose |
|------|---------|
| `scripts/extract_wikipedia_with_metadata.py` | Wikipedia extractor |
| `scripts/extract_books_with_metadata.py` | Books extractor |
| `scripts/build_enhanced_corpus.py` | Corpus builder |
| `scripts/run_wikipedia_extraction.sh` | Wikipedia runner |
| `scripts/run_books_extraction.sh` | Books runner |
| `scripts/run_corpus_builder.sh` | Builder runner |
| `docs/CORPUS_BUILDING.md` | Full documentation |
| `ENHANCED_CORPUS_README.md` | This file |

---

## Why Rebuild?

### Problem with Old Corpus
- Wikipedia entries said "Wikipedia" but no article title
- Books said "paragraph 123" but no chapter name
- Impossible to give meaningful citations

### Solution with Enhanced Corpus
- Wikipedia: Article title + section
- Books: Chapter name + sentence position
- Complete AST metadata
- Parse quality scores

### Result
- Proper citations for RAG answers
- Traceability to source
- Quality filtering
- Ready for training

---

**Ready to start?**

```bash
./scripts/run_wikipedia_extraction.sh
```

**Questions?** Check `docs/CORPUS_BUILDING.md`
