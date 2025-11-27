# Corpus Management System

Complete system for managing Esperanto texts with automatic validation, cleaning, and incremental indexing. Long-running steps (cleaning, building JSONL, indexing) must be resumable via checkpoints and emit line-buffered logs to stdout and rotating files so runs can be monitored or restarted without losing progress.

## Overview

The Corpus Management System provides:
- **Database tracking** of all indexed texts (SQLite)
- **Validation pipeline** ensuring quality Esperanto content
- **Source attribution** tracking which book/line each sentence comes from
- **CLI tools** for adding/removing texts
- **Configurable filtering** for corpus quality control

## Quick Start

### Adding a New Text

```bash
# 1. Validate the text
python -m klareco corpus validate data/raw/my_book.txt

# 2. Add to corpus database
python -m klareco corpus add data/raw/my_book.txt \
  --title "My Book" \
  --type literature

# 3. Build unified corpus with sources
python scripts/build_corpus_with_sources.py \
  --min-length 20

# 4. Index the corpus
python scripts/index_corpus.py \
  --corpus data/corpus_with_sources.jsonl \
  --output data/corpus_index \
  --batch-size 32 \
  --no-resume

# Done! RAG can now search with source attribution
```

### Removing a Text

```bash
# 1. List all texts
python -m klareco corpus list

# 2. Remove by ID or name
python -m klareco corpus remove --id 5
python -m klareco corpus remove --name my_book.txt --force

# 3. Rebuild corpus and re-index
python scripts/build_corpus_with_sources.py --min-length 20
python scripts/index_corpus.py --corpus data/corpus_with_sources.jsonl \
  --output data/corpus_index --no-resume --batch-size 32
```

## Architecture

### Database Schema

**texts table** - Tracks all Esperanto texts:
```sql
CREATE TABLE texts (
    id INTEGER PRIMARY KEY,
    filename TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    source_type TEXT,  -- 'literature', 'wikipedia', 'dictionary'
    language_code TEXT DEFAULT 'eo',
    added_at TIMESTAMP,
    updated_at TIMESTAMP,
    indexed_at TIMESTAMP,
    is_indexed BOOLEAN DEFAULT 0,
    sentence_count INTEGER,
    file_size INTEGER,
    validation_status TEXT,  -- 'valid', 'invalid', 'pending'
    validation_score REAL,   -- 0.0-1.0 parse success rate
    metadata TEXT  -- JSON blob
);
```

**indexed_sentences table** - Maps sentences to source texts:
```sql
CREATE TABLE indexed_sentences (
    id INTEGER PRIMARY KEY,
    text_id INTEGER NOT NULL,
    sentence TEXT NOT NULL,
    line_num INTEGER,
    embedding_idx INTEGER,  -- Index in FAISS
    indexed_at TIMESTAMP,
    FOREIGN KEY (text_id) REFERENCES texts(id) ON DELETE CASCADE
);
```

### Components

1. **CorpusDatabase** (`klareco/corpus_manager.py`)
   - SQLite database wrapper
   - Text and sentence tracking
   - CRUD operations

2. **TextValidator** (`klareco/corpus_manager.py`)
   - Language detection (FastText)
   - Parse rate validation (70% threshold for valid)
   - File size checks

3. **CorpusManager** (`klareco/corpus_manager.py`)
   - High-level API for corpus operations
   - Auto-validation on text addition
   - Statistics and reporting

4. **CLI Commands** (`klareco/cli/corpus.py`)
   - `add` - Add text to corpus
   - `remove` - Remove text from corpus
   - `list` - List all texts
   - `validate` - Validate Esperanto file
   - `stats` - Show corpus statistics

5. **Corpus Builder** (`scripts/build_corpus_with_sources.py`)
   - Combines texts into unified JSONL corpus
   - Configurable filtering (`--min-length`)
   - Source attribution metadata

6. **Corpus Indexer** (`scripts/index_corpus.py`)
   - Tree-LSTM embeddings (512-dim)
   - FAISS similarity search
   - Preserves source metadata

## Configuration Options

### Corpus Builder (`build_corpus_with_sources.py`)

```bash
python scripts/build_corpus_with_sources.py [OPTIONS]

Options:
  --cleaned-dir PATH       Directory with cleaned texts (default: data/cleaned)
  --output PATH           Output JSONL file (default: data/corpus_with_sources.jsonl)
  --min-length INT        Minimum sentence length in characters (default: 20)
  --no-skip-metadata      Include all lines (don't skip metadata)
  --include-wikipedia     Include Wikipedia (warning: 18M+ lines)
  --wikipedia-limit INT   Limit Wikipedia to N lines
```

**min-length parameter**: Filters out short lines that are likely headers, metadata, or fragments. Default is 20 characters. Use higher values (30-40) for cleaner corpus, lower values (10-15) to retain more content.

### Corpus Indexer (`index_corpus.py`)

```bash
python scripts/index_corpus.py [OPTIONS]

Options:
  --corpus PATH           Corpus file (plain text or JSONL with metadata)
  --output PATH          Output directory (default: data/corpus_index)
  --model PATH           Tree-LSTM model checkpoint
  --batch-size INT       Batch size for encoding (default: 32)
  --resume              Resume from checkpoint (default: False)
  --no-resume           Start fresh, overwrite existing index
```

## Validation Thresholds

**Parse Rate Thresholds:**
- ≥70% = Valid (green flag)
- 40-70% = Acceptable (yellow flag, may need cleaning)
- <40% = Invalid (red flag, likely not clean Esperanto)

**File Size Checks:**
- Minimum: 1KB (files smaller than this are rejected)
- Minimum lines: 10 non-empty lines

**Language Detection:**
- Must be detected as Esperanto ('eo')
- Uses FastText language identification
- Tested on first 100 lines

## Source Attribution Format

Every indexed sentence includes source metadata:

```json
{
  "text": "Gandalf estis saĝa majstro.",
  "source": "la_mastro_de_l_ringoj",
  "source_name": "La Mastro de l' Ringoj (Lord of the Rings)",
  "line": 5678
}
```

RAG retrieval returns this metadata so users know which book an answer came from.

## CLI Examples

### List All Texts

```bash
$ python -m klareco corpus list

ID   Title                                      Type         Idx  Valid    Score   Sentences
----------------------------------------------------------------------------------------------------
3    La Mastro de l' Ringoj                     literature   ✅   valid    78.3%   41827
2    La Hobito                                  literature   ✅   valid    82.1%   8132
1    Kadavrejo Strato                           literature   ✅   valid    76.5%   1235

Total: 3 texts, 3 indexed, 51194 sentences
```

### Show Statistics

```bash
$ python -m klareco corpus stats

📊 Corpus Statistics

  Total texts: 7
  Indexed texts: 7
  Total sentences: 49,066
  Total size: 8.4 MB

Indexed texts:
  - La Mastro de l' Ringoj (36797 sentences)
  - La Hobito (7131 sentences)
  - Kadavrejo Strato (1235 sentences)
  - La Korvo (704 sentences)
  - Puto kaj Pendolo (546 sentences)
  - Ses Noveloj (1949 sentences)
  - Usxero Domo (704 sentences)
```

### Validate Before Adding

```bash
$ python -m klareco corpus validate data/raw/new_book.txt

🔍 Validating: new_book.txt

✅ Valid Esperanto (78.3% parse rate)
   Validation score: 78.3%
```

## Workflow Example

Complete workflow for adding "Alice in Wonderland" in Esperanto:

```bash
# Download and save to data/raw/
wget https://example.com/alico_en_mirlando.txt -O data/raw/alico.txt

# Validate it
python -m klareco corpus validate data/raw/alico.txt
# ✅ Valid Esperanto (81.2% parse rate)

# Add to corpus database
python -m klareco corpus add data/raw/alico.txt \
  --title "Alico en Mirlando" \
  --type literature
# ✅ Added 'Alico en Mirlando' (ID: 8). Valid Esperanto (81.2% parse rate)

# Build unified corpus (filters out lines < 20 chars)
python scripts/build_corpus_with_sources.py --min-length 20
# 📖 Processing: Alico en Mirlando
#    ✅ Added 12,543 lines
# ✅ Done! Total sentences: 61,609

# Index the corpus (takes ~3 minutes for 60K sentences)
python scripts/index_corpus.py \
  --corpus data/corpus_with_sources.jsonl \
  --output data/corpus_index \
  --no-resume \
  --batch-size 32
# 2025-11-12 21:00:00 - INFO - Loaded 61609 sentences
# Indexing corpus (success rate: 100.0%): 100%|██████████| 61609/61609 [03:02<00:00, 337.42it/s]
# ✅ Indexed 61609 sentences successfully

# Test RAG query
python scripts/demo_klareco.py --query "Kiu estas Alico?"
# RAG result from: Alico en Mirlando, line 42
# "Alico estas juna knabino kiu falas en kuniklan truon..."
```

## Troubleshooting

### "Validation failed: Poor parse rate"

Your text file may contain:
- Non-Esperanto content
- Too much metadata (headers, footers, production notes)
- Malformed text (encoding issues, OCR errors)

**Solutions:**
1. Clean the text manually (remove metadata)
2. Check encoding (should be UTF-8)
3. Try `--min-length 30` for stricter filtering
4. Use `--no-skip-metadata` to include all content

### "Text is indexed. Rebuild index after removal."

You tried to remove a text that's already indexed. The system prevents this to avoid index corruption.

**Solutions:**
1. Note the text ID
2. Remove it: `python -m klareco corpus remove --id N --force`
3. Rebuild corpus: `python scripts/build_corpus_with_sources.py --min-length 20`
4. Re-index: `python scripts/index_corpus.py --corpus data/corpus_with_sources.jsonl --output data/corpus_index --no-resume --batch-size 32`

### RAG Returns Short Fragments

Your corpus may contain header lines or metadata.

**Solutions:**
1. Increase `--min-length`: Try 30 or 40 characters
2. Rebuild corpus: `python scripts/build_corpus_with_sources.py --min-length 30`
3. Re-index: `python scripts/index_corpus.py ...`

### Indexing is Slow

**Expected speeds:**
- ~300-400 sentences/second on CPU
- ~60K sentences = 2-3 minutes
- ~500K sentences = 20-30 minutes

**Optimizations:**
- Use `--batch-size 64` for faster encoding
- GPU support (coming soon)
- Reduce corpus size with `--min-length 40`

## Future Enhancements

🔲 **Auto-cleaning pipeline** - Automatically clean texts before indexing
🔲 **Incremental indexing** - Add single text without rebuilding everything
🔲 **Rebuild command** - Regenerate index from database
🔲 **MD5 hash tracking** - Detect when files change
🔲 **Watch folder** - Auto-add new files dropped in folder
🔲 **GPU acceleration** - Faster indexing with CUDA
🔲 **Compressed index** - Reduce storage requirements

## Related Documentation

- [CLAUDE.md](../CLAUDE.md) - Project overview and architecture
- [DESIGN.md](../DESIGN.md) - 9-phase implementation roadmap
- [Phase 3 Guide](../PHASE3_CORPUS_INDEXING_GUIDE.md) - RAG system details
- [Corpus Quality Audit](../CORPUS_QUALITY_AUDIT.md) - Quality analysis
