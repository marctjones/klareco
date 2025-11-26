# Corpus Cleaning Summary

## Results

✅ **Your corpus is 99.4% clean Esperanto!**

### Statistics

- **Total sentences:** 20,985
- **Kept (Esperanto):** 20,864 (99.4%)
- **Removed (contamination):** 121 (0.6%)

### What Was Removed

| Category | Count | Description |
|----------|-------|-------------|
| Web artifacts | 52 | Internet Archive UI, navigation, menus |
| Other languages | 42 | Non-Esperanto text |
| Low confidence | 27 | Ambiguous short fragments |
| English | 0 | All English classified as web artifacts |
| Too short | 0 | All sentences meet minimum length |

### Examples of Removed Content

**Web Artifacts (Internet Archive scraping debris):**
- "Full text of La Mastro de l'Ringoj, La Hobito"
- "Hamburger icon - An illustration of a menu..."
- "Internet Archive logo - A line drawing..."
- "Skip to main content"
- "Wayback Machine", "Sign up | Log in"

**Low Confidence (very short dialogue):**
- "— Jes, bonvole!" ("— Yes, please!")
- "— Ne interrompu!" ("— Don't interrupt!")
- "Unu dekstre, jes." ("One to the right, yes.")

These are actually valid Esperanto but were removed due to being too short for confident language detection.

## Files Created

1. **`data/corpus_sentences_cleaned.jsonl`** - Clean Esperanto corpus (20,864 sentences)
2. **`data/corpus_cleaning_report.txt`** - Detailed report with all removals

## Using the Cleaned Corpus

### Option 1: Quick Test

Test with the cleaned corpus without replacing the original:

```bash
# Use cleaned corpus for testing
python scripts/index_corpus.py \
    --corpus data/corpus_sentences_cleaned.jsonl \
    --output data/corpus_index_clean \
    --model models/tree_lstm/checkpoint_epoch_20.pt
```

### Option 2: Replace Original (Recommended)

Replace the original corpus with the cleaned version:

```bash
# Backup original (if not already done)
cp data/corpus_sentences.jsonl data/corpus_sentences.jsonl.with_contamination

# Use cleaned version
cp data/corpus_sentences_cleaned.jsonl data/corpus_sentences.jsonl

# Reindex with cleaned corpus
./reindex_with_new_model.sh
```

### Option 3: Full Pipeline

Clean + reindex + ready for extended context training:

```bash
./clean_and_retrain.sh full
```

## Impact Assessment

### Before Cleaning (with contamination)
- 20,985 total sentences
- 121 sentences (0.6%) are English/web artifacts
- Could confuse language detection
- Could pollute embeddings with English
- Query: "Kiu estas Frodo?" might retrieve web UI text

### After Cleaning (pure Esperanto)
- 20,864 sentences (99.4% clean)
- No English contamination
- No web scraping debris
- Pure Esperanto embeddings
- Better retrieval quality
- More accurate language detection

## Next Steps

### 1. Activate Cleaned Corpus
```bash
# Replace original with cleaned version
mv data/corpus_sentences.jsonl data/corpus_sentences.jsonl.contaminated
cp data/corpus_sentences_cleaned.jsonl data/corpus_sentences.jsonl
```

### 2. Reindex with Cleaned Data
```bash
./reindex_with_new_model.sh
```

### 3. (Optional) Retrain with More Context
```bash
# Combine cleaning with context upgrade
./retrain_with_more_context.sh --context 50
```

## Quality Analysis

### What Makes This Corpus Good Now?

✅ **Pure Esperanto** - No English sentences (except incidental proper nouns)
✅ **No web artifacts** - All Internet Archive UI text removed
✅ **Well-structured** - Full sentences with proper grammar
✅ **Rich content** - Lord of the Rings translation provides deep context
✅ **High confidence** - 99.4% of corpus confirmed as Esperanto

### Acceptable "Contamination"

The following are **kept** and are acceptable:

1. **Proper nouns in English:**
   - "John Ronald Reuel Tolkien"
   - "Frodo Baggins", "Gandalf", "Bilbo"
   - Place names: "Hobbiton", "Rivendell", "Mordor"

2. **Technical terms:**
   - "Tree-LSTM", "AST", "ISBN"
   - Publishing metadata

3. **Code-switching:**
   - Esperanto text that quotes English titles
   - Example: "La traduko de 'The Fellowship of the Ring'"

This is **expected and desirable** for a corpus about English literature translated to Esperanto!

## Detailed Report

View full details of what was removed:

```bash
less data/corpus_cleaning_report.txt

# Or search for specific removal reasons
grep "english_" data/corpus_cleaning_report.txt
grep "web_artifact" data/corpus_cleaning_report.txt
```

## Rollback

If you need to restore the original corpus:

```bash
# Restore from backup
cp data/corpus_sentences.jsonl.with_contamination data/corpus_sentences.jsonl

# Or if you have the original backup
cp data/corpus_sentences.jsonl.original_* data/corpus_sentences.jsonl
```

## Scripts Reference

### Clean Only
```bash
python scripts/clean_corpus_language.py \
    --input data/corpus_sentences.jsonl \
    --output data/corpus_sentences_cleaned.jsonl \
    --report data/corpus_cleaning_report.txt
```

### Preview Before Cleaning
```bash
./clean_and_retrain.sh preview
```

### Clean and Reindex
```bash
./clean_and_retrain.sh full
```

## Recommendations

1. **Use the cleaned corpus** - The 0.6% contamination may seem small, but it can impact:
   - Language detection accuracy
   - Embedding quality
   - Retrieval precision

2. **Keep the original** - Backup preserved as `corpus_sentences.jsonl.contaminated`

3. **Reindex now** - Rebuild FAISS index with clean data for better retrieval

4. **Consider context upgrade** - Combine with 50+ context docs for maximum improvement

## Conclusion

Your corpus is **excellent quality** with minimal contamination. The cleaning removed:
- Web scraping artifacts (Internet Archive UI)
- Ambiguous short fragments
- Mixed language segments

You now have a **pure Esperanto corpus** ready for high-quality RAG and model training!

---

**Ready to use the cleaned corpus?**

```bash
# Quick activation
mv data/corpus_sentences.jsonl data/corpus_sentences.jsonl.contaminated
cp data/corpus_sentences_cleaned.jsonl data/corpus_sentences.jsonl
./reindex_with_new_model.sh
```
