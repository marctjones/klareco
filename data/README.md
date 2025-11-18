# Klareco Data Directory

This directory contains vocabulary and corpus data for the Klareco parser.

## Vocabulary Files

### Current Status

The vocabulary files are initialized with **empty sets** to allow the parser to run with its hardcoded vocabulary (~218 roots). This provides basic functionality while the full vocabulary extraction is optional.

### Files

- `extracted_vocabulary.py` - Roots extracted from Gutenberg English-Esperanto Dictionary
- `merged_vocabulary.py` - Combined vocabulary from all sources
- `__init__.py` - Python module initialization

### Parser Vocabulary Strategy

The parser uses a **fallback strategy**:

1. **Primary**: Import from `data.merged_vocabulary.MERGED_ROOTS` (currently empty)
2. **Fallback**: Import from `data.extracted_vocabulary.DICTIONARY_ROOTS` (currently empty)
3. **Hardcoded**: Built-in `KNOWN_ROOTS` in `klareco/parser.py` (~218 roots)

**Current vocabulary size**: 218 hardcoded roots (sufficient for basic parsing)

## Populating the Vocabulary

To expand the vocabulary beyond the hardcoded roots, you need dictionary source files:

### Option 1: Extract from Gutenberg Dictionary

If you have the Gutenberg English-Esperanto Dictionary:

```bash
# Place dictionary at: data/grammar/gutenberg_dict.txt
# Then run:
python scripts/extract_dictionary_roots.py
```

This will populate `data/extracted_vocabulary.py` with ~8,000+ roots.

### Option 2: Use the Full Vocabulary Pipeline

Run the complete vocabulary extraction pipeline:

```bash
# Download and process Esperanto corpus
python scripts/download_gutenberg_esperanto.py
python scripts/extract_dictionary_roots.py
python scripts/build_morpheme_vocab.py
```

## Parser Performance

The parser achieves **95.7% accuracy** on test corpus with the hardcoded vocabulary. Expanding to the full 8,397-root dictionary would increase coverage significantly.

## Compositional Morphology

Even with 218 roots, the parser can handle **thousands of words** through compositional morphology:

- 7 prefixes (mal-, re-, ge-, etc.)
- 218+ roots (expandable to 8,397)
- 24 suffixes (-ul, -ej, -in, etc.)
- Complete grammatical endings system

**Example**: `malgrandajn` = `mal-` (opposite) + `grand` (big) + `-a` (adjective) + `-j` (plural) + `-n` (accusative)

## Note on .gitignore

The `data/` directory is in `.gitignore` because it contains large corpus files and generated vocabularies. Each user should generate their own vocabulary based on their corpus sources.
