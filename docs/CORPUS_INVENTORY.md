# Klareco Esperanto Corpus Inventory

**Last Updated**: 2025-12-23

## Overview

Klareco uses a curated collection of high-quality Esperanto texts to train and evaluate the AST-first AI system. This document catalogs all available corpus materials and their characteristics.

---

## Complete Corpus Summary

### Total Assets
- **Sentences**: 26,725 complete sentences (Corpus V2)
- **Source Files**: 9 cleaned texts + Wikipedia
- **Total Size**: ~843MB (including Wikipedia)
- **Quality**: 88-94% parse rate (after proper noun handling)

### Corpus Versions
- **V1** (deprecated): Hard-wrapped fragments, poor quality
- **V2** (current): Complete sentences with source tracking
- **V3** (index): FAISS index with AST metadata

---

## Literary Corpus (Fiction & Poetry)

### 1. La Mastro de l' Ringoj (The Lord of the Rings)
- **File**: `data/cleaned/cleaned_la_mastro_de_l_ringoj.txt`
- **Lines**: 73,595
- **Size**: 2.17 MB
- **Type**: Translated fiction (Tolkien)
- **Language**: High-quality Esperanto translation
- **Characters**: Frodo, Bilbo, Gandalf, Saŭrono, etc.
- **Locations**: Ŝajro, Mordoro, Gondoro, Miterreno
- **Value**: Rich narrative, complex sentences, proper noun testing

### 2. La Hobito (The Hobbit)
- **File**: `data/cleaned/cleaned_la_hobito.txt`
- **Lines**: 14,354
- **Size**: 416 KB
- **Type**: Translated fiction (Tolkien)
- **Language**: Same translation quality as LOTR
- **Characters**: Bilbo, Gandalf, Smaŭgo, Golumo
- **Value**: Simpler narrative, good for basic parsing tests

### 3. Edgar Allan Poe Translations
**3a. Puto kaj Pendolo (The Pit and the Pendulum)**
- **File**: `data/cleaned/cleaned_puto_kaj_pendolo.txt`
- **Size**: 31 KB
- **Type**: Translated horror short story

**3b. La Korvo (The Raven)**
- **File**: `data/cleaned/cleaned_la_korvo.txt`
- **Size**: 43 KB
- **Type**: Translated poetry

**3c. La Domo de Uŝero (The Fall of the House of Usher)**
- **File**: `data/cleaned/cleaned_usxero_domo.txt`
- **Size**: 43 KB
- **Type**: Translated gothic horror

**3d. Ses Noveloj (Six Tales)**
- **File**: `data/cleaned/cleaned_ses_noveloj.txt`
- **Size**: 113 KB
- **Type**: Collection of Poe short stories

**Combined Value**: Gothic vocabulary, complex mood and tense constructions, poetic structures

### 4. Kadavrejo Strato (Cemetery Street)
- **File**: `data/cleaned/cleaned_kadavrejo_strato.txt`
- **Size**: 72 KB
- **Type**: Original or translated fiction
- **Value**: Additional narrative diversity

---

## Reference & Dictionary Corpus

### 5. Esperanto-English Dictionary
- **File**: `data/cleaned/cleaned_espdic.txt`
- **Size**: 487 KB
- **Type**: Bilingual dictionary
- **Format**: Word definitions and examples
- **Value**:
  - Root vocabulary extraction
  - Example sentences showing proper usage
  - Etymology and morphology information
  - **Critical**: Can be used to build comprehensive root vocabulary

---

## Authoritative Esperanto Sources (Project Gutenberg)

Based on `data/GUTENBERG_CORPUS_SUMMARY.md`:

### Zamenhof (Founder Works) - MOST AUTHORITATIVE
- **08224** - Fundamenta Krestomatio: 4,337 sentences ⭐
  - *The* definitive Esperanto reference
  - Standard grammar examples
  - Written by the language creator
- **20006** - Dua Libro de l' Lingvo Internacia: 374 sentences
- **11307** - El la Biblio: 480 sentences

**Total Zamenhof**: 5,191 sentences

### Grammar & Reference Books
- **47855** - Esperanta sintakso (Paul Fruictier): 835 sentences
- **52556** - Esperanto-Germana frazlibro: 708 sentences
- **24525** - Karlo: Facila Legolibro: 700 sentences

**Total Grammar**: 2,243 sentences

### Historical Documents & Periodicals
- **38240** - The Esperantist, Complete: 11,664 sentences ⭐ (largest)
- **57184** - Dokumentoj de Esperanto: 2,341 sentences
- **26359** - Vivo de Zamenhof: 1,563 sentences

**Total Historical**: 15,568 sentences

### Original Works by Esperantists
- **42028** - En Rusujo per Esperanto: 1,031 sentences
- **23670** - Nuntempaj Rakontoj: 1,241 sentences
- **42774** - Mondo kaj koro (poetry): 279 sentences
- **48896** - Verdaj fajreroj (poetry): 174 sentences
- **76273** - Por kaj kontraŭ Esperanto: 261 sentences
- **25311** - El la vivo de esperantistoj: 295 sentences

**Total Original**: 3,281 sentences

**GRAND TOTAL (Gutenberg)**: 26,283 sentences

---

## Wikipedia Corpus

### 6. Esperanto Wikipedia
- **File**: `data/cleaned/cleaned_wikipedia.txt`
- **Size**: 843 MB
- **Type**: Encyclopedia articles
- **Language**: Modern Esperanto, diverse topics
- **Value**:
  - Enormous volume for vocabulary extraction
  - Covers modern topics (technology, science, culture)
  - Proper noun rich (people, places, organizations)
  - **Note**: Quality varies, may have foreign borrowings

---

## Processed Corpus Files

### Current Production Corpus (V2)
- **File**: `data/corpus_with_sources_v2.jsonl`
- **Size**: 22.7 GB (full extraction)
- **Format**: JSON Lines with AST metadata
- **Sentences**: 26,725 complete sentences
- **Fields**:
  ```json
  {
    "text": "Frodo estas hobito el la Ŝajro.",
    "source": "la_mastro_de_l_ringoj",
    "ast": { ... },
    "parse_statistics": {
      "success_rate": 0.92,
      "total_words": 7,
      "esperanto_words": 6
    }
  }
  ```

### Production Index (V3)
- **Directory**: `data/corpus_index_v3/`
- **Components**:
  - `faiss_index.bin` - FAISS similarity index
  - `metadata.jsonl` - Sentence metadata with sources
  - `embeddings.npy` - Precomputed embeddings
- **Size**: ~100 MB
- **Coverage**: 26,725 sentences indexed

### Smaller Test Corpus (V3 - Filtered)
- **File**: `data/corpus_with_sources_v3.jsonl`
- **Size**: 104 MB
- **Purpose**: Faster testing, higher quality subset

---

## Training Data (Semantic Similarity)

### Tatoeba EN-EO Parallel Corpus
- **Source**: Tatoeba Project (open translation database)
- **Pairs**: 271,000 EN-EO sentence pairs
- **Usage**: English as "similarity oracle" for training
- **Generated Data**:
  - `data/similarity_pairs_train.jsonl` - Training set (159K pairs)
  - `data/similarity_pairs_val.jsonl` - Validation set
  - `data/similarity_pairs_test.jsonl` - Test set
- **Model Performance**: val_corr=0.84 (Pearson correlation)

---

## Vocabulary Extracts

### Root Vocabulary
- **File**: `data/vocabularies/root_vocab.json`
- **Roots**: 953,000 extracted roots
- **Source**: All corpus texts
- **Format**: `{"root": frequency}`

### Prefix Vocabulary
- **File**: `data/vocabularies/prefix_vocab.json`
- **Prefixes**: 61 identified
- **Known Esperanto Prefixes**: 7 (mal, re, ge, eks, ek, pra, for)

### Suffix Vocabulary
- **File**: `data/vocabularies/suffix_vocab.json`
- **Suffixes**: 38 identified
- **Known Esperanto Suffixes**: 31 standard suffixes

---

## Corpus Quality Metrics

### Parse Success Rates (from Corpus V2)
| Source Type | Parse Rate | Notes |
|-------------|-----------|-------|
| **Fundamenta Krestomatio** | 94% | Highest quality (Zamenhof) |
| **Grammar books** | 93% | Standard constructions |
| **Tolkien (LOTR, Hobbit)** | 91% | Proper nouns challenge |
| **Poe stories** | 88% | Complex Gothic vocabulary |
| **Wikipedia** | Variable | Needs filtering |

### Common Parse Failures
1. **Proper nouns** (40% of failures) - Frodo, Gandalf, etc.
   - *Solution*: Proper noun dictionary (Phase 5.1)
2. **Unknown roots** (35% of failures) - Rare/archaic words
3. **Complex compounds** (15% of failures) - Multi-root words
4. **Foreign borrowings** (10% of failures) - Non-standard Esperanto

---

## Recommended Corpus Usage by Task

### Parser Training & Testing
**Best Sources**:
1. Fundamenta Krestomatio (4,337 sentences) - gold standard
2. Grammar books (2,243 sentences) - clear examples
3. Zamenhof works (5,191 sentences total) - authoritative

**Avoid**: Wikipedia (too noisy), poetry (non-standard structure)

### Vocabulary Expansion
**Best Sources**:
1. espdic.txt (dictionary) - comprehensive roots
2. Wikipedia - modern terminology
3. All corpus combined - frequency analysis

### Semantic Similarity Training
**Best Sources**:
1. Tatoeba EN-EO pairs (271K) - already being used ✅
2. Fundamenta Krestomatio + paraphrase generation
3. Wikipedia + back-translation

### Retrieval & Q&A Testing
**Best Sources**:
1. La Mastro de l' Ringoj - rich narrative for Q&A
2. La Hobito - simpler Q&A
3. Historical documents - factual questions

### Grammar Checker Training
**Best Sources**:
1. Fundamenta Krestomatio - correct examples
2. Grammar books - explicitly correct
3. Synthetic error generation from correct sentences

---

## Future Corpus Expansion

### High Priority
- [ ] Download all 15 Gutenberg Esperanto texts (currently described but not integrated)
- [ ] Build proper noun dictionary from corpus (Phase 5.1)
- [ ] Extract frequency-based root vocabulary for embedding init

### Medium Priority
- [ ] Find more original Esperanto literature (not translations)
- [ ] Collect Esperanto news articles (modern language usage)
- [ ] Extract Esperanto from multilingual datasets (Common Crawl, etc.)

### Low Priority
- [ ] Esperanto poetry corpus (for future creativity module)
- [ ] Esperanto technical documentation (programming, science)
- [ ] Esperanto social media (informal language, but noisy)

---

## Corpus File Locations

```
data/
├── cleaned/                          # Clean Esperanto texts
│   ├── cleaned_la_mastro_de_l_ringoj.txt  (2.17 MB, 73K lines)
│   ├── cleaned_la_hobito.txt              (416 KB, 14K lines)
│   ├── cleaned_puto_kaj_pendolo.txt       (31 KB)
│   ├── cleaned_la_korvo.txt               (43 KB)
│   ├── cleaned_usxero_domo.txt            (43 KB)
│   ├── cleaned_ses_noveloj.txt            (113 KB)
│   ├── cleaned_kadavrejo_strato.txt       (72 KB)
│   ├── cleaned_espdic.txt                 (487 KB - dictionary!)
│   └── cleaned_wikipedia.txt              (843 MB)
│
├── corpus_with_sources_v2.jsonl      # Production corpus (22.7 GB, 26,725 sentences)
├── corpus_with_sources_v3.jsonl      # Test corpus (104 MB, filtered)
├── corpus_index_v3/                  # FAISS production index
│   ├── faiss_index.bin
│   ├── metadata.jsonl
│   └── embeddings.npy
│
├── similarity_pairs_train.jsonl      # Tatoeba training data (159K)
├── similarity_pairs_val.jsonl        # Validation set
├── similarity_pairs_test.jsonl       # Test set
│
├── vocabularies/                     # Extracted vocabularies
│   ├── root_vocab.json               (953K roots)
│   ├── prefix_vocab.json             (61 prefixes)
│   └── suffix_vocab.json             (38 suffixes)
│
└── GUTENBERG_CORPUS_SUMMARY.md       # Gutenberg corpus documentation
```

---

## How to Use This Corpus

### Quick Access to Best Corpus
```bash
# Use Corpus V2 for production
cat data/corpus_with_sources_v2.jsonl | jq -r '.text' | head -100

# Filter by source
cat data/corpus_with_sources_v2.jsonl | jq 'select(.source == "la_mastro_de_l_ringoj")' > lotr_only.jsonl

# Get high-quality sentences (parse rate > 0.9)
cat data/corpus_with_sources_v2.jsonl | jq 'select(.parse_statistics.success_rate > 0.9)' > high_quality.jsonl
```

### Build Custom Index
```bash
# Index specific sources
python scripts/index_corpus.py \
  --corpus data/corpus_with_sources_v2.jsonl \
  --filter-source "la_mastro_de_l_ringoj,la_hobito" \
  --output data/corpus_index_tolkien

# Index only high-quality sentences
python scripts/index_corpus.py \
  --corpus data/corpus_with_sources_v2.jsonl \
  --min-parse-rate 0.9 \
  --output data/corpus_index_high_quality
```

### Extract Vocabulary
```bash
# Get most common roots
jq -r '.root_vocab | to_entries | sort_by(.value) | reverse | .[0:100] | .[].key' \
  data/vocabularies/root_vocab.json > data/top_100_roots.txt

# Find all verbs (roots that appear with -as, -is, -os)
python scripts/extract_verbs.py data/corpus_with_sources_v2.jsonl > data/verb_roots.txt
```

---

## References

### Corpus Sources
- **Project Gutenberg Esperanto**: https://www.gutenberg.org/ebooks/bookshelf/34
- **Tatoeba Project**: https://tatoeba.org/en/downloads
- **Esperanto Wikipedia**: https://eo.wikipedia.org/

### Esperanto Language References
- **16 Grammar Rules**: See `16RULES.MD` (if exists) or `docs/wiki-drafts/Esperanto-Grammar.md`
- **PMEG (Plena Manlibro de Esperanta Gramatiko)**: https://bertilow.com/pmeg/
- **ReVo (Reta Vortaro)**: https://www.reta-vortaro.de/revo/

### Related Documentation
- `CLAUDE.md` - Development guide
- `README.md` - Project overview
- `data/GUTENBERG_CORPUS_SUMMARY.md` - Gutenberg corpus details
- `docs/CORPUS_MANAGEMENT.md` - Corpus management guide
