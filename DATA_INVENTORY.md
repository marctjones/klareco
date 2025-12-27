# Klareco Data Inventory

**Last Updated**: December 2024
**Status**: Audit in progress

---

## Summary

| Category | Status | Tier | Annotated? | Clean? |
|----------|--------|------|------------|--------|
| Fundamento Ekzercaro | **READY** | 1 | Yes | Yes |
| Fundamenta Krestomatio | **READY** | 2 | Yes | Yes |
| Gerda Malaperis | **READY** | 3 | Yes | Yes |
| Reta Vortaro (ReVo) | **READY** | 4 | Yes | Yes |
| Gutenberg Books | Available | 5 | No | Partial |
| Wikipedia | Available | 6 | No | Partial |

---

## Tier 1: Fundamento de Esperanto (READY)

**Location**: `texts/authoritative/fundamento/`
**Status**: Clean, annotated, corpus-ready

| File | Content | Lines | Status |
|------|---------|-------|--------|
| `gramatiko.txt` | 16 grammar rules | 382 | Clean UTF-8 |
| `ekzercaro.txt` | 42 exercises | 3,481 | Clean UTF-8 |
| `metadata.json` | Source annotation | - | Complete |

**Corpus Output**: `data/corpus/authoritative_corpus.jsonl`
- 1,137 sentences with tier=1, weight=10.0
- Citations: `fundamento:ekzercaro:§{n}:{sentence}`

**TODO**:
- [ ] Extract Universala Vortaro for root definitions (scripts/training/extract_fundamento_uv.py exists but needs clean source)

---

## Tier 2: Fundamenta Krestomatio (READY)

**Location**: `texts/authoritative/krestomatio/`
**Status**: Clean, annotated, corpus-ready

| File | Content | Lines | Status |
|------|---------|-------|--------|
| `fundamenta_krestomatio.txt` | Complete anthology | 16,389 | Clean UTF-8 |
| `metadata.json` | Source annotation | - | Complete |

**Corpus Output**: `data/corpus/authoritative_corpus.jsonl`
- 15,258 sentences with tier=2, weight=5.0
- Citations: `krestomatio:{section}:L{line}`

**Notes**: Converted from x-system (Project Gutenberg) to proper Unicode

---

## Tier 3: Gerda Malaperis (READY)

**Location**: `texts/authoritative/gerda_malaperis/`
**Status**: Clean, annotated, corpus-ready

| File | Content | Lines | Status |
|------|---------|-------|--------|
| `gerda_malaperis.txt` | 25 chapters | 1,602 | Clean UTF-8 |
| `metadata.json` | Source annotation | - | Complete |

**Corpus Output**: `data/corpus/authoritative_corpus.jsonl`
- 1,602 sentences with tier=3, weight=3.0
- Citations: `gerda:{chapter}:L{line}`
- Includes: `is_dialogue`, `speaker` annotations

---

## Tier 4: Reta Vortaro / ReVo (READY)

**Location**: `data/revo/`
**Status**: Clean, integrated into training pipeline

| File | Content | Size | Status |
|------|---------|------|--------|
| `revo_definitions_with_roots.json` | 10,766 dictionary entries | 11MB | Complete |
| `revo.db` | SQLite database | 151MB | Complete |

**Features**:
- Clean JSON with definitions, definition_roots, and all_roots
- GPL-licensed from [Reta Vortaro](https://reta-vortaro.de)
- Already integrated into `train_root_embeddings.py`
- Used for Jaccard-similarity pairs in training

**Note**: Replaces old Plena Vortaro (which had OCR artifacts)

---

## Tier 5: Gutenberg Books (PARTIAL)

**Location**: `data/clean_corpus/` and `data/gutenberg_esperanto/`
**Status**: Cleaned, but not source-annotated

### Available Texts

| File | Content | Size | Clean? |
|------|---------|------|--------|
| `la_mastro_de_l_ringoj.txt` | Lord of the Rings | 1.1MB | Yes |
| `la_hobito.txt` | The Hobbit | 472KB | Yes |
| `alicio.txt` | Alice in Wonderland | 150KB | Yes |
| `frankenstejno.txt` | Frankenstein | 267KB | Yes |
| `milito_de_la_mondoj.txt` | War of the Worlds | 236KB | Yes |
| `sorcxisto_de_oz.txt` | Wizard of Oz | 105KB | Yes |
| `jekyll_hyde.txt` | Jekyll & Hyde | 134KB | Yes |
| `kadavrejo_strato.txt` | Murders Rue Morgue | 77KB | Yes |
| `puto_kaj_pendolo.txt` | Pit & Pendulum | 34KB | Yes |
| ... | (more Gutenberg texts) | | |

**Also available** (from Gutenberg Esperanto collection):
- `26359_Vivo_de_Zamenhof.txt` - Zamenhof biography
- `47855_Esperanta_sintakso.txt` - Esperanto syntax
- `57184_Dokumentoj_de_Esperanto.txt` - Historical documents
- `38240_The_Esperantist_Complete.txt` - Esperantist magazine

**TODO**:
- [ ] Add source annotations (tier=5, weight=1.5)
- [ ] Create MANIFEST with source URLs and licenses
- [ ] Distinguish original Esperanto works from translations

---

## Tier 6: Wikipedia (AVAILABLE)

**Location**: `data/clean_corpus/wikipedia.txt`
**Status**: Cleaned, not source-annotated

| File | Size | Status |
|------|------|--------|
| `wikipedia.txt` | 225MB | Clean text |

**In corpus**: ~73% of `corpus_with_sources_v2.jsonl`

**TODO**:
- [ ] Add source annotations (tier=6, weight=1.0)
- [ ] Preserve article titles for citations
- [ ] Track extraction date

---

## Existing Corpus Files

### Main Corpus (v2)

**File**: `data/corpus_with_sources_v2.jsonl`
**Size**: 22GB
**Status**: Has ASTs but no tier annotations

| Source | Approx % | Tier |
|--------|----------|------|
| wikipedia | 73% | 6 |
| la_mastro_de_l_ringoj | 20% | 5 |
| la_hobito | 4% | 5 |
| Other Gutenberg | 3% | 5 |

**Format**:
```json
{
  "text": "...",
  "source": "wikipedia",
  "source_name": "Vikipedio",
  "ast": {...},
  "parse_rate": 0.95
}
```

**Issues**:
- No tier field
- No weight field
- No citation format
- Missing authoritative texts (Fundamento, Krestomatio, Gerda)

### Authoritative Corpus (NEW)

**File**: `data/corpus/authoritative_corpus.jsonl`
**Size**: 38MB
**Status**: Complete with annotations

| Source | Sentences | Tier | Weight |
|--------|-----------|------|--------|
| Fundamento Ekzercaro | 1,137 | 1 | 10.0 |
| Fundamenta Krestomatio | 15,258 | 2 | 5.0 |
| Gerda Malaperis | 1,602 | 3 | 3.0 |
| **Total** | **17,997** | | |

**Format**:
```json
{
  "text": "...",
  "source": {
    "tier": 1,
    "name": "fundamento_ekzercaro",
    "citation": "fundamento:ekzercaro:§5:3",
    "weight": 10.0,
    "verified": true
  },
  "ast": {...},
  "parse_statistics": {...}
}
```

---

## Vocabulary Files

**Location**: `data/vocabularies/` and `data/revo/`

| File | Content | Size | Status |
|------|---------|------|--------|
| `fundamento_roots.json` | UV extracted roots | 475KB | Complete |
| `root_vocabulary.json` | All corpus roots | 21MB | Complete |
| `affix_vocabulary.json` | Affixes | 1.5KB | Complete |
| `revo_definitions_with_roots.json` | ReVo dictionary (Tier 4) | 11MB | Complete |

---

## Action Items

### Critical (Blocking Training)

1. **Merge authoritative corpus with main corpus**
   - Add `authoritative_corpus.jsonl` entries to training
   - Apply tier-based weighting during training

2. **Annotate existing corpus with tiers**
   - Add tier=5 to Gutenberg sources
   - Add tier=6 to Wikipedia sources
   - Update corpus format to match authoritative schema

### Important (Quality Improvement)

3. **Clean Universala Vortaro source**
   - Current raw file has OCR artifacts
   - Need clean multi-lingual root definitions

4. **Add Gutenberg source manifests**
   - Track source URLs
   - Track licenses
   - Distinguish originals from translations

### Nice to Have

5. **Add more Tier 2-3 sources**
   - Other Zamenhof writings
   - Other canonical pedagogical texts

---

## How to Build Complete Corpus

```bash
# 1. Build authoritative corpus (tiers 1-3)
python scripts/build_authoritative_corpus.py \
  --output data/corpus/authoritative_corpus.jsonl

# 2. TODO: Add tier annotations to existing corpus
python scripts/annotate_corpus_tiers.py \
  --input data/corpus_with_sources_v2.jsonl \
  --output data/corpus/annotated_corpus.jsonl

# 3. TODO: Merge all corpora
python scripts/merge_corpora.py \
  --authoritative data/corpus/authoritative_corpus.jsonl \
  --general data/corpus/annotated_corpus.jsonl \
  --output data/corpus/unified_corpus.jsonl
```

---

*See also*: Wiki [[Training-Data-Strategy]] for weighting and training approach
