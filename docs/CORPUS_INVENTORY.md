# Corpus Inventory

**Last Updated**: 2025-12-24

Quick reference for Klareco corpus data sources and locations.

## Current Corpus

**Enhanced Corpus** (with metadata):
- **Location**: `data/enhanced_corpus/corpus_with_metadata.jsonl`
- **Size**: ~5.1M sentences, 15-20 GB
- **Parse Rate**: ~91% average
- **Metadata**: Article titles, chapters, parse quality

## Data Sources

### Wikipedia
- **File**: `data/corpora/eo_wikipedia.xml.bz2`
- **Size**: 348 MB compressed
- **Expected**: ~3.8M sentences
- **Metadata**: Article title, ID, section, timestamp

### Books

| Source | File | Sentences | Parse Rate |
|--------|------|-----------|------------|
| Lord of the Rings | `cleaned_la_mastro_de_l_ringoj.txt` | ~1.1M | 88-91% |
| The Hobbit | `cleaned_la_hobito.txt` | ~200K | 89-92% |
| Poe Stories | `cleaned_kadavrejo_strato.txt` | ~30K | 87-90% |
| The Raven | `cleaned_la_korvo.txt` | ~10K | 87-90% |
| Pit & Pendulum | `cleaned_puto_kaj_pendolo.txt` | ~10K | 87-90% |
| Six Tales | `cleaned_ses_noveloj.txt` | ~20K | 87-90% |
| Usher | `cleaned_usxero_domo.txt` | ~10K | 87-90% |

**All in**: `data/cleaned/`

### Dictionary & Reference
- **Dictionary**: `data/cleaned/cleaned_espdic.txt` (487 KB)
- **Gutenberg texts**: `data/gutenberg/` (Zamenhof works, literature)

## Corpus Versions

| Version | Status | Location | Size |
|---------|--------|----------|------|
| V1 | Deprecated | - | - |
| V2 | Legacy | `corpus_with_sources_v2.jsonl` | 22 GB |
| V3 (Enhanced) | **Current** | `enhanced_corpus/` | 15-20 GB |

## Indexes

| Index | Corpus | Location | Status |
|-------|--------|----------|--------|
| V2 Index | Corpus V2 | `corpus_index_v2/` | Legacy |
| V3 Index | Enhanced | `corpus_index_v3/` | Legacy |
| Enhanced Index | Enhanced | `enhanced_index/` | To be built |

## Building Enhanced Corpus

See `docs/CORPUS_BUILDING.md` for complete guide.

**Quick start**:
```bash
./scripts/run_wikipedia_extraction.sh   # 2-3 hours
./scripts/run_books_extraction.sh       # 5-10 min
./scripts/run_corpus_builder.sh         # 1-2 hours
```

## Citation Metadata

### Wikipedia
```json
{
  "article_title": "AIM",
  "article_id": 1,
  "section": "Priskribo"
}
```

### Books
```json
{
  "chapter": "NEATENDITA FESTO",
  "sentence_in_chapter": 42
}
```

## Related

- **Technical Guide**: `docs/CORPUS_BUILDING.md`
- **Wiki**: [[Corpus Management]]
- **Training Data**: See [[AST-Native Training]]
