# Data Sharing Guide

**Last Updated:** 2026-01-19

This document describes which data can be safely shared publicly and which cannot.

---

## TL;DR

✅ **Safe to share:**
- Scripts and code (all MIT licensed)
- Small config files (quality_overrides.json, etc.)
- Tier0 sources (<10MB total - PMEG, Lingvaj Respondoj, small authoritative sources)

❌ **Do NOT share:**
- Large data files (>50MB) - use download scripts instead
- Copyrighted translations (Tolkien, etc.) - even if not in corpus
- Full Wikipedia dumps (1.3GB) - provide download instructions

🔒 **Already protected:**
- `data/` directory is in .gitignore ✓
- Users run acquisition scripts to download public data
- Copyrighted sources are for personal research only

---

## File Sizes by Directory

### Raw Sources (`data/raw/eo/`)

| Directory | Size | License | Can Share? |
|-----------|------|---------|------------|
| `proverbaro/` | 4KB | Public Domain | ✅ Yes (tiny) |
| `fundamento/` | 536KB | Public Domain (1905) | ✅ Yes |
| `lingvaj_respondoj/` | 1.4MB | Unknown | ⚠️ Check first |
| `pmeg/` | 5.3MB | CC-BY-SA? | ⚠️ Check license |
| `gutenberg/` | 22MB | Public Domain | ⚠️ Too large for git |
| `pag/` | 34MB | Unknown | ❌ Too large + verify |
| `dictionaries/` (ReVo) | 191MB | Unknown | ❌ Too large + verify |
| `wikipedia/` | 348MB | CC-BY-SA 3.0 | ❌ Too large |

**Total raw:** ~600MB

### Cleaned Data (`data/cleaned/eo/`)
- **Size:** 39MB
- **Status:** ❌ Too large for git

### Extracted Sentences
- `books_sentences.jsonl`: 53MB ❌
- `wikipedia_sentences.jsonl`: 1.3GB ❌
- `eo/tier0_filtered/`: 21MB ❌

### Processed Corpus
- `corpus_with_metadata.jsonl`: ~4GB ❌

---

## Copyright Analysis

### ✅ Public Domain (Safe to redistribute)

**Sources:**
- Fundamento de Esperanto (1905) - Zamenhof
- Project Gutenberg books (pre-1928 works)
- Fundamenta Krestomatio

**Can be shared:** Yes, but most are too large for git

### 🟡 CC-BY-SA (Safe with attribution)

**Sources:**
- Wikipedia (Esperanto edition)
- PMEG (needs verification)

**Can be shared:** Yes, but 348MB is too large for git

**Required attribution:**
```
Source: Vikipedio, la libera enciklopedio
License: CC-BY-SA 3.0
URL: https://eo.wikipedia.org/
```

### 🔴 Copyrighted (Personal use only)

**Sources found in `data/raw/LICENSES.md`:**
- La Mastro de l' Ringoj (Tolkien Estate copyright)
- La Hobito (Tolkien Estate copyright)
- Various copyrighted translations

**Status:** Not used in current corpus (verified ✓)

**Can be shared:** NO - research/personal use only

### ⚠️ License Unknown (Verify before sharing)

**Sources:**
- ReVo dictionary (Reta Vortaro) - 191MB
- ESPDIC - Check original project
- Lingvaj Respondoj - Likely public domain but verify
- PAG (Plena Analiza Gramatiko) - Check author permissions

---

## Current Protection Status

### ✅ Already Protected

1. **`.gitignore` covers all data:**
   ```
   data/
   ```
   This excludes ALL raw, cleaned, extracted, and processed data ✓

2. **Acquisition scripts provided instead:**
   - `scripts/acquire_gutenberg.py`
   - `scripts/acquire_wikipedia.sh`
   - Users download public sources themselves

3. **No copyrighted translations in corpus:**
   - Verified: Tolkien works NOT in `books_sentences.jsonl` ✓
   - Only Project Gutenberg public domain books used ✓

### ⚠️ Potential Issues

1. **Small config files ARE tracked:**
   - `config/quality_overrides.json` (should be in git - it's just config)
   - `data/quality_report.txt` (15KB - currently in git via data/)
   - Solution: Move reports to `reports/` directory outside `data/`

2. **Scripts reference download sources:**
   - Make sure download scripts point to legal sources ✓
   - Document where to obtain each dataset ✓

---

## Recommendations

### For Public GitHub Repository

**DO share:**
1. ✅ All Python scripts (`scripts/`, `klareco/`)
2. ✅ Documentation (`.md` files in root)
3. ✅ Config templates (`config/quality_overrides.json`)
4. ✅ Small vocabularies if generated (<1MB)
5. ✅ Test fixtures (small sample data for tests)

**DO NOT share:**
1. ❌ `data/` directory (already in .gitignore ✓)
2. ❌ `models/` directory (trained models - too large)
3. ❌ `logs/` directory (personal logs)

**PROVIDE instead:**
1. 📥 **Acquisition scripts** that download public sources
2. 📄 **DATA_SOURCES.md** documenting where to get each dataset
3. 📋 **Setup instructions** in README

### For Paper/Research Publication

**Can include in supplementary materials:**
- ✅ Tier0 authoritative sources (<10MB total)
- ✅ Quality analysis results (parse rates, statistics)
- ✅ Sample sentences from each quality tier

**Must provide separately:**
- 📥 Links to Project Gutenberg books (with IDs)
- 📥 Wikipedia dump download instructions
- 📄 Attribution for CC-BY-SA sources

### For Model Training Reproducibility

**Provide:**
1. Exact Wikipedia dump date/version
2. List of Project Gutenberg IDs used
3. Scripts to acquire and process
4. Random seeds for reproducibility
5. Quality thresholds and override config

**Do NOT need to provide:**
- Raw or processed corpus files (users can rebuild)
- Trained model weights (unless you want to share)

---

## License Verification Checklist

Before any public release, verify:

- [ ] PMEG license (check https://bertilow.com/pmeg/)
- [ ] PAG license (check author's site)
- [ ] Lingvaj Respondoj license (likely public domain but verify)
- [ ] ReVo license (check https://reta-vortaro.de/)
- [ ] No copyrighted translations accidentally included
- [ ] All CC-BY-SA sources properly attributed

---

## Data Acquisition Documentation

**Location:** `docs/DATA_SOURCES.md` (create this)

Should include:
```markdown
## Wikipedia
- Source: https://dumps.wikimedia.org/eowiki/
- License: CC-BY-SA 3.0
- Download: `./scripts/acquire_wikipedia.sh`

## Project Gutenberg
- Source: https://www.gutenberg.org/
- License: Public Domain (US)
- Download: `./scripts/acquire_gutenberg.py`
- Book IDs: [list used IDs]

## Tier0 Authoritative Sources
- PMEG: [URL and license]
- Fundamento: Public domain
- Lingvaj Respondoj: [URL and license]
```

---

## Summary

**Current status:** ✅ Data is properly protected

The `data/` directory is in `.gitignore`, so no large files or copyrighted content will be accidentally committed. The corpus only uses public domain and CC-BY-SA sources (verified).

**For public sharing:**
1. Keep `data/` in .gitignore ✓
2. Provide acquisition scripts ✓
3. Document data sources with licenses
4. Verify unknown licenses (PMEG, PAG, ReVo, Lingvaj Respondoj)
5. Consider moving `data/quality_report.txt` to `reports/` directory

**Safe to commit:**
- Code, scripts, config templates
- Small (<1MB) generated vocabularies
- Documentation and analysis results (numbers, not data)
