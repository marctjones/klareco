# Data Audit: What to Keep, Commit, and Regenerate

## Summary of Current Data (2.7GB total)

| Directory/File | Size | Source | Copyright Status | Action |
|----------------|------|--------|------------------|--------|
| **data/corpora/** | 357MB | Mixed | **PROBLEMATIC** | ❌ Do NOT commit |
| **data/cleaned/** | 811MB | Processed | Derivative | ❌ Do NOT commit |
| **data/chunked_corpus/** | 1.2GB | Processed | Derivative | ❌ Do NOT commit |
| **data/test_corpus.json** | 20KB | Curated samples | Mixed | ⚠️ Review needed |
| **data/morpheme_vocab.json** | 4KB | Generated | N/A | ✅ Commit this |
| **models/lid.176.bin** | 126MB | FastText | Public | ❌ No longer used |
| **models/cc.eo.300.bin.gz** | 218MB | FastText | Public | ❌ Do NOT commit |
| **models/morpheme_embeddings.pt** | 4KB | Generated | N/A | ❌ Do NOT commit (experimental) |

## Detailed Analysis

### ❌ MUST NOT COMMIT - Copyright Issues

**data/corpora/** (357MB):
- `eo_wikipedia.xml.bz2` (348MB) - Wikipedia dump (CC-BY-SA, but 348MB!)
- `la_hobito.txt` (636KB) - The Hobbit translation (**copyrighted!** Tolkien estate)
- `la_mastro_de_l_ringoj.txt` (2.7MB) - Lord of the Rings (**copyrighted!** Tolkien estate)
- `frankenstejno.txt` (292KB) - Frankenstein (public domain, but processed)
- `jekyll_hyde.txt` (157KB) - Jekyll & Hyde (public domain, but processed)
- `milito_de_la_mondoj.txt` (262KB) - War of the Worlds (public domain, but processed)

**Copyright Status:**
- ✅ **Public Domain (OK)**: Frankenstein, Jekyll/Hyde, War of Worlds, Wizard of Oz, Poe stories
- ⚠️ **Wikipedia (CC-BY-SA)**: Can use, but 348MB is too large for git
- ⚠️ **COPYRIGHTED (Fair Use Locally, DO NOT COMMIT TO GIT)**:
  - **La Hobito** (The Hobbit) - Tolkien estate holds copyright until 2043
  - **La Mastro de l' Ringoj** (Lord of the Rings) - Tolkien estate until 2043
  - **Fair Use Justification**: Personal copies used for ML research/development (transformative, non-commercial)
  - **Critical**: These MUST remain in .gitignore and NEVER be committed to git

**data/cleaned/** (811MB) and **data/chunked_corpus/** (1.2GB):
- Processed versions of above sources
- Inherit copyright issues
- WAY too large for git

### ⚠️ REVIEW NEEDED

**data/test_corpus.json** (20KB):
- Contains samples from multiple sources (seen in head output)
- Includes copyrighted Tolkien text
- **Recommendation**: Replace with ONLY public domain samples

**data/grammar/** (496KB):
- Need to check what this contains

### ✅ SAFE TO COMMIT

**data/morpheme_vocab.json** (4KB):
- Generated vocabulary from corpus analysis
- Just a list of morphemes, no copyrighted text
- Small enough for git
- **Action**: Commit this

### ❌ DO NOT COMMIT - Large Models

**models/lid.176.bin** (126MB):
- FastText language ID model (no longer used - switched to lingua)
- Public domain, but 126MB is too large
- **Action**: Delete (not needed anymore)

**models/cc.eo.300.bin.gz** (218MB):
- FastText word embeddings for Esperanto
- Public domain, but 218MB is too large
- **Action**: Keep locally, don't commit, document how to download

**models/morpheme_embeddings.pt** (4KB):
- Experimental, not used in current system
- **Action**: Don't commit (not production)

## Recommended Actions

### 1. Update .gitignore (DONE)
Already excludes `data/` and `models/` directories.

### 2. Create Data Download/Regeneration Scripts

#### **scripts/download_public_data.sh**
Downloads ONLY public domain sources:
- Project Gutenberg Esperanto texts
- Tekstaro public domain texts
- Public Esperanto dictionaries

#### **scripts/download_wikipedia.sh**
Downloads Esperanto Wikipedia dump (user decision, large file).

#### **scripts/process_corpus.sh**
Runs cleaning, chunking on downloaded data.

### 3. Protect Copyrighted Material from Git

**DO NOT DELETE** - These files provide valuable training data under fair use.

**ENSURE .gitignore protects them** (already configured):
```bash
# Verify protection:
git status --ignored | grep "data/"
# Should show data/ as ignored

# If you ever accidentally stage them:
git rm --cached data/corpora/la_hobito.txt  # (if needed)
```

**NEVER run:**
```bash
git add -f data/  # This would force-add ignored files - DON'T DO THIS
```

**Delete obsolete model:**
```bash
rm models/lid.176.bin  # No longer needed (using lingua now)
```

### 4. Create Curated test_corpus.json

Replace current test corpus with ONLY public domain samples:
- Edgar Allan Poe stories (public domain)
- Mary Shelley's Frankenstein (public domain)
- Wells' War of the Worlds (public domain)
- Simple constructed sentences

**Size**: Should be ~10-20KB (45-50 sentences)

### 5. Commit Only Safe Files

```bash
git add data/morpheme_vocab.json
git add data/test_corpus.json  # After cleaning
git commit -m "Add generated vocabulary and curated test corpus"
```

## Data Regeneration Strategy

### For Public Domain Corpora:

Create `scripts/setup_corpus.sh`:
```bash
#!/bin/bash
# Downloads and processes public domain Esperanto texts

mkdir -p data/corpora

# Download from Project Gutenberg
wget -O data/corpora/frankenstejno.txt "URL_HERE"
wget -O data/corpora/jekyll_hyde.txt "URL_HERE"
# ... etc

# Process corpus
python scripts/clean_corpus.py
python scripts/chunk_corpus.py
python scripts/build_morpheme_vocab.py
```

### For Wikipedia:

Create `scripts/setup_wikipedia.sh`:
```bash
#!/bin/bash
# Downloads Esperanto Wikipedia (348MB - optional)

echo "This will download 348MB of Esperanto Wikipedia"
read -p "Continue? (y/n) " -n 1 -r
if [[ $REPLY =~ ^[Yy]$ ]]; then
    wget -O data/corpora/eo_wikipedia.xml.bz2 \
        "https://dumps.wikimedia.org/eowiki/latest/eowiki-latest-pages-articles.xml.bz2"
    python scripts/clean_corpus.py --wiki
fi
```

### For FastText Embeddings:

Create `scripts/download_fasttext_embeddings.sh`:
```bash
#!/bin/bash
# Downloads FastText Esperanto word embeddings (218MB - optional)

mkdir -p models
wget -O models/cc.eo.300.bin.gz \
    "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.eo.300.bin.gz"
```

## COPYRIGHT COMPLIANCE SUMMARY

### ✅ Can Use/Distribute:
- Public domain texts (pre-1928 in US, varies by country)
- Project Gutenberg Esperanto texts
- Wikipedia (with CC-BY-SA attribution)
- Your own generated/processed data (morpheme_vocab.json)

### ❌ CANNOT Distribute (via git/GitHub):
- Tolkien translations (copyrighted until 2043)
- Any modern copyrighted books
- Unauthorized translations of copyrighted works
- **Even if you own the books** - distribution is different from personal use

### ✅ Fair Use (Keep Locally):
- Using copyrighted books you own for ML research/development
- Training data for non-commercial research projects
- Transformative use (learning patterns, not republishing)
- **Legal standard**: Purpose, nature, amount, market effect
- **Key protection**: .gitignore ensures no accidental distribution

## Recommendation for GitHub

**DO commit:**
- Small generated files: `morpheme_vocab.json` (4KB)
- Curated test corpus from public domain only: `test_corpus.json` (~15KB)
- Scripts to download/regenerate everything

**DO NOT commit:**
- Any corpus text files (even public domain - too large)
- Any model files (too large)
- Any copyrighted material (legal risk)

**Total git repo size with recommendations**: <100KB of data
**Regeneration time with scripts**: ~30 minutes for full corpus

## Next Steps

1. ✅ Verify .gitignore protects data/ and models/ (DONE - confirmed working)
2. ⚠️ Delete obsolete lid.176.bin model (optional - no longer used)
3. ✅ test_corpus.json is clean (verified - only constructed sentences)
4. ⏳ Create setup scripts for corpus regeneration (for other users)
5. ✅ Commit morpheme_vocab.json (safe - generated data, 4KB)
6. ✅ Commit this audit document (safe - no copyrighted text)
7. ⏳ Update README with data setup instructions (for users who clone the repo)
