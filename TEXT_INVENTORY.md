# Esperanto Text Inventory & Quality Assessment

## Summary

**Total: ~810MB of Esperanto text across 3 categories**

---

## 1. PROJECT GUTENBERG (Clean & Ready ✓)
**Location:** `data/gutenberg_esperanto/`  
**Status:** Already cleaned by Project Gutenberg  
**Total:** 15 files, ~4.6 MB

| File | Size | Description |
|------|------|-------------|
| 08224_Fundamenta_Krestomatio.txt | 855K | Zamenhof's foundational text ⭐ |
| 20006_Dua_Libro...txt | 67K | Zamenhof's second book |
| 11307_El_la_Biblio.txt | 118K | Bible excerpts |
| 23670_Nuntempaj_Rakontoj.txt | 150K | Contemporary stories |
| 24525_Karlo__Facila_Legolibro.txt | 85K | Easy reader |
| 25311_El_la_vivo...txt | 78K | Life stories |
| 26359_Vivo_de_Zamenhof.txt | 182K | Zamenhof biography |
| 38240_The_Esperantist_Complete.txt | 2.0M | Complete periodical ⭐ |
| 42028_En_Rusujo...txt | 182K | Travel narrative |
| 42774_Mondo_kaj_koro.txt | 48K | Poetry |
| 47855_Esperanta_sintakso.txt | 154K | Syntax guide |
| 48896_Verdaj_fajreroj.txt | 47K | Poetry |
| 52556_EsperantoGermana_frazlibro.txt | 103K | Phrasebook |
| 57184_Dokumentoj_de_Esperanto.txt | 459K | Documents |
| 76273_Por_kaj_kontraŭ_Esperanto.txt | 55K | Dialogue |

**Parser Performance:** 99.1% word-level success ✓

---

## 2. LITERARY WORKS (Needs HTML Cleanup ⚠)
**Location:** `data/corpora/`  
**Status:** HTML/XML embedded - needs cleaning  
**Total:** ~10 files, ~5 MB

| File | Size | Type | Needs Cleaning |
|------|------|------|----------------|
| la_hobito.txt | 650K | Tolkien | HTML tags ⚠ |
| la_mastro_de_l_ringoj.txt | 3.1M | Tolkien | HTML tags ⚠ |
| kadavrejo_strato.txt | 98K | Edgar Allan Poe | HTML tags ⚠ |
| la_korvo.txt | 63K | Edgar Allan Poe | HTML tags ⚠ |
| puto_kaj_pendolo.txt | 54K | Edgar Allan Poe | HTML tags ⚠ |
| ses_noveloj.txt | 142K | Edgar Allan Poe | HTML tags ⚠ |
| usxero_domo.txt | 63K | Edgar Allan Poe | HTML tags ⚠ |
| alicio.txt | 174K | Lewis Carroll | HTML tags ⚠ |
| frankenstejno.txt | 292K | Mary Shelley | HTML tags ⚠ |
| jekyll_hyde.txt | 157K | R.L. Stevenson | HTML tags ⚠ |
| milito_de_la_mondoj.txt | 262K | H.G. Wells | HTML tags ⚠ |
| sorcxisto_de_oz.txt | 127K | L. Frank Baum | HTML tags ⚠ |

---

## 3. WIKIPEDIA (Clean & Ready ✓)
**Location:** `data/cleaned/cleaned_wikipedia.txt`  
**Status:** Already cleaned  
**Size:** 804 MB (!!)

**Parser Performance:** 87.5% word-level success ✓

---

## 4. CORRUPTED FILES (❌ Skip or Re-download)
**Location:** `data/cleaned/`  
**Issue:** Cleaning script corrupted these files

| File | Original Size | Issue |
|------|---------------|-------|
| cleaned_la_hobito.txt | 407KB | Garbage characters |
| cleaned_la_mastro_de_l_ringoj.txt | 2.1MB | Garbage characters |
| cleaned_kadavrejo_strato.txt | 70KB | Garbage characters |
| cleaned_la_korvo.txt | 42KB | Garbage characters |
| cleaned_puto_kaj_pendolo.txt | 31KB | Garbage characters |
| cleaned_ses_noveloj.txt | 111KB | Garbage characters |
| cleaned_usxero_domo.txt | 42KB | Garbage characters |

**Recommendation:** Delete these and re-clean from `corpora/` originals

---

## Next Steps

### 1. Clean the Literary Works ⚠
Create proper HTML cleaning script for `corpora/` files:
- Remove `<script>`, `<style>`, HTML tags
- Remove URLs
- Keep only Esperanto text
- Save to new `data/clean_corpus/` directory

### 2. Test Parser on Each Source
Once cleaned, test word-level success on:
- Tolkien works (2 files)
- Poe stories (5 files)
- Other classics (5 files)
- Compare to Zamenhof baseline (99.1%)

### 3. Expected Results
- **Standard Esperanto (Zamenhof):** 99% success
- **Wikipedia:** 87% success (proper names, technical terms)
- **Literary works:** 85-95% success (creative language, names)

---

## Priority Actions

**High Priority:**
1. Create HTML cleaning script
2. Clean all `corpora/*.txt` files
3. Delete corrupted `cleaned/` files

**Medium Priority:**
4. Test parser on clean literary works
5. Generate comprehensive success report

**Low Priority:**
6. Extract vocabulary from literary works
7. Identify common proper names for categorization

