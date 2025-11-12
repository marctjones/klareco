# Vocabulary Extraction Summary

**Date:** 2025-11-11
**Task:** Extract missing Esperanto roots from corpus to improve parser performance

---

## Executive Summary

Successfully processed **551MB Wikipedia** (in 28 chunks) and **12 literary texts** using incremental processing to avoid memory issues. Extracted vocabulary reveals both legitimate missing roots and artifacts from imperfect text cleaning.

### Key Findings:
- ✅ **Incremental processing works** - Successfully processed 551MB Wikipedia in 20MB chunks
- ⚠️ **Wikipedia needs deeper cleaning** - 200k+ "roots" are HTML/wiki artifacts
- ✅ **Literary texts relatively clean** - 5,107 missing roots, but many are English/proper names
- 📝 **~100 legitimate roots identified** for addition to parser

---

##Processing Results

### Wikipedia (551MB via 28 chunks)

| Metric | Value |
|--------|-------|
| Files processed | 28 chunks (20MB each) |
| Total unique roots | 904,933 |
| Already in vocabulary | 13,414 |
| Missing roots (≥5 freq) | 203,914 |
| High frequency (≥10) | 118,468 |

**Top Artifacts (need better cleaning):**
- HTML/CSS: `styl` (227k), `nbsp` (157k), `px` (146k), `jpg` (143k), `background` (98k), `color` (82k)
- Wiki markup: `alidirekt` (125k), `vikipedi` (26k), `tpaĝ` (30k)
- Technical: `colsp` (93k), `cellpadd` (33k), `bgcolor` (30k)

**Legitimate roots found:**
- `municip` (140k) - municipality
- `dosi` (90k) - dosage
- `magnitud` (33k) - magnitude
- `arondisment` (34k) - arrondissement (district)
- `periaps`/`apoaps` (31-62k) - astronomy terms
- Country names: `hung`, `franci`, `itali`, `japani`, `hispani`, `rumani`, `slovaki`

### Literary Texts (12 files, ~4MB)

| Metric | Value |
|--------|-------|
| Files processed | 12 (Tolkien, Poe, classics) |
| Total unique roots | 20,489 |
| Already in vocabulary | 4,617 |
| Missing roots (≥3 freq) | 5,107 |
| High frequency (≥10) | 1,757 |

**Categories of "Missing" Roots:**

1. **English Artifacts** (~60%, from Project Gutenberg headers/footers):
   - Common words: `with` (1,751), `was` (1,634), `had` (742), `which` (708), `very` (414)
   - Gutenberg boilerplate: `gutenberg` (977), `project` (890), `copyright` (202), `agreement` (182)

2. **Proper Names** (~20%, character/place names):
   - Tolkien: `frod` (1,087), `bilb` (833), `gandalf` (612), `aragorn` (188), `elrond` (177), `boromir` (141)
   - Alice: `alic` (412)
   - Places: `baginz` (166 - Baggins)

3. **Legitimate Esperanto Roots** (~15-20%, **should be added**):
   - `trankv` (141) - tranquil, calm ✓ (already identified in analyze_failures.py)
   - `spond` (314) - respond
   - `goblen` (204) - goblin
   - `grinĉj` (212) - some creature (verify)
   - `pone` (164) - to place/put (verify)
   - `finf` (155) - compound? (verify)

4. **Ambiguous** (~5%):
   - `gre` (266) - Greek? degree? (investigate)
   - `dow` (244) - down? endowment? (investigate)
   - `archiv` (164) - archive (likely legitimate)
   - `licens` (181) - license (likely legitimate)

---

## Recommendations

### Immediate Actions (High Priority)

1. **Add Verified Esperanto Roots** (~50-100 roots)
   - From `analyze_failures.py` list (52 roots manually identified)
   - From literary extraction, verified roots like: `trankv`, `spond`, `archiv`, `licens`
   - **Impact**: +5-10% parsing success on literary texts

2. **Improve Wikipedia Cleaning**
   - Remove ALL HTML tags, attributes, and entities
   - Strip CSS/style blocks completely
   - Remove wiki template markup (`{{...}}`, `[[...]]`)
   - **Impact**: Reduce noise from 200k to ~10-20k legitimate roots

3. **Improve Literary Cleaning**
   - Better removal of Project Gutenberg headers/footers
   - Remove copyright/license boilerplate
   - **Impact**: Cleaner vocabulary extraction

### Medium Priority

4. **Create Proper Name Database**
   - Extract all character/place names (≥10 occurrences)
   - Add to parser as `KNOWN_PROPER_NAMES` set
   - Categorize as `proper_name` without failing parse
   - **Impact**: Better handling of fantasy/literary texts

5. **Country/Language Name Expansion**
   - Add Esperantized country names: `hungari`, `japani`, `hispani`, etc.
   - Add language names: `franci`, `itali`, `rumani`
   - **Impact**: +2-3% on Wikipedia parsing

### Long-term

6. **Automated Root Validation**
   - Create filter to identify English vs. Esperanto patterns
   - Check against Esperanto morphology rules
   - Cross-reference with ReVo (Reta Vortaro) dictionary
   - **Impact**: Scalable vocabulary expansion

---

## Files Generated

### Wikipedia Extraction
- `/home/marc/klareco/data/wikipedia_chunks/` - 28 chunks (20MB each)
- `/home/marc/klareco/data/wikipedia_chunks_missing_roots.txt` - 203,914 roots
- `/home/marc/klareco/data/wikipedia_chunks_missing_roots.py` - Python set

### Literary Extraction
- `/home/marc/klareco/data/literary_corpus/` - 12 cleaned literary files
- `/home/marc/klareco/data/literary_corpus_missing_roots.txt` - 5,107 roots
- `/home/marc/klareco/data/literary_corpus_missing_roots.py` - Python set

### Tools Created
- `/home/marc/klareco/scripts/extract_vocabulary_incremental.py` - Memory-efficient extraction tool

---

## Technical Details

### Incremental Processing Solution

**Problem:** Original `extract_literary_roots.py` exceeded memory limits (7.5GB RAM) on 551MB Wikipedia file.

**Solution:** Created `extract_vocabulary_incremental.py` that:
- Processes files one at a time
- Reads each file in 1MB chunks
- Aggregates root counts across all files
- Successfully processed 28 Wikipedia chunks without memory issues

**Performance:**
- Wikipedia (28 chunks): ~2-3 minutes total
- Literary (12 files): <30 seconds
- Memory usage: <2GB peak

### Root Extraction Algorithm

```
For each word in text:
  1. Extract word (Esperanto letter sequences)
  2. Strip accusative -n
  3. Strip plural -j
  4. Strip verb endings (as, is, os, us, u, i, participles)
  5. Strip noun/adjective/adverb endings (o, a, e)
  6. Strip prefixes (mal-, re-, etc.)
  7. Strip suffixes (ul, ej, in, et, etc.)
  8. Count frequency across all files
  9. Filter by minimum frequency threshold
```

---

## Next Steps

### For Parser Development

1. **Review and add roots from `analyze_failures.py`** (52 roots)
   - Already manually vetted from failed parses
   - High confidence these are legitimate Esperanto

2. **Manually vet top 200 literary roots** (~2 hours work)
   - Filter out English artifacts
   - Filter out proper names
   - Add legitimate roots to parser

3. **Re-test parser on literary corpus**
   - Measure improvement from root additions
   - Target: 60-70% success rate (currently 42%)

4. **Implement better text cleaning**
   - Wikipedia: Remove HTML/wiki markup more aggressively
   - Literary: Remove Gutenberg boilerplate completely
   - Re-run extraction on cleaner texts

### For Vocabulary Management

1. **Create vocabulary categories**:
   - `CORE_ROOTS` - Common Esperanto (current 33,819)
   - `LITERARY_ROOTS` - Fantasy/literary-specific
   - `TECHNICAL_ROOTS` - Scientific/technical terms
   - `PROPER_NAMES` - Characters, places (don't fail parse)
   - `COUNTRY_NAMES` - Esperantized country/language names

2. **Implement vocabulary loading system**:
   - Move from hardcoded sets to data files
   - Allow dynamic vocabulary expansion
   - Track provenance (where each root came from)

---

## Conclusion

**Incremental processing successfully extracted vocabulary from 555MB of Esperanto text** without memory issues. The Wikipedia extraction reveals need for better HTML/wiki cleaning, while literary extraction shows ~100 legitimate roots that should be added to the parser.

**Immediate next step:** Add the 52 manually-vetted roots from `analyze_failures.py` to the parser and re-test performance.

**Medium-term:** Improve text cleaning and re-run extraction for cleaner vocabulary lists.

**Long-term:** Implement automated root validation and vocabulary categorization system.
