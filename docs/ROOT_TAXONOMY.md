# Root Taxonomy Design

## Purpose

Classification system for all root/word types in the Kuzu database to support:
1. **Training data filtering** - Query for high-quality roots by tier
2. **Model training stages** - Core → Extended → Specialized vocabulary
3. **Linguistic correctness** - Distinguish lexical roots from grammatical morphemes
4. **Quality control** - Identify and exclude garbage/errors
5. **Special handling** - Proper nouns (NER), borrowed terms, scientific vocabulary

## Data Sources Available

| Source | Count | What It Provides |
|--------|-------|------------------|
| Unua Libro (1887) | ~917 roots | Original Esperanto foundation |
| Fundamento (1905) | 2,173 roots | Official base + extensions |
| ReVo dictionary | 23,237 roots | Technical/specialized vocabulary |
| Corpus validated | 76,529 roots | Successfully parsed words |
| Corpus frequency | usage counts | Frequency-based ranking |
| AST annotations | all words | Parse status, proper noun categories |
| Parse failures | 1,019,275 | Garbage/OCR errors/foreign words |
| Official affixes | 45 | Grammatical morphemes (prefixes/suffixes) |

## Classification Challenges

### 1. Esperanto Linguistic Elements

**Lexical roots** (carry semantic meaning):
- Original 917 roots: *hund, dom, bel, far, etc.*
- Extended vocabulary: technical terms, neologisms

**Grammatical morphemes** (compositional, no independent meaning):
- **Affixes**: mal-, re-, -iĝ, -ig, -aĵ, -ej (45 official)
- **Correlatives**: kio, kiu, kie, kiel, etc. (closed class, ~45 forms)
- **Function words**: kaj, de, en, la, mi, estas, etc. (closed class, ~100 words)

**Key question**: Should grammatical morphemes be separate Radiko nodes, or handled only in AST?
- If they're Radiko nodes → need special tier (don't train lexical embeddings on them)
- If they're NOT nodes → current architecture is correct

### 2. Proper Nouns vs Esperantized Names

**Categories from AST**:
- `propranoma_kategorio: 'person'` - Johano, Mario, Einstein-o
- `propranoma_kategorio: 'place'` - Parizo, Londono, Usono, Afriko
- `propranoma_kategorio: 'other'` - Organizations, brands, scientific names

**Usage for NER**:
- These are valuable for Named Entity Recognition
- Should be labeled separately from lexical roots
- May want to train separate embeddings for entity linking

### 3. Borrowed/International Terms

**Examples**:
- Scientific: vitamino, elektrono, mikroskopo
- Technology: komputilo, televido, telefono
- International: demokratio, filozofio, universitato

**Question**: Are these "real Esperanto" or "borrowed terms"?
- If in Fundamento/ReVo → treat as Esperanto
- If frequent in corpus but not in dictionaries → corpus-validated
- Some terms are in Esperanto dictionaries (adapted forms)

### 4. Neologisms (Internet Era)

**Examples**:
- retpoŝto (email), retejo (website), blogero (blogger)
- Often corpus-validated but not in ReVo (dictionary predates internet)

**Should have separate tier?**
- Useful for modern NLP tasks
- May want to train on these for contemporary text understanding
- Or lump into "corpus-validated"?

### 5. Garbage Types

**OCR errors**: 3xrt, hhh, xxx, random characters
**Parse failures**: Words that don't follow Esperanto morphology
**Corrupted text**: Truncated words, encoding errors
**Numbers/codes**: Mixed alphanumeric (unless valid Esperanto like tri, kvar)

**Question**: One garbage tier or subdivide?
- Subdividing might help identify OCR issues vs actual foreign words
- But maybe not worth the complexity?

## Proposed Taxonomy (V1)

### Tier 0: Grammatical Morphemes (Exclude from lexical training)

Total: **169 words across 10 subcategories**

- **tier0_afikso** (45): Official affixes (mal-, re-, -iĝ, -ig, -aĵ, -ej, etc.)
  - *Rationale*: Not lexical roots, compositional meaning only
  - *Training*: Exclude from root embeddings, handle deterministically

- **tier0_korelativo** (45): Table of correlatives (kio, kiu, kia, kie, tio, ĉio, etc.)
  - *Rationale*: Closed grammatical class, systematic compositional structure
  - *Training*: Exclude - compositional (prefix + suffix combinations)

- **tier0_pronomo** (9): Personal pronouns (mi, vi, li, ŝi, ĝi, ni, ili, oni, si)
  - *Rationale*: Closed class, referential function not semantic content
  - *Training*: Exclude from lexical embeddings

- **tier0_prepozicio** (30): Prepositions (de, al, en, sur, sub, kun, pri, etc.)
  - *Rationale*: Grammatical relation markers, closed class
  - *Training*: Exclude from lexical embeddings

- **tier0_konjunkcio** (7): Conjunctions (kaj, aŭ, sed, nek, ĉar, ke, se)
  - *Rationale*: Sentence connectors, closed class
  - *Training*: Exclude from lexical embeddings

- **tier0_partiklo** (25): Particles and interjections (ne, jes, ja, ajn, eĉ, ha!, ho!, etc.)
  - *Rationale*: Grammatical modifiers and expressive elements
  - *Training*: Exclude from lexical embeddings

- **tier0_nombro** (13): Cardinal numbers (unu, du, tri, ..., dek, cent, mil)
  - *Rationale*: Closed class for base numbers, compositional for larger
  - *Training*: May need special numeric embeddings, not standard lexical

- **tier0_finaco** (10): Grammatical endings as standalone words (a, e, i, o, u, as, is, os, us, j)
  - *Rationale*: Appear as words in grammar examples/technical text
  - *Training*: Exclude - not semantic roots
  - *Note*: These ARE Radiko nodes (403K+ Vorto links for "a")

- **tier0_artikolo** (1): The definite article (la)
  - *Rationale*: Only article in Esperanto, purely grammatical
  - *Training*: Exclude from lexical embeddings

- **tier0_komparativo** (4): Comparative markers (pli, plej, malpli, malplej)
  - *Rationale*: Grammatical degree markers
  - *Training*: Exclude from lexical embeddings

### Tier 1: Core Esperanto Vocabulary (High priority training)

- **tier1a_unua_libro**: Original 787 Unua Libro lexical roots (1887)
  - *Rationale*: THE foundation, smallest complete vocabulary
  - *Data source*: Extracted from Unua Libro LaTeX (excludes tier0 grammatical words)
  - *Training*: Priority 1 - ensure perfect embeddings for these
  - *Note*: "917 roots" cited historically includes ~130 grammatical words (now in tier0)

- **tier1b_fundamento**: Extended Fundamento roots (~1,200 additional)
  - *Rationale*: Official standard (1905), universally accepted
  - *Data source*: Fundamento - Unua Libro
  - *Training*: Priority 2 - complete official vocabulary

- **tier1c_ofteco_900**: Top 900 most frequent roots
  - *Rationale*: Frequency-based prioritization (may overlap with 1a/1b)
  - *Data source*: Corpus frequency ranking
  - *Training*: Use for curriculum learning (train frequent first)
  - *Note*: This is a SECONDARY LABEL, not mutually exclusive with 1a/1b

### Tier 2: Extended Esperanto Vocabulary (Legitimate, specialized)

- **tier2a_revo**: ReVo dictionary technical terms (~21K unique)
  - *Rationale*: Accepted technical/specialized vocabulary
  - *Data source*: ReVo - Fundamento
  - *Training*: Priority 3 - technical domains

- **tier2b_korpuso**: Corpus-validated roots (parsed successfully, frequent)
  - *Rationale*: Real Esperanto usage, may include neologisms
  - *Data source*: Successfully parsed (analizstato='sukceso'), not in above tiers
  - *Training*: Priority 4 - contemporary usage
  - *Filter*: Require minimum frequency (e.g., 5+ occurrences) to avoid noise

### Tier 3: Proper Entities (Named Entity Recognition)

- **tier3a_persono**: Person names
  - *Data source*: AST propranoma_kategorio='person'
  - *Training*: May train separate NER embeddings

- **tier3b_loko**: Place names
  - *Data source*: AST propranoma_kategorio='place'
  - *Training*: Geographic entity embeddings

- **tier3c_organizo**: Organizations, companies, brands
  - *Data source*: AST propranoma_kategorio='other' (subset)
  - *Training*: Entity linking

- **tier3d_alia_propranomo**: Other proper nouns
  - *Data source*: AST propranoma_kategorio='other' (remainder)

### Tier 4: Marginal (Use with caution)

- **tier4a_malofta**: Low-frequency corpus words (valid structure, rare)
  - *Rationale*: Possibly valid but too rare to be confident
  - *Data source*: Parsed successfully but frequency < 5
  - *Training*: Maybe exclude to reduce noise

- **tier4b_fremda**: Clear foreign words (not adapted to Esperanto)
  - *Rationale*: Borrowed without Esperanto morphology
  - *Data source*: Manual/heuristic detection (hard to automate)
  - *Training*: Probably exclude

### Tier 5: Garbage (Exclude from training)

- **tier5_rubaĵo**: Parse failures, OCR errors, corrupted text
  - *Rationale*: Not valid Esperanto
  - *Data source*: AST analizstato='malsukceso'
  - *Training*: EXCLUDE
  - *Note*: Might want to subdivide (OCR vs foreign vs corrupt) but maybe not worth it

### Tier 6: Unknown/Unclassified

- **tier6_nekonata**: Couldn't classify, needs review
  - *Rationale*: Fallback for edge cases
  - *Training*: EXCLUDE until manually reviewed

## Secondary Properties (Not mutually exclusive with tiers)

These properties coexist with tier labels to capture additional dimensions:

- **ofteco**: Corpus usage frequency (INT64)
  - *Use*: Curriculum learning, frequency-based sampling
  - *Example*: A tier1a root might have ofteco=50000

- **ofteco_rango**: Frequency rank (INT64, 1=most frequent)
  - *Use*: Query "top 900 roots" regardless of tier
  - *Example*: Top 900 might span tier1a+1b+2a

- **fonto**: Historical source provenance (STRING)
  - *Values*: 'unua_libro', 'fundamento', 'revo', 'korpuso', 'propranomo'
  - *Use*: Track vocabulary origins independent of grammatical role
  - *Example*: "mi" (pronoun) has fonto='unua_libro' AND nivelo='tier0_pronomo'
  - *Important*: Tier0 grammatical words ARE from Unua Libro - don't lose this!

- **jaro_unua_vido**: First year seen in corpus (INT64)
  - *Use*: Identify neologisms (appears only in modern texts)
  - *Example*: "retpoŝto" might have jaro_unua_vido=2000
  - *Note*: NULL for Wikipedia = "modern" (2024)

## Key Insight: Tier vs Source

**Tier** (nivelo) = Grammatical/semantic role → training decisions
- tier0_* = grammatical (exclude from lexical embeddings)
- tier1a = core lexical roots (priority training)
- tier3 = corpus-observed (lower priority)

**Source** (fonto) = Historical provenance → vocabulary origins
- unua_libro = foundational (787 roots + 169 grammatical words)
- fundamento = official extended (1905)
- korpuso = observed usage only

**Example:**
```
"mi" (I/me):
  nivelo: 'tier0_pronomo'        ← grammatical role (exclude from lexical training)
  fonto: 'unua_libro'             ← historical source (foundational word)
  ofteco: 128954                  ← usage frequency

"hundo" (dog):
  nivelo: 'tier1a_unua_libro'    ← core lexical root (priority training)
  fonto: 'unua_libro'             ← historical source (foundational word)
  ofteco: 1901                    ← usage frequency

"blogero" (blogger):
  nivelo: 'tier3_korpuso'        ← corpus-observed only
  fonto: 'korpuso'                ← not in official sources
  ofteco: 42                      ← usage frequency
  jaro_unua_vido: 2005            ← first seen in modern texts
```

**Query examples:**
```cypher
-- All Unua Libro words (lexical + grammatical)
MATCH (r:Radiko) WHERE r.fonto = 'unua_libro' RETURN r

-- Just Unua Libro lexical roots (for training)
MATCH (r:Radiko)
WHERE r.fonto = 'unua_libro' AND r.nivelo = 'tier1a_unua_libro'
RETURN r

-- All grammatical words from Unua Libro
MATCH (r:Radiko)
WHERE r.fonto = 'unua_libro' AND r.nivelo STARTS WITH 'tier0_'
RETURN r

-- Most frequent foundational words (any grammatical role)
MATCH (r:Radiko)
WHERE r.fonto = 'unua_libro'
RETURN r ORDER BY r.ofteco DESC LIMIT 100
```

## Open Questions

1. **Should correlatives/function words be Radiko nodes?**
   - Check database first
   - If yes, need tier0 labels
   - If no, current architecture is correct

2. **How to identify function words automatically?**
   - High frequency + short (≤3 chars) + in grammar rules?
   - Or hardcode official list (~100 words)?

3. **Subdivide garbage tier?**
   - tier5a_misparso, tier5b_okr, tier5c_korupto?
   - Or just tier5_rubaĵo (simpler)?

4. **Neologisms as separate tier?**
   - Or lump into tier2b_korpuso?
   - Hard to distinguish automatically

5. **Borrowed international terms - special handling?**
   - If in Fundamento/ReVo → treat as Esperanto (tier 1/2)
   - If not → tier2b_korpuso or tier4b_fremda?

6. **Should tier1c_ofteco_900 be separate or just use ofteco_rango property?**
   - Using property is cleaner (no overlap with tiers)
   - But might want explicit tier for training data queries

## Recommendation for Implementation

**Start simple, add complexity only if needed:**

1. ✅ Tier 0: Affixes (we have the list)
2. ✅ Tier 1a: Unua Libro roots (need to finish extraction)
3. ✅ Tier 1b: Fundamento extended (Fundamento - Unua Libro)
4. ✅ Tier 2a: ReVo (we have this)
5. ✅ Tier 2b: Corpus validated (parsed successfully, freq ≥ 5)
6. ✅ Tier 3: Proper nouns (use AST propranoma_kategorio)
7. ✅ Tier 5: Garbage (AST analizstato='malsukceso')
8. ✅ Tier 6: Unknown (everything else)

**Add later if needed:**
- Tier 0: Correlatives/function words (check if they're nodes first)
- Tier 1c: Frequency ranking (or just use ofteco_rango property)
- Tier 4: Marginal categories (rare words, foreign words)
- Subdivide garbage (OCR vs foreign vs corrupt)

**Secondary properties:**
- `ofteco`: usage count (always add this)
- `ofteco_rango`: frequency rank (optional, useful for queries)

What do you think? Should we:
1. Go with this simpler version first?
2. Check if correlatives/function words are actually Radiko nodes?
3. Finish extracting the Unua Libro roots?
4. Something else?
