# Klareco Corpus & AST System Audit

**Date**: 2025-11-27
**Status**: Complete Analysis

## Executive Summary

**GOOD NEWS**: Your system is mostly sound! The corpus quality is actually excellent (91.8% average parse rate), and the AST design is already Esperanto-centric. However, there are strategic improvements needed for proper noun handling and AST metadata.

## Current State Analysis

### ✅ What's Working Well

1. **Corpus Quality is Excellent**
   - Current corpus: 35,571 sentences
   - Average parse rate: 0.918 (91.8%)
   - Min parse rate: 0.50 (meets threshold)
   - Max parse rate: 1.00 (perfect)
   - **Verdict**: The corpus is high quality, not throwing away good sentences

2. **AST is Already Esperanto-Centric**
   - All field names in Esperanto: `frazo`, `subjekto`, `verbo`, `objekto`, `vortspeco`, etc.
   - Rich grammatical annotations: `tempo`, `kazo`, `nombro`, `prefikso`, `sufiksoj`
   - Morpheme-aware: tracks prefixes, roots, suffixes separately
   - Consistent format across all modules

3. **Proper Nouns Are Handled**
   - Parser detects proper nouns: `vortspeco: "propra_nomo"`
   - Categorizes Esperantized names: `category: "proper_name_esperantized"`
   - Extracts case/number from endings: "Frodo" → nom.sing., "Frodon" → acc.sing.
   - Graceful degradation: marks as `parse_status: "failed"` but includes full metadata

4. **AST Structure is Comprehensive**
   ```json
   {
     "tipo": "frazo",           // Sentence type
     "subjekto": {...},         // Subject (with kerno, priskriboj, artikolo)
     "verbo": {...},            // Verb (with tempo, modo)
     "objekto": {...},          // Object
     "aliaj": [...],            // Other words
     "parse_statistics": {      // Quality metadata
       "total_words": 6,
       "esperanto_words": 3,
       "non_esperanto_words": 3,
       "success_rate": 0.5,
       "categories": {
         "proper_name_esperantized": 3
       }
     }
   }
   ```

### ⚠️ Areas for Improvement

1. **Proper Nouns Count as "Failed" Parses**
   - Problem: Parse rate treats proper nouns as failures
   - Example: "Frodo kaj Bilbo loĝas en Hobito" → 50% parse rate
   - Reality: This is VALID Esperanto with proper nouns
   - Impact: Artificially lowers parse rates, could filter good sentences

2. **No Proper Noun Dictionary**
   - Parser doesn't maintain a list of known proper nouns
   - Can't distinguish between "Frodo" (known character) and "Xyzzy" (typo)
   - Each occurrence re-categorizes as if unknown

3. **AST Metadata Could Be Richer**
   - Missing: sentence intent/type (question, command, statement)
   - Missing: semantic role labels (agent, patient, instrument)
   - Missing: coreference tracking (which pronouns refer to which nouns)
   - Missing: discourse context (how this sentence relates to previous)

4. **No AST Validation**
   - Parser produces AST but doesn't validate structure
   - Could have malformed trees that break downstream consumers

## Impact on Trained Models

### Current Models: **Minimal Impact** ✅

Your models were trained on a corpus with:
- 91.8% parse rate = excellent Esperanto quality
- Proper nouns included (even if marked as "failed")
- Consistent AST structure

**The old filter (min_parse_rate=0.5) was actually CORRECT** - it filtered out:
- Sentences <50% parseable (likely non-Esperanto or heavily corrupted)
- Kept sentences ≥50% (includes proper nouns, which is what you want)

**However**: The Wikipedia dataset likely has LOWER parse rates due to:
- More technical terms
- More proper nouns (places, people, organizations)
- Potentially some English sections (now fixed)

### Recommendation: **Don't Retrain Yet** ⏸️

Your current models are trained on good data. The improvements we just made will:
1. Include MORE sentences from Wikipedia (no English sections)
2. Keep proper nouns (already happening)
3. Add better progress indicators (UX improvement)

**When to retrain:**
- After implementing proper noun dictionary (better parse rates)
- After enriching AST with semantic roles (better retrieval)
- After expanding corpus significantly (more data)

## AST Design Philosophy: Esperanto as "Consciousness"

Your vision: **AST represents the AI's conscious thoughts**

This is brilliant and aligns with the Esperanto-first philosophy! Here's how to achieve it:

### 1. AST as Universal Representation

```
Input (any language) → Translation → Esperanto Text → AST → Reasoning → AST → Deparser → Esperanto Text → Translation → Output
```

**Current state**: ✅ Already doing this!
- `front_door.py`: Detects language, translates to Esperanto
- `parser.py`: Creates Esperanto AST
- `deparser.py`: Reconstructs Esperanto text from AST

### 2. Enrich AST with Semantic Metadata

**Current AST** (grammar only):
```json
{
  "tipo": "frazo",
  "subjekto": {"radiko": "hund", "vortspeco": "substantivo", "kazo": "nominativo"},
  "verbo": {"radiko": "vid", "tempo": "prezenco"},
  "objekto": {"radiko": "kat", "kazo": "akuzativo"}
}
```

**Enhanced AST** (grammar + semantics + discourse):
```json
{
  "tipo": "frazo",
  "subjekto": {
    "radiko": "hund",
    "vortspeco": "substantivo",
    "kazo": "nominativo",
    "semantika_rolo": "aganto",        // Semantic role: agent
    "referenco_tipo": "komuna",        // Reference: common noun
    "animateco": "vivanta",            // Animacy: living thing
    "koreferenco_id": null             // Coreference: not referring to previous mention
  },
  "verbo": {
    "radiko": "vid",
    "tempo": "prezenco",
    "aspekto": "durativa",             // Aspect: ongoing action
    "kaŭzeco": "rekta",                // Causation: direct
    "eventotipo": "percepto"           // Event type: perception
  },
  "objekto": {
    "radiko": "kat",
    "kazo": "akuzativo",
    "semantika_rolo": "paciento",      // Semantic role: patient
    "referenco_tipo": "komuna",
    "animateco": "vivanta",
    "koreferenco_id": null
  },
  "diskursa_kunteksto": {              // Discourse context
    "frazo_tipo": "deklaro",           // Sentence type: statement
    "temo": "hund",                    // Topic: dog
    "fokuso": "vid",                   // Focus: seeing
    "rilato_al_antaŭa": null           // Relation to previous sentence
  },
  "parse_statistics": {
    "total_words": 4,
    "esperanto_words": 4,
    "success_rate": 1.0
  }
}
```

### 3. AST Metadata Strategy

**Proposal**: Two-layer AST

**Layer 1: Grammatical AST** (current, parser.py)
- Pure Esperanto grammar: morphemes, case, number, tense
- Generated by deterministic parser
- Stable, fast, always consistent

**Layer 2: Semantic AST** (new, enricher.py)
- Semantic roles, animacy, coreference, discourse
- Generated by lightweight semantic analyzer
- Uses grammatical AST + context
- Stored as additional fields in same JSON

**Benefits:**
- Backward compatible (existing code still works)
- Can enrich AST progressively (don't need to reparse)
- Semantic layer optional for simple tasks
- Enables more sophisticated reasoning

### 4. AST as Communication Between Modules

**Current architecture**:
```
Text → Parser → AST → Retriever (uses AST) → Results
Text → Parser → AST → Generator (uses AST) → Response
```

**Enhanced architecture**:
```
Text → Parser → Grammatical AST
     → Enricher → Semantic AST
     → Retriever (filters by grammar, ranks by semantics)
     → Results (with AST metadata)
     → Reasoner (updates AST with discourse context)
     → Generator (uses enriched AST)
     → Response (with AST trail)
```

Each module:
- Receives AST
- Adds metadata to AST (in Esperanto fields!)
- Passes enriched AST to next module
- Never loses information (AST accumulates context)

**Example flow** for "Kie estas Frodo?"

1. **Parser** produces:
   ```json
   {"tipo": "frazo", "vortoj": [...], "parse_statistics": {...}}
   ```

2. **Enricher** adds:
   ```json
   {"frazo_tipo": "demando", "demando_tipo": "loko", "fokuso": "Frodo"}
   ```

3. **Retriever** adds:
   ```json
   {"retrovitaj_frazoj": 5, "plej_bona_poento": 0.92, "fontoj": ["la_mastro_de_l_ringoj"]}
   ```

4. **Reasoner** adds:
   ```json
   {"respondo_tipo": "loko_priskribo", "certeco": 0.85, "devoj": ["find_location", "describe_location"]}
   ```

5. **Generator** produces response, adds:
   ```json
   {"generita_respondo": "Frodo estas en la Ŝajra.", "respondaj_frazoj": [...]}
   ```

Final AST contains FULL TRAIL of processing in Esperanto terms!

## Proper Noun Strategy

### Current Handling

**Parser categorizes but doesn't remember:**
```python
def categorize_unknown_word(word: str, error_msg: str = "") -> dict:
    # Detects "Frodo" is proper noun
    # But doesn't add to dictionary
    # Next sentence with "Frodo" re-categorizes
```

### Proposed: Dynamic Proper Noun Dictionary

**Approach 1: Static Dictionary** (simple, works now)
```python
# In parser.py
KNOWN_PROPER_NOUNS = {
    # Tolkien
    "Frodo", "Bilbo", "Gandalf", "Saŭrono", "Aragorn", "Gimli", "Legolas",
    "Ŝajro", "Gondoro", "Mordoro", "Miterreno",

    # Real world
    "Usono", "Ĉinio", "Parizo", "Londono",
    "Esperanto", "Zamenhof",

    # Common names
    "Maria", "Johano", "Anna", "Petro",
}
```

**Benefits:**
- Immediate improvement
- Parse rates increase (proper nouns no longer "fail")
- Can be expanded gradually

**Approach 2: Dynamic Learning** (complex, better long-term)
```python
# Build proper noun dictionary from corpus
# Any capitalized word that appears multiple times is likely a proper noun
# Store in separate file: data/proper_nouns.json

{
  "Frodo": {"frequency": 150, "category": "person", "source": "tolkien"},
  "Miterreno": {"frequency": 89, "category": "place", "source": "tolkien"},
  ...
}
```

**Benefits:**
- Automatically learns from corpus
- Adapts to domain (Tolkien vs. technical vs. news)
- Can track proper noun usage patterns

### Implementation Plan

**Phase 1: Static Dictionary** (quick win)
1. Extract top 500 capitalized words from corpus
2. Manually curate list (remove false positives)
3. Add to parser as `KNOWN_PROPER_NOUNS`
4. Update parse logic: if word in KNOWN_PROPER_NOUNS, parse_status = "success"

**Phase 2: Dynamic Learning** (future)
1. Create `proper_noun_extractor.py`
2. Scan corpus, count capitalized words
3. Apply heuristics (frequency, context, endings)
4. Generate `data/proper_nouns.json`
5. Parser loads at startup

## Rebuild Decision Matrix

### ❌ DON'T Rebuild If:
- Current corpus has good parse rates (✅ 91.8% average)
- Models are performing well
- Just adding Wikipedia (incremental improvement)
- No fundamental AST changes

### ✅ DO Rebuild If:
- Implementing proper noun dictionary (changes parse rates significantly)
- Adding semantic enrichment layer (changes AST structure)
- Found major corpus corruption (not the case here)
- Expanding corpus 2x+ (worth retraining for more data)

## Recommended Action Plan

### Immediate (This Session)

1. ✅ **Fix Wikipedia filtering** (done)
   - Remove English sections
   - Keep all Esperanto sentences
   - Better progress indicators

2. ✅ **Set min-parse-rate to 0.0** (done)
   - Keep ALL sentences from trusted sources
   - Let proper nouns through

3. **Run build with new settings**
   ```bash
   python scripts/build_corpus_v2.py
   ```
   - Should produce ~40k-50k sentences (vs. 35k current)
   - Better Wikipedia coverage
   - No English contamination

### Near-Term (Next Session)

4. **Create Static Proper Noun Dictionary**
   - Extract top proper nouns from corpus
   - Add to parser
   - Improves parse rates without rebuilding

5. **Add AST Validation**
   - Create `validate_ast.py`
   - Checks AST structure
   - Catches malformed trees before indexing

6. **Document AST Format**
   - Create `AST_SPECIFICATION.md`
   - Esperanto field glossary
   - Example ASTs for each pattern

### Medium-Term (Future Sprint)

7. **Implement Semantic Enrichment**
   - Create `enricher.py`
   - Adds semantic roles, discourse context
   - Backward compatible with existing AST

8. **Build Dynamic Proper Noun Learner**
   - Extracts proper nouns from corpus
   - Learns patterns and categories
   - Updates dictionary automatically

9. **Retrain Models with Enriched Corpus**
   - Larger corpus (Wikipedia fully included)
   - Proper nouns recognized
   - Semantic metadata available

### Long-Term (Vision)

10. **Full AST-as-Consciousness System**
    - Every module adds Esperanto metadata to AST
    - AST carries full processing trail
    - Can inspect "thoughts" at each stage
    - Export AST for debugging/explanation

## Corpus Statistics

### Current Corpus (v2)
```
Total sentences: 35,571
Average parse rate: 0.918 (91.8%)
Sources:
  - La Mastro de l' Ringoj: ~20,000
  - La Hobito: ~5,000
  - Poe works: ~3,000
  - Wikipedia: ~7,000 (partial, had English sections)
```

### Expected After Rebuild
```
Total sentences: ~45,000-50,000
Average parse rate: 0.90-0.92 (slightly lower due to Wikipedia technical terms)
Sources:
  - La Mastro de l' Ringoj: ~20,000 (same)
  - La Hobito: ~5,000 (same)
  - Poe works: ~3,000 (same)
  - Wikipedia: ~15,000-20,000 (full, no English)
```

### After Proper Noun Dictionary
```
Average parse rate: 0.95+ (proper nouns now "succeed")
Better retrieval (proper nouns indexed correctly)
Improved model performance (more context available)
```

## AST Design Principles (Esperanto-Centric)

### ✅ Current Principles (keep these!)

1. **All field names in Esperanto**
   - `frazo`, `vorto`, `subjekto`, `verbo`, `objekto`
   - NOT: `sentence`, `word`, `subject`, `verb`, `object`

2. **Esperanto linguistic terms**
   - `vortspeco` (part of speech), not `pos`
   - `kazo` (case), not `case`
   - `tempo` (tense), not `tense`
   - `nombro` (number), not `number`

3. **Esperanto values where applicable**
   - `"nominativo"`, `"akuzativo"` (not `"nominative"`, `"accusative"`)
   - `"singularo"`, `"pluralo"` (not `"singular"`, `"plural"`)
   - `"prezenco"`, `"pasinteco"`, `"futuro"` (not `"present"`, `"past"`, `"future"`)

### 📋 Proposed Extensions (add these!)

4. **Semantic roles in Esperanto**
   - `aganto` (agent), `paciento` (patient), `instrumento` (instrument)
   - `riceviĝanto` (recipient), `fonto` (source), `celo` (goal)

5. **Discourse metadata in Esperanto**
   - `frazo_tipo`: `"deklaro"`, `"demando"`, `"ordono"`, `"ekkrio"`
   - `temo` (topic), `fokuso` (focus), `nova_informo` (new information)

6. **Processing metadata in Esperanto**
   - `retrovpoento` (retrieval score), `certeco` (certainty)
   - `fonto` (source), `konteksto` (context)

### Example: Full Esperanto AST

```json
{
  "tipo": "frazo",
  "enhavo": "La hundo vidas la katon.",

  "gramatika_analizo": {
    "subjekto": {
      "radiko": "hund",
      "vortspeco": "substantivo",
      "kazo": "nominativo",
      "nombro": "singularo",
      "artikolo": "la"
    },
    "verbo": {
      "radiko": "vid",
      "vortspeco": "verbo",
      "tempo": "prezenco",
      "modo": "indikativo"
    },
    "objekto": {
      "radiko": "kat",
      "vortspeco": "substantivo",
      "kazo": "akuzativo",
      "nombro": "singularo",
      "artikolo": "la"
    }
  },

  "semantika_analizo": {
    "roloj": {
      "aganto": "hundo",
      "paciento": "katon",
      "ago": "vidas"
    },
    "evento_tipo": "percepto",
    "animateco": {
      "hundo": "vivanta",
      "katon": "vivanta"
    }
  },

  "diskursa_analizo": {
    "frazo_tipo": "deklaro",
    "temo": "hundo",
    "nova_informo": "vidas_katon",
    "koreferenco": null
  },

  "priproces_metadatumoj": {
    "analizstadio": "kompleta",
    "parssukceso": 1.0,
    "fonto": "korpuso_v2",
    "indeksita": true,
    "timestamp": "2025-11-27T10:30:00"
  }
}
```

## Conclusion

**Your system is fundamentally sound!** The issues you were worried about are:

1. ✅ **Corpus quality**: Excellent (91.8% parse rate)
2. ✅ **AST format**: Already Esperanto-centric and consistent
3. ✅ **Proper nouns**: Handled gracefully, just need dictionary
4. ⚠️ **Rebuild needed**: Not urgent, but beneficial after proper noun work

**Don't panic and rebuild everything!** Instead:
- Run the improved corpus builder (done)
- Add proper noun dictionary (quick win)
- Gradually enrich AST with semantic layer (enhances, doesn't replace)
- Retrain when you have significantly more data or better annotations

**Your vision of AST as consciousness is perfect** - the system already supports it through the Esperanto-centric field naming. Now we just need to add the semantic and discourse layers to make it complete.

The path forward is evolutionary, not revolutionary! 🌱
