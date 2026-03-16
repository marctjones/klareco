# Implementation Status Assessment - 2026-03-09

**Question**: Do we know what models we need? What each model should do? What training data we need?

**Answer**: YES - With one critical gap identified below.

## Executive Summary

### What We Have ✅

1. **Complete model taxonomy** (9 models, deterministic-first architecture)
2. **Clear interfaces** (ASTAnnotator protocol with tensor support)
3. **Versioning system** (prevents chaos as system evolves)
4. **CLI design** (unified workflow for data → training → evaluation)
5. **Alignment with Klareco purpose** (maximize deterministic, minimize learned)

### Critical Gap Identified ⚠️

**Training data generation scripts are NOT designed yet!**

We know:
- ✅ What data each model needs (conceptually)
- ✅ Where data comes from (v2.1 database)
- ❌ **HOW to extract/format data** (no detailed scripts yet)

**This is the next critical step** - Epic #641 (Data Pipeline).

## Detailed Status by Model

### M0: Parser (Deterministic) ✅

**Status**: COMPLETE - Production ready
**Purpose**: Parse Esperanto text → AST with deterministic features
**Alignment**: 100% - This IS the deterministic foundation

| Question | Answer |
|----------|--------|
| What does it do? | ✅ Parses Esperanto using 16 grammar rules → AST |
| Interface defined? | ✅ Returns AST dict with kazo/nombro/genro/tempo |
| Training data needed? | N/A (rule-based) |
| Aligned with purpose? | ✅ YES - Core deterministic component |

**Implementation**: `klareco/parser.py` (complete)

---

### Stage 1: Root Embeddings (Learned) 🚧

**Status**: READY TO IMPLEMENT - Needs retraining
**Purpose**: Learn semantic representations for content word roots (tier1a+1b+2)
**Alignment**: Good - Minimal learning (64d, ~9,800 roots)

| Question | Answer |
|----------|--------|
| What does it do? | ✅ Maps root words → 64d semantic vectors |
| Interface defined? | ✅ RootEmbeddingsAnnotator (ASTAnnotator) |
| Training data needed? | ⚠️ **PARTIALLY** - Know what (tier1a+1b+2 roots), not HOW to extract |
| Aligned with purpose? | ✅ YES - Minimal learned (only content words, 320K params) |

**Training Data** (Epic #616, Issue #617):
- **What**: Tier1a+1b+2 roots (~9,800 vocabulary)
- **Exclude**: Tier0 function words (187 roots), Tier5 parse failures
- **Format**: JSON list of roots with frequency
- **Source**: v2.1 database (Radiko nodes WHERE nivelo IN ['1a','1b','2'])
- **Size**: ~6.8 MB (76,352 roots including tier3, but we'll filter to 9.8K)

**Gap**: No script yet for extracting this data!

**What We Need**:
```python
# scripts/data/export_roots_v2.1.py (NOT CREATED YET)
"""
Export Root Vocabulary for Training

Query v2.1 database:
  MATCH (r:Radiko)
  WHERE r.nivelo IN ['1a', '1b', '2']
  RETURN r.radiko, r.ofteco, r.fonto, r.nivelo

Output: data/training/roots_tier_filtered.json
Format: [{"radiko": "hund", "ofteco": 1234, "nivelo": "2", "fonto": "korpuso"}, ...]

Validation:
  - No tier0 words (mi, kaj, la, de)
  - All Fundamento roots present (1,403 tier1a+1b)
  - Vocabulary size: ~9,800
"""
```

---

### M1: Selectional Preference (Learned) 🚧

**Status**: NEEDS REDESIGN - Must use ASTAnnotator pattern
**Purpose**: Score plausibility of (subject, verb, object) triples
**Alignment**: Good - Focused learning (plausibility, not grammar)

| Question | Answer |
|----------|--------|
| What does it do? | ✅ Predicts: Is (hund, vid, kat) plausible? → 0.87 |
| Interface defined? | ✅ M1SelectionalAnnotator (reads root_embedding, outputs M1_plausibility) |
| Training data needed? | ⚠️ **PARTIALLY** - Know what (S-V-O triples), not HOW to extract |
| Aligned with purpose? | ✅ YES - Learns semantics, not grammar (grammar from AST) |

**Training Data** (Epic #616, Issue #621):
- **What**: (subject_root, verb_root, object_root) triples with labels
- **Format**: JSONL with {"subject": "hund", "verb": "vid", "object": "kat", "label": 1}
- **Source**: v2.1 database (Frazo → Vorto → Radiko)
- **Positive examples**: Real triples from corpus
- **Negative examples**: Synthetic (random shuffling) or low-frequency
- **Size**: ~10M triples (estimate)

**Gap**: No script yet for extracting triples!

**What We Need**:
```python
# scripts/data/export_m1_triples_v2.1.py (NOT CREATED YET)
"""
Export M1 Selectional Preference Training Data

Query v2.1 database:
  MATCH (s_word:Vorto)-[:EN]->(frazo:Frazoteksto)
  MATCH (s_word)-[:APERIS_EN]->(s_root:Radiko)
  MATCH (v_word:Vorto)-[:EN]->(frazo)
  MATCH (v_word)-[:APERIS_EN]->(v_root:Radiko)
  MATCH (o_word:Vorto)-[:EN]->(frazo)
  MATCH (o_word)-[:APERIS_EN]->(o_root:Radiko)
  WHERE s_word.role = 'subjekto'
    AND v_word.role = 'verbo'
    AND o_word.role = 'objekto'
  RETURN s_root.radiko, v_root.radiko, o_root.radiko

Generate negatives by shuffling

Output: data/training/m1_triples_v2.1.jsonl
Format: {"subject": "hund", "verb": "vid", "object": "kat", "label": 1}

Validation:
  - Balance positive/negative (50/50)
  - No tier0 words in roots
  - Frequency distribution matches corpus
"""
```

---

### M2.1: Taxonomy (Mostly Deterministic) 🚧

**Status**: NEEDS DESIGN - 90% deterministic approach
**Purpose**: Classify entities into taxonomy (PERSONO, ANIMALO, OBJEKTO, etc.)
**Alignment**: EXCELLENT - 90% deterministic, 10% learned fallback

| Question | Answer |
|----------|--------|
| What does it do? | ✅ Assigns hypernyms: hund → ANIMALO → BESTO |
| Interface defined? | ✅ DeterministicAnnotator (90% rules, 10% fallback) |
| Training data needed? | ⚠️ **UNCLEAR** - Deterministic rules use ReVo/ConceptNet, but fallback model needs data |
| Aligned with purpose? | ✅ EXCELLENT - Maximizes deterministic! |

**Deterministic Sources** (90% of cases):
- **ReVo definitions**: "hundo: Dombesto de familio kanedoj"
- **ConceptNet IS-A relations**: hund → animalo → besto
- **Affix rules**: -ist → PERSONO, -ej → LOKO

**Learned Fallback** (10% of cases):
- **OOV words**: Words not in ReVo/ConceptNet
- **Ambiguous**: Words with multiple meanings

**Gap**: No design for deterministic rule system!

**What We Need**:
```python
# klareco/models/taxonomy_deterministic.py (NOT CREATED YET)
"""
Deterministic Taxonomy Classifier

Rule 1: Check ReVo definition
  - "hundo: Dombesto..." → ANIMALO

Rule 2: Check ConceptNet
  - Query: (hund, IsA, ?)
  - Result: animalo → ANIMALO

Rule 3: Check affix rules
  - If ends with -ist: PERSONO
  - If ends with -ej: LOKO
  - If ends with -il: OBJEKTO

Fallback: Use learned classifier for OOV
"""
```

---

### M2.2: Coreference (Mostly Deterministic) 🚧

**Status**: NEEDS DESIGN - 80% deterministic approach
**Purpose**: Resolve pronouns to entities (li → Zamenhof)
**Alignment**: EXCELLENT - 80% deterministic, 20% learned disambiguation

| Question | Answer |
|----------|--------|
| What does it do? | ✅ Resolves: "Zamenhof... Li..." → li refers to Zamenhof |
| Interface defined? | ✅ DeterministicAnnotator (80% grammar, 20% disambiguation) |
| Training data needed? | ⚠️ **UNCLEAR** - Deterministic uses grammar, but disambiguation model needs data |
| Aligned with purpose? | ✅ EXCELLENT - Maximizes deterministic (grammar from AST)! |

**Deterministic Rules** (80% of cases):
- **Gender matching**: li=masc, ŝi=fem, ĝi=neut (from AST!)
- **Number matching**: Plural -j (from AST!)
- **Case matching**: Nominative/accusative (from AST!)
- **Recency**: Most recent entity of matching type

**Example**:
```
"Zamenhof estis viro. Li fondis Esperanton."
        ↓
Pronoun: li (masculine, singular, nominative)
Candidates: [Zamenhof (masculine, singular, nominative)]
Result: Unambiguous! No learning needed!
```

**Learned Disambiguation** (20% of cases):
- Multiple candidates match
- Need semantic similarity
- Need discourse coherence

**Gap**: No design for deterministic rule system!

**What We Need**:
```python
# klareco/models/coreference_deterministic.py (NOT CREATED YET)
"""
Deterministic Coreference Resolution

Rule 1: Extract pronoun features (from AST!)
  - Gender: li/ŝi/ĝi
  - Number: -j or not
  - Case: -n or not

Rule 2: Find candidates with matching features
  - Filter entities by gender/number/case

Rule 3: Apply recency heuristic
  - Most recent entity wins

Rule 4: If still ambiguous, use learned model
"""
```

---

### Entity Classifier (Mostly Deterministic) 🚧

**Status**: NEEDS DESIGN - Tier1-2 deterministic, Tier3 learned
**Purpose**: Classify entities by semantic type
**Alignment**: GOOD - Leverages tier structure

| Question | Answer |
|----------|--------|
| What does it do? | ✅ Classifies: Is this PERSONO/ANIMALO/OBJEKTO/LOKO? |
| Interface defined? | ✅ EntityAnnotator (tier-based classification) |
| Training data needed? | ⚠️ **PARTIALLY** - Tier3 learned classifier needs labels |
| Aligned with purpose? | ✅ YES - Deterministic where possible |

**Tier1-2** (Deterministic):
- Use ReVo/Fundamento definitions
- Use affix rules (-ist, -ej, -il)
- Use ConceptNet

**Tier3** (Learned):
- Corpus-specific words
- Need learned classifier

**Gap**: No design for tier-based classifier!

---

### Reranker (Learned) 🚧

**Status**: NEEDS REDESIGN - Must use ASTAnnotator pattern
**Purpose**: Rerank retrieval results for relevance
**Alignment**: Good - Focused learning (relevance, not grammar)

| Question | Answer |
|----------|--------|
| What does it do? | ✅ Scores: How relevant is document to query? |
| Interface defined? | ✅ RerankerAnnotator (reads AST, outputs relevance_score) |
| Training data needed? | ⚠️ **PARTIALLY** - Know what (query-doc pairs), not HOW to extract |
| Aligned with purpose? | ✅ YES - Learns relevance, grammar from AST |

**Training Data** (Epic #616, Issue #629):
- **What**: (query, document, relevance_label) triples
- **Format**: JSONL
- **Source**: Can we generate from corpus? Or need manual labels?
- **Size**: Unknown

**Gap**: How to generate query-document pairs with labels?

**Options**:
1. **Synthetic**: Generate queries from sentences, label documents
2. **Manual**: Hand-label query-document pairs
3. **Weak supervision**: Use M1 scores as proxy labels

---

## Critical Gaps Summary

### 1. Training Data Extraction Scripts ⚠️

**Status**: NOT DESIGNED YET

We need to create scripts for Epic #641 (Data Pipeline):

| Model | Script Needed | Status |
|-------|---------------|--------|
| Root Embeddings | `data/export_roots_v2.1.py` | ❌ Not created |
| M1 Selectional | `data/export_m1_triples_v2.1.py` | ❌ Not created |
| Entity Classifier | `data/export_entity_labels_v2.1.py` | ❌ Not created |
| Reranker | `data/export_reranker_pairs_v2.1.py` | ❌ Not created |

**Why Critical**: Can't train models without training data!

### 2. Deterministic Rule Systems ⚠️

**Status**: CONCEPTUALLY DESIGNED, NOT IMPLEMENTED

We designed 90%/80% deterministic models but haven't implemented the rule systems:

| Model | Rule System Needed | Status |
|-------|-------------------|--------|
| M2.1 Taxonomy | ReVo + ConceptNet + Affix rules | ❌ Not implemented |
| M2.2 Coreference | Grammar matching + Recency | ❌ Not implemented |
| Entity Classifier | Tier1-2 deterministic classifier | ❌ Not implemented |

**Why Important**: This is THE core Klareco advantage!

### 3. Inspection Tools ⚠️

**Status**: DESIGNED, NOT IMPLEMENTED

Epic #637 (CLI Inspection Tools) is critical for development but not implemented:

| Tool | Purpose | Status |
|------|---------|--------|
| `klareco inspect ast` | Debug parser output | ❌ Not implemented |
| `klareco inspect annotations` | See all annotations | ❌ Not implemented |
| `klareco inspect tensor` | Decode embeddings | ❌ Not implemented |

**Why Blocking**: Can't debug models without inspection tools!

## Alignment with Klareco Purpose

Let me assess each model against Klareco's core principles:

### Core Principles (from VISION.md)

1. **Maximize explainability** - Every prediction traceable to sources
2. **Leverage ASTs** - Grammar/structure deterministic
3. **Maximize Esperanto advantages** - Regular grammar, compositional morphology
4. **Minimize learned parameters** - Focus learned capacity on reasoning, not grammar

### Alignment Assessment

| Model | Learned % | Explainable? | Leverages AST? | Aligns with Purpose? |
|-------|-----------|--------------|----------------|---------------------|
| M0 Parser | 0% | ✅ YES (rules) | ✅ YES (produces AST) | ✅ PERFECT |
| Root Embeddings | 100% (320K params) | ⚠️ Partial (can decode to similar words) | ✅ YES (reads radiko) | ✅ GOOD (minimal learning) |
| M1 Selectional | 100% (~500K params) | ⚠️ Partial (can show similar triples) | ✅ YES (reads AST structure) | ✅ GOOD (learns semantics only) |
| M2.1 Taxonomy | 10% | ✅ EXCELLENT (90% rules) | ✅ YES (uses AST) | ✅ EXCELLENT (maximizes deterministic) |
| M2.2 Coreference | 20% | ✅ EXCELLENT (80% rules) | ✅ YES (reads gender/number/case from AST) | ✅ EXCELLENT (maximizes Esperanto advantages) |
| Entity Classifier | ~30% (Tier3 only) | ✅ GOOD (Tier1-2 deterministic) | ✅ YES (uses AST) | ✅ GOOD (tier-based approach) |
| Reranker | 100% (~1M params) | ⚠️ Partial (can explain features) | ✅ YES (reads AST) | ✅ ACCEPTABLE (relevance requires learning) |

**Overall Alignment**: ✅ **EXCELLENT**

Total learned parameters: ~2.3M (mostly in reranker)
Total deterministic: M2.1 (90%), M2.2 (80%), Entity (70% Tier1-2)

**This is exactly what Klareco aims for!** Most capacity is deterministic, learned models focus on semantic reasoning.

## What We Need Next (Priority Order)

### Phase 0: Foundation (CRITICAL - BLOCKING)

1. **Epic #637: Inspection Tools** - Can't develop without debugging
   - [ ] Issue #638: `klareco inspect ast`
   - [ ] Issue #639: `klareco inspect tensor`
   - [ ] Implement `klareco inspect annotations`
   - **Timeline**: 1 week
   - **Blocks**: All other development

### Phase 1: Data Pipeline (CRITICAL)

2. **Epic #641: Data Export Scripts** - Can't train without data
   - [ ] Create `scripts/data/export_roots_v2.1.py`
   - [ ] Create `scripts/data/export_m1_triples_v2.1.py`
   - [ ] Create data validation scripts
   - **Timeline**: 1 week
   - **Blocks**: All model training

3. **Design Deterministic Rule Systems** - Core Klareco advantage
   - [ ] Design ReVo + ConceptNet taxonomy loader
   - [ ] Design grammar-based coreference resolver
   - [ ] Design tier-based entity classifier
   - **Timeline**: 1 week
   - **Blocks**: M2.1, M2.2, Entity training

### Phase 2: Model Training (DEPENDS ON PHASE 1)

4. **Epic #616, Phase 1: Root Embeddings**
   - [ ] #617: Export root training data
   - [ ] #618: Train root embeddings v3
   - [ ] #619: Test quality (no collapse)
   - [ ] #620: Integration
   - **Timeline**: 1 week
   - **Blocks**: All downstream models

5. **Epic #616, Phase 2: M1 Selectional**
   - [ ] #621: Export M1 training data
   - [ ] #622: Train M1 with frozen roots
   - [ ] #623: Test accuracy
   - [ ] #624: Integration
   - **Timeline**: 1 week
   - **Depends**: Root embeddings complete

### Phase 3: Deterministic Models (PARALLEL WITH PHASE 2)

6. **Implement M2.1 Taxonomy Deterministic**
   - [ ] Load ReVo definitions
   - [ ] Load ConceptNet relations
   - [ ] Implement affix rules
   - [ ] Train fallback model for OOV
   - **Timeline**: 2 weeks
   - **Can start**: After inspection tools ready

7. **Implement M2.2 Coreference Deterministic**
   - [ ] Implement grammar matching
   - [ ] Implement recency heuristics
   - [ ] Train disambiguation model
   - **Timeline**: 2 weeks
   - **Can start**: After inspection tools ready

## Status Summary Table

| Component | Design Complete? | Implementation Complete? | Aligned with Purpose? | Blocking? |
|-----------|------------------|--------------------------|----------------------|-----------|
| **Models** |
| M0 Parser | ✅ YES | ✅ YES | ✅ YES | No |
| Root Embeddings | ✅ YES | ❌ Needs retrain | ✅ YES | Yes (Phase 1) |
| M1 Selectional | ✅ YES | ❌ Needs redesign | ✅ YES | No |
| M2.1 Taxonomy | ⚠️ Partial (90% rules) | ❌ No | ✅ EXCELLENT | No |
| M2.2 Coreference | ⚠️ Partial (80% rules) | ❌ No | ✅ EXCELLENT | No |
| Entity Classifier | ⚠️ Partial (tier-based) | ❌ No | ✅ YES | No |
| Reranker | ✅ YES | ❌ Needs redesign | ✅ YES | No |
| **Infrastructure** |
| ASTAnnotator protocol | ✅ YES | ✅ YES | ✅ YES | No |
| Tensor support | ✅ YES | ✅ YES | N/A | No |
| Versioning system | ✅ YES | ✅ YES | N/A | No |
| CLI design | ✅ YES | ❌ No | N/A | Yes (inspection tools) |
| **Data Pipeline** |
| Root export script | ❌ No | ❌ No | N/A | Yes (Root training) |
| M1 export script | ❌ No | ❌ No | N/A | Yes (M1 training) |
| Entity export script | ❌ No | ❌ No | N/A | Yes (Entity training) |
| Reranker export script | ❌ No | ❌ No | N/A | Yes (Reranker training) |
| **Deterministic Rules** |
| ReVo/ConceptNet loader | ❌ No | ❌ No | N/A | Yes (M2.1) |
| Grammar-based coref | ❌ No | ❌ No | N/A | Yes (M2.2) |
| Tier-based entity | ❌ No | ❌ No | N/A | Yes (Entity) |

## Confidence Assessment

### What We're Confident About ✅

1. **Model taxonomy is correct** - 9 models, right level of granularity
2. **Architecture is sound** - ASTAnnotator protocol, tensor support
3. **Alignment is excellent** - Maximizes deterministic, minimizes learned
4. **Versioning prevents chaos** - Can track compatibility as system evolves

### What We Need to Figure Out ⚠️

1. **Deterministic rule implementations** - Conceptual design exists, need to code
2. **Training data extraction** - Know what data, need to write extraction scripts
3. **Reranker training data** - How to generate query-document pairs with labels?
4. **Fallback model sizes** - How big for M2.1/M2.2 fallbacks (10%/20% cases)?

### What Could Change 🤔

1. **Reranker necessity** - Could skip initially, add later
2. **Entity classifier scope** - Could start with tier1-2 only
3. **M2.1/M2.2 priority** - Could defer to Phase 3

## Recommendation: Critical Path Forward

### Week 1: Inspection Tools (CRITICAL)
**Epic #637** - Can't develop without debugging tools
- Implement `klareco inspect ast`
- Implement `klareco inspect tensor`
- Implement `klareco inspect annotations`

### Week 2: Data Pipeline (CRITICAL)
**Epic #641** - Can't train without data
- Design + implement `export_roots_v2.1.py`
- Design + implement `export_m1_triples_v2.1.py`
- Create data validation scripts

### Week 3: Root Embeddings (FIRST MODEL)
**Epic #616, Phase 1**
- Export root training data (using script from Week 2)
- Train root embeddings v3 (tier-filtered)
- Test quality (no collapse, semantic clusters)

### Week 4: M1 Selectional (SECOND MODEL)
**Epic #616, Phase 2**
- Export M1 training data
- Train M1 with frozen root embeddings
- Test accuracy

### Month 2: Deterministic Models
**M2.1 Taxonomy + M2.2 Coreference**
- Implement deterministic rule systems
- This is the CORE Klareco advantage!

## Final Answer to Your Questions

### "Do we know what models we need?"

**YES** - 9 models clearly defined:
- M0 Parser (complete)
- Root Embeddings (ready to retrain)
- M1 Selectional (ready to retrain)
- M2.1 Taxonomy (90% deterministic - needs design)
- M2.2 Coreference (80% deterministic - needs design)
- Entity Classifier (tier-based - needs design)
- Reranker (needs redesign)
- Compositional Embeddings (uses root embeddings)

### "Do we know what each model should do and is it aligned with Klareco's purpose?"

**YES** - Every model is well-defined and aligned:
- Alignment assessment: ✅ EXCELLENT
- Deterministic percentage: M2.1 (90%), M2.2 (80%), Entity (70%)
- Total learned params: ~2.3M (vs traditional models with 100M+)
- Architecture: All use ASTAnnotator protocol, leverage AST structure

### "Do we know what training data we need for each model?"

**PARTIALLY** - We know WHAT data, but NOT HOW to extract it:

✅ **Know WHAT**:
- Root embeddings: Tier1a+1b+2 roots (~9,800 vocabulary)
- M1: (subject, verb, object) triples from corpus
- Entity: Entity type labels (tier3 only, tier1-2 deterministic)
- Reranker: Query-document relevance pairs

❌ **Don't know HOW**:
- No data extraction scripts written yet (Epic #641)
- No deterministic rule systems implemented yet (M2.1/M2.2)
- Unclear how to generate reranker training data

**Critical Gap**: Epic #641 (Data Pipeline) must be next priority after inspection tools!

## Status: READY TO PROCEED

**Foundation**: ✅ Solid (design complete, interfaces defined, versioning enforced)
**Critical Path**: ✅ Clear (inspection tools → data pipeline → model training)
**Alignment**: ✅ Excellent (maximizes deterministic, minimizes learned)
**Confidence**: ✅ High (know what to build, how to build it)

**Next Action**: Start Epic #637 (Inspection Tools) immediately!
