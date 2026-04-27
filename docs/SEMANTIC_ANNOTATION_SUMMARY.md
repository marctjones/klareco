# Semantic Annotation Summary

**Generated:** 2026-03-31
**Database:** v2.1 Kuzu Full Index
**Total Annotations:** 313 roots

## Overview

The Klareco v2.1 database includes a comprehensive semantic ontology with hierarchical verb taxonomy and entity type classifications. Semantic annotations enable query expansion, synonym matching, and semantic reasoning.

## Annotation Statistics

### Total Coverage

| Category | Count | Percentage of Corpus |
|----------|-------|---------------------|
| **Total Roots in DB** | 1,153,738 | 100% |
| **Annotated Roots** | 313 | 0.027% |
| **Verb Annotations** | 250 | - |
| **Entity Annotations** | 63 | - |

### Coverage by Source

| Source | Verbs | Entities | Total |
|--------|-------|----------|-------|
| **Manual Annotations** | 23 | 0 | 23 |
| **Gazetteer-based** | 109 | 52 | 161 |
| **ReVo Expansion (Depth 1)** | 74 | 6 | 80 |
| **ReVo Expansion (Depth 2)** | 58 | 5 | 63 |
| **TOTAL** | 264* | 63 | 327* |

*Note: Some roots have multiple annotations (e.g., both top-level and fine-grained classes)

## Hierarchical Verb Taxonomy

### Structure

- **8 Top-Level Classes** (broad semantic categories)
- **31 Subclasses** (fine-grained VerbNet-style distinctions)
- **39 Total Verb Classes**

### Top-Level Classes

| Class ID | Name | Description | Example Roots |
|----------|------|-------------|---------------|
| kreado-26 | Kreado | Creation/production | fond, kre, produk, far |
| movo-51 | Movo | Movement/motion | ir, ven, fur, voj |
| pensado-29 | Pensado | Thinking/cognition | pens, sci, kred, komprend |
| perceptado-30 | Perceptado | Perception | vid, aŭd, sent, gust |
| emocio-31 | Emocio | Emotion/feeling | am, ĝoj, tim, trist |
| komunikado-37 | Komunikado | Communication | dir, parol, demand, respond |
| vivo-48 | Vivo | Life processes | viv, mort, nask, kresk |
| profesio-50 | Profesio | Professional activity | labor, instrui, kurac, vend |

### Fine-Grained Subclasses (Examples)

| Subclass ID | Name | Parent | Example Roots |
|-------------|------|--------|---------------|
| vido-30 | Vidado | perceptado-30 | vid, rigar, observ |
| aŭdo-47 | Aŭdado | perceptado-30 | aŭd, aŭskult |
| scio-30 | Sciado | pensado-29 | sci, komprend, kon |
| amo-31 | Amo | emocio-31 | am, ador, ŝat |
| timo-31 | Timo | emocio-31 | tim, angor, panik |
| diro-37 | Parolado | komunikado-37 | dir, parol, rakon |
| ekzisto-47 | Ekzistado | vivo-48 | ekzist, viv, est |

## Entity Type Classifications

### Entity Types

| Type ID | Name | Count | Example Roots |
|---------|------|-------|---------------|
| loko | Location | 41 | pariz, berlin, rom, varsov, prag |
| persono | Person | 11 | aŭtor, verkist, prezident, kuracist |
| tempo | Time | 0 | *(not yet annotated)* |
| organizaĵo | Organization | 0 | *(not yet annotated)* |
| evento | Event | 0 | *(not yet annotated)* |
| profesio | Profession | 0 | *(not yet annotated)* |

### Place Names (Sample)

Annotated location roots include:
- **European Cities:** pariz, berlin, rom, vienn, prag, varsov, moskv
- **Countries:** franc, german, ital, angl, rus, pol, norveg, aŭstr
- **Regions:** eŭrop, mond, ter

### Person Types (Sample)

Annotated person indicators:
- **Professions:** aŭtor, verkist, prezident, kuracist, direktist
- **Roles:** redaktor, tradukist, administr

## Annotation Methods

### 1. Manual Annotation (23 roots)

Hand-curated annotations from `data/annotations/phase_0_*.jsonl`:
- High-quality, verified classifications
- Covers core Fundamento vocabulary
- Used as seed for expansion

### 2. Gazetteer-Based (161 roots)

Automatic annotation from:
- **Verb Classes:** 109 roots from `VerbaKlaso.ekzemplaj_radikoj`
- **Place Names:** 41 roots from `klareco.knowledge.gazetteers.place_names`
- **Person Types:** 11 roots from `klareco.knowledge.gazetteers.person_indicators`

### 3. ReVo Synonym Expansion (143 roots)

Propagated through REVO_SINONIMO relationships:
- **Depth 1:** Direct synonyms (80 roots)
- **Depth 2:** 2-hop synonym chains (63 roots)

**Algorithm:**
```
For each annotated root R with class C:
  For each synonym S of R (via REVO_SINONIMO):
    If S not already annotated:
      Annotate S with class C
```

**Example Expansion:**
```
fond (kreado-26)
  → kre (REVO_SINONIMO)
  → kre inherits kreado-26

fond → kre → produk (2-hop chain)
  → produk inherits kreado-26
```

## Query Examples

### Find All Synonyms of a Verb Class

```cypher
// Find all "creation" verbs (broad)
MATCH (r:Radiko)-[:APARTENAS_AL_VERBA_KLASO]->(v:VerbaKlaso {klaso_id: 'kreado-26'})
RETURN r.radiko
```

### Find Synonyms with Hierarchy

```cypher
// Find all "perception" verbs (including subclasses)
MATCH (r:Radiko)-[:APARTENAS_AL_VERBA_KLASO]->(sub:VerbaKlaso)
WHERE sub.superklaso_id = 'perceptado-30' OR sub.klaso_id = 'perceptado-30'
RETURN sub.klaso_nomo, collect(r.radiko)
```

### Query Expansion for "see"

```cypher
// Find semantic synonyms of "vid" (not just ReVo synonyms)
MATCH (r1:Radiko {radiko: 'vid'})-[:APARTENAS_AL_VERBA_KLASO]->(v:VerbaKlaso)
MATCH (r2:Radiko)-[:APARTENAS_AL_VERBA_KLASO]->(v)
WHERE r2.radiko <> 'vid'
RETURN r2.radiko
```

### Find Location Roots

```cypher
// Get all annotated place names
MATCH (r:Radiko)-[:HAVAS_ENTECAN_TIPON]->(e:EntecaTipo {tipo_id: 'loko'})
RETURN r.radiko
ORDER BY r.radiko
```

## Integration with RAG System

Semantic annotations enhance the RAG pipeline:

### 1. Query Expansion

**Before:**
```
User query: "Kiu fondis Esperanton?"
Retrieval: Search for "fond"
```

**After:**
```
User query: "Kiu fondis Esperanton?"
Expansion: fond → kreado-26 → {fond, kre, produk, far}
Retrieval: Search for {fond, kre, produk, far}
```

### 2. Answer Extraction

**Verb Class Matching:**
```python
# Check if verb matches expected semantic class
def matches_question_type(verb_root, question_type):
    verb_class = get_verb_class(verb_root)

    if question_type == "WHO":
        # WHO questions expect agentive verbs
        return verb_class in ['kreado-26', 'profesio-50']
    elif question_type == "WHERE":
        # WHERE questions expect location verbs
        return verb_class in ['movo-51', 'ekzisto-47']
```

### 3. Importance Scoring

**Schema-Based Weighting:**
```python
# Get importance weight from verb class
verb_class = get_verb_class('fond')  # kreado-26
biography_importance = verb_class.graveco_biografia  # 0.95 (very important)
```

## Expansion Roadmap

### Phase 1: Core Vocabulary (Complete)
- ✅ 313 annotations
- ✅ Hierarchical taxonomy (39 classes)
- ✅ ReVo expansion (2 depths)

### Phase 2: High-Frequency Roots (Next)
- Target: 1,000-2,000 annotations
- Method: Frequency-based with embedding similarity
- Annotate top 1,000 most frequent verbs/nouns

### Phase 3: Comprehensive Coverage (Future)
- Target: 5,000-10,000 annotations
- Method: Semi-automatic with quality review
- Cover Fundamento + modern technical vocabulary

### Phase 4: Domain-Specific (Future)
- Medical terminology
- Technical/scientific vocabulary
- Modern neologisms

## Quality Metrics

### Accuracy Estimate

Based on manual spot-checking:
- **Manual Annotations:** 100% (verified)
- **Gazetteer-Based:** ~95% (some function words incorrectly included)
- **ReVo Depth 1:** ~90% (synonyms generally accurate)
- **ReVo Depth 2:** ~80% (some semantic drift)

### Known Issues

1. **Function Words Misclassified:** Some function words (al, el, pro, dis) inherited kreado-26 via overly aggressive ReVo expansion
2. **Polysemy Not Handled:** Single annotation per root (doesn't handle multiple senses)
3. **Sparse Coverage:** Only 0.027% of roots annotated (need more automation)

## Tools and Scripts

### Annotation Scripts

| Script | Purpose | Output |
|--------|---------|--------|
| `extend_verb_taxonomy_hierarchical.py` | Create 39-class taxonomy | VerbaKlaso nodes |
| `load_semantic_relationships.py` | Load manual annotations | APARTENAS_AL_VERBA_KLASO edges |
| `annotate_core_roots_from_gazetteers.py` | Bulk annotation from examples | 161 annotations |
| `expand_annotations_via_revo.py` | Synonym chain expansion | +143 annotations |

### Query Scripts

| Script | Purpose |
|--------|---------|
| `scripts/demo_semantic_retrieval.py` | Test semantic query expansion |
| `klareco/rag/semantic_query.py` | SemanticQuery API |

## References

- **VerbNet:** https://verbs.colorado.edu/verbnet/
- **FrameNet:** https://framenet.icsi.berkeley.edu/
- **ReVo Dictionary:** http://www.reta-vortaro.de/
- **Fundamento:** https://akademio-de-esperanto.org/fundamento/

## Change Log

- **2026-03-31:** Initial hierarchical taxonomy + ReVo expansion (313 annotations)
- **2026-03-28:** Core gazetteer-based annotation (161 annotations)
- **2026-03-16:** Semantic ontology schema created
- **2026-03-09:** Manual Phase 0 annotations (50 roots)
