# Data Quality Ranking System

**Status**: NEW SYSTEM - Replacing confusing tier 0-6 numbering
**Date**: 2026-01-19
**Motivation**: Old system used arbitrary numbers (0, 2, 5, 6) with undefined gaps

---

## Overview

Klareco uses a **3-level quality system** based on Esperanto language quality:

| Level | Name | Description | Sources | Size |
|-------|------|-------------|---------|------|
| **GOLD** | Authoritative | Expert-written grammar, Q&A, pedagogy | PMEG, Krestomatio, Lingvaj Respondoj | ~22K sentences |
| **SILVER** | Literary | Published literature (high language quality) | Project Gutenberg | ~380K sentences |
| **BRONZE** | Encyclopedic | Community-edited (variable quality) | Wikipedia | ~3.8M sentences |

---

## Quality Level Definitions

### GOLD: Authoritative
**Characteristics**:
- Written or reviewed by Esperanto experts
- Explicitly pedagogical or normative intent
- High grammatical accuracy (near 100%)
- Includes edge cases and grammatical explanations

**Sources**:
- **PMEG** (Plena Manlibro de Esperanta Gramatiko) - Comprehensive grammar reference
- **Krestomatio** (Fundamenta Krestomatio) - Canonical literary anthology
- **Lingvaj Respondoj** - Expert Q&A on grammar questions
- **Ekzercaro** - Grammar exercises

**Use cases**:
- Training selectional preference models
- Grammar pattern learning
- Authoritative examples for RAG

### SILVER: Literary
**Characteristics**:
- Published literary works
- Natural, idiomatic language
- High grammatical quality (~90-95%)
- Rich in narrative/descriptive language

**Sources**:
- **Project Gutenberg** - Public domain books in Esperanto

**Use cases**:
- Natural language patterns
- Discourse-level features
- Stylistic variation
- High-quality training data
- Long-form text

### BRONZE: Encyclopedic
**Characteristics**:
- Community-edited with variable editorial oversight
- Factual/informational content
- High coverage of topics and vocabulary
- Variable grammatical quality (~80-90%)

**Sources**:
- **Wikipedia** (Esperanto edition)

**Use cases**:
- Broad vocabulary coverage
- Domain-specific terminology
- General knowledge corpus
- High-volume training data

---

## Implementation

### Corpus Metadata Structure

Each sentence in the corpus includes quality metadata:

```json
{
  "text": "Esperanto estas planlingvo...",
  "source": {
    "quality": "GOLD",           // Quality level (GOLD/SILVER/BRONZE)
    "name": "pmeg",              // Specific source
    "source_name": "PMEG 2023",  // Full source title
    "source_type": "grammar_reference"
  },
  "ast": {...},
  "parse_rate": 0.95
}
```

### Quality-Aware Training

Training scripts can filter or weight by quality:

```python
# Option 1: Priority loading (load GOLD first)
priority_qualities = ['GOLD']
fill_qualities = ['SILVER', 'BRONZE']

# Option 2: Weighted sampling
quality_weights = {
    'GOLD': 0.15,    # 15% from authoritative sources
    'SILVER': 0.35,  # 35% from literary (high quality)
    'BRONZE': 0.50   # 50% from encyclopedic (volume)
}

# Option 3: Filtering
min_quality = 'SILVER'  # Exclude BRONZE for high-precision task
```

---

## Mapping: Old Tiers → New Quality

The old tier system (0, 2, 5, 6) has been completely replaced:

| Old Tier | New Quality | Notes |
|----------|-------------|-------|
| 0, 2 | GOLD | Authoritative sources (PMEG, Krestomatio, etc) |
| 6 | SILVER | Literary (Gutenberg books - high language quality) |
| 5 | BRONZE | Encyclopedic (Wikipedia - variable quality) |

---

## Future Extensions

### Additional Quality Levels (If Needed)

**PLATINUM** (hypothetical):
- Fundamento de Esperanto (official foundation)
- Academy-approved texts
- Ultra-authoritative (higher than GOLD)

**COPPER** (hypothetical):
- Social media / informal text
- Machine-translated content
- Lower quality than BRONZE

**Current decision**: 3 levels (GOLD/SILVER/BRONZE) are sufficient.
Expand only if clear use case emerges.

---

## Benefits Over Old System

1. **Self-documenting**: "GOLD" conveys meaning, "tier 0" doesn't
2. **No gaps**: 3 contiguous levels, not 0,2,5,6 with gaps
3. **Extensible**: Can add PLATINUM or COPPER without renumbering
4. **Standard terminology**: Gold/Silver/Bronze used in ML community
5. **Quality-first**: Name reflects content quality, not arbitrary number

---

## References

- ISO/IEC 25012 Data Quality Model (inspiration for quality attributes)
- Common ML practice: Gold/Silver/Bronze data tiers
- NLP corpus annotation standards (EAGLES, ISO TC 37)

---

**Document Status**: APPROVED
**Implementation Status**: COMPLETE (quality field replaces tier system)
**Effective Date**: 2026-01-19
