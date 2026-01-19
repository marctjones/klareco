# Data Quality Ranking System

**Status**: NEW SYSTEM - Replacing confusing tier 0-6 numbering
**Date**: 2026-01-19
**Motivation**: Old system used arbitrary numbers (0, 2, 5, 6) with undefined gaps

---

## Overview

Klareco uses a **3-level quality system** based on data source characteristics:

| Level | Name | Description | Sources | Size |
|-------|------|-------------|---------|------|
| **GOLD** | Authoritative | Expert-written grammar, Q&A, pedagogy | PMEG, Krestomatio, Lingvaj Respondoj | ~22K sentences |
| **SILVER** | Encyclopedic | Community-edited encyclopedia | Wikipedia | ~3.8M sentences |
| **BRONZE** | Literary | Published literature (variable quality) | Project Gutenberg | ~380K sentences |

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

### SILVER: Encyclopedic
**Characteristics**:
- Community-edited with editorial oversight
- Factual/informational content
- High coverage of topics and vocabulary
- Medium grammatical quality (~85-95%)

**Sources**:
- **Wikipedia** (Esperanto edition)

**Use cases**:
- Broad vocabulary coverage
- Domain-specific terminology
- General knowledge corpus
- High-volume training data

### BRONZE: Literary
**Characteristics**:
- Published literary works
- Natural, idiomatic language
- Variable quality (author-dependent)
- Rich in narrative/descriptive language

**Sources**:
- **Project Gutenberg** - Public domain books in Esperanto

**Use cases**:
- Natural language patterns
- Discourse-level features
- Stylistic variation
- Long-form text

---

## Implementation

### Corpus Metadata Structure

Each sentence in the corpus includes quality metadata:

```json
{
  "text": "Esperanto estas planlingvo...",
  "source": {
    "quality": "GOLD",           // Quality level
    "name": "pmeg",              // Specific source
    "source_name": "PMEG 2023",  // Full source title
    "source_type": "grammar_reference",
    "tier": 0  // DEPRECATED: kept for backward compatibility
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
    'SILVER': 0.50,  # 50% from encyclopedic
    'BRONZE': 0.35   # 35% from literary
}

# Option 3: Filtering
min_quality = 'SILVER'  # Exclude BRONZE for high-precision task
```

---

## Migration Path

### Phase 1: Add Quality Field (Backward Compatible)
**Status**: IN PROGRESS

Keep existing `tier` field, add new `quality` field:
```python
SOURCE_CONFIGS = {
    'authoritative_grammar': {
        'tier': 0,  # Keep for compatibility
        'quality': 'GOLD',  # New field
        ...
    },
    'wikipedia': {
        'tier': 5,  # Keep for compatibility
        'quality': 'SILVER',  # New field
        ...
    },
    'gutenberg': {
        'tier': 6,  # Keep for compatibility
        'quality': 'BRONZE',  # New field
        ...
    },
}
```

### Phase 2: Update Training Scripts
**Status**: PLANNED

Add quality-based parameters alongside tier parameters:
```bash
# Old style (still works)
--priority-tiers 0
--fill-tiers 5 6

# New style (preferred)
--priority-qualities GOLD
--fill-qualities SILVER BRONZE
```

### Phase 3: Deprecate Tiers (Future)
**Status**: PLANNED

After all scripts use quality-based filtering:
1. Mark `tier` field as deprecated
2. Update documentation
3. Eventually remove (breaking change)

---

## Mapping: Old Tiers → New Quality

| Old Tier | New Quality | Notes |
|----------|-------------|-------|
| 0 | GOLD | Authoritative sources |
| 1 | *(unused)* | Reserved for future |
| 2 | GOLD | Was Krestomatio, now part of tier 0/GOLD |
| 3 | *(unused)* | Reserved for future |
| 4 | *(unused)* | Reserved for future |
| 5 | SILVER | Wikipedia |
| 6 | BRONZE | Gutenberg |

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
**Implementation Status**: Phase 1 (add quality field)
**Migration Timeline**: Phase 2 (Q1 2026), Phase 3 (Q2 2026)
