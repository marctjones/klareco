# Tiered Retrieval Benchmark Design

## Overview

This document describes a corpus-aware, tiered benchmark for testing the ASTAwareRetriever progressively from simple to complex retrieval tasks.

## Problem Statement

The current Q&A benchmark conflates retrieval quality with answer extraction quality. It also doesn't distinguish between:
- Simple retrievals that should work now
- Advanced retrievals requiring features not yet implemented

**Example**: "Kiu fondis Esperanton?" is an **advanced** retrieval because:
1. No document has `fond` + `esperant` + `zamenhof` together
2. Even with synonym expansion (`kre`), no document has all three roots together
3. Requires multi-hop inference or relaxed matching

## Tiered Complexity Model

### Tier 1: Direct Root Match (Simplest)
**Description**: Query contains one or more roots that directly exist in target documents.

**Requirements**:
- Query roots exist verbatim in corpus
- No synonym expansion needed
- No role matching required

**Example Queries**:
- "Kio estas elefanto?" → Documents with `elefant` root
- "Priskribu hundon" → Documents with `hund` root

**Expected Performance**: >90% recall@10

**Corpus Verification**:
- `elefant`: 73 documents (e.g., doc 1823872, 2837790)
- `hund`: 4,638 documents (e.g., doc 6926, 11857)

---

### Tier 2: Multi-Root Conjunction
**Description**: Multiple query roots must ALL appear in target documents.

**Requirements**:
- All query roots present in same document
- No synonym expansion
- No role constraints

**Example Queries**:
- "Kio estas rapida kurado?" → Documents with `rapid` AND `kur` (544 docs)
- "Priskribu bluan ĉielon" → Documents with `blu` AND `ĉiel` (205 docs)
- "Kio estas bona amiko?" → Documents with `bon` AND `amik` (1521 docs)

**Expected Performance**: >80% recall@10

**Corpus Verification**:
- `rapid` + `kur`: 544 docs (e.g., doc 67584, 81921)
- `blu` + `ĉiel`: 205 docs (e.g., doc 523266)
- `bon` + `amik`: 1521 docs (e.g., doc 565248)

---

### Tier 3: Synonym Expansion
**Description**: Query root doesn't appear in document, but a synonym does.

**Requirements**:
- Working synonym graph in Kuzu
- Synonym must be found via graph traversal

**Example Queries**:
- Query with `fond` (found) → Document with only `establ` or `kre`
- Query with `grand` → Document with only `vast` or `gigant`

**Verified Synonym Groups**:
- `fond` → {`kre`, `establ`, `inaŭgur`, `fari`}
- `grand` → {`detal`, `et`, `gigant`, `etern`}

**Expected Performance**: >60% recall@10 (depends on synonym quality)

---

### Tier 4: Role-Aware Matching
**Description**: Roots must appear in specific grammatical roles (subjekto, verbo, objekto).

**Requirements**:
- AST role extraction working
- Role filtering in search

**Example Queries**:
- "Kiu kuris?" → Documents where `kur` is VERBO (1,842 docs)
- "Kion la hundo faris?" → Documents where `hund` is SUBJEKTO (subset of 4,638)
- "Kion vi vidis al la hundo?" → Documents where `hund` is OBJEKTO (290 docs)

**Corpus Verification** (role distribution for `kur`):
- verbo: 1,842 occurrences
- subjekto: 1,332 occurrences
- objekto: 290 occurrences
- aliaj: 6,589 occurrences

**Expected Performance**: >70% precision (correct role matches)

---

### Tier 5: Cross-Document Inference (Advanced)
**Description**: Answer requires combining information that doesn't co-occur in any single document.

**Requirements**:
- Multi-hop reasoning
- OR relaxed matching (partial matches ranked highly)
- OR graph-based entity linking

**Example Queries**:
- "Kiu fondis Esperanton?" (No doc has `fond`/`kre` + `esperant` + `zamenhof` together)
- "Kiu kreis la lingvon internacian?"

**Current Status**: NOT EXPECTED TO WORK until advanced features implemented.

---

## Benchmark JSON Schema

```json
{
  "version": "1.0",
  "created": "2026-01-08",
  "description": "Tiered retrieval benchmark for ASTAwareRetriever",
  "tiers": {
    "tier1_direct": {
      "description": "Direct root match - simplest retrieval",
      "expected_recall_at_10": 0.90,
      "queries": [
        {
          "id": "t1_001",
          "query": "Kio estas elefanto?",
          "query_roots": ["elefant"],
          "expected_doc_ids": [1823872, 2837790, 2377713, 2427457, 2485683],
          "relevance_criteria": "Document mentions elephant",
          "notes": "73 documents in corpus with 'elefant' root"
        }
      ]
    },
    "tier2_conjunction": {
      "description": "Multi-root conjunction",
      "expected_recall_at_10": 0.80,
      "queries": [
        {
          "id": "t2_001",
          "query": "Kio estas rapida kurado?",
          "query_roots": ["rapid", "kur"],
          "expected_doc_ids": [67584, 81921, 186368, 1607684, 73738],
          "relevance_criteria": "Document mentions both 'rapid' and 'kur'",
          "notes": "544 documents with both roots"
        }
      ]
    },
    "tier3_synonym": {
      "description": "Synonym expansion required",
      "expected_recall_at_10": 0.60,
      "queries": [
        {
          "id": "t3_001",
          "query": "Kio estas establita organizo?",
          "query_roots": ["establ", "organiz"],
          "synonym_groups": {"establ": ["fond", "kre", "inaŭgur"]},
          "expected_doc_ids": [],
          "relevance_criteria": "Document about establishing/founding organizations",
          "notes": "Tests synonym graph traversal"
        }
      ]
    },
    "tier4_role": {
      "description": "Role-aware matching",
      "expected_precision_at_10": 0.70,
      "queries": [
        {
          "id": "t4_001",
          "query": "Kiu kuris?",
          "query_roots": ["kur"],
          "required_roles": {"kur": "verbo"},
          "expected_doc_ids": [2718, 6942, 26011, 26012, 30526],
          "relevance_criteria": "Document where 'kur' is the verb",
          "notes": "1,842 docs with 'kur' as verbo"
        }
      ]
    },
    "tier5_inference": {
      "description": "Cross-document inference (advanced)",
      "expected_recall_at_10": 0.30,
      "queries": [
        {
          "id": "t5_001",
          "query": "Kiu fondis Esperanton?",
          "query_roots": ["fond", "esperant"],
          "answer": "Zamenhof",
          "expected_doc_ids": [],
          "relevance_criteria": "Document stating Zamenhof created/founded Esperanto",
          "notes": "No single doc has fond+esperant+zamenhof. Requires relaxed matching."
        }
      ]
    }
  }
}
```

---

## Scoring Metrics

### Per-Query Metrics

| Metric | Description |
|--------|-------------|
| **Recall@K** | Fraction of expected docs found in top K |
| **Precision@K** | Fraction of top K that are relevant |
| **MRR** | Mean Reciprocal Rank of first relevant doc |
| **NDCG@K** | Normalized Discounted Cumulative Gain |

### Per-Tier Aggregates

| Metric | Description |
|--------|-------------|
| **Tier Recall** | Average recall across all queries in tier |
| **Tier Precision** | Average precision across all queries in tier |
| **Pass Rate** | % of queries meeting expected threshold |

### Overall Score

Weighted by tier difficulty:
```
Score = 0.30 * Tier1 + 0.25 * Tier2 + 0.20 * Tier3 + 0.15 * Tier4 + 0.10 * Tier5
```

---

## Implementation Plan

### Phase 1: Create Ground Truth Dataset
1. Select 10-20 queries per tier
2. For each query:
   - Verify query roots exist in corpus
   - Find 5-10 relevant document IDs
   - Record relevance criteria
3. Save as `data/benchmarks/retrieval_benchmark_v1.json`

### Phase 2: Implement Benchmark Script
1. Create `scripts/evaluate_retrieval.py`:
   - Load benchmark JSON
   - Run queries through ASTAwareRetriever
   - Compute per-query and per-tier metrics
   - Generate detailed report
2. Options:
   - `--tier <N>`: Run only specific tier
   - `--verbose`: Show per-query results
   - `--output`: Save results JSON

### Phase 3: Integrate with CI
1. Add retrieval benchmark to test suite
2. Set minimum thresholds per tier
3. Fail build if Tier 1/2 regression

### Phase 4: Iterate on Retriever
1. Focus on passing Tier 1 first (should work now)
2. Debug Tier 2 issues
3. Improve synonym graph for Tier 3
4. Add role filtering for Tier 4
5. Design multi-hop for Tier 5 (future)

---

## Key Insights from Corpus Analysis

### What Should Work Now (Tier 1-2)
- Single root lookups: Working (e.g., `elefant`, `hund`)
- Multi-root conjunction: Should work if docs exist

### What Needs Testing (Tier 3)
- Synonym expansion IS in Kuzu graph
- `fond` → {`kre`, `establ`, `inaŭgur`, `fari`} verified
- Need to verify expansion is used in scoring

### What Doesn't Work Yet (Tier 4-5)
- Role filtering: Index has role data, but search may not use it
- Multi-hop inference: Not implemented

### Corpus Statistics
- Total documents: 4,381,608
- Total roots: 1,158,625
- Common roots have thousands of occurrences
- Rare roots (like `elefant`) have 50-100 docs

---

## Next Steps

1. [ ] Create `retrieval_benchmark_v1.json` with verified queries
2. [ ] Implement `evaluate_retrieval.py` script
3. [ ] Run Tier 1 baseline to establish current performance
4. [ ] Identify specific failures and fix retriever
5. [ ] Progressively enable higher tiers
