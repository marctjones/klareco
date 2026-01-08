---
id: 139
title: 'Phase 2: Add synonym/hypernym graph traversal in Kuzu'
state: open
created: '2026-01-08T00:47:51.990735Z'
labels:
- enhancement
- kuzu
- retrieval
priority: high
---
## Objective

Move SemanticRelationDB from in-memory Python dicts to Kuzu graph edges, enabling single-query synonym expansion.

## Current State

SemanticRelationDB loads all relations into memory at startup:
```python
self.synonyms = {}      # ~2,598 sets
self.hypernyms = {}     # ~2,794 relations  
self.hyponyms = {}      # ~425 relations
```

Synonym expansion requires Python loops:
```python
expanded = set()
for root in query_roots:
    for syn in semantic_db.get_synonyms(root):
        expanded.add(syn)
```

## With Kuzu

Add semantic edges to the graph:
```cypher
CREATE REL TABLE IS_SYNONYM (FROM Root TO Root)
CREATE REL TABLE IS_HYPERNYM (FROM Root TO Root)  -- child → parent
CREATE REL TABLE IS_ANTONYM (FROM Root TO Root)
```

Single-query expansion:
```cypher
MATCH (r:Root {root: 'fond'})-[:IS_SYNONYM*0..2]->(syn:Root)
RETURN collect(DISTINCT syn.root) AS expanded
```

## Benefits

1. **Memory savings**: No in-memory dicts (~10MB saved)
2. **Transitive synonyms**: `*0..2` finds synonyms-of-synonyms
3. **Hypernym chains**: `hundo → besto → vivaĵo` in one query
4. **Unified storage**: One database for index + semantics

## Implementation

### Step 1: Extend schema
Add IS_SYNONYM, IS_HYPERNYM, IS_ANTONYM edge tables

### Step 2: Load semantic relations
- Read `revo_semantic_relations.json`
- Create Root nodes for all roots in relations
- Create edges between related roots

### Step 3: Update retriever
- Replace `semantic_db.get_synonyms()` with Cypher query
- Add transitive synonym option (depth parameter)

### Step 4: Remove SemanticRelationDB
- Once Kuzu handles all lookups
- Delete in-memory loading code

## Success Criteria
- [ ] All 2,598 synonym sets loaded as edges
- [ ] All hypernym/hyponym relations loaded
- [ ] `get_synonyms(root)` returns same results
- [ ] Transitive synonyms working (2-hop)
- [ ] Memory reduced by removing in-memory dicts

## Depends On
- Task #138 (Kuzu migration complete)

## Supersedes
- Task #127 (SemanticRelationDB to Kuzu)
