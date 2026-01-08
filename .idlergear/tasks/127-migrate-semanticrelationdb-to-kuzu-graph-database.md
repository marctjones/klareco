---
id: 127
title: Migrate SemanticRelationDB to Kuzu graph database
state: closed
created: '2026-01-07T03:31:32.269638Z'
labels:
- enhancement
- architecture
- data
priority: medium
---
## Objective
Migrate SemanticRelationDB from in-memory dicts to Kuzu embedded graph database for better support of complex semantic queries and scalability.

## Why Kuzu
- Native graph traversal for semantic relations
- Multi-hop queries (transitive synonyms, hypernym chains)
- Embedded (no server), single file
- Aligns with Klareco's deterministic philosophy
- Scales to full ReVo dictionary + corpus-extracted relations

## Schema Design
```python
# Nodes
Root(name STRING, pos STRING, PRIMARY KEY(name))
Affix(name STRING, type STRING, PRIMARY KEY(name))
Word(form STRING, PRIMARY KEY(form))

# Edges - semantic relations
SYNONYM(FROM Root TO Root)
ANTONYM(FROM Root TO Root)
HYPERNYM(FROM Root TO Root)  # specific → general
HYPONYM(FROM Root TO Root)   # general → specific
RELATED(FROM Root TO Root, weight DOUBLE)

# Edges - morphological
DERIVED_FROM(FROM Root TO Root, affix STRING)
AGENT_NOUN(FROM Root TO Word)
```

## Implementation Phases

### Phase 1: Hybrid Approach
- Keep in-memory dicts for fast simple lookups
- Add Kuzu for complex queries (transitive, pathfinding)
- Load existing JSON relations into Kuzu at startup

### Phase 2: Full Migration
- Move all lookups to Kuzu
- Add proper indexing
- Benchmark performance vs dict approach

### Phase 3: Extended Relations
- Add corpus-extracted co-occurrence relations
- Import full ReVo dictionary
- Add derivational patterns

## New Queries Enabled
```python
# Transitive synonyms (within N hops)
get_synonyms_transitive(root, max_hops=2)

# Hypernym chain: hundo → besto → vivaĵo
get_hypernym_chain(root)

# Find semantic path between two roots
find_semantic_path(root1, root2)

# All related concepts within distance
get_semantic_neighborhood(root, max_distance=3)
```

## Dependencies
- `pip install kuzu`

## Success Criteria
- [ ] Kuzu schema created and documented
- [ ] Migration script from JSON → Kuzu
- [ ] Hybrid SemanticRelationDB with both backends
- [ ] Transitive synonym queries working
- [ ] Hypernym chain traversal working
- [ ] Performance benchmarks (dict vs Kuzu)
- [ ] Integration with AST-aware retriever
