---
id: 138
title: 'Phase 1: Migrate SQLite inverted index to Kuzu graph database'
state: open
created: '2026-01-08T00:33:31.863463Z'
labels:
- architecture
- migration
- high-priority
priority: high
---
## Objective

Replace SQLite-based `root_inverted_index.py` with Kuzu, maintaining exact same functionality first (no new features yet).

## Why Kuzu Over SQLite

Both are disk-based and memory-efficient, but Kuzu provides:
- O(1) hash lookups vs O(log n) B-tree
- Native graph traversals (needed for future features)
- Single storage for index + semantic relations (currently in-memory dict)

## Scope: 1:1 Migration Only

This task does NOT add new features. It replaces SQLite with equivalent Kuzu functionality:

| SQLite | Kuzu Equivalent |
|--------|-----------------|
| `occurrences` table | `HAS_ROOT` edges |
| `root_stats` table | `Root` node properties |
| `metadata` table | Database properties |
| B-tree index on root | Hash index (automatic) |

## Kuzu Schema (Minimal)

```cypher
// NODES
CREATE NODE TABLE Root (
    root STRING PRIMARY KEY,
    doc_freq INT,
    total_freq INT
)

CREATE NODE TABLE Sentence (
    id INT PRIMARY KEY,
    doc_id INT,
    text STRING
)

// EDGES (replaces SQLite occurrences table)
CREATE REL TABLE HAS_ROOT (
    FROM Sentence TO Root,
    role STRING,
    grammar STRING  -- JSON string
)
```

## Implementation Steps

### Step 1: Create Kuzu index builder
- `scripts/build_kuzu_index.py`
- Stream corpus like current `build_root_index.py`
- Insert nodes and edges in batches
- Compute doc_freq after loading

### Step 2: Create Kuzu-backed retriever
- `klareco/rag/kuzu_inverted_index.py`
- Same interface as `RootInvertedIndex`
- Methods: `get_occurrences()`, `has_root()`, `get_doc_frequency()`

### Step 3: Integrate with ASTAwareRetriever
- Add option to use Kuzu backend
- Keep SQLite as fallback during testing

### Step 4: Validate equivalence
- Run same queries on both backends
- Verify identical results
- Benchmark latency

### Step 5: Remove SQLite code
- Once Kuzu validated
- Update build scripts
- Delete SQLite-specific code

## Files to Create
- `scripts/build_kuzu_index.py`
- `klareco/rag/kuzu_inverted_index.py`

## Files to Modify
- `klareco/rag/ast_aware_retriever.py` (add Kuzu backend option)

## Files to Delete (after validation)
- SQLite conversion code in `root_inverted_index.py`

## Success Criteria
- [ ] Kuzu index builds successfully from corpus
- [ ] All 4.2M sentences indexed
- [ ] `get_occurrences(root)` returns same results as SQLite
- [ ] `get_doc_frequency(root)` returns same values
- [ ] Query latency ≤ SQLite
- [ ] Memory usage ≤ 200MB at query time
- [ ] Q&A benchmark scores unchanged

## Memory Efficiency

Kuzu is disk-based like SQLite:
- Data stored on disk, not in RAM
- Buffer manager handles caching
- Spills to disk during bulk loading
- ~100-200MB RAM at query time (vs ~50MB SQLite)

## Dependencies
- `pip install kuzu`

## Blocks (do these AFTER this task)
- Task #139: Synonym traversal in Kuzu
- Task #140: Sentence context edges
- Task #141: Role-aware pattern matching
- Task #142: Multi-hop reasoning queries
