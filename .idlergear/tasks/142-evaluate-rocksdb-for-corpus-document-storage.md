---
id: 142
title: Evaluate RocksDB for corpus document storage
state: open
created: '2026-01-08T02:00:43.312956Z'
labels:
- enhancement
- infrastructure
- storage
priority: low
---
## Background

Currently the 4.4M document corpus is stored as a 19GB JSONL file (`unified_corpus.jsonl`) with a numpy offset file (`doc_offsets.npy`) for O(1) random access via memory mapping.

## Potential Benefits of RocksDB

| Aspect | Current (mmap JSONL) | RocksDB |
|--------|---------------------|---------|
| **Compression** | None (19GB raw) | Built-in LZ4/Snappy (~4-6GB, ~70% smaller) |
| **Concurrent reads** | Single file lock issues | Lock-free reads |
| **Updates/deletes** | Requires full rewrite | In-place updates |
| **Range queries** | Not supported | Native (e.g., docs 1000-2000) |
| **Corruption recovery** | None | WAL + checksums |

## When to Consider

- Running multiple training jobs concurrently
- Frequently updating the corpus (add/remove documents)
- Deploying to production with disk constraints
- Need atomic operations to prevent corruption

## Current Status

The mmap approach works fine for a research project with a mostly-static corpus. This is a "nice to have" optimization, not a blocker.

## Implementation Notes

Python options:
- `rocksdict` - Modern, well-maintained
- `python-rocksdb` - Official bindings

Would replace:
- `documents.jsonl` + `doc_offsets.npy` → `documents.rocksdb/`

Integration points:
- `KuzuInvertedIndex.get_document()`
- `RootInvertedIndex.get_document()`
