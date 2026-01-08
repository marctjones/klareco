---
id: 125
title: Evaluate RocksDB for fast document lookup
state: open
created: '2026-01-07T02:58:34.869840Z'
labels:
- enhancement
- infrastructure
- performance
priority: low
---
## Motivation

Our current document retrieval uses JSONL with offset-based seeking:

```python
def _get_document(self, doc_id: int) -> Dict:
    offset = self.doc_offsets[doc_id]
    with open(index_file, 'rb') as f:
        f.seek(offset)
        line = f.readline()
        return json.loads(line)
```

This works but:
- Requires pre-computed offset cache (24s to build, 17MB numpy file)
- Each lookup does disk seek + JSON parse
- No compression

## Proposed Solution

RocksDB is a fast key-value store (used by CockroachDB, TiKV) that provides:

```python
# Store
db.put(f"doc:{doc_id}".encode(), json.dumps(doc).encode())

# Retrieve - O(1) with LSM tree
doc = json.loads(db.get(f"doc:{doc_id}".encode()))
```

## Benefits

1. **O(1) lookups** - LSM tree instead of file seeking
2. **Built-in compression** - LZ4/Snappy reduces 30GB → ~10GB
3. **No offset cache** - RocksDB handles indexing internally
4. **Atomic writes** - Safe for incremental updates
5. **Column families** - Could separate text/slots/features

## Use Cases in Klareco

1. **Primary**: Replace `slot_index.jsonl` + offset cache
2. **Secondary**: Store parsed ASTs for documents (avoid re-parsing)
3. **Tertiary**: Cache compiled embeddings for common roots

## Schema Design

```
Key                     Value
--------------------------------------------
doc:{id}                {text, source, features}
slots:{id}              {SUBJ: [...], VERB: [...], OBJ: [...]}
embedding:{id}          [128 floats as bytes]
ast:{id}                {tipo: frazo, subjekto: ...}
```

## Tasks

1. [ ] Install python-rocksdb
2. [ ] Write migration script from JSONL → RocksDB
3. [ ] Implement RocksDBDocumentStore class
4. [ ] Benchmark lookup latency vs current seek-based approach
5. [ ] Measure storage size with compression

## Alignment with Architecture

- **Memory efficient**: RocksDB is disk-backed with smart caching
- **No training required**: Pure infrastructure change
- **Compatible with HNSW**: HNSW returns IDs, RocksDB fetches documents
