# Whoosh Full-Text Search Integration

## Problem

Kuzu graph database lacks efficient full-text search capability:
- No BM25/TF-IDF ranking
- LIMIT returns arbitrary subsets (non-deterministic ordering)
- Cannot efficiently filter by keywords
- Retrieval bottleneck in extractive QA (see `RERANKER_DIAGNOSTIC_SESSION.md`)

## Solution

**Hybrid architecture**: Whoosh for retrieval + Kuzu for metadata
- **Whoosh index**: 5.4M sentences, BM25-ranked full-text search
- **Kuzu database**: AST structures, graph relationships, metadata
- **Integration**: Retrieve IDs from Whoosh, fetch ASTs from Kuzu

## Building the Index

```bash
# Build Whoosh index from Kuzu database
python scripts/build_whoosh_index.py

# Custom paths
python scripts/build_whoosh_index.py \
  --db data/indexes/v2.1_kuzu_index_full \
  --output data/indexes/whoosh_fts

# Rebuild from scratch
python scripts/build_whoosh_index.py --overwrite

# Test index
python scripts/build_whoosh_index.py --test
```

**Index Statistics**:
- Sentences indexed: 5,415,600
- Index size: ~1.4 GB (compressed)
- Build time: ~15 minutes (multithreaded)
- Query time: <100ms for typical queries

## Usage

### Python API

```python
from pathlib import Path
from klareco.rag.whoosh_retriever import WhooshRetriever

# Initialize retriever
retriever = WhooshRetriever(
    whoosh_index_dir=Path('data/indexes/whoosh_fts'),
    kuzu_db_path=Path('data/indexes/v2.1_kuzu_index_full')
)

# Search for sentences
query_roots = ['zamenhof', 'kre', 'fond', 'establ']
results = retriever.retrieve(
    query_roots=query_roots,
    top_k=20,
    retrieval_limit=1000
)

# Results include ASTs
for doc in results:
    print(doc['text'])
    print(f"  Matching roots: {doc['matching_roots']}")
    print(f"  Score: {doc['score']}")
    print(f"  AST: {doc['ast']['tipo']}")
```

### Integration with Extractive QA

Modify `scripts/demo_extractive_qa.py` to use Whoosh instead of Kuzu scanning:

```python
from klareco.rag.whoosh_retriever import WhooshRetriever

# Replace retrieve_sentences() function
retriever = WhooshRetriever(
    whoosh_index_dir=Path('data/indexes/whoosh_fts'),
    kuzu_db_path=args.db
)

sentences = retriever.retrieve(
    query_roots=list(query_roots),
    top_k=args.top_k,
    retrieval_limit=1000
)
```

## Architecture

### Index Schema

```python
from whoosh.fields import Schema, ID, TEXT

schema = Schema(
    id=ID(stored=True, unique=True),   # Sentence ID (for Kuzu lookup)
    text=TEXT(stored=True),             # Full sentence text
    text_lower=TEXT                     # Lowercased (for search)
)
```

### Query Flow

```
User Query: "Kiu fondis Esperanton?"
    ↓
Extract roots: [esperant, fond]
    ↓
Expand with synonyms: [esperant, fond, kre, establ, startig]
    ↓
Whoosh query: "esperant OR fond OR kre OR establ OR startig"
    ↓
BM25-ranked results (top 1000)
    ↓
Fetch ASTs from Kuzu by ID
    ↓
Score/filter/rerank → Top 20 sentences
    ↓
Generate answer
```

### Performance Comparison

| Method | Retrieval Time | Correct Answers | Coverage |
|--------|----------------|-----------------|----------|
| **Kuzu LIMIT 1000** | 50ms | 0/5 (0%) | Arbitrary subset |
| **Kuzu LIMIT 10000** | 500ms | 2/5 (40%) | Non-deterministic |
| **Whoosh BM25** | 80ms | Expected: 5/5 (100%) | Full corpus |

## Why Whoosh?

Compared to alternatives:

| Solution | Pros | Cons |
|----------|------|------|
| **SQLite FTS5** | Zero dependencies, fast | Limited flexibility |
| **Whoosh** | Pure Python, BM25, flexible | Slower than Rust |
| **Tantivy** | Very fast (Rust) | Compilation required |
| **Elasticsearch** | Enterprise features | Heavy (JVM), overkill |

**Whoosh wins** because:
1. Pure Python (easy to debug)
2. BM25 ranking built-in
3. Moderate scale (5M docs) is fine
4. `pip install Whoosh` - no system dependencies

## Maintenance

### Rebuilding Index

When corpus changes:

```bash
python scripts/build_whoosh_index.py --overwrite
```

### Incremental Updates

For small changes (adding new sentences):

```python
from whoosh.index import open_dir

ix = open_dir('data/indexes/whoosh_fts')
writer = ix.writer()

# Add new sentences
for sentence_id, text in new_sentences:
    writer.add_document(
        id=str(sentence_id),
        text=text,
        text_lower=text.lower()
    )

writer.commit()
```

### Monitoring

```bash
# Check index size
du -sh data/indexes/whoosh_fts

# Test query performance
python scripts/build_whoosh_index.py --test
```

## Troubleshooting

### Index Not Found

```python
FileNotFoundError: Index not found at data/indexes/whoosh_fts
```

**Solution**: Build the index first:
```bash
python scripts/build_whoosh_index.py
```

### Slow Queries

If queries take >500ms:
1. Check index optimization: `ix.optimize()` (merges segments)
2. Reduce `retrieval_limit` (default 1000 → 500)
3. Consider upgrading to Tantivy for better performance

### Memory Issues

If indexing crashes (OOM):
1. Reduce `limitmb` in `build_whoosh_index.py` (default 512 → 256)
2. Reduce `procs` (default 4 → 2)
3. Index in batches

## Next Steps

1. ✅ Build Whoosh index (5.4M sentences)
2. ⏳ Integrate into `demo_extractive_qa.py`
3. ⏳ Test "Kiu fondis Esperanton?" (should work!)
4. ⏳ Run evaluation on full test set (50 questions)
5. ⏳ Document performance improvements

---

**Last Updated**: 2026-03-25
**Author**: Claude Sonnet 4.5 (with Marc)
**Related**: RERANKER_DIAGNOSTIC_SESSION.md, DETERMINISTIC_VS_NEURAL_QA_TEST.md
