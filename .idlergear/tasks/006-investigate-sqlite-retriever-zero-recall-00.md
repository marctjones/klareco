---
id: 6
title: Investigate sqlite retriever zero recall (0.0% accuracy)
state: closed
created: '2026-01-03T06:24:32.490867Z'
labels:
- bug
- 'priority: high'
---
## Problem

The SQLite retriever completed benchmark but shows **zero recall** on all accuracy metrics:

```
Recall@10:   0.000 (0.0%)
MRR:         0.000
NDCG@10:     0.000
```

This means it's not retrieving any relevant documents for the 50 test queries.

## Evidence

From `benchmark_results/sqlite_20260103_004515.log`:
- Database created successfully: "Created SQLite database with 277 documents"
- Wait, that's wrong! Should be 4,229,277 documents, not just 277!
- Then it says: "Loaded 4,229,277 documents"

This suggests the database creation reported the wrong count initially.

## Root Cause Hypothesis

1. **Database schema issue**: Possible bug in SQLite slot retriever's table schema
2. **Query generation issue**: Queries might not match the database structure
3. **Embedding mismatch**: Embeddings stored vs queried might differ
4. **Index missing**: Database might be missing indexes for efficient lookup

## Next Steps

1. Check `klareco/rag/slot_retriever_sqlite.py` implementation
2. Inspect the created database: `data/indexes/slot_full/slot_index.db`
3. Run manual query against database to verify data is present
4. Compare with working retrievers (check what they return for same query)

## Performance Notes

Despite zero accuracy, the SQLite retriever is fast:
- Mean latency: 152ms
- Memory: 681MB (very efficient)
- Completed 50 queries in 7.7s

So performance is good - just not returning correct results.
