---
id: 145
title: Populate empty synonym_expansion in Tier 3 benchmark query t3_002
state: open
created: '2026-01-08T15:28:36.018712Z'
labels:
- bug
- benchmark
priority: low
---
## Problem
In `data/benchmarks/retrieval_benchmark_v1.json`, query t3_002 has an empty synonym_expansion:

```json
"synonym_expansion": {
  "establ": []
}
```

## Fix
Query the Kuzu graph to get actual synonyms for "establ" and populate the array.

Expected synonyms based on graph: fond, kre, inaŭgur, fari

## File
`data/benchmarks/retrieval_benchmark_v1.json`
