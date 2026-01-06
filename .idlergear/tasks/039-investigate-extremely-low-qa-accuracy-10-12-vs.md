---
id: 39
title: Investigate extremely low Q&A accuracy (10-12% vs expected 60-80%)
state: open
created: '2026-01-05T15:22:17.456765Z'
labels:
- bug
- research
- 'priority: high'
---
## Investigation Complete - Root Cause Identified

### Key Finding
**Answers ARE in the corpus, but retrieval is FAILING to find them**

### Evidence
- MemoryMapped: 0% accuracy (ALL empty results - shape mismatch bug, now fixed)
- Other retrievers: 10-12% accuracy (returning completely irrelevant documents)
- Example: Q "Who founded Esperanto?" → Returns docs about tropical rainforests, Italian novels
- Corpus verification: "ZAMENHOF, Aŭtoro de la lingvo Esperanto" EXISTS in corpus!

### Root Cause
Compositional embeddings not capturing semantic meaning:
1. **Proper nouns** (Zamenhof, Bjalistoko) parsed as unknown → poor embeddings
2. **Slot matching too strict** - fragments without verbs don't match question structure
3. **Possible embedding collapse** - different domains too similar

### Detailed Analysis
See Note #58 for full investigation results, evidence, and recommended fixes (P0-P3 priorities).

### Next Action (P0)
Re-run benchmark with fixed MemoryMapped retriever:
```bash
./scripts/benchmark_qa_all.sh --index data/indexes/slot_verified
```

If accuracy remains low, implement P1 fixes:
- Proper noun boosting
- Full-text fallback for low slot coverage
- Semantic similarity model integration (Stage 1)
