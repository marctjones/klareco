---
id: 62
title: Integrate AST-aware retriever into benchmark evaluation
state: open
created: '2026-01-05T21:25:42.493425Z'
labels:
- enhancement
- evaluation
priority: high
---
## Goal
Integrate the new AST-aware retriever into the Q&A benchmark evaluation system and measure performance.

## Context
- Completed all 5 AST-aware retrieval components (Tasks #49-53)
- 98 tests passing for all components
- Need to evaluate on real Q&A benchmark (33 questions)
- Current baseline: 10-12% accuracy (embedding-based)
- Target: 60-70% accuracy

## Progress
✅ AST-aware retriever integrated into benchmark script
✅ Test script created and verified working
✅ On-the-fly parsing implemented (ASTs not stored in index)
⚠️ Performance: Scans 10k documents with on-the-fly parsing (~30-60s per query)

## Next Steps
1. Run benchmark comparison (AST vs embedding retrievers)
2. Analyze results and failure patterns
3. Consider optimizations:
   - Pre-filtering with HNSW/FAISS before AST matching
   - Caching parsed ASTs
   - Increasing scan limit if needed

## Success Criteria
- Benchmark runs successfully with AST-aware retriever
- Performance metrics collected (accuracy, top-k recall)
- Clear comparison with baseline retrievers
- Failure analysis documented

## Expected Effort
2-3 hours remaining (integration complete, benchmarking next)

## Dependencies
- Requires slot-based index (data/indexes/slot_verified/) ✅
- All AST-aware components tested and working ✅
