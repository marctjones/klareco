# Klareco Project Organization

**Generated**: 2026-01-04
**Purpose**: Master project organization showing all tasks by milestone, priority, and dependencies

---

## Executive Summary

**Total Tasks**: 34 open tasks
**Critical Path**: Task #25 (Wikipedia in corpus) blocks all Q&A functionality
**M2 Goal**: 80% recall on factual Q&A queries

### Milestones Overview

| Milestone | Tasks | Status | Completion |
|-----------|-------|--------|------------|
| **M1: Basic Retrieval** | 2 | Blocked | 0% |
| **M2: 80% Recall Target** | 20 | In Progress | 30% |
| **M3: Future Enhancements** | 7 | Not Started | 0% |
| **Data Quality** | 5 | In Progress | 40% |

---

## 🚨 CRITICAL PATH: Task #25

**BLOCKS ALL M1/M2 WORK**

### Task #25: Build corpus using build_enhanced_corpus.py (includes Wikipedia)
- **Priority**: P0-CRITICAL
- **Status**: Open
- **Blocks**: Tasks #3, #9-34 (all retrieval improvements)
- **Impact**: Wikipedia data (4.2M sentences) exists in extraction but NOT in corpus/index
- **Root Cause**: Wrong corpus builder used (parse_corpus.py vs build_enhanced_corpus.py)
- **Action Required**:
  ```bash
  python scripts/build_enhanced_corpus.py \
      --gutenberg data/extracted/books_sentences.jsonl \
      --wikipedia data/extracted/wikipedia_sentences.jsonl \
      --output data/corpus/unified_corpus_v2.jsonl
  ```
- **Dependencies**: None (can run immediately)
- **Estimated Time**: ~30-60 minutes
- **Success Criteria**: Corpus contains 4.2M+ sentences including Wikipedia

**WHY THIS IS P0**: Without Wikipedia in the corpus:
- ❌ All factual Q&A queries fail ("Kiu kreis Esperanton?" returns wrong answers)
- ❌ 0% recall on Wikipedia-based questions
- ❌ Cannot test any retriever improvements (testing on incomplete data)
- ❌ All M2 work is premature optimization of broken system

**EVERYTHING ELSE DEPENDS ON THIS**

---

## Milestone 1: Basic Retrieval Working

**Goal**: Get deterministic retrieval working with complete data
**Target Recall**: 50-60% (baseline functionality)
**Status**: BLOCKED by Task #25

### M1 Tasks

| ID | Task | Priority | Status | Blocked By |
|----|------|----------|--------|------------|
| #25 | Build corpus with Wikipedia | P0-CRITICAL | Open | - |
| #1 | Include all Gutenberg texts | High | Open | #25 |

**M1 Completion Criteria**:
- [ ] Task #25: Corpus includes Wikipedia (4.2M sentences)
- [ ] Task #1: Corpus includes all 113 Gutenberg files (currently only 7)
- [ ] Corpus rebuilt and indexed
- [ ] Basic Q&A working: "Kiu kreis Esperanton?" → returns Zamenhof

**M1 Dependencies**:
```
#25 (Build corpus) → #1 (Include all texts) → Rebuild index → M1 COMPLETE
```

---

## Milestone 2: 80% Recall Target

**Goal**: Optimize retrievers to achieve 80%+ recall on factual Q&A
**Current**: 85% (FAISS), but on incomplete data
**Status**: In Progress (waiting on M1 completion)

### M2.1: P1 - FAISS Optimization (Quick Wins)

**Priority**: P1 (Highest after Task #25)
**Expected Impact**: 85% → 90% recall, 5ms → 3-4ms latency

| ID | Task | Status | Depends On | Impact |
|----|------|--------|------------|--------|
| #9 | Fix FAISS nlist (√N → 4×√N) | Open | #25 | +3-5% recall |
| #10 | Add HNSW quantizer | Open | #25, #9 | +1-2ms speed |
| #11 | Runtime tunable nprobe/efSearch | Open | #25, #9, #10 | Testing flexibility |
| #16 | **[META] Optimize FAISSSlotRetriever** | Open | #25 | Implements #9-11 |

**Dependencies**:
```
#25 → #16 (implements #9, #10, #11) → Benchmark → P1 COMPLETE
```

**Implementation Order**:
1. Wait for Task #25 completion
2. Implement #16 (includes all P1 optimizations)
3. Rebuild index with new parameters
4. Benchmark on complete data

### M2.2: P2 - Weighted Fusion (Biggest Win)

**Priority**: P2 (After P1 complete)
**Expected Impact**: 75% → 85-90% recall, 5ms → 1-2ms latency (3× faster!)

| ID | Task | Status | Depends On | Impact |
|----|------|--------|------------|--------|
| #12 | Weighted multi-slot fusion | Open | #25 | +10-15% recall vs intersection |
| #17 | **[META] Implement FusedMultiSlotRetriever** | Open | #25 | Implements #12 |

**Why P2 is bigger win than P1**:
- **3× speed improvement** (5ms → 1-2ms)
- Fixes MultiFAISS intersection problem (75% → 85-90%)
- Enables partial query matching
- Parallel slot queries (uses 16 cores)

**Dependencies**:
```
#25 → #17 (implements #12) → Benchmark → P2 COMPLETE
```

### M2.3: P3 - Memory-Mapped with Caching

**Priority**: P3 (Best accuracy, reduce latency)
**Expected Impact**: 90% recall (best!), 44ms → 2-5ms with cache

| ID | Task | Status | Depends On | Impact |
|----|------|--------|------------|--------|
| #7 | Auto-skip mmap for large indexes | Open | #25 | Prevent slow benchmarks |
| #18 | **[META] Add LRU cache to mmap retriever** | Open | #25, #7 | 10× latency on cached queries |

**Why P3**:
- Already has **best accuracy** (90%)
- Just needs caching for speed
- Low priority because other retrievers faster

**Dependencies**:
```
#25 → #7 (detection) + #18 (caching) → P3 COMPLETE
```

### M2.4: Additional Retriever Options

**Priority**: Medium
**Goal**: Provide alternative retrievers for different use cases

| ID | Task | Status | Depends On | Notes |
|----|------|--------|------------|-------|
| #13 | HNSW retriever (IndexHNSWFlat) | Open | #25 | 90-95% recall, simpler than FAISS |
| #14 | FAISS tier filtering (IDSelector) | Open | #25 | Authoritative sources only |

### M2.5: Aggressive Tuning (Optional)

**Priority**: Low (After M2.1-M2.3 complete)
**Goal**: Maximum accuracy for 30GB RAM systems

| ID | Task | Status | Depends On | Notes |
|----|------|--------|------------|-------|
| #19 | Aggressive FAISS tuning | Open | #16 | 90-95% recall, uses all 16 cores |

### M2.6: Slot-Based Improvements

**Priority**: Medium (Core feature)
**Goal**: Improve slot-based retrieval architecture

| ID | Task | Status | Depends On | Notes |
|----|------|--------|------------|-------|
| #3 | Implement slot-based retrieval | Open | #25 | Two-stage retrieval |
| #22 | Benchmark all slot retrievers | Open | #25, #3 | Compare performance |
| #23 | Choose production retriever | Open | #22 | Based on benchmarks |
| #24 | Update demo to use best retriever | Open | #23 | User-facing |
| #29 | Add visualization to benchmark | Open | #22 | Charts/graphs |

**Dependencies**:
```
#25 → #3 (slot retrieval) → #22 (benchmark) → #23 (choose) → #24 (update demo)
                                            ↘ #29 (visualizations)
```

---

## Milestone 3: Future Enhancements

**Goal**: Advanced features and optimizations
**Status**: Not Started
**Priority**: Low (after M2 complete)

### M3 Tasks

| ID | Task | Priority | Notes |
|----|----------|----------|-------|
| #4 | Graph-based embeddings (TreeLSTM/GNN) | Medium | Research needed |
| #15 | Product Quantization (PQ) | Low | Only if scaling >10M docs |
| #26 | ScaNN retriever investigation | Low | Higher accuracy than FAISS |
| #27 | Hybrid FAISS+Mmap exploration | Low | Combine strengths |
| #28 | SQLite retriever evaluation | Low | Database-backed |
| #20 | Add ScaNN to comparison | Low | After #26 |
| #21 | Add Hybrid to comparison | Low | After #27 |

**Note**: These are research/exploration tasks, not blocking M2 completion.

---

## Data Quality Improvements

**Goal**: Improve corpus coverage and quality
**Priority**: Medium (parallel track to M2)

### Wikipedia Data Tasks

| ID | Task | Priority | Status | Depends On |
|----|------|----------|--------|------------|
| #25 | **Build corpus with Wikipedia** | P0 | Open | - |
| #30 | Download complete Wikipedia dump | Medium | Open | #25 |
| #31 | Filter meta/discussion pages | Medium | Open | #25 |
| #32 | Research optimal tier weight | Low | Open | #25 |
| #33 | Add quality filtering | Low | Open | #25 |
| #34 | Add performance test suite | High | Open | #25 |

**Dependencies**:
```
#25 (include Wikipedia) → #30 (update dump)
                       → #31 (filter quality)
                       → #32 (tier weights)
                       → #33 (stub detection)
                       → #34 (test suite)
```

### Gutenberg Data Tasks

| ID | Task | Priority | Status | Depends On |
|----|------|----------|--------|------------|
| #1 | Include all 113 Gutenberg files | High | Open | #25 |

---

## Dependencies Visualization

```
CRITICAL PATH:
#25 (Build corpus with Wikipedia) [P0-CRITICAL]
  │
  ├─→ M1 TRACK
  │   └─→ #1 (Include all Gutenberg texts)
  │       └─→ Rebuild index
  │           └─→ M1 COMPLETE ✓
  │
  ├─→ M2 P1 TRACK (Quick wins)
  │   └─→ #16 (Optimize FAISS)
  │       ├─→ implements #9 (nlist fix)
  │       ├─→ implements #10 (HNSW quantizer)
  │       └─→ implements #11 (runtime tuning)
  │
  ├─→ M2 P2 TRACK (Biggest win - 3× faster!)
  │   └─→ #17 (FusedMultiSlotRetriever)
  │       └─→ implements #12 (weighted fusion)
  │
  ├─→ M2 P3 TRACK (Best accuracy)
  │   ├─→ #7 (Auto-skip mmap for large indexes)
  │   └─→ #18 (LRU cache for mmap)
  │
  ├─→ M2 CORE TRACK
  │   └─→ #3 (Slot-based retrieval)
  │       └─→ #22 (Benchmark all retrievers)
  │           ├─→ #23 (Choose production retriever)
  │           │   └─→ #24 (Update demo)
  │           └─→ #29 (Add visualizations)
  │
  ├─→ M2 OPTIONAL TRACK
  │   ├─→ #13 (HNSW retriever)
  │   ├─→ #14 (FAISS tier filtering)
  │   └─→ #19 (Aggressive tuning - after P1-P3 complete)
  │
  ├─→ DATA QUALITY TRACK
  │   ├─→ #30 (Update Wikipedia dump)
  │   ├─→ #31 (Filter meta pages)
  │   ├─→ #32 (Tier weights)
  │   ├─→ #33 (Quality filtering)
  │   └─→ #34 (Test suite)
  │
  └─→ M3 FUTURE TRACK (after M2 complete)
      ├─→ #4 (Graph embeddings)
      ├─→ #26 (ScaNN investigation)
      │   └─→ #20 (Add to comparison)
      ├─→ #27 (Hybrid investigation)
      │   └─→ #21 (Add to comparison)
      ├─→ #28 (SQLite evaluation)
      └─→ #15 (Product Quantization - only if >10M docs)
```

---

## Recommended Implementation Order

### PHASE 0: UNBLOCK (Week 1)
**CRITICAL**: Must complete before any other work

1. **Task #25**: Build corpus with Wikipedia
   - Run `build_enhanced_corpus.py`
   - Verify Wikipedia data included
   - Rebuild slot index
   - **Estimated**: 1-2 hours
   - **Blocks**: Everything else

### PHASE 1: M1 Completion (Week 1-2)

2. **Task #1**: Include all Gutenberg texts
   - Update extraction script
   - Rebuild books_sentences.jsonl
   - Rebuild corpus and index
   - **Estimated**: 2-4 hours
   - **Depends**: #25

3. **Validate M1**:
   - Test basic Q&A queries
   - Verify "Kiu kreis Esperanton?" returns Zamenhof
   - Measure baseline recall
   - **Estimated**: 1 hour

### PHASE 2: M2 Quick Wins (Week 2-3)

4. **Task #16** (P1): Optimize FAISS
   - Implement nlist=65536
   - Add HNSW quantizer (M=32)
   - Runtime tunable parameters
   - Rebuild FAISS index
   - **Estimated**: 4-6 hours
   - **Impact**: 85% → 90% recall, 5ms → 3-4ms

5. **Task #17** (P2): Fused Multi-Slot Retriever
   - Implement weighted fusion
   - Parallel slot queries
   - Benchmark vs FAISS
   - **Estimated**: 6-8 hours
   - **Impact**: 3× speedup (5ms → 1-2ms)

### PHASE 3: M2 Polish (Week 3-4)

6. **Task #18** (P3): Mmap with LRU cache
   - Add caching layer
   - Benchmark improvement
   - **Estimated**: 3-4 hours
   - **Impact**: 44ms → 2-5ms for cached

7. **Task #7**: Auto-skip mmap for large indexes
   - Detection logic
   - Update benchmark script
   - **Estimated**: 1-2 hours

8. **Task #22**: Benchmark all retrievers
   - Run comprehensive benchmark
   - Generate comparison report
   - **Estimated**: 2-3 hours (mostly compute time)

9. **Task #23**: Choose production retriever
   - Analyze benchmarks
   - Document recommendation
   - **Estimated**: 1 hour

10. **Task #24**: Update demo to use best retriever
    - Update demo script
    - Add parameter examples
    - **Estimated**: 1-2 hours

### PHASE 4: M2 Completion (Week 4)

11. **Task #34**: Add performance test suite
    - Create benchmark queries
    - Guard against regression
    - **Estimated**: 2-3 hours

12. **Task #3**: Document slot-based architecture
    - Update documentation
    - Add architecture diagrams
    - **Estimated**: 2-3 hours

13. **M2 Validation**:
    - Verify 80%+ recall on test queries
    - Measure latency across retrievers
    - Document results

### PHASE 5: Optional Enhancements (Week 5+)

14. **Task #13**: HNSW retriever (if needed)
15. **Task #14**: Tier filtering (if needed)
16. **Task #19**: Aggressive tuning (if needed)
17. **Data quality tasks** (#30-33): Parallel track

---

## Success Metrics

### M1 Success Criteria
- ✅ Corpus includes Wikipedia (4.2M+ sentences)
- ✅ Corpus includes all 113 Gutenberg files
- ✅ Basic Q&A working: "Kiu kreis Esperanton?" → Zamenhof in top-3
- ✅ Baseline recall measured (target: 50-60%)

### M2 Success Criteria
- ✅ **80%+ recall** on factual Q&A queries
- ✅ **<5ms latency** for production retriever
- ✅ At least 3 retrievers benchmarked (FAISS, Fused, Mmap)
- ✅ Production retriever chosen and documented
- ✅ Performance test suite guards against regression
- ✅ Memory usage <4GB (fits on any laptop)

### M3 Success Criteria
- ✅ Research tasks evaluated
- ✅ Graph embeddings prototyped (if beneficial)
- ✅ ScaNN/Hybrid retrievers tested (if beneficial)
- ✅ Documentation complete

---

## Risk Mitigation

### Risk 1: Task #25 takes longer than expected
**Mitigation**: Task #25 is well-understood and straightforward. If issues arise:
- Use existing `build_enhanced_corpus.py` script (already tested)
- Can run in parallel with other work after corpus built

### Risk 2: FAISS optimizations don't improve recall
**Mitigation**: Current FAISS already at 85%. Optimizations are conservative and follow FAISS best practices. Worst case: no regression.

### Risk 3: Weighted fusion underperforms vs intersection
**Mitigation**: Weighted fusion is theoretically superior (handles partial matches). Can always fall back to FAISS if needed.

### Risk 4: Memory usage exceeds 16GB RAM
**Mitigation**: All designs stay under 5GB. Current laptop has 30GB. No risk.

---

## Notes for Future Sessions

### When Resuming Work:

1. **First**: Check if Task #25 is complete
   ```bash
   python3 -c "import json; count=0; wiki=0
   with open('data/corpus/unified_corpus.jsonl') as f:
       for line in f:
           count+=1
           if json.loads(line).get('source_metadata',{}).get('type')=='wikipedia': wiki+=1
   print(f'Total: {count:,}, Wikipedia: {wiki:,} ({wiki/count*100:.1f}%)')"
   ```
   - If Wikipedia > 0%: ✅ Task #25 complete, proceed to M1/M2
   - If Wikipedia = 0%: ❌ Must complete Task #25 first

2. **Second**: Check current milestone status
   - Run benchmarks: `./scripts/benchmark_slot_retrievers.py`
   - Check recall metrics
   - Identify next task from recommended order

3. **Third**: Update this document
   - Mark completed tasks
   - Update priorities based on findings
   - Adjust estimated times

### Key Files to Monitor:
- `data/corpus/unified_corpus.jsonl` - Must include Wikipedia
- `data/indexes/slot_full/` - Current production index
- `benchmark_results/*.json` - Performance metrics
- `.idlergear/tasks/*.md` - Individual task details

---

**Last Updated**: 2026-01-04
**Next Review**: After Task #25 completion
