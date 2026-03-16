# Classification Script Performance Analysis

## Problem

Classify 1.2M Radiko nodes by setting 3 properties (nivelo, fonto, ofteco) in Kuzu database.

**Requirements:**
- Memory-safe (can't load all nodes into RAM)
- Resumable via checkpoints
- Classification logic is fast (Python dict lookups)

## Three Approaches Tested

### Approach 1: Individual UPDATE Queries (classify_roots_v2.py)

**Performance:** 200 nodes/sec (~100 minutes total)

**Method:**
```python
for radiko in batch:
    nivelo, fonto = classify(radiko)  # Fast
    conn.execute(f"MATCH (r:Radiko {{radiko: '{radiko}'}}) SET r.nivelo = '{nivelo}', ...")
```

**Pros:**
- Simple, reliable
- Good PRIMARY KEY index utilization
- Memory-safe (10K fetch batches)
- Checkpoint resume works

**Cons:**
- 1.2M individual UPDATE queries = network/checkpoint overhead
- Takes ~100 minutes

**Why it works:** Kuzu optimizes simple PRIMARY KEY lookups well.

---

### Approach 2: Batched UNWIND Queries (classify_roots_fast.py)

**Performance:** 55 nodes/sec (~6 hours total) ❌ FAILED

**Method:**
```python
# Fetch 50K nodes, classify in memory
# Build large UNWIND query:
query = f"""
    UNWIND [{{radiko: 'a', nivelo: 'tier1', ...}}, {{radiko: 'b', ...}}, ...] AS row
    MATCH (r:Radiko {{radiko: row.radiko}})
    SET r.nivelo = row.nivelo, r.fonto = row.fonto, r.ofteco = row.ofteco
"""
conn.execute(query)  # 100 nodes per query
```

**Pros:**
- Fewer database round-trips in theory

**Cons:**
- **3.6x SLOWER than individual queries!**
- String building/escaping overhead dominates
- Large inline arrays expensive to parse
- Kuzu query planner doesn't optimize this well

**Why it failed:** String manipulation overhead > network overhead savings.

---

### Approach 3: Transaction Batching (classify_roots_optimized.py)

**Performance:** Unknown (needs testing) - Expected 2-3x faster

**Method:**
```python
pending_updates = []
for radiko in batch:
    query = f"MATCH (r:Radiko {{radiko: '{radiko}'}}) SET ..."
    pending_updates.append(query)

    if len(pending_updates) >= 1000:
        # Execute all 1000 UPDATEs in one transaction
        for query in pending_updates:
            conn.execute(query)
        pending_updates.clear()
```

**Theory:**
- [Kuzu checkpoints after each write transaction](https://github.com/kuzudb/kuzu/issues/2529)
- Batching 1000 UPDATEs per transaction reduces checkpoint overhead
- Still uses simple PRIMARY KEY queries (fast)

**Pros:**
- Reduces "stop the world" checkpoint frequency
- Keeps simple, optimized query structure
- Memory-safe

**Cons:**
- **Uncertain if Kuzu Python API supports this** (might auto-commit each execute())
- Needs testing to validate performance gain

---

## Kuzu Performance Characteristics

Based on [Kuzu documentation](https://docs.kuzudb.com/concurrency/) and [GitHub discussions](https://github.com/kuzudb/kuzu/issues/2529):

**Concurrency:**
- ❌ No concurrent write transactions
- Single writer at a time
- Cannot parallelize with multiple connections

**Bulk Operations:**
- ✅ [COPY FROM is fastest](https://docs.kuzudb.com/import/csv/) for bulk inserts
- Optimized for initial loading, not updates
- Would require export → modify → reimport (risky for existing data)

**Transaction Model:**
- Checkpoints after each write transaction
- "Stop the world" phase during checkpoint
- Large transactions use bulk loading techniques

**Query Optimization:**
- PRIMARY KEY lookups are highly optimized
- Simple queries often faster than complex ones
- Vectorized and factorized execution

---

## Recommendations

### Option A: Accept Current Performance (RECOMMENDED for one-time operation)

**Use:** `./scripts/classify_roots.sh --resume`

- 200 nodes/sec
- ~100 minutes total (one-time cost)
- Proven, reliable, already running
- Resume from checkpoint works correctly

**Rationale:** For a one-time classification, 100 minutes is acceptable. Don't over-optimize.

---

### Option B: Test Transaction Batching (Worth trying)

**Use:** `./scripts/classify_roots_optimized.sh`

- Test on small sample first
- Expected 2-3x speedup (if Kuzu supports transaction batching)
- If faster, great! If not, fall back to Option A

**Test command:**
```bash
# Stop current script (Ctrl+C)
./scripts/classify_roots_optimized.sh

# Monitor for 5-10 minutes to verify speed improvement
# If speed is 400-500 nodes/sec → SUCCESS
# If speed is still ~200 nodes/sec → No benefit, use Option A
```

---

### Option C: Export-Transform-Reimport (NOT RECOMMENDED)

**High risk:**
- Must handle HAVAS_RADIKON relationships carefully
- Could corrupt database if not done correctly
- Complex, error-prone

**Only consider if:**
- This becomes a repeated operation (not one-time)
- Need to classify 10M+ nodes regularly

---

## Performance Bottleneck Analysis

**What's fast:**
- ✅ Classification logic (Python dict lookups): microseconds
- ✅ Fetching batches: ~1 second per 10K nodes
- ✅ PRIMARY KEY lookups in Kuzu: optimized

**What's slow:**
- ⏱️ Checkpoint overhead: "stop the world" after each write transaction
- ⏱️ 1.2M individual transactions = 1.2M checkpoints

**Why batched UNWIND failed:**
- String building overhead (Python) > checkpoint savings (Kuzu)
- Kuzu's query planner doesn't optimize large literal arrays well

---

## Conclusion

**For this one-time classification:**
1. Try Option B (optimized with transaction batching)
2. If no improvement, use Option A (current 200 nodes/sec)
3. Avoid Option C (export-reimport) due to risk

**Total time:** 30-100 minutes (acceptable for one-time operation)

**Future optimizations (if this becomes repeated):**
- Work with Kuzu team on bulk UPDATE APIs
- Consider materialized views or triggers
- Pre-classify during initial corpus loading

---

## Sources

- [Kuzu Concurrency Documentation](https://docs.kuzudb.com/concurrency/)
- [Kuzu Transaction Improvements (GitHub Issue)](https://github.com/kuzudb/kuzu/issues/2529)
- [Kuzu COPY FROM Documentation](https://docs.kuzudb.com/import/csv/)
- [Kuzu MERGE Operations](https://docs.kuzudb.com/cypher/data-manipulation-clauses/merge/)
