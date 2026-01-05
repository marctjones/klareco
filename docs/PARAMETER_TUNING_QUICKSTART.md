# Parameter Tuning Quick Start

## TL;DR - Quick Commands

### Compare with Different Speeds

```bash
# Fast mode (2-3x faster, slight accuracy drop)
python scripts/compare_retrievers.py \
    --index data/indexes/slot_full \
    --retrievers scann \
    --prefilter-n 100 --rerank-n 20

# Balanced (default)
python scripts/compare_retrievers.py \
    --index data/indexes/slot_full \
    --retrievers scann

# Accurate mode (2-3x slower, best recall)
python scripts/compare_retrievers.py \
    --index data/indexes/slot_full \
    --retrievers scann \
    --prefilter-n 2000 --rerank-n 500
```

### Automated Parameter Sweep

```bash
# Test all 3 configurations and save results
./scripts/tune_retriever_params.sh
```

## Key Parameters (in order of impact)

| Parameter | Fast | Balanced | Accurate |
|-----------|------|----------|----------|
| `--prefilter-n` | 100 | 500 | 2000 |
| `--rerank-n` | 20 | 100 | 500 |
| `-k` (results) | 5 | 10 | 20 |

## What These Mean

**`--prefilter-n`**: How many candidates to get from vector search (HNSW/ScaNN/FAISS)
- Lower = faster, may miss relevant docs
- Higher = slower, better recall
- Default: 500

**`--rerank-n`**: How many to rerank with slot-based scoring
- Lower = faster, less refinement
- Higher = slower, better quality
- Default: 100

**`-k`**: Number of final results
- Lower = faster
- Higher = more results
- Default: 10

## Expected Performance

Based on 4.2M document index:

### ScaNN
| Mode | Prefilter | Rerank | Latency | Use Case |
|------|-----------|--------|---------|----------|
| Fast | 100 | 20 | ~10ms | Interactive UI |
| Balanced | 500 | 100 | ~18ms | Standard search |
| Accurate | 2000 | 500 | ~50ms | Research/analysis |

### HNSW
| Mode | Prefilter | Rerank | Latency | Use Case |
|------|-----------|--------|---------|----------|
| Fast | 100 | 20 | ~20ms | Quick searches |
| Balanced | 500 | 100 | ~66ms | Standard |
| Accurate | 2000 | 500 | ~150ms | Deep search |

## How to Choose

**Use Fast mode when:**
- Building interactive UI
- Need <30ms response
- Okay with 5-10% recall drop

**Use Balanced mode when:**
- General purpose search
- 50-100ms acceptable
- Want good accuracy

**Use Accurate mode when:**
- Research/analysis tasks
- Recall is critical
- 100-200ms acceptable
- Offline processing

## Testing Your Configuration

```bash
# 1. Run comparison with your params
python scripts/compare_retrievers.py \
    --index data/indexes/slot_full \
    --retrievers scann \
    --prefilter-n 1000 \
    --rerank-n 200 \
    --output my_config.json

# 2. Check the results
cat my_config.json | jq '.[] | {name, avg_time, memory_peak}'

# 3. If good, use in production
```

## Advanced: Build-Time Parameters

For even more control, rebuild indexes with different parameters:

```bash
# HNSW - Fast build (M=8)
# Edit scripts/build_hnsw_index.sh: M=8, ef_construction=100
./scripts/build_hnsw_index.sh

# ScaNN - High accuracy (num_leaves=4000)
# Edit scripts/build_scann_index.sh: num_leaves=4000, reorder_k=200
./scripts/build_scann_index.sh
```

See `docs/RETRIEVER_PARAMETER_TUNING.md` for complete details.

## Common Patterns

### Pattern 1: Progressive Enhancement
Start fast, fall back to accurate if needed:

```python
# Try fast first
results = retriever.search(query, hnsw_top_n=100, slot_top_n=20)

if len(results) < 5:
    # Fall back to accurate
    results = retriever.search(query, hnsw_top_n=2000, slot_top_n=500)
```

### Pattern 2: Query Type Routing
Different params for different query types:

```python
if is_short_query(query):  # "Esperanto"
    # Fast for simple queries
    results = retriever.search(query, hnsw_top_n=100, slot_top_n=20)
else:  # "Kiam Zamenhof kreis Esperanton?"
    # Accurate for complex queries
    results = retriever.search(query, hnsw_top_n=1000, slot_top_n=200)
```

### Pattern 3: User Preference
Let users choose speed vs accuracy:

```python
if user_wants_fast:
    results = retriever.search(query, hnsw_top_n=100, slot_top_n=20)
elif user_wants_thorough:
    results = retriever.search(query, hnsw_top_n=2000, slot_top_n=500)
else:
    results = retriever.search(query)  # Default
```

## Troubleshooting

**Queries too slow?**
- Lower `--prefilter-n` (biggest impact)
- Lower `--rerank-n`
- Consider switching to ScaNN (fastest)

**Results not good enough?**
- Raise `--prefilter-n` (biggest impact)
- Raise `--rerank-n`
- Consider rebuilding index with higher M/num_leaves

**Out of memory?**
- Lower `--prefilter-n`
- Use Hybrid retriever (lowest memory: 1.1GB)

## Next Steps

1. Run `./scripts/tune_retriever_params.sh` to test all modes
2. Choose configuration based on your latency/accuracy needs
3. Update your application code with chosen parameters
4. Monitor performance in production
5. Iterate based on user feedback

For complete parameter reference, see:
- `docs/RETRIEVER_PARAMETER_TUNING.md` - Full parameter guide
- `scripts/compare_retrievers_USAGE.md` - Benchmarking tool usage
