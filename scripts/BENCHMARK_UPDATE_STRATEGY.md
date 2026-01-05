# Benchmark Progress Update Strategy

## Update Triggers (Logical OR)

The benchmark now shows progress updates based on **multiple triggers** - whichever happens first:

### 1. ⏰ Time-Based (1 minute)
**Trigger**: At least 60 seconds since last update

**Purpose**: Ensure regular updates even if queries are slow

**Example**:
```
[3/50] Latency: 2450.1ms | Recall: 0.820 | Memory: 2048MB | ETA: 25m 30s | ⏰ 1min
```
If queries average 30 seconds each, you'd only see 2 queries/minute. Time-based updates ensure you still get feedback.

### 2. 🔢 Query Milestones (every 5 queries)
**Trigger**: Query count is divisible by 5

**Purpose**: Regular progress for fast queries

**Example**:
```
[5/50] Latency: 125.3ms | Recall: 0.850 | Memory: 2048MB | ETA: 5m 30s
[10/50] Latency: 118.7ms | Recall: 0.862 | Memory: 2051MB | ETA: 4m 45s
```

### 3. 💾 Checkpoints (every 10 queries)
**Trigger**: Query count is divisible by 10

**Purpose**: Save progress to disk (resume support)

**Example**:
```
[10/50] Latency: 118.7ms | Recall: 0.862 | Memory: 2051MB | ETA: 4m 45s | 💾 checkpoint
[20/50] Latency: 116.8ms | Recall: 0.870 | Memory: 2350MB | ETA: 3m 30s | 💾 checkpoint
```

Checkpoint also saves:
- Query times so far
- Accuracy metrics
- Completed query count
→ Can resume if interrupted

### 4. ⚠️ Slow Query Events (2x avg latency)
**Trigger**: Query took >2x the average query time

**Purpose**: Immediately flag performance problems

**Example**:
```
# Average query time: 120ms
# Query 7 takes 287ms (2.4x average)
[7/50] Latency: 132.1ms | Recall: 0.845 | Memory: 2049MB | ETA: 5m 15s | ⚠ slow query: 287ms
```

**Why it matters**:
- Identifies problematic queries
- Helps debug index issues
- Catches outliers early

### 5. 📈 Memory Spike Events (>100MB increase)
**Trigger**: Memory increased >100MB in a single query

**Purpose**: Catch memory leaks or inefficient operations

**Example**:
```
# Memory was 2048MB before query
# Memory is 2342MB after query (+294MB)
[18/50] Latency: 117.3ms | Recall: 0.865 | Memory: 2342MB | ETA: 3m 45s | 📈 mem spike: +289MB
```

**Why it matters**:
- Catches memory leaks immediately
- Identifies queries causing high memory usage
- Prevents OOM crashes

### 6. ✅ Completion (final query)
**Trigger**: Last query completed

**Purpose**: Final status update

**Example**:
```
[50/50] Latency: 114.2ms | Recall: 0.880 | Memory: 2358MB | ETA: 0s
```

## Multiple Triggers

Updates can have **multiple indicators** when several triggers fire simultaneously:

```
[30/50] Latency: 113.2ms | Recall: 0.876 | Memory: 2358MB | ETA: 2m 15s | 💾 checkpoint ⏰ 1min
```
This means:
- Checkpoint (every 10 queries)
- AND 1 minute elapsed since last update

## Update Frequency

### Fast Queries (avg 100ms)
- 50 queries complete in ~5 seconds
- Updates: Every 5 queries (0.5s intervals)
- **Very frequent updates**

### Medium Queries (avg 1s)
- 50 queries complete in ~50 seconds
- Updates: Every 5 queries (~5s intervals) + 1 min timer
- **Regular updates**

### Slow Queries (avg 30s)
- 50 queries complete in ~25 minutes
- Updates: Time-based (every 1 min) + checkpoints
- **Guaranteed updates even when slow**

### Variable Speed Queries
- Regular: Every 5 queries
- Slow outliers: Immediate alert when query is 2x slower
- **Adaptive to query performance**

## Example Full Output

```
Starting benchmark run (50 queries)...

[5/50] Latency: 125.3ms | Recall: 0.850 | Memory: 2048MB | ETA: 5m 30s
[7/50] Latency: 132.1ms | Recall: 0.845 | Memory: 2049MB | ETA: 5m 15s | ⚠ slow query: 287ms
[10/50] Latency: 118.7ms | Recall: 0.862 | Memory: 2051MB | ETA: 4m 45s | 💾 checkpoint
[15/50] Latency: 115.2ms | Recall: 0.868 | Memory: 2053MB | ETA: 4m 10s
[18/50] Latency: 117.3ms | Recall: 0.865 | Memory: 2342MB | ETA: 3m 45s | 📈 mem spike: +289MB
[20/50] Latency: 116.8ms | Recall: 0.870 | Memory: 2350MB | ETA: 3m 30s | 💾 checkpoint
[25/50] Latency: 114.9ms | Recall: 0.873 | Memory: 2355MB | ETA: 2m 50s
[30/50] Latency: 113.2ms | Recall: 0.876 | Memory: 2358MB | ETA: 2m 15s | 💾 checkpoint ⏰ 1min
[35/50] Latency: 112.8ms | Recall: 0.878 | Memory: 2360MB | ETA: 1m 40s
[40/50] Latency: 111.9ms | Recall: 0.880 | Memory: 2362MB | ETA: 1m 05s | 💾 checkpoint
[45/50] Latency: 111.2ms | Recall: 0.882 | Memory: 2363MB | ETA: 0m 30s
[50/50] Latency: 110.8ms | Recall: 0.883 | Memory: 2365MB | ETA: 0s | 💾 checkpoint

✓ Benchmark completed in 156.8s
```

## Benefits

1. **Never wondering what's happening**
   - Always get updates within 1 minute maximum
   - See progress even if queries are slow

2. **Catch problems immediately**
   - Slow queries flagged as they happen
   - Memory spikes detected instantly

3. **Optimal update frequency**
   - Fast queries: Every 5 queries
   - Slow queries: Every 1 minute
   - Problem queries: Immediate

4. **Checkpoint safety**
   - Automatic saves every 10 queries
   - Resume from interruptions
   - No lost work

## Configuration

All triggers are hardcoded but tunable in code:

```python
# In benchmark_slot_retrievers.py

# Time-based update interval
is_time_based = time_since_last_update >= 60  # 1 minute

# Query milestone interval
is_query_milestone = (i + 1) % 5 == 0  # Every 5 queries

# Checkpoint interval
is_checkpoint = (i + 1) % 10 == 0  # Every 10 queries

# Slow query threshold
is_slow_query = query_time > avg_query_time * 2  # 2x average

# Memory spike threshold
is_memory_spike = current_mem > prev_mem + 100  # +100MB
```

To customize, edit these values before running.
