#!/bin/bash
# Local-parallel benchmark for evaluate_extractive_qa.py.
# Runs the same 30-Q test set at increasing worker counts, capturing wall
# time so we can compare local-parallel vs Modal-parallel.
#
# Usage:
#   ./scripts/local_parallel_bench.sh
#   ./scripts/local_parallel_bench.sh "1 4 8 16"   # custom worker list
#
# Notes:
#   - First run pays cold OS page cache cost; subsequent runs are warm
#     because Linux caches the Whoosh + Kuzu files in RAM.
#   - We do a tiny warm-up run first so the comparison is fair (warm
#     cache for all worker configs).
set -euo pipefail
cd "$(dirname "$0")/.."

WORKERS="${1:-4 8 16}"
TEST_SET="data/test_sets/general_knowledge_30_keyed.jsonl"
mkdir -p logs/local_bench data/eval_results/local_bench

source .venv/bin/activate 2>/dev/null || source venv/bin/activate 2>/dev/null || true

echo "===== warm-up (first 3 questions, 1 worker, throwaway) ====="
python scripts/evaluate_extractive_qa.py \
    --test-set "$TEST_SET" --limit 3 --workers 1 \
    > /dev/null 2>&1 || true
echo "warm-up done."
echo

# Memory budget: split system RAM across workers so Kuzu's 80%-default
# buffer pool doesn't OOM us — without an explicit cap, N workers each
# try to grow up to 80% of system RAM. Per-worker budget needs to be
# large enough for multi-hop graph traversals (a 4 GB cap was observed
# to OOM on biographical queries against the v2.1 graph).
TOTAL_RAM_MB=$(awk '/MemTotal/ {print int($2/1024)}' /proc/meminfo)
TOTAL_CORES=$(nproc)

echo "===== timed runs (system: ${TOTAL_RAM_MB} MB RAM, ${TOTAL_CORES} cores) ====="
for w in $WORKERS; do
    out="data/eval_results/local_bench/general_knowledge_30_local_w${w}.json"
    log="logs/local_bench/local_w${w}.log"
    # Budget per worker: ~50% of RAM split N ways, capped at 8 GB
    buf_mb=$(( TOTAL_RAM_MB / 2 / w ))
    if [ "$buf_mb" -gt 8192 ]; then buf_mb=8192; fi
    threads=$(( TOTAL_CORES / w ))
    if [ "$threads" -lt 1 ]; then threads=1; fi
    echo "--- workers=$w  (kuzu_buf=${buf_mb}MB/worker, kuzu_threads=${threads}/worker) ---"
    KLARECO_KUZU_BUFFER_MB="$buf_mb" KLARECO_KUZU_MAX_THREADS="$threads" \
    /usr/bin/time -f "wall=%es maxRSS=%MKB" \
        python scripts/evaluate_extractive_qa.py \
        --test-set "$TEST_SET" --workers "$w" --output "$out" \
        > "$log" 2>&1
    grep -E "Wall-clock total|Latency  avg|Answer accuracy|Retrieval recall" "$log" | head -10
    grep "wall=" "$log" | tail -1
    echo
done

echo "===== summary ====="
for w in $WORKERS; do
    log="logs/local_bench/local_w${w}.log"
    wall=$(grep "Wall-clock total" "$log" | awk '{print $3}' | tr -d 's')
    avg=$(grep "Latency  avg" "$log" | awk '{print $4}' | tr -d 's/' | head -1)
    echo "workers=$w  wall=${wall}s  per-q-avg=${avg}s"
done
