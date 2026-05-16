#!/bin/bash
# Warm the OS page cache with the Kuzu graph file before an eval.
#
# Profiling (commit history around the retrieval-query investigation)
# showed the eval's 30-130s/question is dominated by COLD-CACHE disk I/O
# on the 9.9 GB Kuzu file, not the query plan — the same queries run
# 2-16s warm. The file fits in available RAM (~22 GB free), so reading
# it once into page cache before the eval keeps every subsequent Kuzu
# traversal memory-resident.
#
# Zero correctness impact — this only changes where bytes are read from.
#
# Usage:
#   ./scripts/eval/warm_kuzu_cache.sh
#   ./scripts/eval/warm_kuzu_cache.sh data/indexes/v2.1_kuzu_index_full
set -euo pipefail

KUZU_DB="${1:-data/indexes/v2.1_kuzu_index_full}"

if [[ ! -e "$KUZU_DB" ]]; then
    echo "Kuzu DB not found: $KUZU_DB" >&2
    exit 1
fi

SIZE=$(du -h "$KUZU_DB" | cut -f1)
AVAIL_GB=$(free -g | awk '/^Mem:/{print $7}')
echo "Warming page cache: $KUZU_DB ($SIZE), ${AVAIL_GB}G RAM available"

t0=$(date +%s)
# Read the whole file sequentially into page cache.
cat "$KUZU_DB" > /dev/null
t1=$(date +%s)

echo "Cache warmed in $((t1 - t0))s"
# Report how much of the file is now resident if we can tell cheaply.
if command -v vmtouch >/dev/null 2>&1; then
    vmtouch "$KUZU_DB" 2>/dev/null | grep -E "Resident|percent" || true
fi
