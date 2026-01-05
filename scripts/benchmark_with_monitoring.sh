#!/bin/bash
#
# Memory-safe benchmark wrapper
# Monitors memory usage and kills if exceeds threshold
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Configuration
MEMORY_THRESHOLD_GB=25  # Kill if exceeds 25GB (leave 5GB buffer)
CHECK_INTERVAL=10        # Check every 10 seconds

# Parse args to pass through
BENCHMARK_ARGS="$@"

echo "════════════════════════════════════════════════════════════════════"
echo "Memory-Safe Benchmark Wrapper"
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "Configuration:"
echo "  Memory threshold: ${MEMORY_THRESHOLD_GB}GB"
echo "  Check interval:   ${CHECK_INTERVAL}s"
echo "  System RAM:       $(free -h | awk '/^Mem:/ {print $2}')"
echo "  Available RAM:    $(free -h | awk '/^Mem:/ {print $7}')"
echo ""
echo "════════════════════════════════════════════════════════════════════"
echo ""

# Start benchmark in background
echo "Starting benchmark..."
./scripts/benchmark_all_retrievers.sh $BENCHMARK_ARGS &
BENCHMARK_PID=$!

echo "Benchmark PID: $BENCHMARK_PID"
echo ""
echo "Monitoring memory usage (Ctrl+C to stop)..."
echo ""

# Monitor memory
while kill -0 $BENCHMARK_PID 2>/dev/null; do
    # Get memory usage in GB
    USED_GB=$(free -g | awk '/^Mem:/ {print $3}')
    AVAILABLE_GB=$(free -g | awk '/^Mem:/ {print $7}')

    # Get benchmark process memory if exists
    BENCH_MEM_MB=$(ps -p $BENCHMARK_PID -o rss= 2>/dev/null | awk '{print int($1/1024)}' || echo "0")

    # Check threshold
    if [ "$USED_GB" -gt "$MEMORY_THRESHOLD_GB" ]; then
        echo ""
        echo "⚠️  MEMORY THRESHOLD EXCEEDED!"
        echo "  Used: ${USED_GB}GB / Threshold: ${MEMORY_THRESHOLD_GB}GB"
        echo "  Benchmark process: ${BENCH_MEM_MB}MB"
        echo ""
        echo "Killing benchmark to prevent system freeze..."

        # Kill the entire process group
        kill -TERM -$BENCHMARK_PID 2>/dev/null || true
        sleep 2
        kill -KILL -$BENCHMARK_PID 2>/dev/null || true

        echo ""
        echo "✗ Benchmark terminated due to memory usage"
        echo ""
        echo "Recommendation:"
        echo "  - Use a smaller index for testing"
        echo "  - Skip memory-intensive retrievers (mmap)"
        echo "  - Add more RAM or swap space"
        exit 1
    fi

    # Print status every 10 checks (100 seconds)
    if [ $((SECONDS % 100)) -lt $CHECK_INTERVAL ]; then
        echo "[$(date +%H:%M:%S)] Memory: ${USED_GB}GB used, ${AVAILABLE_GB}GB available | Benchmark: ${BENCH_MEM_MB}MB"
    fi

    sleep $CHECK_INTERVAL
done

# Benchmark completed
BENCH_EXIT_CODE=$?

echo ""
if [ $BENCH_EXIT_CODE -eq 0 ]; then
    echo "✓ Benchmark completed successfully"
else
    echo "✗ Benchmark failed with exit code $BENCH_EXIT_CODE"
fi

exit $BENCH_EXIT_CODE
