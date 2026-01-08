#!/bin/bash
# Tiered Retrieval Benchmark Script
#
# Runs the tiered retrieval benchmark in the background with logging.
#
# Usage:
#   ./scripts/evaluate_retrieval.sh              # Run all tiers
#   ./scripts/evaluate_retrieval.sh --tier 1     # Run only Tier 1
#   ./scripts/evaluate_retrieval.sh --tier 1 2   # Run Tiers 1 and 2
#   ./scripts/evaluate_retrieval.sh -v           # Verbose output

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Activate venv
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Error: No virtual environment found"
    exit 1
fi

# Create logs directory
LOG_DIR="$PROJECT_ROOT/logs/benchmark"
mkdir -p "$LOG_DIR"

# Generate log filename with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/retrieval_${TIMESTAMP}.log"

echo "Running tiered retrieval benchmark..."
echo "Log file: $LOG_FILE"
echo ""

# Run the benchmark
python scripts/evaluate_retrieval.py "$@" 2>&1 | tee "$LOG_FILE"

echo ""
echo "Benchmark complete. Log saved to: $LOG_FILE"
