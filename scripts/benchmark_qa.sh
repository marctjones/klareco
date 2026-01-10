#!/bin/bash
# Run Q&A benchmark evaluation
# Usage: ./scripts/benchmark_qa.sh [--verbose]

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
    echo "No virtual environment found"
    exit 1
fi

# Create logs directory
mkdir -p logs

# Generate timestamp for log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/benchmark_qa_${TIMESTAMP}.log"

echo "=============================================="
echo "Q&A BENCHMARK EVALUATION"
echo "=============================================="
echo ""
echo "Started: $(date)"
echo "Log file: $LOG_FILE"
echo ""

# Run benchmark with unbuffered output
PYTHONUNBUFFERED=1 python scripts/evaluate_qa.py --system klareco "$@" 2>&1 | tee "$LOG_FILE"

echo ""
echo "=============================================="
echo "Complete! Log saved to: $LOG_FILE"
echo "=============================================="
