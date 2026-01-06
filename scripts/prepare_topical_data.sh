#!/bin/bash
# Wrapper script for prepare_topical_pairs.py
# Generates topical training pairs from unified corpus

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
    echo "Error: No virtual environment found (.venv or venv)"
    exit 1
fi

# Parse flags
FRESH_FLAG=""
RESUME_FLAG=""
MAX_SENTENCES=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --fresh)
            FRESH_FLAG="--fresh"
            shift
            ;;
        --resume)
            RESUME_FLAG="--resume"
            shift
            ;;
        --max-sentences)
            MAX_SENTENCES="--max-sentences $2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--fresh|--resume] [--max-sentences N]"
            exit 1
            ;;
    esac
done

# Create log directory
mkdir -p logs/training

# Run with logging
LOG_FILE="logs/training/prepare_topical_$(date +%Y%m%d_%H%M%S).log"

echo "Generating topical training pairs..."
echo "Log: $LOG_FILE"
echo ""

python scripts/data/prepare_topical_pairs.py \
    $FRESH_FLAG $RESUME_FLAG $MAX_SENTENCES \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "Complete! Log saved to $LOG_FILE"
