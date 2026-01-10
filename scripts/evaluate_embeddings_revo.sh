#!/bin/bash
# Evaluate embedding quality using ReVo semantic relations
#
# Usage:
#   ./scripts/evaluate_embeddings_revo.sh              # Hybrid embeddings
#   ./scripts/evaluate_embeddings_revo.sh --linguistic # Linguistic only
#   ./scripts/evaluate_embeddings_revo.sh --topical    # Topical only
#   ./scripts/evaluate_embeddings_revo.sh --verbose    # Detailed output

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "ERROR: No virtual environment found"
    exit 1
fi

# Parse arguments
MODEL_TYPE="hybrid"
VERBOSE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --linguistic|-l)
            MODEL_TYPE="linguistic"
            shift
            ;;
        --topical|-t)
            MODEL_TYPE="topical"
            shift
            ;;
        --hybrid|-h)
            MODEL_TYPE="hybrid"
            shift
            ;;
        --verbose|-v)
            VERBOSE="--verbose"
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--linguistic|--topical|--hybrid] [--verbose]"
            exit 1
            ;;
    esac
done

echo "Evaluating $MODEL_TYPE embeddings using ReVo relations..."
python scripts/evaluate_embeddings_revo.py --model "$MODEL_TYPE" $VERBOSE
