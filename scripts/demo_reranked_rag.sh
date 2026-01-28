#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Activate venv if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
fi

# Set PYTHONPATH to include project root
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Default settings
INDEX_DIR="data/indexes/kuzu_index"
TOP_K=5
QUERY=""
MAX_LENGTH=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --query)
            QUERY="$2"
            shift 2
            ;;
        --index-dir)
            INDEX_DIR="$2"
            shift 2
            ;;
        --top-k)
            TOP_K="$2"
            shift 2
            ;;
        --max-length)
            MAX_LENGTH="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --query TEXT        Query to test (default: runs all test queries)"
            echo "  --index-dir PATH    Path to Kuzu index (default: data/indexes/kuzu_index)"
            echo "  --top-k N           Number of results (default: 5)"
            echo "  --max-length N      Truncate text to N characters (default: no truncation)"
            echo "  --help              Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0"
            echo "  $0 --query 'Kio estas hundo?'"
            echo "  $0 --query 'Kie vivas la homoj?' --top-k 10"
            echo "  $0 --max-length 100  # Truncate long sentences"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "============================================================"
echo "RAG Demo with Reranking"
echo "============================================================"
echo "Index: $INDEX_DIR"
echo "Model: models/reranker/best_model.pt"
echo "Top-K: $TOP_K"
if [ -n "$QUERY" ]; then
    echo "Query: $QUERY"
fi
echo ""

# Build command
CMD="python scripts/demo_reranked_rag.py --index-dir $INDEX_DIR --top-k $TOP_K"
if [ -n "$QUERY" ]; then
    CMD="$CMD --query \"$QUERY\""
fi
if [ -n "$MAX_LENGTH" ]; then
    CMD="$CMD --max-length $MAX_LENGTH"
fi

# Run the demo
eval $CMD
