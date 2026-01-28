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
TOP_K=10
RERANK_TOP_K=50
QUERIES_FILE=""
OUTPUT_FILE=""
SHOW_EXAMPLES=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --queries)
            QUERIES_FILE="$2"
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
        --rerank-top-k)
            RERANK_TOP_K="$2"
            shift 2
            ;;
        --output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --show-examples)
            SHOW_EXAMPLES="--show-examples"
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --queries FILE          File with queries (one per line)"
            echo "  --index-dir PATH        Path to Kuzu index (default: data/indexes/kuzu_index)"
            echo "  --top-k N               Number of results to return (default: 10)"
            echo "  --rerank-top-k N        Number of candidates to rerank (default: 50)"
            echo "  --output FILE           Save results to JSON file"
            echo "  --show-examples         Show detailed ranking examples"
            echo "  --help                  Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0"
            echo "  $0 --show-examples"
            echo "  $0 --queries test_queries.txt --output results.json"
            echo "  $0 --top-k 20 --rerank-top-k 100"
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
echo "Reranker Benchmark"
echo "============================================================"
echo "Index: $INDEX_DIR"
echo "Top-K: $TOP_K"
echo "Rerank Top-K: $RERANK_TOP_K"
if [ -n "$QUERIES_FILE" ]; then
    echo "Queries: $QUERIES_FILE"
else
    echo "Queries: Default test set"
fi
if [ -n "$OUTPUT_FILE" ]; then
    echo "Output: $OUTPUT_FILE"
fi
echo ""

# Build command
CMD="python scripts/benchmark_reranker.py --index-dir $INDEX_DIR --top-k $TOP_K --rerank-top-k $RERANK_TOP_K"

if [ -n "$QUERIES_FILE" ]; then
    CMD="$CMD --queries $QUERIES_FILE"
fi

if [ -n "$OUTPUT_FILE" ]; then
    CMD="$CMD --output $OUTPUT_FILE"
fi

if [ -n "$SHOW_EXAMPLES" ]; then
    CMD="$CMD --show-examples"
fi

# Run benchmark
eval $CMD
