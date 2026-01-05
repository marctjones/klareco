#!/bin/bash
# Demo script for slot-based retrieval
# Run this in a separate terminal to test retrievers

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

# Default values
INDEX="data/indexes/slot_full"
TOP_K=5
RERANK_TOP_N=500
INTERACTIVE=false
TRANSLATE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --index)
            INDEX="$2"
            shift 2
            ;;
        -k|--top-k)
            TOP_K="$2"
            shift 2
            ;;
        --rerank-top-n)
            RERANK_TOP_N="$2"
            shift 2
            ;;
        -i|--interactive)
            INTERACTIVE=true
            shift
            ;;
        --translate)
            TRANSLATE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --index PATH          Path to slot index (default: data/indexes/slot_full)"
            echo "  -k, --top-k N         Number of results (default: 5)"
            echo "  --rerank-top-n N      Reranking candidates (default: 500)"
            echo "  -i, --interactive     Interactive mode"
            echo "  --translate           Enable EO→EN translations"
            echo "  -h, --help            Show this help"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Demo mode, 5 results"
            echo "  $0 -i                                 # Interactive mode"
            echo "  $0 -i --translate                     # Interactive with translations"
            echo "  $0 -k 10 --rerank-top-n 1000         # More thorough search"
            echo "  $0 --index data/indexes/slot_test    # Use test index"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Build command
CMD="python scripts/demo_slot_retrieval.py --index $INDEX -k $TOP_K --rerank-top-n $RERANK_TOP_N"

if [ "$INTERACTIVE" = true ]; then
    CMD="$CMD -i"
fi

if [ "$TRANSLATE" = true ]; then
    CMD="$CMD --translate"
fi

# Run with logging
LOG_DIR="logs/demos"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/slot_demo_$(date +%Y%m%d_%H%M%S).log"

echo "========================================"
echo "Slot-Based Retrieval Demo"
echo "========================================"
echo "Index: $INDEX"
echo "Top-k: $TOP_K"
echo "Rerank candidates: $RERANK_TOP_N"
echo "Interactive: $INTERACTIVE"
echo "Translations: $TRANSLATE"
echo "Log file: $LOG_FILE"
echo "========================================"
echo ""

# Run the demo (with tee to both terminal and log file)
$CMD 2>&1 | tee "$LOG_FILE"
