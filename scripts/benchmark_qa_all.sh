#!/bin/bash
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
    echo "ERROR: No venv found. Run: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Default arguments
INDEX_DIR="data/indexes/slot_full"
BENCHMARK_FILE="data/benchmarks/datasets/qa_benchmark_v1.jsonl"
OUTPUT_DIR="benchmark_results/qa"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TOP_K=10

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --index)
            INDEX_DIR="$2"
            shift 2
            ;;
        --benchmark)
            BENCHMARK_FILE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --top-k)
            TOP_K="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Benchmark all retriever implementations on Q&A task."
            echo ""
            echo "Options:"
            echo "  --index DIR           Index directory (default: data/indexes/slot_full)"
            echo "  --benchmark FILE      Q&A benchmark file (default: qa_benchmark_v1.jsonl)"
            echo "  --output-dir DIR      Output directory for results (default: benchmark_results/qa)"
            echo "  --top-k N             Number of results per query (default: 10)"
            echo "  --help                Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 --index data/indexes/slot_full --top-k 10"
            echo ""
            echo "This script tests 5 retrievers:"
            echo "  • mmap       - Memory-mapped (slow for large indexes)"
            echo "  • multifaiss - Multi-FAISS with IVF (fastest)"
            echo "  • hybrid     - FAISS + mmap fallback"
            echo "  • hnsw       - HNSW graph search"
            echo "  • scann      - Google ScaNN"
            echo ""
            echo "Measures:"
            echo "  • Top-1/5/10 accuracy: % questions with answer in top-N results"
            echo "  • Latency: Average query time in milliseconds"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate files exist
if [ ! -f "$BENCHMARK_FILE" ]; then
    echo "ERROR: Benchmark file not found: $BENCHMARK_FILE"
    echo ""
    echo "Available benchmarks:"
    ls -lh data/benchmarks/datasets/*.jsonl 2>/dev/null || echo "  (none found)"
    exit 1
fi

if [ ! -d "$INDEX_DIR" ]; then
    echo "ERROR: Index directory not found: $INDEX_DIR"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Print banner
echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║          Q&A RETRIEVAL BENCHMARK - ALL RETRIEVERS                 ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Configuration:"
echo "  Index:        $INDEX_DIR"
echo "  Benchmark:    $BENCHMARK_FILE"
echo "  Output:       $OUTPUT_DIR"
echo "  Top-K:        $TOP_K"
echo "  Timestamp:    $TIMESTAMP"
echo ""

# Count benchmark questions
NUM_QUESTIONS=$(wc -l < "$BENCHMARK_FILE")
echo "Benchmark: $NUM_QUESTIONS questions"
echo ""

# Check available retrievers
echo "Checking available retrievers..."
AVAILABLE_RETRIEVERS=()

if [ -d "$INDEX_DIR/mmap" ]; then
    AVAILABLE_RETRIEVERS+=("mmap")
    echo "  ✓ mmap"
fi

if [ -d "$INDEX_DIR/multifaiss" ]; then
    AVAILABLE_RETRIEVERS+=("multifaiss")
    echo "  ✓ multifaiss"
fi

if [ -d "$INDEX_DIR/faiss" ] && [ -d "$INDEX_DIR/mmap" ]; then
    AVAILABLE_RETRIEVERS+=("hybrid")
    echo "  ✓ hybrid (faiss + mmap)"
fi

if [ -d "$INDEX_DIR/hnsw" ] && [ -d "$INDEX_DIR/mmap" ]; then
    AVAILABLE_RETRIEVERS+=("hnsw")
    echo "  ✓ hnsw"
fi

if [ -d "$INDEX_DIR/scann" ] && [ -d "$INDEX_DIR/mmap" ]; then
    AVAILABLE_RETRIEVERS+=("scann")
    echo "  ✓ scann"
fi

if [ ${#AVAILABLE_RETRIEVERS[@]} -eq 0 ]; then
    echo ""
    echo "ERROR: No retrievers found in $INDEX_DIR"
    echo ""
    echo "Expected directory structure:"
    echo "  $INDEX_DIR/mmap/"
    echo "  $INDEX_DIR/multifaiss/"
    echo "  $INDEX_DIR/faiss/"
    echo "  $INDEX_DIR/hnsw/"
    echo "  $INDEX_DIR/scann/"
    exit 1
fi

echo ""
echo "Found ${#AVAILABLE_RETRIEVERS[@]} retrievers: ${AVAILABLE_RETRIEVERS[*]}"
echo ""

# Run benchmark
OUTPUT_FILE="$OUTPUT_DIR/qa_benchmark_${TIMESTAMP}.json"
LOG_FILE="$OUTPUT_DIR/qa_benchmark_${TIMESTAMP}.log"

echo "════════════════════════════════════════════════════════════════════"
echo "Running Q&A Benchmark"
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "This will test all retrievers on $NUM_QUESTIONS questions."
echo "Expected time: ~5-10 minutes for ${#AVAILABLE_RETRIEVERS[@]} retrievers"
echo ""
echo "Results will be saved to:"
echo "  JSON: $OUTPUT_FILE"
echo "  Log:  $LOG_FILE"
echo ""
echo "Starting at $(date +'%H:%M:%S')..."
echo ""

START_TIME=$(date +%s)

# Run Q&A evaluation
if python scripts/evaluate_qa.py \
    --output "$OUTPUT_FILE" 2>&1 | tee "$LOG_FILE"; then

    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    ELAPSED_MIN=$((ELAPSED / 60))
    ELAPSED_SEC=$((ELAPSED % 60))

    echo ""
    echo "╔════════════════════════════════════════════════════════════════════╗"
    echo "║                   ✓ Q&A BENCHMARK COMPLETE!                       ║"
    echo "╚════════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Completed in ${ELAPSED_MIN}m ${ELAPSED_SEC}s"
    echo ""
    echo "Output Files:"
    echo "  📊 Results: $OUTPUT_FILE"
    echo "  📝 Log:     $LOG_FILE"
    echo ""
    echo "Next Steps:"
    echo "  1. Review results in: $OUTPUT_FILE"
    echo "  2. Analyze with Claude Code (pass the JSON file)"
    echo "  3. Use evaluate_qa_with_llm.py for deeper analysis"
    echo ""
else
    EXIT_CODE=$?
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))

    echo ""
    echo "✗ Benchmark failed with exit code $EXIT_CODE after ${ELAPSED}s"
    echo "  Check log: $LOG_FILE"
    echo ""
    exit $EXIT_CODE
fi
