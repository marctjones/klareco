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
OUTPUT_DIR="benchmark_results"
NUM_QUERIES=50
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --index)
            INDEX_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --num-queries)
            NUM_QUERIES="$2"
            shift 2
            ;;
        --include-baseline)
            INCLUDE_BASELINE=1
            shift 1
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Benchmark all slot-based retriever implementations."
            echo ""
            echo "Options:"
            echo "  --index DIR           Index directory (default: data/indexes/slot_full)"
            echo "  --output-dir DIR      Output directory for results (default: benchmark_results)"
            echo "  --num-queries N       Number of test queries (default: 50)"
            echo "  --include-baseline    Include baseline retriever (WARNING: loads full index into RAM)"
            echo "  --help                Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 --index data/indexes/slot_full --num-queries 100"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate index exists
if [ ! -f "$INDEX_DIR/slot_index.jsonl" ]; then
    echo "ERROR: Index not found at $INDEX_DIR/slot_index.jsonl"
    echo ""
    echo "Create an index first:"
    echo "  python scripts/index_slot_based.py \\"
    echo "    --corpus data/corpus/unified_corpus.jsonl \\"
    echo "    --output $INDEX_DIR"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║          SLOT RETRIEVER BENCHMARK SUITE                           ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Configuration:"
echo "  Index:        $INDEX_DIR"
echo "  Output:       $OUTPUT_DIR"
echo "  Queries:      $NUM_QUERIES"
echo "  Timestamp:    $TIMESTAMP"
echo ""

# Count index size
echo "Checking index..."
INDEX_SIZE=$(wc -l < "$INDEX_DIR/slot_index.jsonl")
INDEX_SIZE_MB=$(du -h "$INDEX_DIR/slot_index.jsonl" | cut -f1)
echo "  ✓ Found $INDEX_SIZE documents ($INDEX_SIZE_MB)"
echo ""

# Step 1: Create benchmark queries
echo "════════════════════════════════════════════════════════════════════"
echo "STEP 1: Creating Benchmark Queries"
echo "════════════════════════════════════════════════════════════════════"
echo ""

QUERIES_FILE="$OUTPUT_DIR/queries_${TIMESTAMP}.jsonl"

# Check if queries already exist
if [ -f "$INDEX_DIR/benchmark_queries.jsonl" ]; then
    echo "Found existing queries at $INDEX_DIR/benchmark_queries.jsonl"
    EXISTING_QUERY_COUNT=$(wc -l < "$INDEX_DIR/benchmark_queries.jsonl")
    if [ "$EXISTING_QUERY_COUNT" -eq "$NUM_QUERIES" ]; then
        echo "  ✓ Query count matches ($EXISTING_QUERY_COUNT queries)"
        echo "  → Reusing existing queries"
        cp "$INDEX_DIR/benchmark_queries.jsonl" "$QUERIES_FILE"
        echo ""
    else
        echo "  ⚠ Query count mismatch (found $EXISTING_QUERY_COUNT, need $NUM_QUERIES)"
        echo "  → Creating new queries"
        echo ""
        python scripts/benchmark_slot_retrievers.py \
            --index "$INDEX_DIR" \
            --create-queries \
            --num-queries "$NUM_QUERIES" \
            --queries "$QUERIES_FILE" \
            --solution baseline
    fi
else
    echo "Creating $NUM_QUERIES benchmark queries from index..."
    echo ""
    python scripts/benchmark_slot_retrievers.py \
        --index "$INDEX_DIR" \
        --create-queries \
        --num-queries "$NUM_QUERIES" \
        --queries "$QUERIES_FILE" \
        --solution baseline
fi

echo ""
echo "✓ Queries ready: $QUERIES_FILE"
echo ""

# Step 2: Benchmark each solution
# NOTE: Baseline skipped by default for large indexes (loads entire index into RAM)
# NOTE: mmap skipped for indexes >1M docs (too slow - 160s/query brute-force search)

# Determine which solutions to run based on index size
if [ "$INDEX_SIZE" -gt 1000000 ]; then
    echo "════════════════════════════════════════════════════════════════════"
    echo "ℹ️  Large Index Detected ($INDEX_SIZE docs)"
    echo "════════════════════════════════════════════════════════════════════"
    echo ""
    echo "Automatically skipping 'mmap' retriever (too slow for >1M docs):"
    echo "  • mmap uses brute-force search: ~160s per query on 4M docs"
    echo "  • Would take >2 hours for 50 queries"
    echo ""
    echo "Testing optimized retrievers only: faiss, multifaiss, sqlite"
    echo ""
    SOLUTIONS=("faiss" "multifaiss" "sqlite")
else
    echo "Testing all retrievers: mmap, faiss, multifaiss, sqlite"
    echo ""
    SOLUTIONS=("mmap" "faiss" "multifaiss" "sqlite")
fi

if [ "${INCLUDE_BASELINE:-0}" = "1" ]; then
    echo "════════════════════════════════════════════════════════════════════"
    echo "⚠️  WARNING: BASELINE RETRIEVER ENABLED"
    echo "════════════════════════════════════════════════════════════════════"
    echo ""
    echo "The baseline retriever loads the ENTIRE index into RAM."
    echo "  • For $INDEX_SIZE docs: ~$((INDEX_SIZE / 1000000 * 7))GB RAM required"
    echo "  • This may freeze your system if insufficient RAM"
    echo ""
    read -p "Press Ctrl+C to cancel, or Enter to continue..."
    echo ""
    SOLUTIONS=("baseline" "${SOLUTIONS[@]}")
fi

echo "════════════════════════════════════════════════════════════════════"
echo "STEP 2: Benchmarking Retrievers"
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "Testing ${#SOLUTIONS[@]} retrievers: ${SOLUTIONS[*]}"
echo "Expected total time: ~$((${#SOLUTIONS[@]} * 15))-$((${#SOLUTIONS[@]} * 25)) minutes"
echo ""

SOLUTION_NUM=0
for SOLUTION in "${SOLUTIONS[@]}"; do
    SOLUTION_NUM=$((SOLUTION_NUM + 1))

    echo ""
    echo "────────────────────────────────────────────────────────────────────"
    echo "[$SOLUTION_NUM/${#SOLUTIONS[@]}] Testing: $SOLUTION"
    echo "────────────────────────────────────────────────────────────────────"
    echo ""

    RESULT_FILE="$OUTPUT_DIR/${SOLUTION}_${TIMESTAMP}.json"
    LOG_FILE="$OUTPUT_DIR/${SOLUTION}_${TIMESTAMP}.log"

    # Check if already completed
    if [ -f "$RESULT_FILE" ]; then
        echo "ℹ️  Found existing results: $RESULT_FILE"
        RESULT_QUERIES=$(grep -o '"num_queries":[0-9]*' "$RESULT_FILE" | cut -d: -f2)
        if [ "$RESULT_QUERIES" = "$NUM_QUERIES" ]; then
            echo "  ✓ Result is complete ($RESULT_QUERIES queries)"
            echo "  → Skipping (delete file to re-run)"
            echo ""
            continue
        else
            echo "  ⚠ Result incomplete (found $RESULT_QUERIES, need $NUM_QUERIES)"
            echo "  → Re-running benchmark"
            echo ""
        fi
    fi

    echo "Starting $SOLUTION benchmark at $(date +'%H:%M:%S')..."
    echo ""

    START_TIME=$(date +%s)

    # Run benchmark with timeout (30 minutes max to accommodate initialization + queries)
    if timeout 1800 python scripts/benchmark_slot_retrievers.py \
        --index "$INDEX_DIR" \
        --queries "$QUERIES_FILE" \
        --solution "$SOLUTION" \
        --output "$RESULT_FILE" 2>&1 | tee "$LOG_FILE"; then

        END_TIME=$(date +%s)
        ELAPSED=$((END_TIME - START_TIME))
        ELAPSED_MIN=$((ELAPSED / 60))
        ELAPSED_SEC=$((ELAPSED % 60))

        echo ""
        echo "✓ $SOLUTION completed successfully in ${ELAPSED_MIN}m ${ELAPSED_SEC}s"
        echo "  Results: $RESULT_FILE"
        echo "  Log:     $LOG_FILE"
        echo ""
    else
        EXIT_CODE=$?
        END_TIME=$(date +%s)
        ELAPSED=$((END_TIME - START_TIME))

        if [ $EXIT_CODE -eq 124 ]; then
            echo ""
            echo "✗ $SOLUTION timed out after ${ELAPSED}s (>30 minutes)"
        else
            echo ""
            echo "✗ $SOLUTION failed with exit code $EXIT_CODE after ${ELAPSED}s"
        fi
        echo "  Check log: $LOG_FILE"
        echo ""
    fi

    echo ""
done

# Step 3: Combine results
echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "STEP 3: Combining Results"
echo "════════════════════════════════════════════════════════════════════"
echo ""

COMBINED_FILE="$OUTPUT_DIR/combined_${TIMESTAMP}.json"

# Create combined JSON from individual results
python -c "
import json
import sys
from pathlib import Path

output_dir = Path('$OUTPUT_DIR')
timestamp = '$TIMESTAMP'
# Only include solutions that were actually tested
solutions = [s for s in ['baseline', 'mmap', 'faiss', 'multifaiss', 'sqlite']
             if (output_dir / f'{s}_{timestamp}.json').exists()]

combined = {
    'index_path': '$INDEX_DIR',
    'queries_path': '$QUERIES_FILE',
    'num_queries': $NUM_QUERIES,
    'timestamp': '$TIMESTAMP',
    'solutions': []
}

for solution in solutions:
    result_file = output_dir / f'{solution}_{timestamp}.json'
    if result_file.exists():
        try:
            with open(result_file) as f:
                data = json.load(f)
                # Extract just the solution data
                if 'solutions' in data and len(data['solutions']) > 0:
                    combined['solutions'].append(data['solutions'][0])
        except Exception as e:
            print(f'Warning: Could not load {result_file}: {e}', file=sys.stderr)
            combined['solutions'].append({
                'retriever': solution,
                'error': str(e)
            })
    else:
        combined['solutions'].append({
            'retriever': solution,
            'error': 'benchmark did not complete'
        })

with open('$COMBINED_FILE', 'w') as f:
    json.dump(combined, f, indent=2)

print(f'Saved combined results to: $COMBINED_FILE')
"

echo ""

# Step 4: Generate visualizations
echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "STEP 4: Generating Visualizations"
echo "════════════════════════════════════════════════════════════════════"
echo ""

REPORT_FILE="$OUTPUT_DIR/report_${TIMESTAMP}.html"
CHARTS_DIR="$OUTPUT_DIR/charts_${TIMESTAMP}"

python scripts/visualize_benchmark_results.py \
    --results "$COMBINED_FILE" \
    --output "$REPORT_FILE"

echo ""
echo "✓ HTML report generated: $REPORT_FILE"
echo "✓ Charts saved to: $CHARTS_DIR/"
echo ""

# Step 5: Print summary
echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "FINAL SUMMARY"
echo "════════════════════════════════════════════════════════════════════"
echo ""

# Extract and print summary table
python -c "
import json

with open('$COMBINED_FILE') as f:
    results = json.load(f)

print('Solution         Mean (ms)    P95 (ms)     Memory (MB)  Recall@10')
print('-' * 70)

for sol in results['solutions']:
    if 'error' in sol:
        print(f\"{sol['retriever']:<16} ERROR: {sol['error']}\")
        continue

    name = sol['retriever']
    mean_lat = sol['latency']['mean_ms']
    p95_lat = sol['latency']['p95_ms']
    memory = sol['memory']['delta_mb']
    recall = sol.get('accuracy', {}).get('recall_at_10', 0.0)

    print(f'{name:<16} {mean_lat:<12.2f} {p95_lat:<12.2f} {memory:<12.1f} {recall:<10.3f}')

print()
"

echo ""
echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║                   ✓ BENCHMARK COMPLETE!                           ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Output Files:"
echo "  📊 Combined JSON:  $COMBINED_FILE"
echo "  📈 HTML Report:    $REPORT_FILE"
echo "  📉 Charts:         $CHARTS_DIR/"
echo ""
echo "View Results:"
echo "  firefox $REPORT_FILE"
echo "  # or"
echo "  google-chrome $REPORT_FILE"
echo ""
echo "Benchmark completed at $(date +'%H:%M:%S')"
echo ""
