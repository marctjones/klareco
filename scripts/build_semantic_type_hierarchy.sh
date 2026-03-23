#!/bin/bash
#
# Semantic Type Hierarchy Pipeline
#
# Builds automated semantic type hierarchy from corpus SVO patterns.
# Zero human annotation required.
#
# Pipeline:
#   1. Extract SVO triples from corpus
#   2. Cluster roots by verb co-occurrence patterns
#   3. Generate verb selectional preference constraints
#
# Usage:
#   ./scripts/build_semantic_type_hierarchy.sh [--fresh]
#
# Options:
#   --fresh  Start from scratch (re-extract SVO triples)
#
# Output:
#   data/semantic_types/svo_triples_full.jsonl      - SVO triples
#   data/semantic_types/semantic_types.json         - Root → type mappings
#   data/semantic_types/verb_constraints.json       - Verb selectional preferences
#   data/semantic_types/cluster_stats.json          - Clustering statistics
#   data/semantic_types/constraint_stats.json       - Constraint statistics
#
# Time estimate:
#   - SVO extraction: 4-8 hours (5.4M sentences)
#   - Clustering: 10-30 minutes (depends on corpus size)
#   - Constraints: 5-10 minutes
#   Total: ~5-9 hours
#

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Activate venv
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "❌ No virtual environment found"
    exit 1
fi

# Parse arguments
FRESH=false
if [[ "$1" == "--fresh" ]]; then
    FRESH=true
    echo "🔄 Running in FRESH mode (will re-extract SVO triples)"
fi

# Create output directory
OUTPUT_DIR="data/semantic_types"
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

# Logging
LOG_FILE="logs/semantic_type_hierarchy_$(date +%Y%m%d_%H%M%S).log"

echo "📊 Semantic Type Hierarchy Pipeline"
echo "======================================"
echo "Output: $OUTPUT_DIR"
echo "Log: $LOG_FILE"
echo ""

# Step 1: Extract SVO triples (if needed or --fresh)
SVO_FILE="$OUTPUT_DIR/svo_triples_full.jsonl"

if [ ! -f "$SVO_FILE" ] || [ "$FRESH" = true ]; then
    echo "Step 1/3: Extracting SVO triples from corpus..."
    echo "⏱️  Estimated time: 4-8 hours for 5.4M sentences"
    echo ""

    # Check if Kuzu database exists
    KUZU_DB="data/indexes/kuzu_v2.1"
    CORPUS_FILE="data/extracted/wikipedia_sentences.jsonl"

    if [ -d "$KUZU_DB" ]; then
        echo "Using Kuzu database (faster)"
        python scripts/extract_svo_triples.py \
            --source kuzu \
            --kuzu-path "$KUZU_DB" \
            --output "$SVO_FILE" \
            2>&1 | tee -a "$LOG_FILE"
    elif [ -f "$CORPUS_FILE" ]; then
        echo "Using JSONL corpus (comprehensive, slower)"
        python scripts/extract_svo_triples.py \
            --source jsonl \
            --corpus "$CORPUS_FILE" \
            --output "$SVO_FILE" \
            --max-sentences 5400000 \
            2>&1 | tee -a "$LOG_FILE"
    else
        echo "❌ No corpus source found (need Kuzu DB or JSONL corpus)"
        exit 1
    fi

    echo "✅ SVO extraction complete"
    echo ""
else
    echo "Step 1/3: SVO triples already exist ($SVO_FILE)"
    echo "         Use --fresh to re-extract"
    echo ""
fi

# Step 2: Cluster semantic types
TYPES_FILE="$OUTPUT_DIR/semantic_types.json"

echo "Step 2/3: Clustering semantic types..."
echo "⏱️  Estimated time: 10-30 minutes"
echo ""

python scripts/cluster_semantic_types.py \
    --input "$SVO_FILE" \
    --output "$TYPES_FILE" \
    --num-clusters 18 \
    --min-frequency 10 \
    --save-matrix \
    2>&1 | tee -a "$LOG_FILE"

echo "✅ Clustering complete"
echo ""

# Step 3: Generate verb constraints
CONSTRAINTS_FILE="$OUTPUT_DIR/verb_constraints.json"

echo "Step 3/3: Generating verb selectional constraints..."
echo "⏱️  Estimated time: 5-10 minutes"
echo ""

python scripts/generate_verb_constraints.py \
    --triples "$SVO_FILE" \
    --semantic-types "$TYPES_FILE" \
    --output "$CONSTRAINTS_FILE" \
    --min-frequency 5 \
    --smoothing-alpha 0.1 \
    2>&1 | tee -a "$LOG_FILE"

echo "✅ Constraint generation complete"
echo ""

# Summary
echo "======================================"
echo "✅ Pipeline complete!"
echo ""
echo "📁 Outputs:"
echo "   SVO triples:     $SVO_FILE"
echo "   Semantic types:  $TYPES_FILE"
echo "   Verb constraints: $CONSTRAINTS_FILE"
echo ""
echo "📊 Statistics:"
echo "   Cluster stats:   $OUTPUT_DIR/cluster_stats.json"
echo "   Constraint stats: $OUTPUT_DIR/constraint_stats.json"
echo ""
echo "📋 Log: $LOG_FILE"
echo ""
echo "🚀 Next step: Train Semantic Fact Validator"
echo "   python scripts/train_semantic_fact_validator.py"
