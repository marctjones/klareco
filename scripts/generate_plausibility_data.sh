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
    echo "No venv found. Please create one:"
    echo "  python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Parse arguments
MODE="full"  # full or test
if [[ "$1" == "--test" ]]; then
    MODE="test"
    shift
fi

# Configuration
if [[ "$MODE" == "test" ]]; then
    echo "Running in TEST mode (100K examples)"
    NUM_POSITIVES=50000
    NUM_NEGATIVES=50000
    OUTPUT_DIR="data/plausibility_training_test"
else
    echo "Running in FULL mode (9M examples)"
    NUM_POSITIVES=4500000
    NUM_NEGATIVES=4500000
    OUTPUT_DIR="data/plausibility_training"
fi

SVO_TRIPLES="data/semantic_types/svo_triples_all.jsonl"

# Check if SVO triples exist
if [ ! -f "$SVO_TRIPLES" ]; then
    echo "ERROR: SVO triples file not found: $SVO_TRIPLES"
    echo "Please run SVO extraction first:"
    echo "  python scripts/extract_svo_triples.py \\"
    echo "    --source kuzu \\"
    echo "    --db-path data/indexes/v2.1_kuzu_index_full \\"
    echo "    --output $SVO_TRIPLES"
    exit 1
fi

# Create log directory
LOG_DIR="logs/plausibility_data"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/generation_$(date +%Y%m%d_%H%M%S).log"

echo "Starting plausibility training data generation..."
echo "Mode: $MODE"
echo "Positives: $NUM_POSITIVES"
echo "Negatives: $NUM_NEGATIVES"
echo "Output: $OUTPUT_DIR"
echo "Log: $LOG_FILE"
echo ""

# Run generation
python scripts/generate_plausibility_training_data.py \
    --svo-triples "$SVO_TRIPLES" \
    --output-dir "$OUTPUT_DIR" \
    --num-positives $NUM_POSITIVES \
    --num-negatives $NUM_NEGATIVES \
    --train-split 0.9 \
    --seed 42 \
    --log-level INFO \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "Generation complete!"
echo "Training data: $OUTPUT_DIR/train.jsonl"
echo "Validation data: $OUTPUT_DIR/val.jsonl"
echo "Statistics: $OUTPUT_DIR/stats.json"
echo "Log: $LOG_FILE"
