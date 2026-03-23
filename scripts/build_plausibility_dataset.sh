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

# Configuration
SVO_OUTPUT="data/semantic_types/svo_triples_all.jsonl"
PLAUSIBILITY_OUTPUT="data/plausibility_training"
LOG_DIR="logs/plausibility_pipeline"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/build_$(date +%Y%m%d_%H%M%S).log"

echo "============================================================" | tee "$LOG_FILE"
echo "PLAUSIBILITY DATASET PIPELINE" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Step 1: Extract SVO triples from corpus
if [ -f "$SVO_OUTPUT" ]; then
    EXISTING_TRIPLES=$(wc -l < "$SVO_OUTPUT")
    echo "Found existing SVO triples file: $SVO_OUTPUT ($EXISTING_TRIPLES triples)" | tee -a "$LOG_FILE"
    echo "Skipping SVO extraction (file exists)." | tee -a "$LOG_FILE"
    echo "To re-extract, delete: $SVO_OUTPUT" | tee -a "$LOG_FILE"
else
    echo "Step 1: Extracting SVO triples from corpus..." | tee -a "$LOG_FILE"
    echo "This will take 1-2 hours for 5.4M sentences" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"

    PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" python scripts/extract_svo_triples.py \
        --source jsonl \
        --corpus data/extracted/wikipedia_sentences.jsonl \
        --corpus data/extracted/books_sentences.jsonl \
        --output "$SVO_OUTPUT" \
        --log-level INFO \
        2>&1 | tee -a "$LOG_FILE"

    EXTRACTED_TRIPLES=$(wc -l < "$SVO_OUTPUT")
    echo "" | tee -a "$LOG_FILE"
    echo "SVO extraction complete: $EXTRACTED_TRIPLES triples" | tee -a "$LOG_FILE"
fi

echo "" | tee -a "$LOG_FILE"

# Step 2: Generate plausibility training dataset
echo "Step 2: Generating plausibility training dataset..." | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Determine dataset size based on extracted triples
TOTAL_TRIPLES=$(wc -l < "$SVO_OUTPUT")
# Use all available triples for positives, generate same number for negatives
NUM_POSITIVES=$TOTAL_TRIPLES
NUM_NEGATIVES=$TOTAL_TRIPLES

echo "Dataset size:" | tee -a "$LOG_FILE"
echo "  Positive examples: $NUM_POSITIVES" | tee -a "$LOG_FILE"
echo "  Negative examples: $NUM_NEGATIVES" | tee -a "$LOG_FILE"
echo "  Total: $((NUM_POSITIVES + NUM_NEGATIVES))" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" python scripts/generate_plausibility_training_data.py \
    --svo-triples "$SVO_OUTPUT" \
    --output-dir "$PLAUSIBILITY_OUTPUT" \
    --num-positives "$NUM_POSITIVES" \
    --num-negatives "$NUM_NEGATIVES" \
    --train-split 0.9 \
    --seed 42 \
    --log-level INFO \
    2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "PIPELINE COMPLETE" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Output files:" | tee -a "$LOG_FILE"
echo "  SVO triples: $SVO_OUTPUT" | tee -a "$LOG_FILE"
echo "  Training data: $PLAUSIBILITY_OUTPUT/train.jsonl" | tee -a "$LOG_FILE"
echo "  Validation data: $PLAUSIBILITY_OUTPUT/val.jsonl" | tee -a "$LOG_FILE"
echo "  Statistics: $PLAUSIBILITY_OUTPUT/stats.json" | tee -a "$LOG_FILE"
echo "  Log: $LOG_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
