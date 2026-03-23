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
    echo "No venv found!"
    exit 1
fi

# Configuration
SVO_OUTPUT="data/semantic_types/svo_triples_quality.jsonl"
PLAUSIBILITY_OUTPUT="data/plausibility_training_quality"
LOG_DIR="logs/plausibility_quality_pipeline"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/pipeline_$(date +%Y%m%d_%H%M%S).log"

echo "============================================================" | tee "$LOG_FILE"
echo "QUALITY PLAUSIBILITY DATASET PIPELINE" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Step 1: Extract SVO triples (500K sentences → ~125K triples)
if [ -f "$SVO_OUTPUT" ]; then
    EXISTING_TRIPLES=$(wc -l < "$SVO_OUTPUT")
    echo "Found existing SVO triples: $SVO_OUTPUT ($EXISTING_TRIPLES triples)" | tee -a "$LOG_FILE"
    echo "Skipping extraction. To re-extract, delete: $SVO_OUTPUT" | tee -a "$LOG_FILE"
else
    echo "Step 1: Extracting SVO triples from 500K sentences..." | tee -a "$LOG_FILE"
    echo "Expected: ~125K triples (~25% extraction rate)" | tee -a "$LOG_FILE"
    echo "ETA: ~15-20 minutes" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"

    PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" python scripts/extract_svo_triples.py \
        --source jsonl \
        --corpus data/extracted/wikipedia_sentences.jsonl \
        --corpus data/extracted/books_sentences.jsonl \
        --output "$SVO_OUTPUT" \
        --max-sentences 500000 \
        --log-level INFO \
        2>&1 | tee -a "$LOG_FILE"

    EXTRACTED_TRIPLES=$(wc -l < "$SVO_OUTPUT")
    echo "" | tee -a "$LOG_FILE"
    echo "SVO extraction complete: $EXTRACTED_TRIPLES triples" | tee -a "$LOG_FILE"
fi

echo "" | tee -a "$LOG_FILE"

# Step 2: Generate quality-focused dataset (200K examples)
echo "Step 2: Generating quality-focused plausibility dataset..." | tee -a "$LOG_FILE"
echo "Target: 200K examples (100K positive + 100K negative)" | tee -a "$LOG_FILE"
echo "ETA: ~5-10 minutes" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" python scripts/generate_plausibility_training_data_quality.py \
    --svo-triples "$SVO_OUTPUT" \
    --output-dir "$PLAUSIBILITY_OUTPUT" \
    --num-examples 200000 \
    --min-confidence 0.9 \
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
echo "Next step: Train plausibility scorer" | tee -a "$LOG_FILE"
