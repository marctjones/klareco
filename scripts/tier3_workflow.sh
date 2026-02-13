#!/bin/bash
#
# Tier 3: Active Learning Workflow
#
# Complete workflow for highest quality model training through
# iterative active learning.
#
# Usage:
#   ./scripts/tier3_workflow.sh <iteration> <command>
#
# Commands:
#   init              - Initialize iteration 0 (extract initial data)
#   train             - Train model on current iteration data
#   find-uncertain    - Find uncertain predictions for annotation
#   annotate          - Prepare annotation interface
#   merge             - Merge annotations and prepare next iteration
#   evaluate          - Evaluate current model
#   status            - Show pipeline status
#

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
    echo "ERROR: No virtual environment found"
    exit 1
fi

# Defaults
ITERATION="${1:-0}"
COMMAND="${2:-status}"

# Paths
OUTPUT_DIR="data/training/active_learning"
MODEL_DIR="models/active_learning"

# Functions

function show_help() {
    echo "Tier 3: Active Learning Workflow"
    echo ""
    echo "Usage:"
    echo "  $0 <iteration> <command>"
    echo ""
    echo "Commands:"
    echo "  init              - Initialize iteration 0 (extract initial data)"
    echo "  train             - Train model on current iteration data"
    echo "  find-uncertain    - Find uncertain predictions for annotation"
    echo "  annotate          - Prepare annotation interface"
    echo "  merge             - Merge annotations and prepare next iteration"
    echo "  evaluate          - Evaluate current model"
    echo "  status            - Show pipeline status"
    echo ""
    echo "Typical workflow:"
    echo "  1. ./scripts/tier3_workflow.sh 0 init"
    echo "  2. ./scripts/tier3_workflow.sh 0 train"
    echo "  3. ./scripts/tier3_workflow.sh 1 find-uncertain"
    echo "  4. ./scripts/tier3_workflow.sh 1 annotate"
    echo "  5. [Manually annotate in browser, download annotations.json]"
    echo "  6. ./scripts/tier3_workflow.sh 1 merge"
    echo "  7. ./scripts/tier3_workflow.sh 1 train"
    echo "  8. ./scripts/tier3_workflow.sh 1 evaluate"
    echo "  9. Repeat steps 3-8 for iterations 2, 3, ... until accuracy plateaus"
    echo ""
}

function cmd_init() {
    echo "="
    echo "ITERATION 0: INITIALIZE"
    echo "="
    echo ""
    echo "Extracting initial training data from corpus..."
    echo "  Focus: Semantic gap (confidence < 0.7)"
    echo "  Format: Real sentences with full context"
    echo "  Target: 10,000 examples"
    echo ""

    python scripts/active_learning_pipeline.py \
        --iteration 0 \
        --action extract \
        --target-examples 10000 \
        --conf-high 0.7 \
        --max-sentences 100000

    echo ""
    echo "✓ Iteration 0 initialized"
    echo ""
    echo "Next step: Train initial model"
    echo "  ./scripts/tier3_workflow.sh 0 train"
}

function cmd_train() {
    echo "="
    echo "ITERATION $ITERATION: TRAIN MODEL"
    echo "="
    echo ""

    TRAIN_DATA="$OUTPUT_DIR/iteration_${ITERATION}_train.jsonl"
    VAL_DATA="$OUTPUT_DIR/iteration_${ITERATION}_val.jsonl"
    MODEL_OUTPUT="$MODEL_DIR/iteration_${ITERATION}"

    if [ ! -f "$TRAIN_DATA" ]; then
        echo "ERROR: Training data not found: $TRAIN_DATA"
        echo "Run: ./scripts/tier3_workflow.sh $ITERATION init"
        exit 1
    fi

    echo "Training on:"
    echo "  Train: $TRAIN_DATA"
    echo "  Val: $VAL_DATA"
    echo "  Output: $MODEL_OUTPUT"
    echo ""

    # Create symlinks for training script (expects train.jsonl and val.jsonl)
    TEMP_DATA_DIR="$OUTPUT_DIR/temp_iteration_$ITERATION"
    mkdir -p "$TEMP_DATA_DIR"
    ln -sf "$(realpath $TRAIN_DATA)" "$TEMP_DATA_DIR/train.jsonl"
    ln -sf "$(realpath $VAL_DATA)" "$TEMP_DATA_DIR/val.jsonl"

    python scripts/train_entity_classifier.py \
        --data "$TEMP_DATA_DIR" \
        --output "$MODEL_OUTPUT" \
        --epochs 50 \
        --batch-size 32 \
        --patience 5

    # Cleanup temp dir
    rm -rf "$TEMP_DATA_DIR"

    echo ""
    echo "✓ Model trained"
    echo ""
    if [ "$ITERATION" -eq 0 ]; then
        echo "Next step: Find uncertain predictions for annotation"
        echo "  ./scripts/tier3_workflow.sh 1 find-uncertain"
    else
        echo "Next step: Evaluate model"
        echo "  ./scripts/tier3_workflow.sh $ITERATION evaluate"
    fi
}

function cmd_find_uncertain() {
    if [ "$ITERATION" -eq 0 ]; then
        echo "ERROR: Cannot find uncertain for iteration 0"
        echo "Run: ./scripts/tier3_workflow.sh 1 find-uncertain"
        exit 1
    fi

    echo "="
    echo "ITERATION $ITERATION: FIND UNCERTAIN PREDICTIONS"
    echo "="
    echo ""

    PREV_ITERATION=$((ITERATION - 1))
    echo "Using model from iteration $PREV_ITERATION to find uncertain cases..."
    echo ""

    python scripts/active_learning_pipeline.py \
        --iteration "$ITERATION" \
        --action find-uncertain \
        --uncertainty-threshold 0.7 \
        --sample-size 500 \
        --max-sentences 50000

    echo ""
    echo "✓ Found uncertain predictions"
    echo ""
    echo "Next step: Prepare annotation interface"
    echo "  ./scripts/tier3_workflow.sh $ITERATION annotate"
}

function cmd_annotate() {
    echo "="
    echo "ITERATION $ITERATION: PREPARE ANNOTATION"
    echo "="
    echo ""

    HTML_OUTPUT="$OUTPUT_DIR/iteration_${ITERATION}_annotate.html"

    python scripts/active_learning_pipeline.py \
        --iteration "$ITERATION" \
        --action prepare-annotation \
        --output-html "$HTML_OUTPUT"

    echo ""
    echo "✓ Annotation interface ready"
    echo ""
    echo "To annotate:"
    echo "  1. Open: file://$(realpath $HTML_OUTPUT)"
    echo "  2. Review each example in context"
    echo "  3. Select correct tier3_type"
    echo "  4. Download annotations.json when done"
    echo ""
    echo "Next step (after annotating):"
    echo "  ./scripts/tier3_workflow.sh $ITERATION merge"
}

function cmd_merge() {
    echo "="
    echo "ITERATION $ITERATION: MERGE ANNOTATIONS"
    echo "="
    echo ""

    # Check for annotations file
    ANNOTATIONS_FILE="annotations.json"
    if [ ! -f "$ANNOTATIONS_FILE" ]; then
        echo "ERROR: annotations.json not found in current directory"
        echo ""
        echo "Please:"
        echo "  1. Complete annotation in browser"
        echo "  2. Download annotations.json"
        echo "  3. Move to project root: $PROJECT_ROOT"
        echo "  4. Re-run: ./scripts/tier3_workflow.sh $ITERATION merge"
        exit 1
    fi

    echo "Merging annotations from: $ANNOTATIONS_FILE"
    echo ""

    python scripts/active_learning_pipeline.py \
        --iteration "$ITERATION" \
        --action merge-annotations \
        --annotations-file "$ANNOTATIONS_FILE"

    # Backup annotations file
    BACKUP_FILE="$OUTPUT_DIR/iteration_${ITERATION}_annotations_backup.json"
    cp "$ANNOTATIONS_FILE" "$BACKUP_FILE"
    echo "✓ Backed up annotations to: $BACKUP_FILE"
    echo ""

    echo "✓ Annotations merged"
    echo ""
    echo "Next step: Train on merged data"
    echo "  ./scripts/tier3_workflow.sh $ITERATION train"
}

function cmd_evaluate() {
    echo "="
    echo "ITERATION $ITERATION: EVALUATE MODEL"
    echo "="
    echo ""

    MODEL_PATH="$MODEL_DIR/iteration_${ITERATION}"
    VAL_DATA="$OUTPUT_DIR/iteration_${ITERATION}_val.jsonl"

    if [ ! -d "$MODEL_PATH" ]; then
        echo "ERROR: Model not found: $MODEL_PATH"
        echo "Run: ./scripts/tier3_workflow.sh $ITERATION train"
        exit 1
    fi

    echo "Evaluating:"
    echo "  Model: $MODEL_PATH"
    echo "  Validation: $VAL_DATA"
    echo ""

    python scripts/evaluate_entity_classifier.py \
        --model "$MODEL_PATH" \
        --data "$VAL_DATA" \
        --output "$OUTPUT_DIR/iteration_${ITERATION}_metrics.json"

    echo ""
    echo "✓ Evaluation complete"
    echo ""

    # Show metrics
    if [ -f "$OUTPUT_DIR/iteration_${ITERATION}_metrics.json" ]; then
        echo "Results:"
        python -m json.tool "$OUTPUT_DIR/iteration_${ITERATION}_metrics.json" | grep -E "(accuracy|f1_score)" || true
        echo ""
    fi

    NEXT_ITERATION=$((ITERATION + 1))
    echo "Next iteration: $NEXT_ITERATION"
    echo "  ./scripts/tier3_workflow.sh $NEXT_ITERATION find-uncertain"
    echo ""
    echo "Or stop if accuracy has plateaued (< 1% improvement)"
}

function cmd_status() {
    echo "="
    echo "TIER 3: ACTIVE LEARNING STATUS"
    echo "="
    echo ""

    echo "Iterations completed:"
    echo ""

    for i in {0..10}; do
        ITER_TRAIN="$OUTPUT_DIR/iteration_${i}_train.jsonl"
        ITER_MODEL="$MODEL_DIR/iteration_${i}"
        ITER_METRICS="$OUTPUT_DIR/iteration_${i}_metrics.json"

        if [ -f "$ITER_TRAIN" ]; then
            TRAIN_COUNT=$(wc -l < "$ITER_TRAIN")
            echo -n "  Iteration $i: "

            if [ -d "$ITER_MODEL" ]; then
                echo -n "✓ Trained "
            else
                echo -n "○ Data ready, not trained "
            fi

            if [ -f "$ITER_METRICS" ]; then
                ACCURACY=$(python -c "import json; print(json.load(open('$ITER_METRICS')).get('accuracy', 'N/A'))")
                echo -n "- Accuracy: $ACCURACY "
            fi

            echo "($TRAIN_COUNT examples)"
        fi
    done

    echo ""
    echo "Data directory: $OUTPUT_DIR"
    echo "Model directory: $MODEL_DIR"
    echo ""
}

# Main
case "$COMMAND" in
    init)
        cmd_init
        ;;
    train)
        cmd_train
        ;;
    find-uncertain)
        cmd_find_uncertain
        ;;
    annotate)
        cmd_annotate
        ;;
    merge)
        cmd_merge
        ;;
    evaluate)
        cmd_evaluate
        ;;
    status)
        cmd_status
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo "ERROR: Unknown command: $COMMAND"
        echo ""
        show_help
        exit 1
        ;;
esac
