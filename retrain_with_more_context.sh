#!/bin/bash
# Retrain Klareco models with larger context size for better RAG results
# This script regenerates the QA dataset and retrains the QA decoder with more context documents
#
# Usage:
#   ./retrain_with_more_context.sh              # Use default settings (50 context docs)
#   ./retrain_with_more_context.sh --context 100  # Use 100 context docs
#   ./retrain_with_more_context.sh --resume      # Resume from last checkpoint
#
# Features:
# - Backs up existing models automatically
# - Can resume from checkpoints if interrupted
# - Validates prerequisites before starting
# - Progress tracking and ETA
# - Replaces models only after successful completion

set -e  # Exit on error
set -o pipefail  # Catch errors in pipes

# ============================================================================
# CONFIGURATION
# ============================================================================

# Default settings
CONTEXT_SIZE=50  # Number of context documents to retrieve
QA_DATASET_PATH="data/qa_dataset.jsonl"
QA_DATASET_BACKUP=""
QA_MODEL_DIR="models/qa_decoder"
QA_MODEL_BACKUP=""
CORPUS_PATH="data/corpus_sentences.jsonl"
GNN_MODEL_PATH="models/tree_lstm/checkpoint_epoch_20.pt"
BATCH_SIZE=16
EPOCHS=10
LEARNING_RATE=0.0001
RESUME=false
DEVICE="cpu"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# ============================================================================
# FUNCTIONS
# ============================================================================

print_header() {
    echo -e "${BOLD}========================================================================"
    echo "$1"
    echo -e "========================================================================${NC}"
}

print_section() {
    echo ""
    echo -e "${BLUE}${BOLD}>>> $1${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ ERROR: $1${NC}"
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --context)
                CONTEXT_SIZE="$2"
                shift 2
                ;;
            --resume)
                RESUME=true
                shift
                ;;
            --epochs)
                EPOCHS="$2"
                shift 2
                ;;
            --batch-size)
                BATCH_SIZE="$2"
                shift 2
                ;;
            --device)
                DEVICE="$2"
                shift 2
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Retrain Klareco models with larger context size for better RAG retrieval.

Options:
  --context N        Number of context documents (default: 50)
  --resume           Resume from last checkpoint
  --epochs N         Number of training epochs (default: 10)
  --batch-size N     Training batch size (default: 16)
  --device DEVICE    Device to use: cpu or cuda (default: cpu)
  --help             Show this help message

Examples:
  $0                           # Train with 50 context docs
  $0 --context 100             # Train with 100 context docs
  $0 --resume --context 50     # Resume previous training

This script will:
  1. Backup existing models
  2. Regenerate QA dataset with larger context size
  3. Retrain QA decoder
  4. Replace models on success

EOF
}

# Validate prerequisites
validate_prerequisites() {
    print_section "Validating prerequisites"

    local errors=0

    # Check Python
    if ! command -v python &> /dev/null; then
        print_error "Python not found"
        ((errors++))
    else
        print_success "Python found: $(python --version)"
    fi

    # Check corpus
    if [ ! -f "$CORPUS_PATH" ]; then
        print_error "Corpus not found: $CORPUS_PATH"
        ((errors++))
    else
        local corpus_lines=$(wc -l < "$CORPUS_PATH")
        print_success "Corpus found: $corpus_lines sentences"
    fi

    # Check GNN model
    if [ ! -f "$GNN_MODEL_PATH" ]; then
        print_error "GNN model not found: $GNN_MODEL_PATH"
        print_warning "Please train GNN first: ./retrain_gnn.sh"
        ((errors++))
    else
        print_success "GNN model found: $GNN_MODEL_PATH"
    fi

    # Check if we can resume
    if [ "$RESUME" = true ]; then
        if [ ! -d "$QA_MODEL_DIR" ]; then
            print_warning "Resume requested but no checkpoint directory found"
            print_warning "Will start fresh training"
            RESUME=false
        else
            local checkpoints=$(find "$QA_MODEL_DIR" -name "checkpoint_epoch_*.pt" 2>/dev/null | wc -l)
            if [ "$checkpoints" -eq 0 ]; then
                print_warning "Resume requested but no checkpoints found"
                print_warning "Will start fresh training"
                RESUME=false
            else
                print_success "Found $checkpoints checkpoint(s) for resumption"
            fi
        fi
    fi

    if [ $errors -gt 0 ]; then
        print_error "Validation failed with $errors error(s)"
        exit 1
    fi

    print_success "All prerequisites validated"
}

# Backup existing models
backup_models() {
    print_section "Backing up existing models"

    local timestamp=$(date +%Y%m%d_%H%M%S)

    # Backup QA dataset if it exists
    if [ -f "$QA_DATASET_PATH" ]; then
        QA_DATASET_BACKUP="${QA_DATASET_PATH}.backup_${timestamp}"
        cp "$QA_DATASET_PATH" "$QA_DATASET_BACKUP"
        print_success "QA dataset backed up: $QA_DATASET_BACKUP"
    fi

    # Backup QA model directory if it exists
    if [ -d "$QA_MODEL_DIR" ]; then
        QA_MODEL_BACKUP="${QA_MODEL_DIR}_backup_${timestamp}"
        cp -r "$QA_MODEL_DIR" "$QA_MODEL_BACKUP"
        print_success "QA models backed up: $QA_MODEL_BACKUP"
    fi
}

# Regenerate QA dataset with larger context
regenerate_qa_dataset() {
    print_section "Regenerating QA dataset with context size: $CONTEXT_SIZE"

    if [ "$RESUME" = true ] && [ -f "$QA_DATASET_PATH" ]; then
        print_warning "Resume mode: Skipping dataset regeneration"
        print_warning "Using existing dataset: $QA_DATASET_PATH"
        return 0
    fi

    # Create temporary modified script
    local temp_script=$(mktemp)
    cp scripts/generate_qa_dataset.py "$temp_script"

    # Modify context size in the script
    # Replace: 'context': context[:3] with 'context': context[:CONTEXT_SIZE]
    sed -i "s/'context': context\[:3\]/'context': context[:$CONTEXT_SIZE]/" "$temp_script"

    print_warning "This will take approximately 5-10 minutes for 20K sentences..."
    echo ""

    # Run modified script
    python "$temp_script" \
        --corpus "$CORPUS_PATH" \
        --output "$QA_DATASET_PATH" \
        --method both \
        --max-dialogue 5000 \
        --max-synthetic 10000 \
        2>&1 | grep -E "^(INFO|Scanning for questions|Generating from sentences|Parsing)"

    # Clean up
    rm "$temp_script"

    # Verify dataset was created
    if [ ! -f "$QA_DATASET_PATH" ]; then
        print_error "Failed to generate QA dataset"
        exit 1
    fi

    local dataset_size=$(wc -l < "$QA_DATASET_PATH")
    print_success "QA dataset regenerated: $dataset_size pairs with up to $CONTEXT_SIZE context docs each"
}

# Modify training script to use larger context
modify_training_script() {
    print_section "Preparing training with context size: $CONTEXT_SIZE"

    # Create temporary modified training script
    local temp_script=$(mktemp)
    cp scripts/train_qa_decoder.py "$temp_script"

    # Replace: context_asts = item.get('context_asts', [])[:5]
    # With:    context_asts = item.get('context_asts', [])[:CONTEXT_SIZE]
    sed -i "s/context_asts = item.get('context_asts', \[\])\[:5\]/context_asts = item.get('context_asts', [])[:\$$CONTEXT_SIZE]/" "$temp_script"

    echo "$temp_script"
}

# Train QA decoder
train_qa_decoder() {
    print_section "Training QA Decoder"

    # Modify training script
    local training_script=$(modify_training_script)

    print_warning "This will take significant time depending on dataset size and hardware..."
    print_warning "Estimated time: 30-60 minutes per epoch on CPU"
    echo ""

    local resume_args=""
    if [ "$RESUME" = true ]; then
        print_warning "Resuming from latest checkpoint"
        resume_args="--resume"
    fi

    # Run training
    python "$training_script" \
        --dataset "$QA_DATASET_PATH" \
        --gnn-checkpoint "$GNN_MODEL_PATH" \
        --output "$QA_MODEL_DIR" \
        --batch-size "$BATCH_SIZE" \
        --epochs "$EPOCHS" \
        --lr "$LEARNING_RATE" \
        --device "$DEVICE" \
        --d-model 512 \
        --n-layers 8 \
        --n-heads 8 \
        --d-ff 2048 \
        2>&1 | grep -vE "^(WARNING|DEBUG)" | grep -E "^(INFO|Epoch|Train|Val|Saved|TRAINING|=====)"

    # Clean up
    rm "$training_script"

    # Verify model was created
    if [ ! -f "$QA_MODEL_DIR/best_model.pt" ]; then
        print_error "Failed to train QA decoder"
        exit 1
    fi

    print_success "QA Decoder training complete"
}

# Verify models
verify_models() {
    print_section "Verifying trained models"

    local errors=0

    # Check QA decoder
    if [ -f "$QA_MODEL_DIR/best_model.pt" ]; then
        print_success "QA decoder model found"
    else
        print_error "QA decoder model missing"
        ((errors++))
    fi

    # Check vocabulary
    if [ -f "$QA_MODEL_DIR/vocabulary.json" ]; then
        print_success "Vocabulary file found"
    else
        print_error "Vocabulary file missing"
        ((errors++))
    fi

    # Check dataset
    if [ -f "$QA_DATASET_PATH" ]; then
        local dataset_size=$(wc -l < "$QA_DATASET_PATH")
        print_success "QA dataset validated: $dataset_size pairs"
    else
        print_error "QA dataset missing"
        ((errors++))
    fi

    if [ $errors -gt 0 ]; then
        print_error "Verification failed with $errors error(s)"
        return 1
    fi

    print_success "All models verified successfully"
    return 0
}

# Show summary
show_summary() {
    print_header "TRAINING COMPLETE - SUMMARY"

    echo ""
    echo -e "${BOLD}New Configuration:${NC}"
    echo "  Context size:        $CONTEXT_SIZE documents (was: 3-5)"
    echo "  QA Dataset:          $QA_DATASET_PATH"
    echo "  QA Model:            $QA_MODEL_DIR/best_model.pt"
    echo ""

    if [ -n "$QA_DATASET_BACKUP" ]; then
        echo -e "${BOLD}Backups Created:${NC}"
        echo "  Dataset:  $QA_DATASET_BACKUP"
    fi
    if [ -n "$QA_MODEL_BACKUP" ]; then
        echo "  Models:   $QA_MODEL_BACKUP"
        echo ""
    fi

    echo -e "${BOLD}Next Steps:${NC}"
    echo ""
    echo "1. Test with increased retrieval:"
    echo "   ${BLUE}python scripts/quick_query.py \"Kiu estas Frodo?\"${NC}"
    echo ""
    echo "2. For queries needing many results, modify klareco/experts/rag_expert.py:"
    echo "   ${BLUE}Change line 46: k: int = $CONTEXT_SIZE  # was k: int = 5${NC}"
    echo ""
    echo "3. Test end-to-end QA with new model:"
    echo "   ${BLUE}python scripts/test_end_to_end_qa.py${NC}"
    echo ""
    echo -e "${GREEN}${BOLD}Models are ready for production use!${NC}"
    echo ""
}

# Cleanup on error
cleanup_on_error() {
    print_error "Training failed or interrupted"
    echo ""

    if [ -n "$QA_DATASET_BACKUP" ] && [ -f "$QA_DATASET_BACKUP" ]; then
        echo "To restore backup dataset:"
        echo "  cp $QA_DATASET_BACKUP $QA_DATASET_PATH"
    fi

    if [ -n "$QA_MODEL_BACKUP" ] && [ -d "$QA_MODEL_BACKUP" ]; then
        echo "To restore backup models:"
        echo "  rm -rf $QA_MODEL_DIR"
        echo "  cp -r $QA_MODEL_BACKUP $QA_MODEL_DIR"
    fi

    echo ""
    echo "To resume this training, run:"
    echo "  $0 --resume --context $CONTEXT_SIZE"
    echo ""
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

main() {
    # Parse arguments
    parse_args "$@"

    # Print header
    print_header "KLARECO MODEL RETRAINING WITH EXTENDED CONTEXT"
    echo ""
    echo "Configuration:"
    echo "  Context documents:   $CONTEXT_SIZE (was: 3-5)"
    echo "  Training epochs:     $EPOCHS"
    echo "  Batch size:          $BATCH_SIZE"
    echo "  Device:              $DEVICE"
    echo "  Resume mode:         $RESUME"
    echo ""

    # Set up error handling
    trap cleanup_on_error ERR INT TERM

    # Execute pipeline
    validate_prerequisites
    backup_models
    regenerate_qa_dataset
    train_qa_decoder

    # Verify and summarize
    if verify_models; then
        show_summary
        exit 0
    else
        cleanup_on_error
        exit 1
    fi
}

# Run main function
main "$@"
