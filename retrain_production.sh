#!/bin/bash
# Production-Quality Retraining Script
# Retrains QA Decoder with optimal parameters for best quality
#
# Time estimate: 6-8 hours on average CPU
# Use --quick for 2-hour test run

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================================================"
echo "KLARECO - PRODUCTION QUALITY RETRAINING"
echo "========================================================================"
echo ""

# Parse arguments
QUICK_MODE=false
if [[ "$1" == "--quick" ]]; then
    QUICK_MODE=true
    echo "${YELLOW}⚡ QUICK TEST MODE${NC}"
    echo "   Using reduced parameters for 2-hour test"
    echo ""
fi

# Set parameters based on mode
if [ "$QUICK_MODE" = true ]; then
    # Quick test parameters (2 hours)
    MAX_PAIRS=5000
    CONTEXT_SIZE=20
    EPOCHS=10
    BATCH_SIZE=16
    OUTPUT_SUFFIX="_test"
    echo "Parameters:"
    echo "  - QA pairs: 5,000 (quick test)"
    echo "  - Context size: k=20"
    echo "  - Epochs: 10"
    echo "  - Batch size: 16"
    echo "  - Time estimate: ~2 hours"
else
    # Production parameters (6-8 hours)
    MAX_PAIRS=20000
    CONTEXT_SIZE=75
    EPOCHS=50
    BATCH_SIZE=32
    OUTPUT_SUFFIX="_production"
    echo "Parameters:"
    echo "  - QA pairs: 20,000 (production)"
    echo "  - Context size: k=75"
    echo "  - Epochs: 50"
    echo "  - Batch size: 32"
    echo "  - Time estimate: ~6-8 hours"
fi

echo ""
echo "========================================================================"
echo ""

# Create backup of existing models
BACKUP_DIR=".model_backups/$(date +%Y%m%d_%H%M%S)"
echo "${GREEN}Step 0: Backing up existing models${NC}"
echo "Backup location: $BACKUP_DIR"
mkdir -p "$BACKUP_DIR"

if [ -d "models/qa_decoder" ]; then
    echo "  Backing up QA decoder..."
    cp -r models/qa_decoder "$BACKUP_DIR/"
    echo "  ✓ QA decoder backed up"
fi

if [ -f "data/qa_dataset.jsonl" ]; then
    echo "  Backing up QA dataset..."
    cp data/qa_dataset.jsonl "$BACKUP_DIR/"
    echo "  ✓ QA dataset backed up"
fi

echo ""
echo "========================================================================"
echo ""

# Step 1: Generate QA dataset with optimal context size
echo "${GREEN}Step 1: Generating QA dataset (k=${CONTEXT_SIZE})${NC}"
echo "This will take 1-2 hours..."
echo ""

DATASET_FILE="data/qa_dataset${OUTPUT_SUFFIX}.jsonl"

# Check if dataset already exists
if [ -f "$DATASET_FILE" ]; then
    echo "${YELLOW}⚠️  Dataset already exists: $DATASET_FILE${NC}"
    read -p "Regenerate? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping dataset generation (using existing)"
    else
        rm "$DATASET_FILE"
        python scripts/generate_qa_dataset.py \
            --corpus data/corpus_sentences.jsonl \
            --output "$DATASET_FILE" \
            --max-pairs $MAX_PAIRS \
            --max-dialogue $MAX_PAIRS \
            --max-synthetic $MAX_PAIRS \
            --context-size $CONTEXT_SIZE
    fi
else
    python scripts/generate_qa_dataset.py \
        --corpus data/corpus_sentences.jsonl \
        --output "$DATASET_FILE" \
        --max-pairs $MAX_PAIRS \
        --max-dialogue $MAX_PAIRS \
        --max-synthetic $MAX_PAIRS \
        --context-size $CONTEXT_SIZE
fi

# Check if dataset was created successfully
if [ ! -f "$DATASET_FILE" ]; then
    echo "${RED}✗ Dataset generation failed!${NC}"
    exit 1
fi

DATASET_SIZE=$(wc -l < "$DATASET_FILE")
echo ""
echo "${GREEN}✓ Dataset generated: $DATASET_SIZE pairs${NC}"
echo ""

echo "========================================================================"
echo ""

# Step 2: Build vocabulary from dataset
echo "${GREEN}Step 2: Building vocabulary from dataset${NC}"
echo "This will take 5-10 minutes..."
echo ""

VOCAB_FILE="models/qa_decoder/vocabulary${OUTPUT_SUFFIX}.json"

# Check if build_vocabulary.py exists
if [ ! -f "scripts/build_vocabulary.py" ]; then
    echo "${YELLOW}⚠️  build_vocabulary.py not found, using train_qa_decoder.py's vocab builder${NC}"
    echo "Vocabulary will be built during training"
else
    python scripts/build_vocabulary.py \
        --dataset "$DATASET_FILE" \
        --output "$VOCAB_FILE" \
        --min-frequency 2

    echo "${GREEN}✓ Vocabulary built${NC}"

    # Show vocab stats
    if [ -f "$VOCAB_FILE" ]; then
        VOCAB_SIZE=$(python -c "import json; print(len(json.load(open('$VOCAB_FILE'))['token2id']))")
        echo "  Vocabulary size: $VOCAB_SIZE tokens"
    fi
fi

echo ""
echo "========================================================================"
echo ""

# Step 3: Train QA Decoder
echo "${GREEN}Step 3: Training QA Decoder${NC}"
if [ "$QUICK_MODE" = true ]; then
    echo "Quick mode: ~1-2 hours"
else
    echo "Production mode: ~4-8 hours"
fi
echo ""
echo "You can monitor progress in another terminal with:"
echo "  tail -f models/qa_decoder${OUTPUT_SUFFIX}/training.log"
echo ""

OUTPUT_DIR="models/qa_decoder${OUTPUT_SUFFIX}"
mkdir -p "$OUTPUT_DIR"

# Build training command
TRAIN_CMD="python scripts/train_qa_decoder.py \
    --dataset $DATASET_FILE \
    --gnn-checkpoint models/tree_lstm/checkpoint_epoch_20.pt \
    --output $OUTPUT_DIR \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr 1e-4"

# Add validation split for production mode
if [ "$QUICK_MODE" = false ]; then
    TRAIN_CMD="$TRAIN_CMD --val-split 0.2"
fi

echo "Starting training..."
echo "Command: $TRAIN_CMD"
echo ""

# Run training (redirect output to log file and console)
$TRAIN_CMD 2>&1 | tee "$OUTPUT_DIR/training.log"

echo ""
echo "${GREEN}✓ Training complete!${NC}"
echo ""

echo "========================================================================"
echo ""

# Step 4: Summary
echo "${GREEN}Step 4: Training Summary${NC}"
echo ""

if [ -f "$OUTPUT_DIR/best_model.pt" ]; then
    echo "${GREEN}✓ Model saved successfully${NC}"

    # Check model size
    MODEL_SIZE=$(du -h "$OUTPUT_DIR/best_model.pt" | cut -f1)
    echo "  Model size: $MODEL_SIZE"

    # Check final epoch
    if [ -f "$OUTPUT_DIR/training.log" ]; then
        FINAL_EPOCH=$(grep -o "Epoch [0-9]*" "$OUTPUT_DIR/training.log" | tail -1 | grep -o "[0-9]*")
        echo "  Final epoch: $FINAL_EPOCH"

        # Check final loss
        FINAL_LOSS=$(grep "val_loss" "$OUTPUT_DIR/training.log" | tail -1 | grep -o "val_loss: [0-9.]*" | grep -o "[0-9.]*")
        if [ ! -z "$FINAL_LOSS" ]; then
            echo "  Final validation loss: $FINAL_LOSS"
        fi
    fi

    echo ""
    echo "Model location: $OUTPUT_DIR/best_model.pt"
    echo "Vocabulary: $VOCAB_FILE"
    echo "Dataset: $DATASET_FILE"
else
    echo "${RED}✗ Model file not found!${NC}"
    echo "Training may have failed. Check logs at: $OUTPUT_DIR/training.log"
    exit 1
fi

echo ""
echo "========================================================================"
echo ""

# Step 5: Test the model
echo "${GREEN}Step 5: Testing the model${NC}"
echo ""
echo "You can test the model with:"
echo ""
if [ "$QUICK_MODE" = true ]; then
    echo "  # Modify ask.sh to point to: $OUTPUT_DIR/best_model.pt"
    echo "  ./ask.sh \"Kiu estas Frodo?\""
else
    echo "  # Production model is ready!"
    echo "  # Replace old model:"
    echo "  mv models/qa_decoder models/qa_decoder_old"
    echo "  mv $OUTPUT_DIR models/qa_decoder"
    echo ""
    echo "  # Then test:"
    echo "  ./ask.sh \"Kiu estas Frodo?\""
    echo "  ./ask.sh \"Kiu estas Gandalfo?\""
    echo "  ./ask.sh \"Kio estas hobito?\""
fi

echo ""
echo "========================================================================"
echo ""

# Final summary
echo "${GREEN}✓ RETRAINING COMPLETE!${NC}"
echo ""
echo "Summary:"
echo "  - Dataset: $DATASET_SIZE QA pairs with k=$CONTEXT_SIZE context"
if [ -f "$VOCAB_FILE" ]; then
    echo "  - Vocabulary: $VOCAB_SIZE tokens"
fi
echo "  - Training: $EPOCHS epochs, batch_size=$BATCH_SIZE"
echo "  - Output: $OUTPUT_DIR/"
echo ""
echo "Backup of old models: $BACKUP_DIR/"
echo ""

if [ "$QUICK_MODE" = true ]; then
    echo "${YELLOW}Note: This was a QUICK TEST run.${NC}"
    echo "For production quality, run:"
    echo "  ./retrain_production.sh"
    echo ""
else
    echo "${GREEN}This is a PRODUCTION-QUALITY model!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Test the model (see commands above)"
    echo "  2. Compare to old model quality"
    echo "  3. If satisfied, replace old model"
    echo "  4. If not satisfied, check training logs"
fi

echo ""
echo "========================================================================"
