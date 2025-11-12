#!/bin/bash
#
# Overnight GNN Training Script
#
# This script runs pre-flight checks and starts Tree-LSTM training
# with checkpoint resumption enabled.
#
# Usage:
#   ./train_overnight.sh           # Start training with default settings
#   ./train_overnight.sh 50 32     # Custom epochs and batch size
#

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default parameters
EPOCHS=${1:-50}        # Default: 50 epochs
BATCH_SIZE=${2:-16}    # Default: 16 batch size
LR=${3:-0.001}         # Default: 0.001 learning rate

echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}  KLARECO TREE-LSTM OVERNIGHT TRAINING${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo ""
echo -e "${YELLOW}Training Configuration:${NC}"
echo -e "  Epochs:     ${EPOCHS}"
echo -e "  Batch Size: ${BATCH_SIZE}"
echo -e "  Learn Rate: ${LR}"
echo ""
echo -e "${YELLOW}Output:${NC}"
echo -e "  Checkpoints: models/tree_lstm/"
echo -e "  Logs:        training_overnight.log"
echo ""

# Step 1: Run pre-flight checks
echo -e "${BLUE}Step 1: Running pre-flight checks...${NC}"
echo ""

if ! python scripts/preflight_check_training.py --training-data data/training_pairs; then
    echo ""
    echo -e "${RED}❌ Pre-flight checks FAILED${NC}"
    echo -e "${RED}Please fix the issues above before training.${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✅ Pre-flight checks PASSED${NC}"
echo ""

# Step 2: Start training
echo -e "${BLUE}Step 2: Starting training...${NC}"
echo ""

# Build the training command
TRAIN_CMD="python scripts/train_tree_lstm.py \
    --training-data data/training_pairs \
    --output models/tree_lstm \
    --resume auto \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --lr ${LR}"

echo -e "${YELLOW}Command:${NC}"
echo "  $TRAIN_CMD"
echo ""

# Check if we're in a tmux session
if [ -n "$TMUX" ]; then
    echo -e "${YELLOW}Detected tmux session - running in foreground${NC}"
    echo -e "${YELLOW}Training output will be shown below.${NC}"
    echo -e "${YELLOW}To detach: Press Ctrl-B, then D${NC}"
    echo ""
    sleep 2

    # Run in foreground (already in tmux)
    $TRAIN_CMD 2>&1 | tee training_overnight.log

else
    echo -e "${YELLOW}Not in tmux - would you like to:${NC}"
    echo ""
    echo "  1) Run in background (with nohup)"
    echo "  2) Start in new tmux session (recommended)"
    echo "  3) Run in foreground (current terminal)"
    echo ""
    read -p "Choice (1/2/3): " choice

    case $choice in
        1)
            echo ""
            echo -e "${YELLOW}Starting training in background with nohup...${NC}"
            nohup $TRAIN_CMD > training_overnight.log 2>&1 &
            PID=$!
            echo ""
            echo -e "${GREEN}✅ Training started!${NC}"
            echo ""
            echo -e "${YELLOW}Process ID:${NC} $PID"
            echo -e "${YELLOW}Log file:${NC}    training_overnight.log"
            echo ""
            echo -e "${YELLOW}To monitor progress:${NC}"
            echo "  tail -f training_overnight.log"
            echo ""
            echo -e "${YELLOW}To check if still running:${NC}"
            echo "  ps -p $PID"
            echo ""
            ;;
        2)
            echo ""
            echo -e "${YELLOW}Starting new tmux session 'training'...${NC}"
            echo ""
            echo -e "${YELLOW}Commands to know:${NC}"
            echo "  Detach:   Ctrl-B, then D"
            echo "  Reattach: tmux attach -t training"
            echo ""
            sleep 2

            # Create tmux session and run training
            tmux new-session -s training "$TRAIN_CMD 2>&1 | tee training_overnight.log; echo 'Press Enter to exit'; read"
            ;;
        3)
            echo ""
            echo -e "${YELLOW}Starting training in foreground...${NC}"
            echo -e "${YELLOW}Press Ctrl-C to stop (will lose progress if stopped before checkpoint)${NC}"
            echo ""
            sleep 2

            $TRAIN_CMD 2>&1 | tee training_overnight.log
            ;;
        *)
            echo ""
            echo -e "${RED}Invalid choice. Exiting.${NC}"
            exit 1
            ;;
    esac
fi

echo ""
echo -e "${GREEN}======================================================================${NC}"
echo -e "${GREEN}  Training session complete or detached${NC}"
echo -e "${GREEN}======================================================================${NC}"
echo ""
