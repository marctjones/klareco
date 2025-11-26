#!/bin/bash
# Watch training progress in real-time
# Run this in a separate terminal while retrain_production.sh is running

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

clear

echo "========================================================================"
echo "KLARECO - TRAINING MONITOR"
echo "========================================================================"
echo ""
echo "Watching for training logs..."
echo ""

# Check which model is being trained
if [ -d "models/qa_decoder_production" ]; then
    LOG_FILE="models/qa_decoder_production/training.log"
    MODE="PRODUCTION"
elif [ -d "models/qa_decoder_test" ]; then
    LOG_FILE="models/qa_decoder_test/training.log"
    MODE="QUICK TEST"
else
    LOG_FILE="models/qa_decoder_production/training.log"
    MODE="PRODUCTION (waiting)"
fi

echo "${GREEN}Mode: $MODE${NC}"
echo "Log file: $LOG_FILE"
echo ""

# Wait for log file to appear
while [ ! -f "$LOG_FILE" ]; do
    echo -ne "${YELLOW}Waiting for training to start...${NC}\r"
    sleep 2
done

echo ""
echo "${GREEN}✓ Training started! Showing live progress...${NC}"
echo "========================================================================"
echo ""

# Follow the log file
tail -f "$LOG_FILE" 2>/dev/null || tail -F "$LOG_FILE"
