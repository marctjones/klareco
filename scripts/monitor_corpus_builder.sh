#!/bin/bash
#
# Monitor Corpus Builder Progress
#
# Usage:
#   ./scripts/monitor_corpus_builder.sh
#
# This script monitors the corpus builder's progress in real-time
# and shows current status, memory usage, and logs.
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

DATA_DIR="$PROJECT_ROOT/data"
LOG_DIR="$PROJECT_ROOT/logs"
CHECKPOINT_FILE="$DATA_DIR/build_corpus_v2_checkpoint.json"
OUTPUT_FILE="$DATA_DIR/corpus_with_sources_v2.jsonl"
LATEST_LOG="$LOG_DIR/corpus_builder_latest.log"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

clear
echo "============================================================================"
echo "Klareco Corpus Builder - Progress Monitor"
echo "============================================================================"
echo ""

# Check if builder is running
if pgrep -f "build_corpus_v2.py" > /dev/null; then
    echo -e "${GREEN}✓ Corpus builder is RUNNING${NC}"
    PID=$(pgrep -f "build_corpus_v2.py")
    echo "  Process ID: $PID"
else
    echo -e "${YELLOW}○ Corpus builder is NOT running${NC}"
fi

echo ""
echo "============================================================================"
echo "Current Progress"
echo "============================================================================"

# Check checkpoint
if [ -f "$CHECKPOINT_FILE" ]; then
    echo -e "${BLUE}Checkpoint found:${NC}"
    cat "$CHECKPOINT_FILE" | python3 -m json.tool 2>/dev/null || cat "$CHECKPOINT_FILE"
    echo ""
else
    echo "No checkpoint file found (either not started or completed)"
    echo ""
fi

# Check output file
if [ -f "$OUTPUT_FILE" ]; then
    SENTENCE_COUNT=$(wc -l < "$OUTPUT_FILE")
    FILE_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
    echo -e "${BLUE}Output file:${NC}"
    echo "  Location: $OUTPUT_FILE"
    echo "  Sentences: $SENTENCE_COUNT"
    echo "  Size: $FILE_SIZE"
    echo ""
else
    echo "No output file yet"
    echo ""
fi

echo "============================================================================"
echo "System Resources"
echo "============================================================================"

# Memory usage
if command -v free &> /dev/null; then
    echo "Memory usage:"
    free -h | grep -E "^Mem:|^Swap:"
    echo ""
fi

# Disk space
echo "Disk space (data directory):"
df -h "$DATA_DIR" | tail -n 1
echo ""

# CPU usage (if builder is running)
if pgrep -f "build_corpus_v2.py" > /dev/null; then
    PID=$(pgrep -f "build_corpus_v2.py")
    echo "CPU usage (corpus builder):"
    ps -p $PID -o %cpu,%mem,cmd 2>/dev/null || echo "  (process monitoring not available)"
    echo ""
fi

echo "============================================================================"
echo "Recent Log Output (last 20 lines)"
echo "============================================================================"

if [ -f "$LATEST_LOG" ]; then
    tail -n 20 "$LATEST_LOG"
else
    echo "No log file found"
fi

echo ""
echo "============================================================================"
echo "Commands"
echo "============================================================================"
echo ""
echo "View full log:"
echo "  tail -f $LATEST_LOG"
echo ""
echo "Check progress every 5 seconds:"
echo "  watch -n 5 $0"
echo ""
echo "Stop builder (gracefully):"
echo "  pkill -INT -f build_corpus_v2.py"
echo ""
echo "View this monitor again:"
echo "  $0"
echo ""
