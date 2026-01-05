#!/bin/bash
# =============================================================================
# Reparse Training Corpus
# =============================================================================
# Re-parses the combined_training.jsonl with the updated parser v2.
#
# This updates the ASTs in place without rebuilding the corpus from source.
# Useful after parser improvements to get new fields like:
#   - prefiksoj (list format)
#   - fraztipo (sentence type)
#   - demandotipo (question type)
#   - participo_voĉo/tempo (participle info)
#   - korelativo_prefikso/sufikso (correlative decomposition)
#
# The script has checkpoint support for restartability.
#
# Usage:
#   ./scripts/run_reparse_training.sh           # Resume from checkpoint
#   ./scripts/run_reparse_training.sh --fresh   # Start fresh
#
# Output:
#   data/training/combined_training_v3.jsonl
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Parse arguments
FRESH_START=false
for arg in "$@"; do
    case $arg in
        --fresh)
            FRESH_START=true
            shift
            ;;
    esac
done

# Activate virtual environment
echo -e "${BLUE}=== Activating Python Environment ===${NC}"
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "Using .venv"
elif [ -d "venv" ]; then
    source venv/bin/activate
    echo "Using venv"
else
    echo -e "${RED}No virtual environment found. Create one with: python -m venv .venv${NC}"
    exit 1
fi

# Verify Python
python --version
echo ""

# Paths
INPUT_FILE="$PROJECT_ROOT/data/training/combined_training.jsonl"
OUTPUT_FILE="$PROJECT_ROOT/data/training/combined_training_v3.jsonl"
CHECKPOINT_FILE="$PROJECT_ROOT/data/training/reparse_checkpoint.json"

# Log file
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/reparse_training_$(date +%Y%m%d_%H%M%S).log"

echo -e "${BLUE}=== Reparse Training Corpus ===${NC}"
echo "Input:  $INPUT_FILE"
echo "Output: $OUTPUT_FILE"
echo "Log:    $LOG_FILE"
echo "Fresh:  $FRESH_START"
echo ""

# Check input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo -e "${RED}Error: Input file not found: $INPUT_FILE${NC}"
    exit 1
fi

# Get input file info
INPUT_LINES=$(wc -l < "$INPUT_FILE")
INPUT_SIZE=$(du -h "$INPUT_FILE" | cut -f1)
echo "Input file: $INPUT_LINES lines, $INPUT_SIZE"
echo ""

# Estimate time (roughly 500 lines/second)
ESTIMATED_MINUTES=$((INPUT_LINES / 500 / 60))
echo -e "${YELLOW}Estimated time: ~$ESTIMATED_MINUTES minutes${NC}"
echo ""

# Confirm before starting
echo -e "${YELLOW}This will reparse $INPUT_LINES sentences.${NC}"
echo "Press Enter to continue or Ctrl+C to cancel..."
read -r

# If fresh start, remove checkpoint
if [ "$FRESH_START" = true ]; then
    echo -e "${YELLOW}Fresh start requested - removing checkpoint${NC}"
    rm -f "$CHECKPOINT_FILE"
fi

# Run the reparse
echo -e "${BLUE}=== Starting Reparse ===${NC}"
echo "Logging to: $LOG_FILE"
echo ""

python scripts/reparse_corpus.py \
    --input "$INPUT_FILE" \
    --output "$OUTPUT_FILE" \
    --batch-size 10000 \
    --log-level INFO \
    2>&1 | tee "$LOG_FILE"

# Check if output was created
if [ -f "$OUTPUT_FILE" ]; then
    OUTPUT_LINES=$(wc -l < "$OUTPUT_FILE")
    OUTPUT_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)

    echo ""
    echo -e "${GREEN}=== Reparse Complete ===${NC}"
    echo "Input:  $INPUT_LINES lines, $INPUT_SIZE"
    echo "Output: $OUTPUT_LINES lines, $OUTPUT_SIZE"
    echo ""

    # Show a sample of the new format
    echo "Sample of reparsed entry:"
    head -5 "$OUTPUT_FILE" | tail -1 | python -c "import json, sys; d = json.load(sys.stdin); print(json.dumps({k: d.get(k) for k in ['text', 'parser_version', 'reparsed_at']}, indent=2))"
    echo ""

    # Ask about replacing the original
    echo -e "${YELLOW}Would you like to replace the original file?${NC}"
    echo "  Original: $INPUT_FILE"
    echo "  New:      $OUTPUT_FILE"
    echo ""
    echo "To replace, run:"
    echo "  mv $OUTPUT_FILE $INPUT_FILE"
    echo ""
else
    echo -e "${RED}Error: Output file not created${NC}"
    exit 1
fi

echo "Log file: $LOG_FILE"
echo -e "${GREEN}Done!${NC}"
