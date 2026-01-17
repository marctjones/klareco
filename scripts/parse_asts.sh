#!/bin/bash
#
# Parse Entire Corpus to AST Dataset
#
# This script reads clean corpus files and parses every sentence into
# a structured AST, saving them as JSONL for GNN encoder training.
#
# Features:
# - Progress updates every 100 sentences
# - Error logging for failed parses
# - Parse quality statistics
#
# Usage:
#   ./scripts/parse_asts.sh
#   ./scripts/parse_asts.sh --input data/cleaned/eo --output data/corpus/asts
#
# Output:
#   data/corpus/asts/*_asts.jsonl         # AST files per source
#   data/corpus/asts/parsing_statistics.json  # Quality stats
#   corpus_parsing_errors.log             # Failed sentence log

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Default parameters
INPUT_DIR="data/cleaned/eo"
OUTPUT_DIR="data/corpus/asts"
ERROR_LOG="corpus_parsing_errors.log"
DEBUG_FLAG=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input)
            INPUT_DIR="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --error-log)
            ERROR_LOG="$2"
            shift 2
            ;;
        --debug)
            DEBUG_FLAG="--debug"
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Corpus AST Parser - Phase 3${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${YELLOW}→${NC} Project root: $PROJECT_ROOT"

# Activate virtual environment
echo -e "${YELLOW}→${NC} Activating virtual environment..."
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
else
    echo -e "${RED}✗${NC} No virtual environment found (.venv or venv)"
    exit 1
fi
echo -e "${GREEN}✓${NC} Virtual environment activated"

# Check input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo -e "${RED}✗${NC} Input directory not found: $INPUT_DIR"
    echo -e "${YELLOW}→${NC} Run cleaning scripts first (./scripts/clean_all.sh)"
    exit 1
fi

# Count input files
INPUT_COUNT=$(ls -1 "$INPUT_DIR"/*.txt 2>/dev/null | wc -l)
if [ $INPUT_COUNT -eq 0 ]; then
    echo -e "${RED}✗${NC} No .txt files found in $INPUT_DIR"
    exit 1
fi

echo -e "${GREEN}✓${NC} Input directory: $INPUT_DIR ($INPUT_COUNT files)"

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

echo ""
echo -e "${YELLOW}→${NC} This will parse all corpus sentences to structured ASTs"
echo -e "${YELLOW}→${NC} Estimated time: 10-30 minutes depending on corpus size"
echo -e "${YELLOW}→${NC} Progress updates every 100 sentences"
echo ""

# Set up logging
LOG_FILE="logs/parse_asts_$(date +%Y%m%d_%H%M%S).log"
echo -e "${YELLOW}→${NC} Log file: $LOG_FILE"
echo -e "${YELLOW}→${NC} Error log: $ERROR_LOG"
echo ""

# Run parsing script with logging
python scripts/parse_asts.py \
    --input "$INPUT_DIR" \
    --output "$OUTPUT_DIR" \
    --error-log "$ERROR_LOG" \
    $DEBUG_FLAG \
    2>&1 | tee "$LOG_FILE"

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}✓ Corpus parsing complete!${NC}"
    echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
    echo ""

    # Show output statistics
    STATS_FILE="$OUTPUT_DIR/parsing_statistics.json"
    if [ -f "$STATS_FILE" ]; then
        echo -e "${BLUE}Parse Statistics:${NC}"
        python3 -c "
import json
with open('$STATS_FILE') as f:
    stats = json.load(f)
    total = stats['total_sentences']
    success = stats['successful_sentences']
    failed = stats['failed_sentences']
    success_pct = 100 * success / total if total > 0 else 0

    print(f'  Total sentences: {total:,}')
    print(f'  Successful: {success:,} ({success_pct:.1f}%)')
    print(f'  Failed: {failed:,} ({100-success_pct:.1f}%)')
    print()

    if 'total_words' in stats:
        total_words = stats['total_words']
        eo_words = stats['esperanto_words']
        non_eo = stats['non_esperanto_words']
        print(f'  Total words: {total_words:,}')
        print(f'  Esperanto words: {eo_words:,} ({100*eo_words/total_words:.1f}%)')
        print(f'  Non-Esperanto: {non_eo:,} ({100*non_eo/total_words:.1f}%)')
"
    fi

    # Count output files
    OUTPUT_COUNT=$(ls -1 "$OUTPUT_DIR"/*_asts.jsonl 2>/dev/null | wc -l)
    echo ""
    echo -e "${GREEN}✓${NC} Output files: $OUTPUT_COUNT JSONL files"
    echo -e "${GREEN}✓${NC} Location: $OUTPUT_DIR/"

    # Show failed sentence count if any
    if [ -f "$ERROR_LOG" ]; then
        ERROR_COUNT=$(grep -c "Error:" "$ERROR_LOG" 2>/dev/null || echo "0")
        if [ $ERROR_COUNT -gt 0 ]; then
            echo -e "${YELLOW}⚠${NC}  Failed sentences logged: $ERROR_LOG ($ERROR_COUNT errors)"
        fi
    fi

    echo ""
    echo -e "${YELLOW}→${NC} Next step: Build unified corpus with metadata"
    echo -e "${YELLOW}→${NC} Command: ./scripts/parse_corpus.sh"
else
    echo ""
    echo -e "${RED}✗ Corpus parsing failed${NC}"
    echo -e "${YELLOW}→${NC} Check logs:"
    echo -e "     - $LOG_FILE"
    echo -e "     - $ERROR_LOG"
    exit 1
fi
