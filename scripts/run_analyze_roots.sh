#!/bin/bash
#
# Run Issue #41: Analyze Root Frequency Distribution in M1 Corpus
#
# This script analyzes the M1 corpus to determine optimal vocabulary size
# for morpheme-aware embeddings.
#
# Usage:
#   ./scripts/run_analyze_roots.sh
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Paths
VENV_DIR="$PROJECT_ROOT/.venv"
CORPUS_FILE="$PROJECT_ROOT/data/corpus_enhanced_m1.jsonl"
OUTPUT_FILE="/tmp/root_frequency_analysis.json"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_ROOT/data/logs"
LOG_FILE="$LOG_DIR/root_analysis_${TIMESTAMP}.log"

# Create log directory
mkdir -p "$LOG_DIR"

echo -e "${BLUE}=====================================================================${NC}"
echo -e "${BLUE}Issue #41: Analyze Root Frequency Distribution${NC}"
echo -e "${BLUE}=====================================================================${NC}"
echo ""
echo -e "${GREEN}Project root:${NC} $PROJECT_ROOT"
echo -e "${GREEN}Virtual env:${NC} $VENV_DIR"
echo -e "${GREEN}Corpus:${NC} $CORPUS_FILE"
echo -e "${GREEN}Output:${NC} $OUTPUT_FILE"
echo -e "${GREEN}Log file:${NC} $LOG_FILE"
echo ""

# Check virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${RED}ERROR: Virtual environment not found at $VENV_DIR${NC}"
    exit 1
fi

# Check corpus exists
if [ ! -f "$CORPUS_FILE" ]; then
    echo -e "${RED}ERROR: M1 corpus not found at $CORPUS_FILE${NC}"
    echo -e "${YELLOW}Please run Issue #28 first:${NC}"
    echo "  ./scripts/run_build_corpus_m1.sh"
    exit 1
fi

CORPUS_SIZE=$(du -h "$CORPUS_FILE" | cut -f1)
CORPUS_LINES=$(wc -l < "$CORPUS_FILE")
echo -e "${GREEN}Corpus size:${NC} $CORPUS_SIZE ($CORPUS_LINES sentences)"
echo ""

# Activate virtual environment
echo -e "${BLUE}Activating virtual environment...${NC}"
source "$VENV_DIR/bin/activate"

# Check Python version
PYTHON_VERSION=$(python --version)
echo -e "${GREEN}Python version: $PYTHON_VERSION${NC}"
echo ""

# Confirm before running
echo -e "${YELLOW}=====================================================================${NC}"
echo -e "${YELLOW}Ready to start root frequency analysis${NC}"
echo -e "${YELLOW}=====================================================================${NC}"
echo ""
echo -e "${YELLOW}This will:${NC}"
echo "  1. Read all 4.2M sentences from M1 corpus"
echo "  2. Extract and count all unique roots"
echo "  3. Calculate coverage statistics (top 100, 1K, 5K, 10K)"
echo "  4. Recommend optimal vocabulary size"
echo ""
echo -e "${YELLOW}Expected time:${NC} 1-2 minutes"
echo ""
echo -e "${YELLOW}Output:${NC}"
echo "  - Analysis report: $OUTPUT_FILE"
echo "  - Log file: $LOG_FILE"
echo ""

# Ask for confirmation (skip if running in background)
if [ -t 0 ]; then
    read -p "$(echo -e ${GREEN}Continue? [y/N]: ${NC})" -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Aborted by user${NC}"
        exit 0
    fi
fi

# Record start time
START_TIME=$(date)
START_TIMESTAMP=$(date +%s)

echo ""
echo -e "${BLUE}=====================================================================${NC}"
echo -e "${BLUE}Starting root frequency analysis at $START_TIME${NC}"
echo -e "${BLUE}=====================================================================${NC}"
echo ""

# Run analysis
python "$SCRIPT_DIR/analyze_root_frequencies.py" \
    --corpus "$CORPUS_FILE" \
    --output "$OUTPUT_FILE" \
    --verbose \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

# Record end time
END_TIME=$(date)
END_TIMESTAMP=$(date +%s)
DURATION=$((END_TIMESTAMP - START_TIMESTAMP))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

echo ""
echo -e "${BLUE}=====================================================================${NC}"

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}Root frequency analysis completed successfully!${NC}"
    echo -e "${BLUE}=====================================================================${NC}"
    echo ""
    echo -e "${GREEN}Start time:${NC} $START_TIME"
    echo -e "${GREEN}End time:${NC} $END_TIME"
    echo -e "${GREEN}Duration:${NC} ${MINUTES}m ${SECONDS}s"
    echo ""

    # Show output files
    echo -e "${GREEN}Output files:${NC}"
    echo "  - Analysis: $OUTPUT_FILE"
    echo "  - Log: $LOG_FILE"
    echo ""

    # Show recommendation if available
    if [ -f "$OUTPUT_FILE" ]; then
        echo -e "${GREEN}Recommendation:${NC}"
        VOCAB_SIZE=$(jq -r '.recommendation.vocabulary_size' "$OUTPUT_FILE")
        COVERAGE=$(jq -r '.recommendation.coverage' "$OUTPUT_FILE")
        REASON=$(jq -r '.recommendation.reason' "$OUTPUT_FILE")
        echo "  Vocabulary size: $VOCAB_SIZE roots"
        echo "  Coverage: $(python -c "print(f'{$COVERAGE:.1%}')")"
        echo "  Reason: $REASON"
        echo ""
    fi

    # Next steps
    echo -e "${YELLOW}Next steps:${NC}"
    echo "  1. Review analysis:"
    echo "     cat $OUTPUT_FILE | jq '.'"
    echo ""
    echo "  2. Proceed to Issue #42 (Create affix classification):"
    echo "     # Create config/affix_classification.json manually"
    echo ""
    echo "  3. Then Issue #43 (Implement MorphemeAwareEmbedding)"
    echo ""

else
    echo -e "${RED}Root frequency analysis failed with exit code $EXIT_CODE${NC}"
    echo -e "${BLUE}=====================================================================${NC}"
    echo ""
    echo -e "${RED}Start time:${NC} $START_TIME"
    echo -e "${RED}End time:${NC} $END_TIME"
    echo -e "${RED}Duration:${NC} ${MINUTES}m ${SECONDS}s"
    echo ""
    echo -e "${YELLOW}Check log for errors:${NC}"
    echo "  - Log file: $LOG_FILE"
    echo ""
    echo -e "${YELLOW}Last 20 lines of log:${NC}"
    tail -20 "$LOG_FILE"
    echo ""
fi

echo -e "${BLUE}=====================================================================${NC}"

# Deactivate virtual environment
deactivate

exit $EXIT_CODE
