#!/bin/bash
#
# Run Issue #28: Build Enhanced M1 Corpus
#
# This script:
# - Activates the Python virtual environment
# - Runs the corpus building script
# - Captures all output to timestamped log files
# - Handles errors gracefully
#
# Usage:
#   ./scripts/run_build_corpus_m1.sh
#
# Or run in background:
#   nohup ./scripts/run_build_corpus_m1.sh &
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
PYTHON_SCRIPT="$PROJECT_ROOT/scripts/build_enhanced_corpus_m1.py"
DATA_DIR="$PROJECT_ROOT/data"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_LOG="$DATA_DIR/corpus_build_run_${TIMESTAMP}.log"

echo -e "${BLUE}=====================================================================${NC}"
echo -e "${BLUE}Issue #28: Build Enhanced M1 Corpus${NC}"
echo -e "${BLUE}=====================================================================${NC}"
echo ""
echo -e "${GREEN}Project root:${NC} $PROJECT_ROOT"
echo -e "${GREEN}Virtual env:${NC} $VENV_DIR"
echo -e "${GREEN}Python script:${NC} $PYTHON_SCRIPT"
echo -e "${GREEN}Run log:${NC} $RUN_LOG"
echo ""

# Check virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${RED}ERROR: Virtual environment not found at $VENV_DIR${NC}"
    echo -e "${YELLOW}Please create it first:${NC}"
    echo "  cd $PROJECT_ROOT"
    echo "  python3 -m venv .venv"
    echo "  source .venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Check Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo -e "${RED}ERROR: Python script not found at $PYTHON_SCRIPT${NC}"
    exit 1
fi

# Check data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo -e "${RED}ERROR: Data directory not found at $DATA_DIR${NC}"
    exit 1
fi

# Check input files exist
WIKIPEDIA_INPUT="$DATA_DIR/extracted/wikipedia_sentences.jsonl"
BOOKS_INPUT="$DATA_DIR/extracted/books_sentences.jsonl"

if [ ! -f "$WIKIPEDIA_INPUT" ]; then
    echo -e "${RED}ERROR: Wikipedia input not found at $WIKIPEDIA_INPUT${NC}"
    echo -e "${YELLOW}Please run corpus extraction first${NC}"
    exit 1
fi

if [ ! -f "$BOOKS_INPUT" ]; then
    echo -e "${YELLOW}WARNING: Books input not found at $BOOKS_INPUT${NC}"
    echo -e "${YELLOW}Continuing with Wikipedia only...${NC}"
fi

# Check disk space
AVAILABLE_SPACE=$(df -BG "$DATA_DIR" | awk 'NR==2 {print $4}' | sed 's/G//')
if [ "$AVAILABLE_SPACE" -lt 5 ]; then
    echo -e "${RED}ERROR: Insufficient disk space${NC}"
    echo -e "${YELLOW}Available: ${AVAILABLE_SPACE}GB, Needed: ~5GB${NC}"
    exit 1
fi

echo -e "${GREEN}Disk space available: ${AVAILABLE_SPACE}GB${NC}"
echo ""

# Activate virtual environment
echo -e "${BLUE}Activating virtual environment...${NC}"
source "$VENV_DIR/bin/activate"

# Check Python version
PYTHON_VERSION=$(python --version)
echo -e "${GREEN}Python version: $PYTHON_VERSION${NC}"
echo ""

# Check required packages
echo -e "${BLUE}Checking required packages...${NC}"
python -c "import tqdm" 2>/dev/null || {
    echo -e "${RED}ERROR: tqdm not installed${NC}"
    echo -e "${YELLOW}Installing tqdm...${NC}"
    pip install tqdm
}

python -c "from klareco.parser import parse" 2>/dev/null || {
    echo -e "${RED}ERROR: klareco package not importable${NC}"
    echo -e "${YELLOW}Make sure you're in the right directory and klareco is installed${NC}"
    exit 1
}

echo -e "${GREEN}All packages available${NC}"
echo ""

# Confirm before running
echo -e "${YELLOW}=====================================================================${NC}"
echo -e "${YELLOW}Ready to start corpus building${NC}"
echo -e "${YELLOW}=====================================================================${NC}"
echo ""
echo -e "${YELLOW}This will:${NC}"
echo "  - Process ~4.2 million sentences"
echo "  - Take approximately 2-4 hours"
echo "  - Create data/corpus_enhanced_m1.jsonl (~2-3GB)"
echo "  - Use 1 CPU core at 100%"
echo ""
echo -e "${YELLOW}Logs will be written to:${NC}"
echo "  - Script log (this output): $RUN_LOG"
echo "  - Python log (detailed): $DATA_DIR/corpus_enhanced_m1_build.log"
echo ""
echo -e "${YELLOW}You can monitor progress with:${NC}"
echo "  tail -f $DATA_DIR/corpus_enhanced_m1_build.log"
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
echo -e "${BLUE}Starting corpus build at $START_TIME${NC}"
echo -e "${BLUE}=====================================================================${NC}"
echo ""

# Run Python script with output to both console and log file
# tee captures stdout, 2>&1 also captures stderr
python "$PYTHON_SCRIPT" 2>&1 | tee "$RUN_LOG"

# Capture exit code
EXIT_CODE=${PIPESTATUS[0]}

# Record end time
END_TIME=$(date)
END_TIMESTAMP=$(date +%s)
DURATION=$((END_TIMESTAMP - START_TIMESTAMP))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo ""
echo -e "${BLUE}=====================================================================${NC}"

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}Corpus build completed successfully!${NC}"
    echo -e "${BLUE}=====================================================================${NC}"
    echo ""
    echo -e "${GREEN}Start time:${NC} $START_TIME"
    echo -e "${GREEN}End time:${NC} $END_TIME"
    echo -e "${GREEN}Duration:${NC} ${HOURS}h ${MINUTES}m ${SECONDS}s"
    echo ""

    # Show output file info
    OUTPUT_FILE="$DATA_DIR/corpus_enhanced_m1.jsonl"
    if [ -f "$OUTPUT_FILE" ]; then
        FILE_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
        LINE_COUNT=$(wc -l < "$OUTPUT_FILE")
        echo -e "${GREEN}Output file:${NC} $OUTPUT_FILE"
        echo -e "${GREEN}File size:${NC} $FILE_SIZE"
        echo -e "${GREEN}Sentences:${NC} $(printf "%'d" $LINE_COUNT)"
        echo ""
    fi

    # Show logs
    echo -e "${GREEN}Logs:${NC}"
    echo "  - Run log: $RUN_LOG"
    echo "  - Build log: $DATA_DIR/corpus_enhanced_m1_build.log"
    echo ""

    # Next steps
    echo -e "${YELLOW}Next steps:${NC}"
    echo "  1. Review logs for any issues"
    echo "  2. Build FAISS index:"
    echo "     python scripts/index_corpus.py --corpus data/corpus_enhanced_m1.jsonl --output data/corpus_index_m1"
    echo "  3. Test retrieval:"
    echo "     python scripts/demo_rag.py --index data/corpus_index_m1 --interactive"
    echo ""

else
    echo -e "${RED}Corpus build failed with exit code $EXIT_CODE${NC}"
    echo -e "${BLUE}=====================================================================${NC}"
    echo ""
    echo -e "${RED}Start time:${NC} $START_TIME"
    echo -e "${RED}End time:${NC} $END_TIME"
    echo -e "${RED}Duration:${NC} ${HOURS}h ${MINUTES}m ${SECONDS}s"
    echo ""
    echo -e "${YELLOW}Check logs for errors:${NC}"
    echo "  - Run log: $RUN_LOG"
    echo "  - Build log: $DATA_DIR/corpus_enhanced_m1_build.log"
    echo ""
    echo -e "${YELLOW}Last 20 lines of run log:${NC}"
    tail -20 "$RUN_LOG"
    echo ""
fi

echo -e "${BLUE}=====================================================================${NC}"

# Deactivate virtual environment
deactivate

exit $EXIT_CODE
