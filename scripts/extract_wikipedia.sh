#!/bin/bash
#
# Extract Wikipedia articles with metadata
#
# This script:
# - Activates Python virtual environment
# - Runs Wikipedia extraction with progress indicators
# - Logs all output to logs/wikipedia_extraction.log
# - Can be run in background or foreground
#
# Usage:
#   ./scripts/extract_wikipedia.sh           # Resume from checkpoint (if exists)
#   ./scripts/extract_wikipedia.sh --fresh   # Start fresh (delete checkpoint and output)
#   ./scripts/extract_wikipedia.sh &         # Run in background

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Wikipedia Extraction with Metadata${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo ""

# Change to project root
cd "$PROJECT_ROOT"
echo -e "${YELLOW}→${NC} Project root: $PROJECT_ROOT"

# Check if venv exists
if [ ! -d ".venv" ]; then
    echo -e "${RED}✗${NC} Virtual environment not found at .venv"
    echo -e "${YELLOW}→${NC} Creating virtual environment..."
    python3 -m venv .venv
    echo -e "${GREEN}✓${NC} Virtual environment created"
fi

# Activate virtual environment
echo -e "${YELLOW}→${NC} Activating virtual environment..."
source .venv/bin/activate
echo -e "${GREEN}✓${NC} Virtual environment activated"

# Check if klareco is installed
if ! python -c "import klareco" 2>/dev/null; then
    echo -e "${YELLOW}→${NC} Installing klareco package..."
    pip install -e . > /dev/null 2>&1
    echo -e "${GREEN}✓${NC} Klareco package installed"
fi

# Create necessary directories
echo -e "${YELLOW}→${NC} Creating directories..."
mkdir -p data/extracted
mkdir -p logs
echo -e "${GREEN}✓${NC} Directories created"

# Check if Wikipedia dump exists
WIKI_DUMP="data/raw/eo/wikipedia/eo_wikipedia.xml.bz2"
if [ ! -f "$WIKI_DUMP" ]; then
    echo -e "${RED}✗${NC} Wikipedia dump not found: $WIKI_DUMP"
    echo -e "${YELLOW}→${NC} Please download Wikipedia dump first"
    exit 1
fi

# Show file info
WIKI_SIZE=$(du -h "$WIKI_DUMP" | cut -f1)
echo -e "${GREEN}✓${NC} Wikipedia dump found: $WIKI_SIZE"
echo ""

# Parse command line arguments
FRESH_FLAG=""
if [[ "$1" == "--fresh" ]]; then
    FRESH_FLAG="--fresh"
    echo -e "${YELLOW}⚠${NC}  Fresh start requested - will delete checkpoint and output"
    echo ""
fi

# Check if resuming from checkpoint (only if not fresh)
CHECKPOINT="data/extracted/wikipedia_checkpoint.json"
if [ -f "$CHECKPOINT" ] && [ -z "$FRESH_FLAG" ]; then
    echo -e "${YELLOW}⚠${NC}  Checkpoint found - will resume from previous run"
    echo -e "${YELLOW}→${NC} To start fresh, run: ./scripts/extract_wikipedia.sh --fresh"
    echo ""
fi

echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Starting extraction...${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${YELLOW}→${NC} Logs: logs/wikipedia_extraction.log"
echo -e "${YELLOW}→${NC} Output: data/extracted/wikipedia_sentences.jsonl"
echo -e "${YELLOW}→${NC} Press Ctrl+C to pause (will checkpoint automatically)"
echo ""

# Run extraction
python scripts/extract_wikipedia.py \
    --xml "$WIKI_DUMP" \
    --output data/extracted/wikipedia_sentences.jsonl \
    --checkpoint "$CHECKPOINT" \
    --checkpoint-interval 1000 \
    $FRESH_FLAG

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}✓ Wikipedia extraction complete!${NC}"
    echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
    echo ""

    # Show output file size
    if [ -f "data/extracted/wikipedia_sentences.jsonl" ]; then
        OUTPUT_SIZE=$(du -h data/extracted/wikipedia_sentences.jsonl | cut -f1)
        OUTPUT_LINES=$(wc -l < data/extracted/wikipedia_sentences.jsonl)
        echo -e "${GREEN}✓${NC} Output file: $OUTPUT_SIZE ($OUTPUT_LINES sentences)"
        echo -e "${GREEN}✓${NC} Location: data/extracted/wikipedia_sentences.jsonl"
    fi

    echo ""
    echo -e "${YELLOW}→${NC} Next step: Run books extraction"
    echo -e "${YELLOW}→${NC} Command: ./scripts/extract_gutenberg.sh"
else
    echo ""
    echo -e "${RED}════════════════════════════════════════════════════════════${NC}"
    echo -e "${RED}✗ Extraction failed or interrupted${NC}"
    echo -e "${RED}════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "${YELLOW}→${NC} Check logs: logs/wikipedia_extraction.log"
    echo -e "${YELLOW}→${NC} To resume, run this script again"
    exit 1
fi
