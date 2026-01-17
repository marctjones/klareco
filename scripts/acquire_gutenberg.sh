#!/bin/bash
#
# Download Esperanto texts from Project Gutenberg
#
# This script downloads all Esperanto texts from Project Gutenberg.
# Supports resuming from checkpoint if interrupted.
#
# Usage:
#   ./scripts/acquire_gutenberg.sh              # Resume from checkpoint
#   ./scripts/acquire_gutenberg.sh --fresh      # Start fresh, ignore checkpoint
#
# Output:
#   data/raw/eo/gutenberg/*.txt                 # Downloaded ebook texts
#   data/raw/eo/gutenberg/_download_checkpoint.json  # Resume checkpoint

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

# Parse arguments
FRESH_FLAG=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --fresh)
            FRESH_FLAG="--fresh"
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Project Gutenberg Esperanto Text Downloader${NC}"
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

# Create output directory
mkdir -p data/raw/eo/gutenberg
mkdir -p logs

echo ""
echo -e "${YELLOW}→${NC} This will download Esperanto texts from Project Gutenberg"
echo -e "${YELLOW}→${NC} Resume capability: Will skip already downloaded texts"
echo -e "${YELLOW}→${NC} Estimated time: 5-10 minutes (depends on network speed)"
echo ""

# Set up logging
LOG_FILE="logs/acquire_gutenberg_$(date +%Y%m%d_%H%M%S).log"
echo -e "${YELLOW}→${NC} Log file: $LOG_FILE"
echo ""

# Run download script with logging
python scripts/acquire_gutenberg.py \
    $FRESH_FLAG \
    2>&1 | tee "$LOG_FILE"

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}✓ Download complete!${NC}"
    echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
    echo ""

    # Show downloaded files
    TOTAL_FILES=$(ls -1 data/raw/eo/gutenberg/*.txt 2>/dev/null | wc -l)
    echo -e "${GREEN}✓${NC} Downloaded texts: $TOTAL_FILES"
    echo -e "${GREEN}✓${NC} Location: data/raw/eo/gutenberg/"
    echo ""
    echo -e "${YELLOW}→${NC} Next step: Clean the downloaded texts"
    echo -e "${YELLOW}→${NC} Command: ./scripts/clean_gutenberg.sh"
else
    echo ""
    echo -e "${RED}✗ Download failed${NC}"
    echo -e "${YELLOW}→${NC} Check logs: $LOG_FILE"
    exit 1
fi
