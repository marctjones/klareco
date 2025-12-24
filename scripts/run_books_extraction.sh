#!/bin/bash
#
# Extract book sentences with chapter metadata
#
# This script:
# - Activates Python virtual environment
# - Runs book extraction with chapter detection
# - Logs all output to logs/books_extraction.log
#
# Usage:
#   ./scripts/run_books_extraction.sh

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
echo -e "${BLUE}Book Extraction with Chapter Metadata${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo ""

# Change to project root
cd "$PROJECT_ROOT"
echo -e "${YELLOW}→${NC} Project root: $PROJECT_ROOT"

# Activate virtual environment
echo -e "${YELLOW}→${NC} Activating virtual environment..."
source .venv/bin/activate
echo -e "${GREEN}✓${NC} Virtual environment activated"

# Create necessary directories
mkdir -p data/extracted
mkdir -p logs

# Check if cleaned books exist
CLEANED_DIR="data/cleaned"
if [ ! -d "$CLEANED_DIR" ]; then
    echo -e "${RED}✗${NC} Cleaned books directory not found: $CLEANED_DIR"
    exit 1
fi

# Count books
BOOK_COUNT=$(ls -1 "$CLEANED_DIR"/cleaned_*.txt 2>/dev/null | wc -l)
echo -e "${GREEN}✓${NC} Found $BOOK_COUNT cleaned book files"
echo ""

echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Starting extraction...${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${YELLOW}→${NC} Logs: logs/books_extraction.log"
echo -e "${YELLOW}→${NC} Output: data/extracted/books_sentences.jsonl"
echo ""

# Run extraction
python scripts/extract_books_with_metadata.py \
    --cleaned-dir "$CLEANED_DIR" \
    --output data/extracted/books_sentences.jsonl

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}✓ Book extraction complete!${NC}"
    echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
    echo ""

    # Show output file size
    if [ -f "data/extracted/books_sentences.jsonl" ]; then
        OUTPUT_SIZE=$(du -h data/extracted/books_sentences.jsonl | cut -f1)
        OUTPUT_LINES=$(wc -l < data/extracted/books_sentences.jsonl)
        echo -e "${GREEN}✓${NC} Output file: $OUTPUT_SIZE ($OUTPUT_LINES sentences)"
        echo -e "${GREEN}✓${NC} Location: data/extracted/books_sentences.jsonl"
    fi

    echo ""
    echo -e "${YELLOW}→${NC} Next step: Build enhanced corpus"
    echo -e "${YELLOW}→${NC} Command: ./scripts/run_corpus_builder.sh"
else
    echo ""
    echo -e "${RED}✗ Extraction failed${NC}"
    echo -e "${YELLOW}→${NC} Check logs: logs/books_extraction.log"
    exit 1
fi
