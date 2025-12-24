#!/bin/bash
#
# Run full corpus extraction pipeline
#
# This script:
# 1. Archives any existing extracted data
# 2. Runs Wikipedia extraction (2-3 hours)
# 3. Runs book extraction (5-10 minutes)
# 4. Shows summary and next steps
#
# Usage:
#   ./scripts/run_full_extraction.sh
#   ./scripts/run_full_extraction.sh --no-archive  # Skip archiving

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

cd "$PROJECT_ROOT"

echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Full Corpus Extraction Pipeline${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "This will:"
echo -e "  1. Archive existing extracted data (if any)"
echo -e "  2. Extract Wikipedia articles (~2-3 hours, ~3.8M sentences)"
echo -e "  3. Extract book sentences (~5-10 min, ~1.3M sentences)"
echo -e ""
echo -e "${YELLOW}⚠${NC}  Total time: ~2-3 hours"
echo -e "${YELLOW}⚠${NC}  You can monitor progress in another terminal with:"
echo -e "     tail -f logs/wikipedia_extraction.log"
echo -e "     tail -f logs/books_extraction.log"
echo ""

# Check if user wants to skip archiving
SKIP_ARCHIVE=false
if [ "$1" = "--no-archive" ]; then
    SKIP_ARCHIVE=true
fi

# Archive existing data
if [ "$SKIP_ARCHIVE" = false ]; then
    ARCHIVE_DIR="data/archive/extraction_$(date +%Y%m%d_%H%M%S)"

    # Check if there's anything to archive
    if [ -d "data/extracted" ] && [ "$(ls -A data/extracted 2>/dev/null)" ]; then
        echo -e "${YELLOW}→${NC} Archiving existing extracted data..."
        mkdir -p "$ARCHIVE_DIR"

        # Archive extracted sentences
        if [ -f "data/extracted/wikipedia_sentences.jsonl" ]; then
            SIZE=$(du -h data/extracted/wikipedia_sentences.jsonl | cut -f1)
            LINES=$(wc -l < data/extracted/wikipedia_sentences.jsonl)
            cp data/extracted/wikipedia_sentences.jsonl "$ARCHIVE_DIR/"
            echo -e "${GREEN}✓${NC} Archived Wikipedia: $SIZE ($LINES sentences)"
        fi

        if [ -f "data/extracted/books_sentences.jsonl" ]; then
            SIZE=$(du -h data/extracted/books_sentences.jsonl | cut -f1)
            LINES=$(wc -l < data/extracted/books_sentences.jsonl)
            cp data/extracted/books_sentences.jsonl "$ARCHIVE_DIR/"
            echo -e "${GREEN}✓${NC} Archived Books: $SIZE ($LINES sentences)"
        fi

        # Archive checkpoint
        if [ -f "data/extracted/wikipedia_checkpoint.json" ]; then
            cp data/extracted/wikipedia_checkpoint.json "$ARCHIVE_DIR/"
        fi

        echo -e "${GREEN}✓${NC} Archive location: $ARCHIVE_DIR"
        echo ""
    else
        echo -e "${YELLOW}→${NC} No existing data to archive"
        echo ""
    fi

    # Clean extracted directory
    echo -e "${YELLOW}→${NC} Cleaning extracted data directory..."
    rm -rf data/extracted
    mkdir -p data/extracted
    echo -e "${GREEN}✓${NC} Clean slate ready"
    echo ""
else
    echo -e "${YELLOW}→${NC} Skipping archive step (--no-archive flag)"
    echo ""
fi

# Confirm before starting
read -p "Ready to start extraction? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}→${NC} Cancelled by user"
    exit 0
fi

echo ""
START_TIME=$(date +%s)

# Step 1: Wikipedia extraction
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Step 1/2: Wikipedia Extraction${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${YELLOW}⚠${NC}  This will take 2-3 hours"
echo -e "${YELLOW}→${NC}  Monitor: tail -f logs/wikipedia_extraction.log"
echo ""

./scripts/run_wikipedia_extraction.sh

if [ $? -ne 0 ]; then
    echo -e "${RED}✗${NC} Wikipedia extraction failed"
    exit 1
fi

# Step 2: Book extraction
echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Step 2/2: Book Extraction${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${YELLOW}⚠${NC}  This will take 5-10 minutes"
echo -e "${YELLOW}→${NC}  Monitor: tail -f logs/books_extraction.log"
echo ""

./scripts/run_books_extraction.sh

if [ $? -ne 0 ]; then
    echo -e "${RED}✗${NC} Book extraction failed"
    exit 1
fi

# Summary
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))

echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✓ Full Extraction Complete!${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo ""

# Show results
if [ -f "data/extracted/wikipedia_sentences.jsonl" ]; then
    WIKI_SIZE=$(du -h data/extracted/wikipedia_sentences.jsonl | cut -f1)
    WIKI_LINES=$(wc -l < data/extracted/wikipedia_sentences.jsonl)
    echo -e "${GREEN}✓${NC} Wikipedia: $WIKI_SIZE ($WIKI_LINES sentences)"
fi

if [ -f "data/extracted/books_sentences.jsonl" ]; then
    BOOKS_SIZE=$(du -h data/extracted/books_sentences.jsonl | cut -f1)
    BOOKS_LINES=$(wc -l < data/extracted/books_sentences.jsonl)
    echo -e "${GREEN}✓${NC} Books: $BOOKS_SIZE ($BOOKS_LINES sentences)"
fi

echo ""
echo -e "${YELLOW}→${NC} Total time: ${HOURS}h ${MINUTES}m"
echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Next Steps${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "1. Build enhanced corpus with ASTs:"
echo -e "   ${YELLOW}./scripts/run_corpus_builder.sh${NC}"
echo -e ""
echo -e "2. Index for retrieval:"
echo -e "   ${YELLOW}python scripts/index_corpus.py \\${NC}"
echo -e "   ${YELLOW}     --corpus data/enhanced_corpus/corpus_with_metadata.jsonl \\${NC}"
echo -e "   ${YELLOW}     --output data/corpus_index${NC}"
echo ""
