#!/bin/bash
#
# Build enhanced corpus with full metadata
#
# This script:
# - Parses extracted sentences to ASTs
# - Filters by parse quality
# - Combines all sources into single corpus
# - Shows progress and logs errors
#
# Usage:
#   ./scripts/run_corpus_builder.sh [--min-parse-rate 0.5]

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default parameters
MIN_PARSE_RATE=0.5

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --min-parse-rate)
            MIN_PARSE_RATE="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Enhanced Corpus Builder${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo ""

# Change to project root
cd "$PROJECT_ROOT"
echo -e "${YELLOW}→${NC} Project root: $PROJECT_ROOT"

# Activate virtual environment
echo -e "${YELLOW}→${NC} Activating virtual environment..."
source .venv/bin/activate
echo -e "${GREEN}✓${NC} Virtual environment activated"

# Create directories
mkdir -p data/enhanced_corpus
mkdir -p logs

echo ""
echo -e "${YELLOW}Settings:${NC}"
echo -e "  Min parse rate: ${BLUE}$MIN_PARSE_RATE${NC}"
echo ""

# Check if extracted files exist
WIKI_EXTRACTED="data/extracted/wikipedia_sentences.jsonl"
BOOKS_EXTRACTED="data/extracted/books_sentences.jsonl"

if [ ! -f "$WIKI_EXTRACTED" ]; then
    echo -e "${RED}✗${NC} Wikipedia extraction not found: $WIKI_EXTRACTED"
    echo -e "${YELLOW}→${NC} Run: ./scripts/run_wikipedia_extraction.sh"
    exit 1
fi

if [ ! -f "$BOOKS_EXTRACTED" ]; then
    echo -e "${RED}✗${NC} Books extraction not found: $BOOKS_EXTRACTED"
    echo -e "${YELLOW}→${NC} Run: ./scripts/run_books_extraction.sh"
    exit 1
fi

# Show input file sizes
WIKI_SIZE=$(du -h "$WIKI_EXTRACTED" | cut -f1)
WIKI_LINES=$(wc -l < "$WIKI_EXTRACTED")
BOOKS_SIZE=$(du -h "$BOOKS_EXTRACTED" | cut -f1)
BOOKS_LINES=$(wc -l < "$BOOKS_EXTRACTED")

echo -e "${GREEN}✓${NC} Wikipedia extracted: $WIKI_SIZE ($WIKI_LINES sentences)"
echo -e "${GREEN}✓${NC} Books extracted: $BOOKS_SIZE ($BOOKS_LINES sentences)"
echo ""

echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Starting corpus building...${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${YELLOW}→${NC} This will parse all sentences to ASTs"
echo -e "${YELLOW}→${NC} Estimated time: 2-4 hours for full corpus"
echo -e "${YELLOW}→${NC} Logs: logs/corpus_building.log"
echo -e "${YELLOW}→${NC} Progress updates every 100 sentences"
echo ""
echo -e "${YELLOW}→${NC} Press Ctrl+C to pause (will checkpoint)"
echo ""

# Run corpus builder
python scripts/build_enhanced_corpus.py \
    --stage all \
    --min-parse-rate "$MIN_PARSE_RATE" \
    --output-dir data/enhanced_corpus

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}✓ Enhanced corpus complete!${NC}"
    echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
    echo ""

    # Show final corpus stats
    CORPUS_FILE="data/enhanced_corpus/corpus_with_metadata.jsonl"
    if [ -f "$CORPUS_FILE" ]; then
        CORPUS_SIZE=$(du -h "$CORPUS_FILE" | cut -f1)
        CORPUS_LINES=$(wc -l < "$CORPUS_FILE")
        echo -e "${GREEN}✓${NC} Final corpus: $CORPUS_SIZE"
        echo -e "${GREEN}✓${NC} Total sentences: $CORPUS_LINES"
        echo -e "${GREEN}✓${NC} Location: $CORPUS_FILE"
        echo ""

        # Show sample entry
        echo -e "${BLUE}Sample entry (first sentence):${NC}"
        head -1 "$CORPUS_FILE" | python3 -m json.tool | head -30
        echo ""

        # Calculate statistics
        echo -e "${YELLOW}→${NC} Calculating corpus statistics..."
        python3 << 'PYEOF'
import json

wiki_count = 0
books_count = 0
total_words = 0
parse_rates = []

with open('data/enhanced_corpus/corpus_with_metadata.jsonl', 'r') as f:
    for line in f:
        entry = json.loads(line)
        if entry['source'] == 'wikipedia':
            wiki_count += 1
        else:
            books_count += 1
        total_words += entry.get('word_count', 0)
        parse_rates.append(entry.get('parse_rate', 0))

avg_parse_rate = sum(parse_rates) / len(parse_rates) if parse_rates else 0

print(f'')
print(f'Corpus Statistics:')
print(f'  Wikipedia: {wiki_count:,} sentences')
print(f'  Books: {books_count:,} sentences')
print(f'  Total words: {total_words:,}')
print(f'  Average parse rate: {avg_parse_rate:.2%}')
PYEOF
    fi

    echo ""
    echo -e "${YELLOW}→${NC} Next step: Index the corpus for retrieval"
    echo -e "${YELLOW}→${NC} Command: python scripts/index_corpus.py --corpus data/enhanced_corpus/corpus_with_metadata.jsonl"
else
    echo ""
    echo -e "${RED}✗ Corpus building failed${NC}"
    echo -e "${YELLOW}→${NC} Check logs: logs/corpus_building.log"
    exit 1
fi
