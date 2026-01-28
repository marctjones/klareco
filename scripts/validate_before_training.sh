#!/bin/bash
#
# Pre-Training Validation Workflow
#
# Validates everything before starting the expensive training process:
# 1. Corpus quality
# 2. Kùzu index integrity
# 3. Training data quality
# 4. Training data freshness (matches current corpus)
#
# Usage:
#   ./scripts/validate_before_training.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}Pre-Training Validation${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""

# Activate venv
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
fi

# Step 1: Validate corpus
echo -e "${GREEN}=== Step 1: Validate Corpus ===${NC}"
echo "Checking corpus quality (quick sample)..."
python scripts/validate_corpus.py --quick
echo ""

# Step 2: Validate Kùzu index
echo -e "${GREEN}=== Step 2: Validate Kùzu Index ===${NC}"
echo "Checking index integrity..."
python scripts/validate_kuzu_index.py
echo ""

# Step 3: Check training data freshness
echo -e "${GREEN}=== Step 3: Check Training Data Freshness ===${NC}"

CORPUS_FILE="data/enhanced_corpus/corpus_with_metadata.jsonl"
CORPUS_TIME=$(stat -c %Y "$CORPUS_FILE" 2>/dev/null || stat -f %m "$CORPUS_FILE")

echo "Corpus timestamp: $(date -d @$CORPUS_TIME 2>/dev/null || date -r $CORPUS_TIME)"
echo ""

# Check M1 training data
M1_DIRS=(
    "data/training/m1_semantic_full"
    "data/training/m1_with_tier0"
    "data/training/m1_tier0_only"
)

STALE_DATA=false
for dir in "${M1_DIRS[@]}"; do
    if [ -f "$dir/train.jsonl" ]; then
        TRAIN_TIME=$(stat -c %Y "$dir/train.jsonl" 2>/dev/null || stat -f %m "$dir/train.jsonl")
        DIR_NAME=$(basename "$dir")

        if [ $TRAIN_TIME -lt $CORPUS_TIME ]; then
            echo -e "${YELLOW}⚠️  STALE: $DIR_NAME (older than corpus)${NC}"
            STALE_DATA=true
        else
            echo -e "${GREEN}✓ FRESH: $DIR_NAME${NC}"
        fi
    fi
done

# Check root embeddings training data
if [ -f "data/training/ekzercaro_sentences.jsonl" ]; then
    EKZERCARO_TIME=$(stat -c %Y "data/training/ekzercaro_sentences.jsonl" 2>/dev/null || stat -f %m "data/training/ekzercaro_sentences.jsonl")

    if [ $EKZERCARO_TIME -lt $CORPUS_TIME ]; then
        echo -e "${YELLOW}⚠️  STALE: ekzercaro_sentences.jsonl (older than corpus)${NC}"
        STALE_DATA=true
    else
        echo -e "${GREEN}✓ FRESH: ekzercaro_sentences.jsonl${NC}"
    fi
fi

echo ""

if [ "$STALE_DATA" = true ]; then
    echo -e "${YELLOW}============================================================${NC}"
    echo -e "${YELLOW}ACTION REQUIRED: Regenerate Training Data${NC}"
    echo -e "${YELLOW}============================================================${NC}"
    echo ""
    echo "Training data is older than corpus. You must regenerate it:"
    echo ""
    echo "  # Regenerate M1 training data"
    echo "  ./scripts/train_m1_semantic_tier_priority.sh"
    echo "  # (This will regenerate data automatically before training)"
    echo ""
    echo "  # For root embeddings, ekzercaro data is extracted on-the-fly"
    echo "  # from corpus, so no regeneration needed"
    echo ""
    exit 1
fi

# Step 4: Validate training data quality (if fresh)
echo -e "${GREEN}=== Step 4: Validate Training Data Quality ===${NC}"

# Validate M1 data (if exists)
if [ -d "data/training/m1_semantic_full" ]; then
    echo "Checking M1 semantic data..."
    python scripts/validate_training_data.py data/training/m1_semantic_full
    echo ""
fi

# Validate ekzercaro data (if exists)
if [ -f "data/training/ekzercaro_sentences.jsonl" ]; then
    echo "Checking ekzercaro data..."
    python scripts/validate_training_data.py data/training/ekzercaro_sentences.jsonl
    echo ""
fi

echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}✓ All Validations Passed!${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo "Ready to train models:"
echo "  1. ./scripts/train_roots.sh --fresh"
echo "  2. ./scripts/train_m1_semantic_tier_priority.sh"
echo ""
