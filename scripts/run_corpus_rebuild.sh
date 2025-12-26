#!/bin/bash
# =============================================================================
# Corpus Rebuild Pipeline
# =============================================================================
# Rebuilds the entire corpus with updated parser (prefiksoj format).
#
# This script is restartable - each step has checkpoint support.
# Run with --fresh to start from scratch, otherwise it resumes.
#
# Usage:
#   ./scripts/run_corpus_rebuild.sh           # Resume from checkpoint
#   ./scripts/run_corpus_rebuild.sh --fresh   # Start fresh
#
# Steps:
#   1. Build authoritative corpus (Fundamento, Krestomatio, Gerda)
#   2. Build general corpus (Gutenberg books, Wikipedia)
#   3. Annotate general corpus with tiers
#   4. Merge into unified corpus
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
DATA_DIR="$PROJECT_ROOT/data"
CORPUS_DIR="$DATA_DIR/corpus"
CLEANED_DIR="$DATA_DIR/cleaned"

# Output files
AUTHORITATIVE_CORPUS="$CORPUS_DIR/authoritative_corpus.jsonl"
GENERAL_CORPUS="$CORPUS_DIR/general_corpus.jsonl"
TIERED_CORPUS="$CORPUS_DIR/tiered_general_corpus.jsonl"
UNIFIED_CORPUS="$CORPUS_DIR/unified_corpus.jsonl"

# Create corpus directory
mkdir -p "$CORPUS_DIR"

# Log file
LOG_FILE="$PROJECT_ROOT/logs/corpus_rebuild_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$PROJECT_ROOT/logs"

echo -e "${BLUE}=== Corpus Rebuild Pipeline ===${NC}"
echo "Project root: $PROJECT_ROOT"
echo "Log file: $LOG_FILE"
echo "Fresh start: $FRESH_START"
echo ""

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# =============================================================================
# Step 1: Build Authoritative Corpus
# =============================================================================
echo -e "${YELLOW}=== Step 1: Build Authoritative Corpus ===${NC}"
log "Starting authoritative corpus build..."

AUTH_CHECKPOINT="$CORPUS_DIR/authoritative_corpus.checkpoint.json"

if [ "$FRESH_START" = true ]; then
    FRESH_FLAG="--fresh"
    log "Fresh start requested - removing old checkpoints"
else
    FRESH_FLAG=""
fi

python scripts/build_authoritative_corpus.py \
    --output "$AUTHORITATIVE_CORPUS" \
    $FRESH_FLAG \
    2>&1 | tee -a "$LOG_FILE"

AUTH_COUNT=$(wc -l < "$AUTHORITATIVE_CORPUS")
log "Authoritative corpus: $AUTH_COUNT entries"
echo -e "${GREEN}Step 1 complete: $AUTH_COUNT entries${NC}"
echo ""

# =============================================================================
# Step 2: Build General Corpus
# =============================================================================
echo -e "${YELLOW}=== Step 2: Build General Corpus ===${NC}"
log "Starting general corpus build..."

if [ "$FRESH_START" = true ]; then
    CLEAN_FLAG="--clean"
else
    CLEAN_FLAG=""
fi

python scripts/build_corpus_v2.py \
    --cleaned-dir "$CLEANED_DIR" \
    --output "$GENERAL_CORPUS" \
    --checkpoint "$DATA_DIR/build_corpus_v2_checkpoint.json" \
    --min-parse-rate 0.0 \
    --batch-size 50 \
    --throttle 0.05 \
    $CLEAN_FLAG \
    2>&1 | tee -a "$LOG_FILE"

GEN_COUNT=$(wc -l < "$GENERAL_CORPUS")
log "General corpus: $GEN_COUNT entries"
echo -e "${GREEN}Step 2 complete: $GEN_COUNT entries${NC}"
echo ""

# =============================================================================
# Step 3: Annotate General Corpus with Tiers
# =============================================================================
echo -e "${YELLOW}=== Step 3: Annotate with Tiers ===${NC}"
log "Starting tier annotation..."

if [ "$FRESH_START" = true ]; then
    FRESH_FLAG="--fresh"
else
    FRESH_FLAG=""
fi

python scripts/annotate_corpus_tiers.py \
    --input "$GENERAL_CORPUS" \
    --output "$TIERED_CORPUS" \
    $FRESH_FLAG \
    2>&1 | tee -a "$LOG_FILE"

TIERED_COUNT=$(wc -l < "$TIERED_CORPUS")
log "Tiered corpus: $TIERED_COUNT entries"
echo -e "${GREEN}Step 3 complete: $TIERED_COUNT entries${NC}"
echo ""

# =============================================================================
# Step 4: Merge Corpora
# =============================================================================
echo -e "${YELLOW}=== Step 4: Merge Corpora ===${NC}"
log "Starting corpus merge..."

if [ "$FRESH_START" = true ]; then
    FRESH_FLAG="--fresh"
else
    FRESH_FLAG=""
fi

python scripts/merge_corpora.py \
    --authoritative "$AUTHORITATIVE_CORPUS" \
    --general "$TIERED_CORPUS" \
    --output "$UNIFIED_CORPUS" \
    --deduplicate \
    --min-parse-rate 0.5 \
    $FRESH_FLAG \
    2>&1 | tee -a "$LOG_FILE"

UNIFIED_COUNT=$(wc -l < "$UNIFIED_CORPUS")
log "Unified corpus: $UNIFIED_COUNT entries"
echo -e "${GREEN}Step 4 complete: $UNIFIED_COUNT entries${NC}"
echo ""

# =============================================================================
# Summary
# =============================================================================
echo -e "${BLUE}=== Pipeline Complete ===${NC}"
echo ""
echo "Output files:"
echo "  Authoritative: $AUTHORITATIVE_CORPUS ($AUTH_COUNT entries)"
echo "  General:       $GENERAL_CORPUS ($GEN_COUNT entries)"
echo "  Tiered:        $TIERED_CORPUS ($TIERED_COUNT entries)"
echo "  Unified:       $UNIFIED_CORPUS ($UNIFIED_COUNT entries)"
echo ""
echo "Metadata: $UNIFIED_CORPUS.meta.json"
echo "Log file: $LOG_FILE"
echo ""

# Show unified corpus metadata
if [ -f "${UNIFIED_CORPUS%.jsonl}.meta.json" ]; then
    echo "Unified corpus metadata:"
    cat "${UNIFIED_CORPUS%.jsonl}.meta.json"
fi

log "Pipeline complete!"
echo -e "${GREEN}Done!${NC}"
