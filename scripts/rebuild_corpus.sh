#!/bin/bash
#
# Rebuild Unified Corpus with Hybrid Quality System
#
# This script:
# 1. Backs up existing corpus
# 2. Rebuilds corpus with automatic quality assessment (GOLD/SILVER/BRONZE/COPPER)
# 3. Logs all output for debugging
# 4. Supports checkpoint/resume for restartability
#
# Usage:
#   ./scripts/rebuild_corpus.sh              # Resume from checkpoint if available
#   ./scripts/rebuild_corpus.sh --fresh      # Start fresh (backup old corpus)
#   ./scripts/rebuild_corpus.sh --resume     # Explicit resume from checkpoint
#
# Quality System:
#   GOLD:   parse_rate >= 0.98 (exceptional)
#   SILVER: parse_rate >= 0.95 (high quality)
#   BRONZE: parse_rate >= 0.90 (good quality)
#   COPPER: parse_rate < 0.90  (fair quality)
#
# Overrides:
#   Edit config/quality_overrides.json for manual quality adjustments
#

set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Activate venv
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
else
    echo -e "${RED}❌ No virtual environment found (.venv or venv)${NC}"
    exit 1
fi

# Parse arguments
MODE="resume"  # Default: resume from checkpoint if available

for arg in "$@"; do
    case $arg in
        --fresh)
            MODE="fresh"
            shift
            ;;
        --resume)
            MODE="resume"
            shift
            ;;
        *)
            echo -e "${RED}Unknown argument: $arg${NC}"
            echo "Usage: $0 [--fresh|--resume]"
            exit 1
            ;;
    esac
done

# Setup paths
CORPUS_OUTPUT="data/enhanced_corpus/corpus_with_metadata.jsonl"
QUALITY_OVERRIDES="config/quality_overrides.json"
CHECKPOINT_FILE="data/enhanced_corpus/.build_corpus_checkpoint.json"

# Setup logging
LOG_DIR="logs/corpus_rebuild"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/rebuild_${TIMESTAMP}.log"

# Print banner
echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}                    CORPUS REBUILD - HYBRID QUALITY SYSTEM${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo ""
echo -e "${GREEN}Mode:${NC}              $MODE"
echo -e "${GREEN}Output corpus:${NC}     $CORPUS_OUTPUT"
echo -e "${GREEN}Quality config:${NC}    $QUALITY_OVERRIDES"
echo -e "${GREEN}Log file:${NC}          $LOG_FILE"
echo ""

# Check if checkpoint exists
if [ -f "$CHECKPOINT_FILE" ] && [ "$MODE" = "resume" ]; then
    echo -e "${YELLOW}⚠️  Checkpoint found - will resume from where we left off${NC}"
    echo ""
elif [ -f "$CHECKPOINT_FILE" ] && [ "$MODE" = "fresh" ]; then
    echo -e "${YELLOW}⚠️  Checkpoint found but --fresh specified - will start over${NC}"
    echo ""
elif [ "$MODE" = "resume" ]; then
    echo -e "${YELLOW}⚠️  No checkpoint found - starting from beginning${NC}"
    MODE="fresh"  # No checkpoint, so treat as fresh
    echo ""
fi

# Check if corpus already exists
if [ -f "$CORPUS_OUTPUT" ] && [ "$MODE" = "fresh" ]; then
    BACKUP_PATH="${CORPUS_OUTPUT}.backup_${TIMESTAMP}"
    echo -e "${YELLOW}📦 Backing up existing corpus to:${NC}"
    echo "   $BACKUP_PATH"
    cp "$CORPUS_OUTPUT" "$BACKUP_PATH"
    echo ""
fi

# Print quality system info
echo -e "${BLUE}Quality Assessment System:${NC}"
echo "  GOLD:   parse_rate >= 0.98 (exceptional quality)"
echo "  SILVER: parse_rate >= 0.95 (high quality)"
echo "  BRONZE: parse_rate >= 0.90 (good quality)"
echo "  COPPER: parse_rate < 0.90  (fair quality)"
echo ""

# Check for quality overrides
if [ -f "$QUALITY_OVERRIDES" ]; then
    OVERRIDE_COUNT=$(jq '[.overrides | to_entries[] | select(.key | startswith("_") | not) | select(.value.enabled // true)] | length' "$QUALITY_OVERRIDES" 2>/dev/null || echo "0")
    EXCLUDE_COUNT=$(jq '[.exclude | to_entries[] | select(.key | startswith("_") | not) | select(.value.enabled // true)] | length' "$QUALITY_OVERRIDES" 2>/dev/null || echo "0")

    if [ "$OVERRIDE_COUNT" -gt 0 ] || [ "$EXCLUDE_COUNT" -gt 0 ]; then
        echo -e "${GREEN}✓ Quality overrides loaded:${NC}"
        echo "  Manual overrides: $OVERRIDE_COUNT"
        echo "  Exclusions: $EXCLUDE_COUNT"
    else
        echo -e "${YELLOW}ℹ️  No manual quality overrides configured${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  No quality overrides file found - using automatic assessment only${NC}"
fi
echo ""

# Confirm before starting (only for fresh builds of large corpora)
if [ "$MODE" = "fresh" ] && [ -f "$CORPUS_OUTPUT" ]; then
    CORPUS_SIZE=$(du -h "$CORPUS_OUTPUT" | cut -f1)
    echo -e "${YELLOW}⚠️  This will rebuild the entire corpus (current size: $CORPUS_SIZE)${NC}"
    echo -e "${YELLOW}   Estimated time: Several hours${NC}"
    echo ""
    read -p "Continue? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${RED}Aborted by user${NC}"
        exit 0
    fi
    echo ""
fi

# Build flags
BUILD_FLAGS=""
if [ "$MODE" = "fresh" ]; then
    BUILD_FLAGS="--fresh"
elif [ "$MODE" = "resume" ]; then
    BUILD_FLAGS="--resume"
fi

# Run corpus builder
echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}                           STARTING CORPUS BUILD${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo ""
echo -e "${GREEN}Command:${NC} python scripts/build_unified_corpus.py $BUILD_FLAGS"
echo ""
echo "Logging to: $LOG_FILE"
echo ""
echo -e "${YELLOW}This will take several hours. You can:${NC}"
echo "  - Monitor progress: tail -f $LOG_FILE"
echo "  - Check statistics: grep 'Added:' $LOG_FILE"
echo "  - Stop safely: Ctrl+C (checkpoint will save progress)"
echo ""
echo -e "${GREEN}Starting build...${NC}"
echo ""

# Run with logging
if python scripts/build_unified_corpus.py \
    --output "$CORPUS_OUTPUT" \
    --overrides "$QUALITY_OVERRIDES" \
    $BUILD_FLAGS \
    2>&1 | tee "$LOG_FILE"; then

    # Success!
    echo ""
    echo -e "${BLUE}================================================================================${NC}"
    echo -e "${GREEN}✓ CORPUS BUILD COMPLETE!${NC}"
    echo -e "${BLUE}================================================================================${NC}"
    echo ""

    # Show summary from log
    echo -e "${GREEN}Summary:${NC}"
    grep -E "(Total sentences|GOLD|SILVER|BRONZE|COPPER|Size:)" "$LOG_FILE" | tail -6
    echo ""

    echo -e "${GREEN}Output:${NC} $CORPUS_OUTPUT"
    echo -e "${GREEN}Log:${NC}    $LOG_FILE"
    echo ""

    echo -e "${BLUE}Next steps:${NC}"
    echo "  1. Review quality distribution above"
    echo "  2. Rebuild Kuzu index:"
    echo "     ./scripts/index_kuzu.sh --fresh"
    echo "  3. Train models with new corpus:"
    echo "     ./scripts/train_m1_semantic_tier_priority.sh"
    echo ""

    exit 0
else
    # Build failed
    EXIT_CODE=$?
    echo ""
    echo -e "${BLUE}================================================================================${NC}"
    echo -e "${RED}✗ CORPUS BUILD FAILED${NC}"
    echo -e "${BLUE}================================================================================${NC}"
    echo ""
    echo -e "${RED}Exit code: $EXIT_CODE${NC}"
    echo ""
    echo -e "${YELLOW}Check the log for details:${NC}"
    echo "  cat $LOG_FILE"
    echo ""

    # Check if checkpoint exists
    if [ -f "$CHECKPOINT_FILE" ]; then
        echo -e "${GREEN}✓ Checkpoint saved - you can resume with:${NC}"
        echo "  ./scripts/rebuild_corpus.sh --resume"
        echo ""
    fi

    exit $EXIT_CODE
fi
