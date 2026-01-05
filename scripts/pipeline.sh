#!/bin/bash
#
# Full Klareco Pipeline
#
# Runs the complete data processing pipeline:
# 1. Acquire - Download raw data
# 2. Clean - Clean/normalize text
# 3. Extract - Extract sentences with metadata
# 4. Parse - Parse to ASTs and build unified corpus
# 5. Index - Build FAISS indexes
# 6. Train - Train embedding models
# 7. Validate - Validate quality
#
# Usage:
#   ./scripts/pipeline.sh              # Run full pipeline
#   ./scripts/pipeline.sh --from clean # Start from clean stage
#   ./scripts/pipeline.sh --only index # Run only index stage
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Parse arguments
FROM_STAGE=""
ONLY_STAGE=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --from)
            FROM_STAGE="$2"
            shift 2
            ;;
        --only)
            ONLY_STAGE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Stage order
STAGES=("acquire" "clean" "extract" "parse" "index" "train" "validate")

# Check if we should run a stage
should_run() {
    local stage=$1

    if [[ -n "$ONLY_STAGE" ]]; then
        [[ "$stage" == "$ONLY_STAGE" ]]
        return
    fi

    if [[ -n "$FROM_STAGE" ]]; then
        local found=false
        for s in "${STAGES[@]}"; do
            if [[ "$s" == "$FROM_STAGE" ]]; then
                found=true
            fi
            if $found && [[ "$s" == "$stage" ]]; then
                return 0
            fi
        done
        return 1
    fi

    return 0
}

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Klareco Full Pipeline${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Activate venv
if [[ -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
elif [[ -f "venv/bin/activate" ]]; then
    source venv/bin/activate
fi

# Stage 1: Acquire
if should_run "acquire"; then
    echo -e "${GREEN}=== Stage 1: ACQUIRE ===${NC}"
    echo "Downloading raw data..."
    # Note: Wikipedia must be downloaded manually from dumps.wikimedia.org
    # ./scripts/acquire_gutenberg.sh  # Uncomment if needed
    echo -e "${YELLOW}Note: Run acquire_gutenberg.py manually if needed${NC}"
    echo ""
fi

# Stage 2: Clean
if should_run "clean"; then
    echo -e "${GREEN}=== Stage 2: CLEAN ===${NC}"
    ./scripts/clean_all.sh
    echo ""
fi

# Stage 3: Extract
if should_run "extract"; then
    echo -e "${GREEN}=== Stage 3: EXTRACT ===${NC}"
    ./scripts/extract_all.sh
    echo ""
fi

# Stage 4: Parse
if should_run "parse"; then
    echo -e "${GREEN}=== Stage 4: PARSE ===${NC}"
    ./scripts/parse_corpus.sh
    echo ""
fi

# Stage 5: Index
if should_run "index"; then
    echo -e "${GREEN}=== Stage 5: INDEX ===${NC}"
    ./scripts/index_compositional.sh
    echo ""
fi

# Stage 6: Train
if should_run "train"; then
    echo -e "${GREEN}=== Stage 6: TRAIN ===${NC}"
    ./scripts/train_roots.sh
    ./scripts/train_affixes.sh
    echo ""
fi

# Stage 7: Validate
if should_run "validate"; then
    echo -e "${GREEN}=== Stage 7: VALIDATE ===${NC}"
    ./scripts/validate_all.sh
    echo ""
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Pipeline Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "  - Run demo: python scripts/demo_rag.py --interactive"
echo "  - Check indexes: ls -la data/indexes/"
echo ""
