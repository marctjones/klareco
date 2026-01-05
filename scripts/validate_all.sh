#!/bin/bash
#
# Run all validation checks
#
# This script runs all validation steps:
# 1. Validate vocabulary coverage
# 2. Validate corpus quality
# 3. Validate Stage 1 embeddings (if trained)
#
# Usage:
#   ./scripts/validate_all.sh
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

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Run All Validation Checks${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Activate venv
if [[ -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
elif [[ -f "venv/bin/activate" ]]; then
    source venv/bin/activate
fi

PASSED=0
FAILED=0

# Validate vocabulary
echo -e "${GREEN}Step 1: Validating vocabulary...${NC}"
if python scripts/validate_vocabulary.py; then
    ((PASSED++))
else
    ((FAILED++))
fi
echo ""

# Validate corpus
echo -e "${GREEN}Step 2: Validating corpus...${NC}"
if python scripts/validate_corpus.py; then
    ((PASSED++))
else
    ((FAILED++))
fi
echo ""

# Validate Stage 1 (if models exist)
if [[ -f "models/root_embeddings/best_model.pt" ]]; then
    echo -e "${GREEN}Step 3: Validating Stage 1 embeddings...${NC}"
    if python scripts/validate_stage1.py; then
        ((PASSED++))
    else
        ((FAILED++))
    fi
else
    echo -e "${YELLOW}Step 3: Skipping Stage 1 validation (no trained model)${NC}"
fi
echo ""

# Summary
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Validation Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${GREEN}Passed:${NC} $PASSED"
if [[ $FAILED -gt 0 ]]; then
    echo -e "${YELLOW}Failed:${NC} $FAILED"
else
    echo -e "${GREEN}Failed:${NC} $FAILED"
fi
echo ""

exit $FAILED
