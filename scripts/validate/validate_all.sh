#!/bin/bash
#
# Run all validation checks
#
# Currently runs:
#   1. Corpus quality (validate_corpus.py)
#   2. Kuzu graph integrity (validate_kuzu_v2.1.py) — only if a v2.1
#      Kuzu graph is present at data/indexes/v2.1_kuzu_index_full
#
# Usage:
#   ./scripts/validate/validate_all.sh
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

# Validate corpus
echo -e "${GREEN}Step 1: Validating corpus...${NC}"
if python scripts/validate/validate_corpus.py; then
    ((PASSED++))
else
    ((FAILED++))
fi
echo ""

# (Kuzu graph validation removed 2026-07-18 — Kuzu retired; the store is DuckDB.
#  Use `python scripts/index/validate_duckdb_store.py` for store integrity.)

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
