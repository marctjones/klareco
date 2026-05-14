#!/bin/bash
#
# Clean all raw Esperanto texts
#
# This script runs all cleaning steps:
# 1. Clean Gutenberg texts (remove headers/footers)
# 2. Clean ReVo vocabulary
#
# Usage:
#   ./scripts/clean/clean_all.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Clean All Esperanto Texts${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Activate venv
if [[ -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
elif [[ -f "venv/bin/activate" ]]; then
    source venv/bin/activate
fi

# Clean Gutenberg texts
echo -e "${GREEN}Step 1: Cleaning Gutenberg texts...${NC}"
python scripts/clean/clean_gutenberg.py \
    --input data/raw/eo/gutenberg \
    --output data/cleaned/eo

# Clean ReVo vocabulary
echo -e "${GREEN}Step 2: Cleaning ReVo vocabulary...${NC}"
python scripts/clean/clean_revo.py

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Cleaning Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${GREEN}Output:${NC} data/cleaned/eo/"
echo ""
echo -e "${BLUE}Next step:${NC} ./scripts/extract/extract_all.sh"
