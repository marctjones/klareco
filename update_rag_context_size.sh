#!/bin/bash
# Update RAG Expert default context size to match trained models
#
# This script updates the RAG Expert configuration to use the same
# context size that the models were trained with.
#
# Usage:
#   ./update_rag_context_size.sh 50    # Set k=50 in RAGExpert

set -e

CONTEXT_SIZE=${1:-50}
RAG_EXPERT_FILE="klareco/experts/rag_expert.py"

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "========================================================================"
echo "UPDATING RAG EXPERT CONTEXT SIZE"
echo "========================================================================"
echo ""
echo "Setting default k=$CONTEXT_SIZE in $RAG_EXPERT_FILE"
echo ""

# Check if file exists
if [ ! -f "$RAG_EXPERT_FILE" ]; then
    echo -e "${YELLOW}ERROR: File not found: $RAG_EXPERT_FILE${NC}"
    exit 1
fi

# Create backup
BACKUP="${RAG_EXPERT_FILE}.backup_$(date +%Y%m%d_%H%M%S)"
cp "$RAG_EXPERT_FILE" "$BACKUP"
echo -e "${GREEN}✓ Backup created: $BACKUP${NC}"
echo ""

# Update the default k value
# Line 46: k: int = 5,
# Change to: k: int = CONTEXT_SIZE,
sed -i "s/k: int = [0-9]\+,/k: int = $CONTEXT_SIZE,/" "$RAG_EXPERT_FILE"

# Verify change
if grep -q "k: int = $CONTEXT_SIZE," "$RAG_EXPERT_FILE"; then
    echo -e "${GREEN}✓ Successfully updated RAG Expert default k=$CONTEXT_SIZE${NC}"
    echo ""
    echo "Modified line:"
    grep "k: int = " "$RAG_EXPERT_FILE" | head -1
else
    echo -e "${YELLOW}WARNING: Could not verify change${NC}"
    echo "Please manually check $RAG_EXPERT_FILE line 46"
    exit 1
fi

echo ""
echo "========================================================================"
echo "✓ RAG EXPERT UPDATED"
echo "========================================================================"
echo ""
echo "The RAG Expert will now retrieve $CONTEXT_SIZE context documents by default."
echo ""
echo "To restore the original:"
echo "  cp $BACKUP $RAG_EXPERT_FILE"
echo ""
