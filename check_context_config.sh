#!/bin/bash
# Quick check of current context size configuration
# Shows current limits and recommends upgrades

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

echo ""
echo -e "${BOLD}========================================================================"
echo "KLARECO CONTEXT SIZE CONFIGURATION CHECK"
echo -e "========================================================================${NC}"
echo ""

# Check generate_qa_dataset.py
echo -e "${BLUE}1. QA Dataset Generation (scripts/generate_qa_dataset.py:151)${NC}"
if [ -f "scripts/generate_qa_dataset.py" ]; then
    context_line=$(grep -n "'context': context\[:" scripts/generate_qa_dataset.py | head -1)
    current_value=$(echo "$context_line" | sed -n "s/.*\[::\?\([0-9]*\)\].*/\1/p")

    if [ -z "$current_value" ]; then
        current_value=$(echo "$context_line" | sed -n "s/.*\[:\([0-9]*\)\].*/\1/p")
    fi

    if [ -n "$current_value" ]; then
        if [ "$current_value" -lt 20 ]; then
            echo -e "   ${RED}Current: $current_value context sentences (LOW)${NC}"
            echo -e "   ${YELLOW}Recommendation: Upgrade to 50+${NC}"
        else
            echo -e "   ${GREEN}Current: $current_value context sentences (GOOD)${NC}"
        fi
    else
        echo -e "   ${YELLOW}Could not detect value${NC}"
    fi
else
    echo -e "   ${RED}File not found${NC}"
fi
echo ""

# Check train_qa_decoder.py
echo -e "${BLUE}2. QA Decoder Training (scripts/train_qa_decoder.py:209)${NC}"
if [ -f "scripts/train_qa_decoder.py" ]; then
    training_line=$(grep -n "context_asts = item.get('context_asts" scripts/train_qa_decoder.py | head -1)
    current_value=$(echo "$training_line" | sed -n "s/.*\[:\([0-9]*\)\].*/\1/p")

    if [ -n "$current_value" ]; then
        if [ "$current_value" -lt 20 ]; then
            echo -e "   ${RED}Current: $current_value context ASTs (LOW)${NC}"
            echo -e "   ${YELLOW}Recommendation: Upgrade to 50+${NC}"
        else
            echo -e "   ${GREEN}Current: $current_value context ASTs (GOOD)${NC}"
        fi
    else
        echo -e "   ${YELLOW}Could not detect value${NC}"
    fi
else
    echo -e "   ${RED}File not found${NC}"
fi
echo ""

# Check RAG Expert
echo -e "${BLUE}3. RAG Expert Retrieval (klareco/experts/rag_expert.py:46)${NC}"
if [ -f "klareco/experts/rag_expert.py" ]; then
    rag_line=$(grep -n "k: int = " klareco/experts/rag_expert.py | head -1)
    current_value=$(echo "$rag_line" | sed -n "s/.*k: int = \([0-9]*\).*/\1/p")

    if [ -n "$current_value" ]; then
        if [ "$current_value" -lt 20 ]; then
            echo -e "   ${RED}Current: k=$current_value documents (LOW)${NC}"
            echo -e "   ${YELLOW}Recommendation: Upgrade to 50+${NC}"
        else
            echo -e "   ${GREEN}Current: k=$current_value documents (GOOD)${NC}"
        fi
    else
        echo -e "   ${YELLOW}Could not detect value${NC}"
    fi
else
    echo -e "   ${RED}File not found${NC}"
fi
echo ""

# Check for existing models
echo -e "${BLUE}4. Trained Models Status${NC}"
if [ -f "models/qa_decoder/best_model.pt" ]; then
    model_date=$(stat -c %y "models/qa_decoder/best_model.pt" 2>/dev/null | cut -d' ' -f1)
    echo -e "   ${GREEN}QA Decoder found (trained: $model_date)${NC}"
else
    echo -e "   ${YELLOW}QA Decoder not found${NC}"
fi

if [ -f "data/qa_dataset.jsonl" ]; then
    dataset_size=$(wc -l < "data/qa_dataset.jsonl")
    echo -e "   ${GREEN}QA Dataset found ($dataset_size pairs)${NC}"
else
    echo -e "   ${YELLOW}QA Dataset not found${NC}"
fi
echo ""

# Check for backups
echo -e "${BLUE}5. Available Backups${NC}"
backup_count=0
if compgen -G "data/qa_dataset.jsonl.backup_*" > /dev/null 2>&1; then
    latest_backup=$(ls -t data/qa_dataset.jsonl.backup_* 2>/dev/null | head -1)
    echo -e "   ${GREEN}Dataset backups found${NC}"
    echo -e "   Latest: $latest_backup"
    ((backup_count++))
fi

if compgen -G "models/qa_decoder_backup_*" > /dev/null 2>&1; then
    latest_backup=$(ls -td models/qa_decoder_backup_* 2>/dev/null | head -1)
    echo -e "   ${GREEN}Model backups found${NC}"
    echo -e "   Latest: $latest_backup"
    ((backup_count++))
fi

if [ $backup_count -eq 0 ]; then
    echo -e "   ${YELLOW}No backups found${NC}"
fi
echo ""

# Summary and recommendations
echo -e "${BOLD}========================================================================"
echo "SUMMARY & RECOMMENDATIONS"
echo -e "========================================================================${NC}"
echo ""

# Detect if upgrade is needed
needs_upgrade=false
if grep -q "'context': context\[:3\]" scripts/generate_qa_dataset.py 2>/dev/null; then
    needs_upgrade=true
fi
if grep -q "context_asts = item.get('context_asts', \[\])\[:5\]" scripts/train_qa_decoder.py 2>/dev/null; then
    needs_upgrade=true
fi

if [ "$needs_upgrade" = true ]; then
    echo -e "${YELLOW}${BOLD}⚠ UPGRADE RECOMMENDED${NC}"
    echo ""
    echo "Your system is configured for only 3-5 context documents."
    echo "For better results on information-rich queries (like 'Who is Frodo?'),"
    echo "upgrade to 50+ context documents."
    echo ""
    echo -e "${GREEN}Quick Upgrade:${NC}"
    echo "  ./retrain_with_more_context.sh              # Upgrade to 50 docs"
    echo "  ./retrain_with_more_context.sh --context 100  # Upgrade to 100 docs"
    echo ""
    echo -e "${BLUE}Documentation:${NC}"
    echo "  cat CONTEXT_SIZE_UPGRADE_GUIDE.md"
else
    echo -e "${GREEN}${BOLD}✓ CONFIGURATION LOOKS GOOD${NC}"
    echo ""
    echo "Your system appears to be configured for extended context."
    echo ""
    echo -e "${BLUE}To test:${NC}"
    echo "  python scripts/quick_query.py \"Kiu estas Frodo?\""
    echo "  python scripts/test_end_to_end_qa.py"
fi
echo ""
echo -e "${BOLD}========================================================================${NC}"
echo ""
