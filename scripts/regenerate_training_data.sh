#!/bin/bash
#
# Regenerate Training Data (Without Training)
#
# Regenerates all training data from the current corpus:
# 1. Ekzercaro data for root embeddings
# 2. M1 semantic data for selectional preference
#
# Usage:
#   ./scripts/regenerate_training_data.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}Regenerate Training Data${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""

# Activate venv
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
fi

CORPUS_FILE="data/enhanced_corpus/corpus_with_metadata.jsonl"

# =============================================================================
# Step 1: Extract Ekzercaro Training Data (for root embeddings)
# =============================================================================
echo -e "${GREEN}=== Step 1: Extract Ekzercaro Training Data ===${NC}"
echo "Extracting GOLD-quality sentences for root embeddings..."
echo ""

EKZERCARO_FILE="data/training/ekzercaro_sentences.jsonl"

python -c "
import json
import sys
from pathlib import Path

corpus_path = Path('$CORPUS_FILE')
output_path = Path('$EKZERCARO_FILE')
output_path.parent.mkdir(parents=True, exist_ok=True)

print(f'Extracting GOLD sentences from {corpus_path}...')

def extract_roots(node):
    '''Recursively extract all roots from AST.'''
    roots = []
    if isinstance(node, dict):
        if node.get('tipo') == 'vorto' and node.get('radiko'):
            roots.append(node['radiko'])
        for v in node.values():
            roots.extend(extract_roots(v))
    elif isinstance(node, list):
        for item in node:
            roots.extend(extract_roots(item))
    return roots

count = 0
written = 0

with open(corpus_path) as f_in, open(output_path, 'w') as f_out:
    for line in f_in:
        entry = json.loads(line)
        count += 1

        # Only use GOLD quality
        if entry.get('source', {}).get('quality') != 'GOLD':
            continue

        # Must have good parse
        if entry.get('parse_rate', 0) < 0.90:
            continue

        # Extract roots
        roots = extract_roots(entry.get('ast', {}))
        if len(roots) < 2:
            continue

        # Write to output
        f_out.write(json.dumps(entry, ensure_ascii=False) + '\n')
        written += 1

        if written % 10000 == 0:
            print(f'  Extracted {written:,} sentences...')

print(f'\nProcessed: {count:,} sentences')
print(f'Extracted: {written:,} GOLD sentences')
print(f'Output:    {output_path}')
"

echo ""
echo -e "${GREEN}✓ Ekzercaro data extracted${NC}"
echo ""

# =============================================================================
# Step 2: Generate M1 Training Data (tier priority)
# =============================================================================
echo -e "${GREEN}=== Step 2: Generate M1 Training Data ===${NC}"
echo "Generating tier-prioritized selectional preference data..."
echo ""

DATA_DIR="data/training/m1_semantic_tier_priority"
LOG_DIR="logs/training"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DATA_LOG="$LOG_DIR/prepare_m1_tier_priority_${TIMESTAMP}.log"

# Check if we have a trained root embeddings model
if [ -f "models/root_embeddings/best_model.pt" ]; then
    STAGE1_MODEL="models/root_embeddings/best_model.pt"
elif [ -f "models/root_embeddings_tier0/best_model.pt" ]; then
    STAGE1_MODEL="models/root_embeddings_tier0/best_model.pt"
else
    echo -e "${YELLOW}⚠️  No root embeddings model found${NC}"
    echo "   M1 training will use random root embeddings"
    echo "   Recommendation: Train root embeddings first for better M1 quality"
    echo ""
    STAGE1_MODEL="none"
fi

echo "Corpus:       $CORPUS_FILE"
echo "Output:       $DATA_DIR"
echo "Stage1 model: $STAGE1_MODEL"
echo "Log:          $DATA_LOG"
echo ""

if [ "$STAGE1_MODEL" = "none" ]; then
    # Generate without stage1 model (will use random embeddings)
    python scripts/prepare_m1_training_data_tier_priority.py \
        --corpus "$CORPUS_FILE" \
        --output-dir "$DATA_DIR" \
        --max-triples 200000 \
        --priority-tiers 0 \
        --fill-tiers 5 6 \
        --similarity-threshold 0.15 \
        --min-parse-rate 0.0 \
        2>&1 | tee "$DATA_LOG"
else
    # Generate with stage1 model
    python scripts/prepare_m1_training_data_tier_priority.py \
        --corpus "$CORPUS_FILE" \
        --stage1-model "$STAGE1_MODEL" \
        --output-dir "$DATA_DIR" \
        --max-triples 200000 \
        --priority-tiers 0 \
        --fill-tiers 5 6 \
        --similarity-threshold 0.15 \
        --min-parse-rate 0.0 \
        2>&1 | tee "$DATA_LOG"
fi

echo ""
echo -e "${GREEN}✓ M1 training data generated${NC}"
echo ""

# =============================================================================
# Summary
# =============================================================================
echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}Training Data Generation Complete${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""
echo "Generated files:"
echo "  1. Ekzercaro (root embeddings): $EKZERCARO_FILE"
echo "  2. M1 semantic data:            $DATA_DIR"
echo ""
echo "Next steps:"
echo "  1. Validate training data:"
echo "     python scripts/validate_training_data.py $EKZERCARO_FILE"
echo "     python scripts/validate_training_data.py $DATA_DIR"
echo ""
echo "  2. Train models:"
echo "     ./scripts/train_roots.sh --fresh"
echo "     ./scripts/train_m1_semantic_tier_priority.sh --skip-data"
echo ""
