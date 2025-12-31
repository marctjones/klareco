#!/bin/bash
# =============================================================================
# Train Root Embeddings
# =============================================================================
# Trains root embeddings using the Fundamento-Centered approach.
#
# Prerequisites:
#   - Unified corpus built: data/corpus/unified_corpus.jsonl
#   - Clean vocabulary: data/vocabularies/clean_roots.json
#   - ReVo definitions: data/revo/revo_definitions_with_roots.json
#
# This script:
#   1. Extracts Ekzercaro-style training data from Tier 1-3 sentences
#   2. Trains root embeddings with contrastive learning
#
# Usage:
#   ./scripts/run_root_training.sh           # Resume from checkpoint
#   ./scripts/run_root_training.sh --fresh   # Start fresh
#
# Output:
#   models/root_embeddings/best_model.pt
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
CORPUS_FILE="$PROJECT_ROOT/data/corpus/unified_corpus.jsonl"
EKZERCARO_FILE="$PROJECT_ROOT/data/training/ekzercaro_sentences.jsonl"
CLEAN_VOCAB="$PROJECT_ROOT/data/vocabularies/clean_roots.json"
FUNDAMENTO_ROOTS="$PROJECT_ROOT/data/vocabularies/fundamento_roots.json"
REVO_DEFINITIONS="$PROJECT_ROOT/data/revo/revo_definitions_with_roots.json"
OUTPUT_DIR="$PROJECT_ROOT/models/root_embeddings"

# Log file
LOG_DIR="$PROJECT_ROOT/logs/training"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/root_training_$(date +%Y%m%d_%H%M%S).log"

echo -e "${BLUE}=== Root Embedding Training ===${NC}"
echo "Corpus:      $CORPUS_FILE"
echo "Ekzercaro:   $EKZERCARO_FILE"
echo "Clean Vocab: $CLEAN_VOCAB"
echo "Output:      $OUTPUT_DIR"
echo "Log:         $LOG_FILE"
echo "Fresh:       $FRESH_START"
echo ""

# Check prerequisites
if [ ! -f "$CORPUS_FILE" ]; then
    echo -e "${RED}Error: Corpus file not found: $CORPUS_FILE${NC}"
    echo "Run ./scripts/run_corpus_rebuild.sh first"
    exit 1
fi

if [ ! -f "$CLEAN_VOCAB" ]; then
    echo -e "${RED}Error: Clean vocabulary not found: $CLEAN_VOCAB${NC}"
    exit 1
fi

if [ ! -f "$REVO_DEFINITIONS" ]; then
    echo -e "${RED}Error: ReVo definitions not found: $REVO_DEFINITIONS${NC}"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "$(dirname "$EKZERCARO_FILE")"

# =============================================================================
# Step 1: Extract Ekzercaro-style training data (if needed or fresh)
# =============================================================================
if [ "$FRESH_START" = true ] || [ ! -f "$EKZERCARO_FILE" ]; then
    echo -e "${YELLOW}=== Step 1: Extract Ekzercaro Training Data ===${NC}"

    # Extract Tier 1-3 sentences with roots for co-occurrence training
    python -c "
import json
import sys
from pathlib import Path

corpus_path = Path('$CORPUS_FILE')
output_path = Path('$EKZERCARO_FILE')

print(f'Extracting Tier 1-3 sentences from {corpus_path}...')

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
        count += 1
        if count % 100000 == 0:
            print(f'  Processed {count:,} sentences, extracted {written:,}...')

        try:
            entry = json.loads(line)
        except:
            continue

        # Only use Tier 1-3 (authoritative sources)
        tier = entry.get('tier', 6)
        if tier > 3:
            continue

        ast = entry.get('ast')
        if not ast:
            continue

        roots = extract_roots(ast)
        if len(roots) >= 2:  # Need at least 2 roots for co-occurrence
            out_entry = {
                'text': entry.get('text', ''),
                'roots': roots,
                'source': entry.get('source', 'unknown'),
                'tier': tier
            }
            f_out.write(json.dumps(out_entry, ensure_ascii=False) + '\n')
            written += 1

print(f'Done! Extracted {written:,} sentences from {count:,} total')
"

    EKZERCARO_COUNT=$(wc -l < "$EKZERCARO_FILE")
    echo -e "${GREEN}Step 1 complete: $EKZERCARO_COUNT training sentences${NC}"
    echo ""
else
    EKZERCARO_COUNT=$(wc -l < "$EKZERCARO_FILE")
    echo -e "${GREEN}Using existing Ekzercaro data: $EKZERCARO_COUNT sentences${NC}"
    echo ""
fi

# =============================================================================
# Step 2: Train Root Embeddings
# =============================================================================
echo -e "${YELLOW}=== Step 2: Train Root Embeddings ===${NC}"

if [ "$FRESH_START" = true ]; then
    FRESH_FLAG="--fresh"
    echo -e "${YELLOW}Fresh start requested - ignoring checkpoints${NC}"
else
    FRESH_FLAG=""
fi

echo "Starting training... (logging to $LOG_FILE)"
echo ""

python scripts/training/train_root_embeddings.py \
    --fundamento-roots "$FUNDAMENTO_ROOTS" \
    --revo-definitions "$REVO_DEFINITIONS" \
    --ekzercaro "$EKZERCARO_FILE" \
    --clean-vocab "$CLEAN_VOCAB" \
    --output-dir "$OUTPUT_DIR" \
    --log-dir "$LOG_DIR" \
    --embedding-dim 64 \
    --epochs 100 \
    --batch-size 128 \
    --learning-rate 0.001 \
    --patience 15 \
    --margin 0.3 \
    $FRESH_FLAG \
    2>&1 | tee "$LOG_FILE"

# =============================================================================
# Summary
# =============================================================================
echo ""
echo -e "${BLUE}=== Training Complete ===${NC}"
echo ""

if [ -f "$OUTPUT_DIR/best_model.pt" ]; then
    echo -e "${GREEN}Model saved to: $OUTPUT_DIR/best_model.pt${NC}"

    # Show model info
    python -c "
import torch
checkpoint = torch.load('$OUTPUT_DIR/best_model.pt', map_location='cpu')
print(f'Vocabulary size: {checkpoint[\"vocab_size\"]:,} roots')
print(f'Embedding dim: {checkpoint[\"embedding_dim\"]}')
print(f'Best correlation: {checkpoint.get(\"correlation\", 0):.4f}')
print(f'Training epochs: {checkpoint[\"epoch\"]}')
"
else
    echo -e "${RED}Warning: best_model.pt not found${NC}"
fi

echo ""
echo "Log file: $LOG_FILE"
echo -e "${GREEN}Done!${NC}"
