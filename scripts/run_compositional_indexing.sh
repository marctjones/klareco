#!/bin/bash
# =============================================================================
# Build FAISS Index with Compositional Embeddings
# =============================================================================
# Indexes the corpus using the trained Stage 1 models:
# - Root embeddings: models/root_embeddings/best_model.pt
# - Affix transforms: models/affix_transforms/best_model.pt (transformation matrices)
#
# The key difference from the old approach:
# - Affixes are TRANSFORMATIONS, not additive embeddings
# - mal- flips polarity by transforming the embedding
# - -ej adds "place" semantics through transformation
# - Composition: prefixes → root → suffixes
#
# Usage:
#   ./scripts/run_compositional_indexing.sh           # Resume from checkpoint
#   ./scripts/run_compositional_indexing.sh --fresh   # Start fresh
#
# Monitor progress in another terminal:
#   tail -f data/corpus_index_compositional/indexing.log
#
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Compositional Corpus Indexing${NC}"
echo -e "${BLUE}  (Transform-based Affixes)${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Activate venv
if [[ -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
    echo "Using .venv"
elif [[ -f "venv/bin/activate" ]]; then
    source venv/bin/activate
    echo "Using venv"
else
    echo -e "${RED}No virtual environment found${NC}"
    exit 1
fi

# Configuration
CORPUS="data/corpus/unified_corpus.jsonl"
ROOT_MODEL="models/root_embeddings/best_model.pt"
AFFIX_MODEL="models/affix_transforms_v2/best_model.pt"
OUTPUT_DIR="data/corpus_index_compositional"

# Parse arguments
FRESH_FLAG=""
if [[ "$1" == "--fresh" ]]; then
    FRESH_FLAG="--fresh"
    echo -e "${YELLOW}Starting fresh (ignoring checkpoint)${NC}"
fi

# Check files exist
if [[ ! -f "$CORPUS" ]]; then
    echo -e "${RED}Error: Corpus not found: $CORPUS${NC}"
    echo "Run ./scripts/run_corpus_rebuild.sh first"
    exit 1
fi

if [[ ! -f "$ROOT_MODEL" ]]; then
    echo -e "${RED}Error: Root model not found: $ROOT_MODEL${NC}"
    echo "Run: ./scripts/run_root_training.sh"
    exit 1
fi

if [[ ! -f "$AFFIX_MODEL" ]]; then
    echo -e "${RED}Error: Affix transforms not found: $AFFIX_MODEL${NC}"
    echo "Run: ./scripts/run_affix_training.sh"
    exit 1
fi

echo ""
echo -e "${GREEN}Corpus:${NC}        $CORPUS"
echo -e "${GREEN}Root model:${NC}    $ROOT_MODEL"
echo -e "${GREEN}Affix model:${NC}   $AFFIX_MODEL"
echo -e "${GREEN}Output:${NC}        $OUTPUT_DIR"
echo ""

# Show model info
echo -e "${BLUE}=== Model Info ===${NC}"
python3 -c "
import torch
root = torch.load('$ROOT_MODEL', map_location='cpu', weights_only=False)
affix = torch.load('$AFFIX_MODEL', map_location='cpu', weights_only=False)
print(f'Root embeddings: {root[\"vocab_size\"]:,} roots x {root[\"embedding_dim\"]}d')
print(f'Affix transforms: {len(affix[\"prefixes\"])} prefixes, {len(affix[\"suffixes\"])} suffixes (rank={affix[\"rank\"]})')
"
echo ""

# Count sentences
SENTENCE_COUNT=$(wc -l < "$CORPUS")
echo -e "${GREEN}Sentences:${NC}     $SENTENCE_COUNT"

# Check for checkpoint
if [[ -f "$OUTPUT_DIR/indexing_checkpoint.json" ]] && [[ -z "$FRESH_FLAG" ]]; then
    PROCESSED=$(python3 -c "import json; print(json.load(open('$OUTPUT_DIR/indexing_checkpoint.json'))['processed'])")
    echo -e "${YELLOW}Found checkpoint: $PROCESSED/$SENTENCE_COUNT processed${NC}"
    echo -e "${YELLOW}Will resume from checkpoint${NC}"
    echo ""
fi

# Estimate time (~300 sentences/sec with transforms)
ESTIMATED_SECS=$((SENTENCE_COUNT / 300))
echo -e "${GREEN}Estimated time:${NC} ~$((ESTIMATED_SECS / 60)) minutes"
echo ""

echo -e "${BLUE}To monitor progress:${NC}"
echo -e "  ${GREEN}tail -f $OUTPUT_DIR/indexing.log${NC}"
echo ""

# Run
echo -e "${GREEN}Starting indexing...${NC}"
echo ""

python scripts/index_corpus_compositional.py \
    --corpus "$CORPUS" \
    --root-model "$ROOT_MODEL" \
    --affix-model "$AFFIX_MODEL" \
    --output-dir "$OUTPUT_DIR" \
    $FRESH_FLAG

if [[ $? -eq 0 ]]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  Indexing Complete!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo -e "${GREEN}Output files:${NC}"
    ls -lh "$OUTPUT_DIR/" 2>/dev/null | grep -E "\.(npy|bin|jsonl)$" | awk '{print "  " $9 " (" $5 ")"}'
    echo ""
    echo -e "${GREEN}Embeddings shape:${NC}"
    python3 -c "import numpy as np; emb = np.load('$OUTPUT_DIR/embeddings.npy'); print(f'  {emb.shape}')" 2>/dev/null || echo "  (not found)"
    echo ""
    echo -e "${BLUE}Next steps:${NC}"
    echo "  1. Test retrieval: python scripts/demo_rag.py --index $OUTPUT_DIR"
    echo "  2. Run demo: python scripts/demo_pipeline.py"
    echo ""
else
    echo ""
    echo -e "${YELLOW}Indexing interrupted or failed${NC}"
    echo -e "${YELLOW}To resume: ./scripts/run_compositional_indexing.sh${NC}"
    exit 1
fi
