#!/bin/bash
# =============================================================================
# Train Affix Transformations (V2 - Contrastive Learning)
# =============================================================================
# Uses contrastive learning to prevent embedding collapse.
#
# Key differences from v1:
#   - REMOVED: Problematic context loss that caused collapse
#   - ADDED: Separation loss (different roots with same affix stay apart)
#   - ADDED: Anti-collapse metrics during training
#   - ADDED: Larger initialization (gain=0.5)
#
# Prerequisites:
#   - Trained root embeddings: models/root_embeddings/best_model.pt
#   - Unified corpus: data/corpus/unified_corpus.jsonl
#
# Usage:
#   ./scripts/run_affix_training_v2.sh           # Resume from checkpoint
#   ./scripts/run_affix_training_v2.sh --fresh   # Start fresh
#
# Output:
#   models/affix_transforms_v2/best_model.pt
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
ROOT_MODEL="$PROJECT_ROOT/models/root_embeddings/best_model.pt"
CORPUS_FILE="$PROJECT_ROOT/data/corpus/unified_corpus.jsonl"
OUTPUT_DIR="$PROJECT_ROOT/models/affix_transforms_v2"

# Log file
LOG_DIR="$PROJECT_ROOT/logs/training"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/affix_training_v2_$(date +%Y%m%d_%H%M%S).log"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Affix Transformation Training (V2)${NC}"
echo -e "${BLUE}  Contrastive Learning - No Collapse${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${GREEN}Key improvements over V1:${NC}"
echo "  - Separation loss keeps different roots apart"
echo "  - No context similarity loss (was causing collapse)"
echo "  - Larger weight initialization"
echo "  - Anti-collapse metrics tracked during training"
echo ""
echo "Root Model:  $ROOT_MODEL"
echo "Corpus:      $CORPUS_FILE"
echo "Output:      $OUTPUT_DIR"
echo "Log:         $LOG_FILE"
echo "Fresh:       $FRESH_START"
echo ""

# Check prerequisites
if [ ! -f "$ROOT_MODEL" ]; then
    echo -e "${RED}Error: Root embeddings not found: $ROOT_MODEL${NC}"
    echo "Run ./scripts/run_root_training.sh first"
    exit 1
fi

if [ ! -f "$CORPUS_FILE" ]; then
    echo -e "${RED}Error: Corpus file not found: $CORPUS_FILE${NC}"
    echo "Run ./scripts/run_corpus_rebuild.sh first"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Show root model info
echo -e "${BLUE}=== Root Model Info ===${NC}"
python -c "
import torch
checkpoint = torch.load('$ROOT_MODEL', map_location='cpu', weights_only=False)
print(f'Vocabulary size: {checkpoint[\"vocab_size\"]:,} roots')
print(f'Embedding dim: {checkpoint[\"embedding_dim\"]}')
print(f'Best correlation: {checkpoint.get(\"correlation\", 0):.4f}')
"
echo ""

# Build training command
if [ "$FRESH_START" = true ]; then
    FRESH_FLAG="--fresh"
    echo -e "${YELLOW}Fresh start requested - ignoring checkpoints${NC}"
else
    FRESH_FLAG=""
fi

echo -e "${YELLOW}=== Starting Training ===${NC}"
echo "Logging to: $LOG_FILE"
echo ""
echo -e "${BLUE}Watch for these anti-collapse metrics:${NC}"
echo "  mal_mean_sim < 0.5   (mal-words shouldn't all cluster)"
echo "  embedding_diversity  (should stay high)"
echo ""

python scripts/training/train_affix_transforms_v2.py \
    --root-embeddings "$ROOT_MODEL" \
    --corpus "$CORPUS_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --log-dir "$LOG_DIR" \
    --rank 4 \
    --epochs 50 \
    --batch-size 64 \
    --learning-rate 0.001 \
    --patience 10 \
    --max-samples 500000 \
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
checkpoint = torch.load('$OUTPUT_DIR/best_model.pt', map_location='cpu', weights_only=False)
print(f'Embedding dim: {checkpoint[\"embedding_dim\"]}')
print(f'Rank: {checkpoint[\"rank\"]}')
print(f'Prefixes: {len(checkpoint[\"prefixes\"])}')
print(f'Suffixes: {len(checkpoint[\"suffixes\"])}')
print(f'Training epochs: {checkpoint[\"epoch\"]}')
print(f'Final loss: {checkpoint[\"loss\"]:.4f}')
if 'metrics' in checkpoint:
    m = checkpoint['metrics']
    print(f'')
    print(f'Anti-collapse metrics:')
    if 'mal_mean_sim' in m:
        status = '✓' if m['mal_mean_sim'] < 0.5 else '⚠️ HIGH'
        print(f'  mal_mean_sim: {m[\"mal_mean_sim\"]:.4f} {status}')
    if 'embedding_diversity' in m:
        print(f'  embedding_diversity: {m[\"embedding_diversity\"]:.4f}')
"

    echo ""
    echo -e "${BLUE}Next steps:${NC}"
    echo "  1. Rebuild index with new transforms:"
    echo "     python scripts/index_corpus_compositional.py \\"
    echo "         --affix-model $OUTPUT_DIR/best_model.pt"
    echo ""
    echo "  2. Test RAG:"
    echo "     python scripts/demo_rag_compositional.py -i"
else
    echo -e "${RED}Warning: best_model.pt not found${NC}"
fi

echo ""
echo "Log file: $LOG_FILE"
echo -e "${GREEN}Done!${NC}"
