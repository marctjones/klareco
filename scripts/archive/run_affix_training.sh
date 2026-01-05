#!/bin/bash
# =============================================================================
# Train Affix Transformations
# =============================================================================
# Trains low-rank transformation matrices for Esperanto affixes.
#
# Prerequisites:
#   - Trained root embeddings: models/root_embeddings/best_model.pt
#   - Unified corpus: data/corpus/unified_corpus.jsonl
#
# This approach treats affixes as TRANSFORMATIONS rather than embeddings:
#   - mal- learns to flip polarity (bon -> malbona)
#   - ej- learns to add "place" semantics (lerni -> lernejo)
#   - Each affix is a low-rank matrix (dim x rank x dim)
#
# Usage:
#   ./scripts/run_affix_training.sh           # Resume from checkpoint
#   ./scripts/run_affix_training.sh --fresh   # Start fresh
#
# Output:
#   models/affix_transforms/best_model.pt
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
OUTPUT_DIR="$PROJECT_ROOT/models/affix_transforms"

# Log file
LOG_DIR="$PROJECT_ROOT/logs/training"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/affix_training_$(date +%Y%m%d_%H%M%S).log"

echo -e "${BLUE}=== Affix Transformation Training ===${NC}"
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
checkpoint = torch.load('$ROOT_MODEL', map_location='cpu')
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

python scripts/training/train_affix_transforms.py \
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
    if 'mal_distance' in m:
        print(f'mal- distance: {m[\"mal_distance\"]:.4f}')
    if 'ej_cluster_sim' in m:
        print(f'ej cluster sim: {m[\"ej_cluster_sim\"]:.4f}')
"
else
    echo -e "${RED}Warning: best_model.pt not found${NC}"
fi

echo ""
echo "Log file: $LOG_FILE"
echo -e "${GREEN}Done!${NC}"
