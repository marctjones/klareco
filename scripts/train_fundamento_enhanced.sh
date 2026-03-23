#!/usr/bin/bash
#
# Train Enhanced Fundamento Root Embeddings
#
# APPROACH: AST-aware + Co-occurrence (best of both worlds!)
# - AST pairs: 35K (modifier-head, subject-object, antonyms)
# - Co-occurrence pairs: 91K (distributional semantics)
# - Total: 126K high-quality pairs from GOLD sources
#
# ARCHITECTURE:
# - Vocabulary: ~2,750 roots (Fundamento + ReVo in GOLD sources)
# - Embedding dim: 64D (same as AST-only model)
# - Training pairs: 126K (3.6x more than AST-only)
# - Epochs: 30 (same as before)
#
# GOAL:
# - Maintain AST structure awareness (antonyms!)
# - Add distributional semantics (better clustering!)
#
# Usage:
#   ./scripts/train_fundamento_enhanced.sh
#   ./scripts/train_fundamento_enhanced.sh --fresh   # Ignore checkpoint

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Activate venv
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "ERROR: No virtual environment found (.venv or venv)"
    exit 1
fi

# Configuration
TRAINING_DIR="data/training/fundamento_enhanced"
OUTPUT_DIR="models/root_embeddings_fundamento_enhanced"
LOG_DIR="logs/fundamento_enhanced_training"

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# Parse arguments
RESUME_FLAG="--resume"
if [ "$1" == "--fresh" ]; then
    RESUME_FLAG=""
fi

# Timestamp for logging
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/training_$TIMESTAMP.log"

echo "============================================================================"
echo "Enhanced Fundamento Root Embeddings Training"
echo "============================================================================"
echo "Training data: $TRAINING_DIR"
echo "Output: $OUTPUT_DIR"
echo "Log: $LOG_FILE"
echo ""
echo "APPROACH: AST-First + Distributional Semantics"
echo "- AST pairs: 35K (modifier-head, subject-object, antonyms)"
echo "- Co-occurrence: 91K (window-based semantic similarity)"
echo "- Total: 126K pairs from GOLD sources only"
echo ""
echo "TRAINING:"
echo "- Embedding dim: 64D"
echo "- Training pairs: 126K (3.6x more than AST-only)"
echo "- Epochs: 30"
echo "- Negative samples: 5"
echo "- Batch size: 256"
echo "============================================================================"

# Check if training data exists
if [ ! -f "$TRAINING_DIR/pairs.jsonl" ]; then
    echo "ERROR: Training pairs not found: $TRAINING_DIR/pairs.jsonl"
    echo ""
    echo "Run merge first:"
    echo "  python scripts/merge_training_pairs.py \\\\"
    echo "    --ast-pairs data/training/fundamento_ast_pairs/pairs.jsonl \\\\"
    echo "    --cooccurrence-pairs data/training/fundamento_cooccurrence/pairs.jsonl \\\\"
    echo "    --output data/training/fundamento_enhanced/pairs.jsonl"
    exit 1
fi

echo ""
echo "Training with skip-gram model..."
echo ""

python scripts/train_root_embeddings_skipgram_v2_1.py \
    --training-pairs "$TRAINING_DIR/pairs.jsonl" \
    --vocabulary "$TRAINING_DIR/vocab.json" \
    --output "$OUTPUT_DIR" \
    --embedding-dim 64 \
    --epochs 30 \
    --batch-size 256 \
    --dataset-fraction 1.0 \
    --learning-rate 0.025 \
    --negative-samples 5 \
    --patience 5 \
    --min-delta 0.001 \
    --collapse-threshold 0.7 \
    $RESUME_FLAG \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "============================================================================"
echo "Training Complete - Enhanced Fundamento Model"
echo "============================================================================"
echo "Best model: $OUTPUT_DIR/root_embeddings_best.pt"
echo "Final model: $OUTPUT_DIR/root_embeddings_final.pt"
echo "Log file: $LOG_FILE"
echo ""
echo "QUALITY: AST-aware + Distributional Semantics"
echo "- 64D embeddings (~350K parameters)"
echo "- ~2,750 Fundamento/ReVo validated roots"
echo "- AST structure (antonyms, grammar) + Co-occurrence (semantic clustering)"
echo "- 126K training pairs (35K AST + 91K co-occurrence)"
echo ""
echo "Next steps:"
echo "1. Evaluate quality:"
echo "   python scripts/evaluate_fundamento_ast_model.py \\\\"
echo "     --model $OUTPUT_DIR/root_embeddings_best.pt \\\\"
echo "     --output results/fundamento_enhanced_eval.json"
echo ""
echo "2. Compare to previous models:"
echo "   - Production (positional window): 73.3/100"
echo "   - AST-only: 73.3/100"
echo "   - Enhanced (this model): ???"
echo "============================================================================"
