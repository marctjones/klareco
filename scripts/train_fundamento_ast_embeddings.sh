#!/usr/bin/bash
#
# Train Fundamento AST-Aware Root Embeddings
#
# APPROACH: AST-aware extraction from GOLD sources only
# - Fundamento + ReVo Tier 1-2 vocabulary (~9,800 roots)
# - AST-aware semantic pairing (not bag-of-words!)
# - GOLD sources: Zamenhof, PMEG, classic translations
# - Includes ReVo semantic relations + mal- antonyms
#
# ARCHITECTURE:
# - Vocabulary: ~2,400 roots (appear in GOLD sources)
# - Embedding dim: 64D (smaller vocab needs less dims)
# - Training pairs: ~35K high-quality AST pairs
# - Dataset: 100% (only 35K pairs, all valuable)
# - Epochs: 30 (smaller dataset needs more epochs)
#
# QUALITY:
# - All pairs from authoritative sources
# - AST-aware (modifier-head, subject-object)
# - Includes antonyms (mal- prefix)
# - Includes ReVo synonyms/antonyms/hypernyms
#
# Usage:
#   ./scripts/train_fundamento_ast_embeddings.sh
#   ./scripts/train_fundamento_ast_embeddings.sh --fresh   # Ignore checkpoint

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
TRAINING_DIR="data/training/fundamento_ast_pairs"
OUTPUT_DIR="models/root_embeddings_fundamento_ast"
LOG_DIR="logs/fundamento_ast_training"

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
echo "Fundamento AST-Aware Root Embeddings Training"
echo "============================================================================"
echo "Training data: $TRAINING_DIR"
echo "Output: $OUTPUT_DIR"
echo "Log: $LOG_FILE"
echo ""
echo "APPROACH: AST-First, Fundamento-Focused"
echo "- Vocabulary: Tier 1-2 (Fundamento + ReVo: ~9,800 roots)"
echo "- Sources: GOLD only (Zamenhof, PMEG, classics)"
echo "- Pairs: AST-aware (modifier-head, subject-object)"
echo "- Semantic: ReVo relations + mal- antonyms"
echo ""
echo "TRAINING:"
echo "- Embedding dim: 128D (UPGRADED to match Production model)"
echo "- Training pairs: ~35K"
echo "- Epochs: 30 (smaller dataset)"
echo "- Negative samples: 5"
echo "- Batch size: 256"
echo "============================================================================"

# Check if training data exists
if [ ! -f "$TRAINING_DIR/pairs.jsonl" ]; then
    echo "ERROR: Training pairs not found: $TRAINING_DIR/pairs.jsonl"
    echo ""
    echo "Run extraction first:"
    echo "  python scripts/extract_fundamento_ast_pairs.py \\"
    echo "    --db-path data/indexes/v2.1_kuzu_index_full \\"
    echo "    --vocabulary data/vocabularies/fundamento_revo_tier12.json \\"
    echo "    --output data/training/fundamento_ast_pairs/pairs.jsonl"
    exit 1
fi

echo ""
echo "Training with skip-gram model..."
echo ""

python scripts/train_root_embeddings_skipgram_v2_1.py \
    --training-pairs "$TRAINING_DIR/pairs.jsonl" \
    --vocabulary "$TRAINING_DIR/vocab.json" \
    --output "$OUTPUT_DIR" \
    --embedding-dim 128 \
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
echo "Training Complete - Fundamento AST-Aware Model"
echo "============================================================================"
echo "Best model: $OUTPUT_DIR/root_embeddings_best.pt"
echo "Final model: $OUTPUT_DIR/root_embeddings_final.pt"
echo "Log file: $LOG_FILE"
echo ""
echo "QUALITY: AST-aware, Fundamento-focused embeddings"
echo "- 128D embeddings (UPGRADED - matches Production model)"
echo "- ~2,400 Fundamento/ReVo validated roots"
echo "- AST-aware semantic structure (not bag-of-words)"
echo "- Includes antonyms and ReVo relations"
echo ""
echo "Next steps:"
echo "1. Evaluate quality:"
echo "   python scripts/improvements/evaluate_embeddings.py \\"
echo "     --model $OUTPUT_DIR/root_embeddings_best.pt"
echo ""
echo "2. Compare to current model:"
echo "   models/root_embeddings_phase1_fast/ (positional window, 6.7K roots)"
echo "   vs"
echo "   $OUTPUT_DIR (AST-aware, 2.4K roots)"
echo "============================================================================"
