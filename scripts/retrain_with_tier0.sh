#!/bin/bash
# Retrain models with tier0 corpus and semantic relations
#
# This script:
# 1. Regenerates M1 training data from tier0-enhanced corpus
# 2. Retrains Stage 1 root embeddings with tier0 + ReVo semantic relations
# 3. Retrains M1 selectional model with new embeddings + tier0 data
#
# Usage:
#   ./scripts/retrain_with_tier0.sh           # Full retraining
#   ./scripts/retrain_with_tier0.sh --skip-data    # Skip data generation
#   ./scripts/retrain_with_tier0.sh --stage1-only # Only retrain Stage 1
#   ./scripts/retrain_with_tier0.sh --m1-only     # Only retrain M1

set -e
set -o pipefail  # Exit if any command in a pipeline fails

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Activate venv
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Error: No virtual environment found (.venv or venv)"
    exit 1
fi

# Create logs directory
mkdir -p logs/retraining

# Master log
MASTER_LOG="logs/retraining/retrain_$(date +%Y%m%d_%H%M%S).log"

echo "========================================================================"
echo "RETRAIN MODELS WITH TIER0 CORPUS + SEMANTIC RELATIONS"
echo "========================================================================"
echo ""
echo "This will:"
echo "  1. Regenerate M1 training data from tier0-enhanced corpus"
echo "  2. Retrain Stage 1 root embeddings with:"
echo "     - Tier0 corpus co-occurrence (weight=15.0)"
echo "     - ReVo semantic relations (weight=2.0-8.0)"
echo "     - Ekzercaro co-occurrence (weight=10.0)"
echo "  3. Retrain M1 selectional model with new embeddings"
echo ""
echo "Master log: $MASTER_LOG"
echo ""

# Parse flags
SKIP_DATA=false
STAGE1_ONLY=false
M1_ONLY=false

for arg in "$@"; do
    case $arg in
        --skip-data)
            SKIP_DATA=true
            shift
            ;;
        --stage1-only)
            STAGE1_ONLY=true
            shift
            ;;
        --m1-only)
            M1_ONLY=true
            shift
            ;;
    esac
done

# Step 1: Generate M1 training data from tier0 corpus
if [ "$SKIP_DATA" = false ] && [ "$STAGE1_ONLY" = false ]; then
    echo "========================================================================"
    echo "STEP 1: Regenerate M1 training data (tier0 + full corpus)"
    echo "========================================================================"
    echo ""

    LOG_FILE="logs/retraining/m1_data_$(date +%Y%m%d_%H%M%S).log"
    if python scripts/prepare_m1_training_data.py \
        --corpus data/enhanced_corpus/corpus_full_with_tier0.jsonl \
        --output data/training/m1_with_tier0 \
        --max-triples 200000 \
        --negatives-per-positive 1 \
        2>&1 | tee "$LOG_FILE"; then
        echo ""
        echo "✓ M1 data generation complete"
    else
        echo ""
        echo "✗ M1 data generation failed"
        exit 1
    fi
fi

# Step 2: Retrain Stage 1 root embeddings
if [ "$M1_ONLY" = false ]; then
    echo ""
    echo "========================================================================"
    echo "STEP 2: Retrain Stage 1 root embeddings"
    echo "========================================================================"
    echo ""
    echo "Training with:"
    echo "  - Tier0 corpus: data/enhanced_corpus/corpus_with_tier0.jsonl"
    echo "  - ReVo relations: data/raw/eo/dictionaries/revo/revo_semantic_relations.json"
    echo "  - Ekzercaro: data/training/ekzercaro_sentences.jsonl"
    echo ""

    LOG_FILE="logs/retraining/stage1_$(date +%Y%m%d_%H%M%S).log"
    if python scripts/train_root_embeddings.py \
        --tier0-corpus data/enhanced_corpus/corpus_with_tier0.jsonl \
        --revo-relations data/raw/eo/dictionaries/revo/revo_semantic_relations.json \
        --output-dir models/root_embeddings_tier0 \
        --log-dir logs/training \
        --epochs 100 \
        --patience 15 \
        --fresh \
        2>&1 | tee "$LOG_FILE"; then
        echo ""
        echo "✓ Stage 1 training complete"
    else
        echo ""
        echo "✗ Stage 1 training failed"
        exit 1
    fi
fi

# Step 3: Retrain M1 selectional model
if [ "$STAGE1_ONLY" = false ]; then
    echo ""
    echo "========================================================================"
    echo "STEP 3: Retrain M1 selectional model"
    echo "========================================================================"
    echo ""
    echo "Using:"
    echo "  - Stage 1 embeddings: models/root_embeddings_tier0/best_model.pt"
    echo "  - Training data: data/training/m1_with_tier0/"
    echo ""

    # Use new embeddings if we just trained them
    if [ "$M1_ONLY" = true ]; then
        STAGE1_MODEL="models/root_embeddings/best_model.pt"
    else
        STAGE1_MODEL="models/root_embeddings_tier0/best_model.pt"
    fi

    LOG_FILE="logs/retraining/m1_$(date +%Y%m%d_%H%M%S).log"
    if python scripts/train_m1_selectional.py \
        --stage1-model "$STAGE1_MODEL" \
        --data-dir data/training/m1_with_tier0 \
        --output-dir models/m1_selectional_tier0 \
        --log-dir logs/training \
        --epochs 50 \
        --patience 10 \
        --fresh \
        2>&1 | tee "$LOG_FILE"; then
        echo ""
        echo "✓ M1 training complete"
    else
        echo ""
        echo "✗ M1 training failed"
        exit 1
    fi
fi

# Summary
echo ""
echo "========================================================================"
echo "RETRAINING COMPLETE"
echo "========================================================================"
echo ""

if [ "$M1_ONLY" = false ]; then
    echo "Stage 1 root embeddings saved to:"
    echo "  models/root_embeddings_tier0/best_model.pt"
    echo ""
fi

if [ "$STAGE1_ONLY" = false ]; then
    echo "M1 selectional model saved to:"
    echo "  models/m1_selectional_tier0/best_model.pt"
    echo ""
fi

echo "Logs saved to:"
echo "  $MASTER_LOG"
echo "  logs/retraining/"
echo ""
echo "Next steps:"
echo "  1. Evaluate model quality: pytest tests/test_stage1_model_quality.py"
echo "  2. Test semantic queries with new embeddings"
echo "  3. Update production models if quality improves"
echo ""
