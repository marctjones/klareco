#!/bin/bash
#
# Train Phase 1 Root Embeddings - OPTIMIZED OFFICIAL ROOTS (~2-3 hours on CPU)
#
# CURRENT APPROACH (v2.1): Positional skip-gram with cross-sentence context
# - Uses 8-word context windows (extended for more semantic relationships)
# - Cross-sentence pairs for paragraph coherence: weight 0.5
# - Function word filtering: excludes artikolo, prepozicio, konjunkcio, pronomo
# - Content roots only: substantivo, verbo, adjektivo, adverbo
# - Mikolov subsampling for high-frequency words (threshold 1e-3)
#
# OPTIMIZED SETTINGS (official Fundamento roots):
# - 128D embeddings (optimal for 6.7K vocabulary, avoids divergence)
# - Window size 8 (captures broader context relationships)
# - 10 negative samples (better discrimination between similar/dissimilar)
# - 15 epochs with early stopping (ensure full convergence)
# - 100% dataset (~34M training pairs from full corpus)
# - Vocabulary: ~6.7K official roots (production_semantic_roots_15k.json filtered)
#
# PERFORMANCE:
# - Extraction time: ~40 minutes (full corpus with window=8)
# - Training time: ~1.5-2 hours on CPU (128D, 34M pairs, 15 epochs)
# - Total: ~2-3 hours end-to-end
#
# WHY 128D instead of 256D:
# - 256D was too large for 6.7K vocabulary (caused loss divergence)
# - 128D is optimal: 10-20 dims per 1K words = 67-134D for 6.7K vocab
# - Faster training, better convergence, proven to work
#
# FUTURE OPTIMIZATION (See issue #677):
# - AST-aware semantic pairing approach theoretically better
# - Currently too slow (OPTIONAL MATCH query bottleneck in Kuzu)
# - Need to optimize graph queries before using AST approach
#
# Vocabulary strategy:
# - Official Fundamento + validated corpus roots (production_semantic_roots_15k.json)
# - Only 6.7K of 15K appear in corpus (others too rare)
# - Ensures all embeddings are for official Esperanto roots
#
# Optimal training parameters:
# - 96D embeddings (appropriate for 15K vocabulary)
# - 10 epochs, 10% data sampling (~50-200 pairs/root, matches Word2vec literature)
# - 2.9M parameters
#
# Total time: ~4-5 hours (extraction 60-90min + training 3h)
# Result: Pure semantic embeddings + production quality
#
# Usage:
#   ./scripts/train_phase1_embeddings_fast.sh                    # Full pipeline (auto-skips extraction if exists)
#   ./scripts/train_phase1_embeddings_fast.sh --fresh            # Force re-extract even if pairs exist
#   ./scripts/train_phase1_embeddings_fast.sh --extract-only     # Only extract pairs
#   ./scripts/train_phase1_embeddings_fast.sh --train-only       # Only train (pairs must exist)
#

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
DB_PATH="data/indexes/v2.1_kuzu_index_full"
TRAINING_DIR="data/training/phase1_embeddings_fast"
OUTPUT_DIR="models/root_embeddings_phase1_fast"
LOG_DIR="logs/phase1_embeddings_fast"

# Create directories
mkdir -p "$TRAINING_DIR"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# Parse arguments
EXTRACT=true
TRAIN=true
FRESH=false

if [ "$1" == "--extract-only" ]; then
    TRAIN=false
elif [ "$1" == "--train-only" ]; then
    EXTRACT=false
elif [ "$1" == "--fresh" ]; then
    FRESH=true
fi

# Auto-detect existing training pairs (unless --fresh or --extract-only)
if [ "$FRESH" = false ] && [ "$EXTRACT" = true ]; then
    if [ -f "$TRAINING_DIR/root_embedding_pairs.jsonl" ] && \
       [ -f "$TRAINING_DIR/root_embedding_pairs_vocab.json" ] && \
       [ -f "$TRAINING_DIR/root_embedding_pairs_stats.json" ]; then
        echo ""
        echo "Found existing training pairs, skipping extraction (use --fresh to re-extract)"
        EXTRACT=false
    fi
fi

# Timestamp for logging
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/training_$TIMESTAMP.log"

echo "============================================================================"
echo "Phase 1 Root Embeddings Training Pipeline - POSITIONAL WINDOW v2.1"
echo "============================================================================"
echo "Database: $DB_PATH"
echo "Training data: $TRAINING_DIR"
echo "Output: $OUTPUT_DIR"
echo "Log: $LOG_FILE"
echo ""
echo "POSITIONAL WINDOW SKIP-GRAM (v2.1) - MAXIMUM QUALITY:"
echo "- 8-word context windows (extended for broader semantic relationships)"
echo "- Cross-sentence context for paragraph coherence: weight 0.5"
echo "- Function word filtering (artikolo, prepozicio, konjunkcio, pronomo)"
echo "- Content roots only (substantivo, verbo, adjektivo, adverbo)"
echo "- Mikolov subsampling for high-frequency words (threshold 1e-3)"
echo ""
echo "OPTIMIZED SETTINGS (official Fundamento roots):"
echo "- Vocabulary: ~6.7K official roots (Fundamento + validated corpus)"
echo "  * All Fundamento roots that appear in corpus"
echo "  * High-frequency validated corpus roots"
echo "  * Using production_semantic_roots_15k.json filter"
echo "  * Function words auto-excluded by parser"
echo "- Embedding dim: 128D → ~1.7M params (optimal for 6.7K vocab)"
echo "- Window size: 8 (broader context relationships)"
echo "- Dataset: ~34M training pairs (window=8, filtered to official roots)"
echo "- Negative samples: 10 (better discrimination)"
echo "- Epochs: 15 with early stopping (ensure full convergence)"
echo "- Batch size: 512"
echo "- Smart per-word sampling: Mikolov et al. 2013 subsampling"
echo "- Estimated time: ~2-3 hours (extraction 40min + training 1.5-2h)"
echo "- Quality: Official Esperanto roots (128D avoids divergence)"
echo ""
echo "NOTE: AST-aware semantic pairing approach deferred (see issue #677)"
echo "============================================================================"

# Step 1: Extract training pairs (if needed)
if [ "$EXTRACT" = true ]; then
    echo ""
    echo "Step 1: Extracting training pairs with positional windows..."
    echo "--------------------------------------------------------------------------------"

    python scripts/extract_embedding_training_pairs.py \
        --db-path "$DB_PATH" \
        --output "$TRAINING_DIR/root_embedding_pairs.jsonl" \
        --vocab-output "$TRAINING_DIR/root_embedding_pairs_vocab.json" \
        --target-vocabulary data/vocabularies/production_semantic_roots_15k.json \
        --window-size 8 \
        --cross-sentence-weight 0.5 \
        --subsample-threshold 1e-3 \
        2>&1 | tee -a "$LOG_FILE"

    echo "Training pairs extracted."
else
    echo "Skipping extraction (using existing pairs)"
fi

# Check if training pairs exist
if [ ! -f "$TRAINING_DIR/root_embedding_pairs.jsonl" ]; then
    echo "ERROR: Training pairs not found: $TRAINING_DIR/root_embedding_pairs.jsonl"
    echo "Run without --train-only flag to extract pairs first"
    exit 1
fi

# Step 2: Train skip-gram model (if needed)
if [ "$TRAIN" = true ]; then
    echo ""
    echo "Step 2: Training skip-gram model with negative sampling (MAXIMUM QUALITY)..."
    echo "--------------------------------------------------------------------------------"

    python scripts/train_root_embeddings_skipgram_v2_1.py \
        --training-pairs "$TRAINING_DIR/root_embedding_pairs.jsonl" \
        --vocabulary "$TRAINING_DIR/root_embedding_pairs_vocab.json" \
        --output "$OUTPUT_DIR" \
        --embedding-dim 128 \
        --epochs 15 \
        --batch-size 512 \
        --dataset-fraction 1.0 \
        --learning-rate 0.005 \
        --negative-samples 10 \
        --patience 3 \
        --min-delta 0.001 \
        --collapse-threshold 0.7 \
        --resume \
        2>&1 | tee -a "$LOG_FILE"

    echo "Training complete."
else
    echo "Skipping training"
fi

echo ""
echo "============================================================================"
echo "Pipeline Complete - MAXIMUM QUALITY v2.1"
echo "============================================================================"

if [ "$TRAIN" = true ]; then
    echo "Best model: $OUTPUT_DIR/root_embeddings_best.pt"
    echo "Final model: $OUTPUT_DIR/root_embeddings_final.pt"
    echo "Checkpoint: $OUTPUT_DIR/root_embeddings_checkpoint.pt"
fi

echo "Training pairs: $TRAINING_DIR/root_embedding_pairs.jsonl"
echo "Vocabulary: $TRAINING_DIR/root_embedding_pairs_vocab.json"
echo "Statistics: $TRAINING_DIR/root_embedding_pairs_stats.json"
echo "Log file: $LOG_FILE"
echo ""
echo "QUALITY: Maximum quality skip-gram root embeddings"
echo "- 256D embeddings (7.3M parameters)"
echo "- 8-word context windows (broader semantic relationships)"
echo "- 10 negative samples (better discrimination)"
echo "- 100% dataset (5.7M training pairs - full corpus)"
echo "- Cross-sentence pairs for paragraph coherence (weight 0.5)"
echo "- Function words filtered automatically"
echo "- ~400 pairs/root (excellent coverage)"
echo ""
echo "NOTE: AST-aware semantic pairing deferred to issue #677"
echo "============================================================================"
