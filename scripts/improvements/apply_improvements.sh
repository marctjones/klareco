#!/bin/bash
# =============================================================================
# Apply RootEmbedder v1.0 Improvements
# =============================================================================
# This script:
# 1. Adds antonym pairs to training
# 2. Retrains root embeddings
# 3. Freezes the model
# 4. Evaluates quality
#
# Usage:
#   ./scripts/improvements/apply_improvements.sh
# =============================================================================

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

echo -e "${BLUE}=== RootEmbedder v1.0 Improvements ===${NC}"
echo ""
echo "This will:"
echo "  1. Add antonym pairs (~4K mal- prefix pairs)"
echo "  2. Retrain root embeddings (1-2 hours)"
echo "  3. Freeze model for downstream use"
echo "  4. Evaluate embedding quality"
echo ""
echo -e "${YELLOW}WARNING: This will overwrite models/root_embeddings/best_model.pt${NC}"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted"
    exit 1
fi

# =============================================================================
# Step 1: Backup existing model
# =============================================================================
echo -e "${YELLOW}=== Step 1: Backup Existing Model ===${NC}"

if [ -f "models/root_embeddings/best_model.pt" ]; then
    BACKUP_DIR="models/root_embeddings/backups"
    mkdir -p "$BACKUP_DIR"
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    cp models/root_embeddings/best_model.pt "$BACKUP_DIR/best_model_${TIMESTAMP}.pt"
    echo -e "${GREEN}Backed up to $BACKUP_DIR/best_model_${TIMESTAMP}.pt${NC}"
else
    echo "No existing model to backup"
fi

# =============================================================================
# Step 2: Check if antonym pairs are integrated
# =============================================================================
echo -e "\n${YELLOW}=== Step 2: Check Antonym Integration ===${NC}"

if grep -q "Systematic antonym pairs" scripts/train_root_embeddings.py; then
    echo -e "${GREEN}✓ Antonym pairs already integrated${NC}"
else
    echo -e "${YELLOW}⚠ Antonym pairs not yet integrated${NC}"
    echo ""
    echo "Please add the following to scripts/train_root_embeddings.py"
    echo "in the build_similarity_pairs() function (around line 460):"
    echo ""
    echo "----------------------------------------------------------------------"
    cat << 'EOF'
    # =========================================================================
    # 4. Systematic antonym pairs (mal- prefix)
    # =========================================================================
    logger.info("Generating systematic antonym pairs (mal- prefix)...")

    antonym_count = 0
    for root in root_to_idx:
        if not root.startswith('mal'):
            continue

        positive_root = root[3:]

        if len(positive_root) < 2:
            continue
        if root in FUNCTION_WORDS or positive_root in FUNCTION_WORDS:
            continue

        if positive_root not in root_to_idx:
            continue

        idx1, idx2 = root_to_idx[root], root_to_idx[positive_root]
        pair_key = (min(idx1, idx2), max(idx1, idx2))

        target = -0.7  # Negative = antonyms!
        weight = 20.0

        if pair_key not in pair_targets or target < pair_targets[pair_key]:
            pair_targets[pair_key] = target
            pairs.append((idx1, idx2, target))
            weights.append(weight)
            antonym_count += 1

    logger.info(f"Created {antonym_count} antonym pairs (target=-0.7, weight=20.0)")
EOF
    echo "----------------------------------------------------------------------"
    echo ""
    read -p "Have you added the code? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Please add the code and run this script again"
        exit 1
    fi
fi

# =============================================================================
# Step 3: Retrain with antonyms
# =============================================================================
echo -e "\n${YELLOW}=== Step 3: Retrain Root Embeddings ===${NC}"
echo "This will take 1-2 hours..."
echo ""

./scripts/train_roots.sh --fresh

if [ ! -f "models/root_embeddings/best_model.pt" ]; then
    echo -e "${YELLOW}Error: Training failed - no model produced${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Training complete${NC}"

# =============================================================================
# Step 4: Freeze the model
# =============================================================================
echo -e "\n${YELLOW}=== Step 4: Freeze Model ===${NC}"

python scripts/improvements/freeze_model.py \
    --model models/root_embeddings/best_model.pt \
    --output models/root_embedder/frozen_v1.0.pt \
    --version "v1.0" \
    --description "Root embeddings with antonym pairs and function word filtering" \
    --validate

if [ ! -f "models/root_embedder/frozen_v1.0.pt" ]; then
    echo -e "${YELLOW}Error: Freezing failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Model frozen${NC}"

# =============================================================================
# Step 5: Evaluate quality
# =============================================================================
echo -e "\n${YELLOW}=== Step 5: Evaluate Quality ===${NC}"

mkdir -p results

python scripts/improvements/evaluate_embeddings.py \
    --model models/root_embedder/frozen_v1.0.pt \
    --output results/root_embedder_v1.0_eval.json

echo -e "${GREEN}✓ Evaluation complete${NC}"
echo ""
echo "Results saved to results/root_embedder_v1.0_eval.json"

# =============================================================================
# Summary
# =============================================================================
echo ""
echo -e "${BLUE}=== Summary ===${NC}"
echo ""
echo "✓ Trained model:    models/root_embeddings/best_model.pt"
echo "✓ Frozen model:     models/root_embedder/frozen_v1.0.pt"
echo "✓ Metadata:         models/root_embedder/frozen_v1.0.json"
echo "✓ Evaluation:       results/root_embedder_v1.0_eval.json"
echo ""
echo "Next steps:"
echo "  1. Review evaluation results"
echo "  2. If score >85/100 → Proceed to MorphemeComposer (#698)"
echo "  3. If score <85/100 → Consider AST-aware redesign with Opus"
echo ""
echo -e "${GREEN}Done!${NC}"
