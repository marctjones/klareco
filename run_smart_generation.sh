#!/bin/bash
# Smart generation of topical pairs from scratch
# Coverage-based sampling during generation (not post-processing)

set -e
cd "$(dirname "$0")"

# Activate venv
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Error: No virtual environment found"
    exit 1
fi

# Create directories
mkdir -p logs/training data/training data/vocabularies

echo "=========================================="
echo "Smart Topical Pair Generation"
echo "=========================================="
echo ""
echo "Strategy: Coverage-Based Generation"
echo "  - Processes ALL 4.3M sentences"
echo "  - Samples during generation (not after)"
echo "  - Target: 300 pairs per root (balanced)"
echo "  - Negative ratio: 2:1 (quality over quantity)"
echo "  - Expected output: ~20-25M pairs (~2-3GB)"
echo ""
echo "Why this is better:"
echo "  ✓ Full corpus coverage (not just first 23%)"
echo "  ✓ Balanced per-root representation"
echo "  ✓ Frequency-weighted negative sampling"
echo "  ✓ No post-processing needed"
echo "  ✓ Single-pass streaming (memory efficient)"
echo ""
echo "Time: ~1.5-2 hours"
echo ""
echo "Starting..."
echo ""

python scripts/data/generate_topical_pairs_smart.py \
    --corpus data/corpus/unified_corpus.jsonl \
    --output data/training/topical_pairs_smart.jsonl \
    --vocab-output data/vocabularies/topical_vocab.json \
    --target-per-root 300 \
    --window-size 5 \
    --negative-ratio 2.0 \
    --min-frequency 5 \
    --min-root-freq 50 \
    2>&1 | tee logs/training/smart_generation_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "=========================================="
echo "COMPLETE!"
echo "=========================================="
echo ""
echo "Output: data/training/topical_pairs_smart.jsonl"
echo "Vocabulary: data/vocabularies/topical_vocab.json"
echo ""
echo "Ready for Phase 2: Training dual embeddings"
echo ""
echo "Next step:"
echo "  ./scripts/train_dual_embeddings.sh"
