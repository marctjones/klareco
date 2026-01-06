#!/bin/bash
# Smart sampling of topical pairs with coverage guarantees
# Run this in a separate terminal

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
mkdir -p logs data/training

echo "=========================================="
echo "Smart Topical Pair Sampling"
echo "=========================================="
echo ""
echo "Strategy:"
echo "  - Target 300 pairs per root (balanced coverage)"
echo "  - Minimum 50 pairs per root (skip very rare roots)"
echo "  - Negative ratio 2:1 (quality over quantity)"
echo "  - Max 30M total pairs (manageable size)"
echo ""
echo "Input:  data/training/topical_pairs.jsonl (20GB, 181M pairs)"
echo "Output: data/training/topical_pairs_sampled.jsonl"
echo ""
echo "This will take ~10-15 minutes (2-pass algorithm)"
echo ""
read -p "Press ENTER to start..."
echo ""

python scripts/data/smart_sample_topical_pairs.py \
    --input data/training/topical_pairs.jsonl \
    --output data/training/topical_pairs_sampled.jsonl \
    --target-per-root 300 \
    --min-per-root 50 \
    --negative-ratio 2.0 \
    --max-pairs 30000000 \
    2>&1 | tee logs/smart_sampling_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "=========================================="
echo "COMPLETE!"
echo "=========================================="
echo "Output: data/training/topical_pairs_sampled.jsonl"
echo ""
echo "Use this file for training topical embeddings."
