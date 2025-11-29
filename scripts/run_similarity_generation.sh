#!/bin/bash
#
# Generate Semantic Similarity Training Pairs
# Run this in a separate terminal - it will take ~10-15 minutes
#
# Usage: ./scripts/run_similarity_generation.sh
#

set -e

cd /home/marc/klareco

echo "=============================================="
echo "Generating Semantic Similarity Training Pairs"
echo "=============================================="
echo ""
echo "This will:"
echo "  1. Load 271K Esperanto-English parallel sentences"
echo "  2. Find Esperanto paraphrases (same English translation)"
echo "  3. Compute English embeddings to find similar sentences"
echo "  4. Output training pairs containing ONLY Esperanto"
echo ""
echo "Estimated time: 10-15 minutes"
echo ""

# Run the generation script
python3 scripts/generate_similarity_pairs.py \
    --en-file data/tatoeba/Tatoeba.en-eo.en \
    --eo-file data/tatoeba/Tatoeba.en-eo.eo \
    --output data/similarity_pairs.jsonl \
    --model paraphrase-MiniLM-L6-v2 \
    --max-paraphrase-pairs 200000 \
    --max-similarity-pairs 100000 \
    --seed 42

echo ""
echo "=============================================="
echo "DONE!"
echo "=============================================="
echo ""
echo "Output files:"
ls -lh data/similarity_pairs_*.jsonl 2>/dev/null || echo "  (no files found)"
echo ""
echo "Sample from training data:"
head -3 data/similarity_pairs_train.jsonl 2>/dev/null | python3 -c "
import sys, json
for line in sys.stdin:
    d = json.loads(line)
    print(f\"  A: {d['sentence_a']}\")
    print(f\"  B: {d['sentence_b']}\")
    print(f\"  Similarity: {d['similarity']}\")
    print()
" || echo "  (could not read)"
echo ""
echo "Next step: Return to Claude Code and continue with training."
