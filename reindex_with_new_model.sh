#!/bin/bash
# Re-index corpus with newly trained GNN model
# Run this after GNN training completes

echo "========================================================================"
echo "RE-INDEXING CORPUS WITH NEW GNN MODEL"
echo "========================================================================"
echo ""

# Check if new model exists
if [ ! -f "models/tree_lstm/checkpoint_epoch_20.pt" ]; then
    echo "ERROR: New model not found at models/tree_lstm/checkpoint_epoch_20.pt"
    echo "Please ensure GNN training completed successfully."
    echo ""
    exit 1
fi

echo "✓ Found new model: models/tree_lstm/checkpoint_epoch_20.pt"
echo ""

# Archive old index
if [ -d "data/corpus_index" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    echo "Archiving old index → data/corpus_index_before_retrain_${TIMESTAMP}"
    mv data/corpus_index "data/corpus_index_before_retrain_${TIMESTAMP}"
    echo "✓ Old index archived"
    echo ""
fi

# Re-index corpus
echo "Re-indexing corpus with new GNN model..."
echo "  Corpus: data/corpus_sentences.jsonl (20,985 sentences)"
echo "  Model:  models/tree_lstm/checkpoint_epoch_20.pt"
echo "  Output: data/corpus_index/"
echo ""
echo "This will take approximately 5 minutes..."
echo ""
echo "========================================================================"
echo ""

python scripts/index_corpus.py \
    --corpus data/corpus_sentences.jsonl \
    --output data/corpus_index \
    --model models/tree_lstm/checkpoint_epoch_20.pt \
    --batch-size 32 2>&1 | \
    stdbuf -oL grep -E "^(INFO|Indexing corpus.*[0-9]+%|Building FAISS|Done)" | \
    grep -v "pbar"

echo ""
echo "========================================================================"
echo "✓ Re-indexing complete!"
echo "========================================================================"
echo ""
echo "New corpus index ready at: data/corpus_index/"
echo ""
echo "You can now query with the improved RAG system:"
echo "  python scripts/quick_query.py \"Kiu estas Frodo?\""
echo ""
