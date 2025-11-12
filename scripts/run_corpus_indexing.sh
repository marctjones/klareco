#!/bin/bash
#
# Run full corpus indexing with Tree-LSTM embeddings
#
# This script:
# - Indexes the full 74K-sentence Gutenberg Esperanto corpus
# - Saves embeddings, metadata, and FAISS index to data/corpus_index/
# - Automatically resumes from checkpoint if interrupted
# - Logs everything to data/corpus_index/indexing.log
#
# Usage:
#   ./scripts/run_corpus_indexing.sh
#
# Monitor progress in another terminal:
#   tail -f data/corpus_index/indexing.log
#

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Klareco Corpus Indexing${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Configuration
CORPUS_FILE="data/gutenberg_sentences.txt"
OUTPUT_DIR="data/corpus_index"
BATCH_SIZE=32
MODEL_PATH="models/tree_lstm/checkpoint_epoch_12.pt"

# Check if corpus exists
if [ ! -f "$CORPUS_FILE" ]; then
    echo -e "${YELLOW}Error: Corpus file not found: $CORPUS_FILE${NC}"
    exit 1
fi

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${YELLOW}Error: Model checkpoint not found: $MODEL_PATH${NC}"
    exit 1
fi

# Count sentences in corpus
SENTENCE_COUNT=$(wc -l < "$CORPUS_FILE")
echo -e "${GREEN}Corpus:${NC} $CORPUS_FILE"
echo -e "${GREEN}Sentences:${NC} $SENTENCE_COUNT"
echo -e "${GREEN}Output:${NC} $OUTPUT_DIR"
echo -e "${GREEN}Batch size:${NC} $BATCH_SIZE"
echo ""

# Check for existing checkpoint
if [ -f "$OUTPUT_DIR/indexing_checkpoint.json" ]; then
    PROCESSED=$(python3 -c "import json; print(json.load(open('$OUTPUT_DIR/indexing_checkpoint.json'))['processed'])")
    echo -e "${YELLOW}Found existing checkpoint: $PROCESSED/$SENTENCE_COUNT sentences processed${NC}"
    echo -e "${YELLOW}Will resume from checkpoint${NC}"
    echo ""
fi

# Estimate time
ESTIMATED_TIME=$((SENTENCE_COUNT / 400))
echo -e "${GREEN}Estimated time:${NC} ~${ESTIMATED_TIME} seconds (~$((ESTIMATED_TIME / 60)) minutes)"
echo ""

# Show monitoring command
echo -e "${BLUE}To monitor progress in another terminal:${NC}"
echo -e "  ${GREEN}tail -f $OUTPUT_DIR/indexing.log${NC}"
echo ""

# Countdown
echo -e "${YELLOW}Starting indexing in 3 seconds...${NC}"
sleep 1
echo "2..."
sleep 1
echo "1..."
sleep 1
echo ""

# Run indexing
echo -e "${GREEN}Starting corpus indexing...${NC}"
echo ""

python scripts/index_corpus.py \
    --corpus "$CORPUS_FILE" \
    --output "$OUTPUT_DIR" \
    --batch-size "$BATCH_SIZE" \
    --resume

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  Indexing Complete!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""

    # Show results
    echo -e "${GREEN}Output files:${NC}"
    ls -lh "$OUTPUT_DIR/" | grep -E "(embeddings|metadata|faiss_index)" | awk '{print "  " $9 " (" $5 ")"}'
    echo ""

    # Show final stats from checkpoint
    if [ -f "$OUTPUT_DIR/indexing_checkpoint.json" ]; then
        echo -e "${GREEN}Final statistics:${NC}"
        python3 << EOF
import json
with open('$OUTPUT_DIR/indexing_checkpoint.json') as f:
    stats = json.load(f)
print(f"  Total sentences: {stats['total_sentences']}")
print(f"  Successfully encoded: {stats['successful']} ({stats['successful']/stats['total_sentences']*100:.1f}%)")
print(f"  Failed: {stats['failed']}")
EOF
        echo ""
    fi

    # Show embedding shape
    echo -e "${GREEN}Embeddings shape:${NC}"
    python3 -c "import numpy as np; print(f\"  {np.load('$OUTPUT_DIR/embeddings.npy').shape}\")"
    echo ""

    echo -e "${BLUE}Next steps:${NC}"
    echo -e "  1. Review failed sentences: cat $OUTPUT_DIR/failed_sentences.jsonl"
    echo -e "  2. Build RAG retrieval interface"
    echo -e "  3. Test semantic search queries"
    echo ""
else
    echo ""
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}  Indexing Interrupted or Failed${NC}"
    echo -e "${YELLOW}========================================${NC}"
    echo ""
    echo -e "${YELLOW}To resume, simply run this script again:${NC}"
    echo -e "  ${GREEN}./scripts/run_corpus_indexing.sh${NC}"
    echo ""
    echo -e "${YELLOW}Check logs for details:${NC}"
    echo -e "  ${GREEN}tail -50 $OUTPUT_DIR/indexing.log${NC}"
    echo ""
    exit 1
fi
