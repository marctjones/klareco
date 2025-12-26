#!/bin/bash
#
# Run Issue #34: Retrain Compositional Embeddings on M1 Corpus
#
# This script:
# 1. Builds M1 vocabulary from enhanced corpus
# 2. Trains compositional embeddings + Tree-LSTM
# 3. Logs all output for monitoring
#
# Usage:
#   ./scripts/run_train_embeddings_m1.sh
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Paths
VENV_DIR="$PROJECT_ROOT/.venv"
DATA_DIR="$PROJECT_ROOT/data"
CORPUS_FILE="$DATA_DIR/corpus_enhanced_m1.jsonl"
VOCAB_DIR="$DATA_DIR/vocabularies_m1"
MODEL_DIR="$PROJECT_ROOT/models/embeddings_m1"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$DATA_DIR/logs"
RUN_LOG="$LOG_DIR/embedding_training_run_${TIMESTAMP}.log"
TRAIN_LOG="$LOG_DIR/embedding_training.log"

# Create directories
mkdir -p "$VOCAB_DIR"
mkdir -p "$MODEL_DIR"
mkdir -p "$LOG_DIR"

echo -e "${BLUE}=====================================================================${NC}"
echo -e "${BLUE}Issue #34: Retrain Compositional Embeddings on M1 Corpus${NC}"
echo -e "${BLUE}=====================================================================${NC}"
echo ""
echo -e "${GREEN}Project root:${NC} $PROJECT_ROOT"
echo -e "${GREEN}Virtual env:${NC} $VENV_DIR"
echo -e "${GREEN}Corpus:${NC} $CORPUS_FILE"
echo -e "${GREEN}Vocabulary output:${NC} $VOCAB_DIR"
echo -e "${GREEN}Model output:${NC} $MODEL_DIR"
echo -e "${GREEN}Run log:${NC} $RUN_LOG"
echo -e "${GREEN}Training log:${NC} $TRAIN_LOG"
echo ""

# Check virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${RED}ERROR: Virtual environment not found at $VENV_DIR${NC}"
    exit 1
fi

# Check corpus exists
if [ ! -f "$CORPUS_FILE" ]; then
    echo -e "${RED}ERROR: M1 corpus not found at $CORPUS_FILE${NC}"
    echo -e "${YELLOW}Please run Issue #28 first:${NC}"
    echo "  ./scripts/run_build_corpus_m1.sh"
    exit 1
fi

CORPUS_SIZE=$(du -h "$CORPUS_FILE" | cut -f1)
CORPUS_LINES=$(wc -l < "$CORPUS_FILE")
echo -e "${GREEN}Corpus size:${NC} $CORPUS_SIZE ($CORPUS_LINES sentences)"
echo ""

# Activate virtual environment
echo -e "${BLUE}Activating virtual environment...${NC}"
source "$VENV_DIR/bin/activate"

# Check Python version
PYTHON_VERSION=$(python --version)
echo -e "${GREEN}Python version: $PYTHON_VERSION${NC}"
echo ""

# Check required packages
echo -e "${BLUE}Checking required packages...${NC}"
python -c "import torch; import torch_geometric" 2>/dev/null || {
    echo -e "${RED}ERROR: PyTorch and PyTorch Geometric required${NC}"
    echo -e "${YELLOW}Installing...${NC}"
    pip install torch torch-geometric
}
echo -e "${GREEN}All packages available${NC}"
echo ""

# Confirm before running
echo -e "${YELLOW}=====================================================================${NC}"
echo -e "${YELLOW}Ready to start embedding training${NC}"
echo -e "${YELLOW}=====================================================================${NC}"
echo ""
echo -e "${YELLOW}This will:${NC}"
echo "  1. Build M1 vocabulary (~15-20K roots vs Tatoeba's 10K)"
echo "  2. Sample 20% of corpus for faster training (~850K sentences)"
echo "  3. Train Compositional Embeddings (640K params)"
echo "  4. Train Tree-LSTM Encoder (2M params)"
echo "  5. Run 5 epochs (~1-1.5 hours per epoch)"
echo "  6. Total time: 5-8 hours"
echo "  7. Use GPU if available, otherwise CPU"
echo ""
echo -e "${YELLOW}Training progress:${NC}"
echo "  - Run log: $RUN_LOG"
echo "  - Training log: $TRAIN_LOG"
echo ""
echo -e "${YELLOW}Monitor with:${NC}"
echo "  tail -f $TRAIN_LOG"
echo ""

# Ask for confirmation (skip if running in background)
if [ -t 0 ]; then
    read -p "$(echo -e ${GREEN}Continue? [y/N]: ${NC})" -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Aborted by user${NC}"
        exit 0
    fi
fi

# Record start time
START_TIME=$(date)
START_TIMESTAMP=$(date +%s)

echo ""
echo -e "${BLUE}=====================================================================${NC}"
echo -e "${BLUE}Starting embedding training at $START_TIME${NC}"
echo -e "${BLUE}=====================================================================${NC}"
echo ""

# Step 1: Build M1 vocabulary
echo -e "${BLUE}Step 1/2: Building M1 vocabulary from corpus...${NC}"
echo "This will take ~5-10 minutes for 4.2M sentences"
echo ""

python "$SCRIPT_DIR/extract_root_vocabulary.py" \
    --corpus "$CORPUS_FILE" \
    --output-dir "$VOCAB_DIR" \
    2>&1 | tee -a "$RUN_LOG"

VOCAB_EXIT_CODE=${PIPESTATUS[0]}

if [ $VOCAB_EXIT_CODE -ne 0 ]; then
    echo -e "${RED}ERROR: Vocabulary extraction failed${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✓ Vocabulary built successfully${NC}"
echo ""

# Show vocabulary statistics
if [ -f "$VOCAB_DIR/root_vocabulary.json" ]; then
    ROOT_COUNT=$(jq 'length' "$VOCAB_DIR/root_vocabulary.json")
    echo -e "${GREEN}Root vocabulary size: $ROOT_COUNT roots${NC}"
fi
if [ -f "$VOCAB_DIR/affix_vocabulary.json" ]; then
    PREFIX_COUNT=$(jq '.prefixes | length' "$VOCAB_DIR/affix_vocabulary.json")
    SUFFIX_COUNT=$(jq '.suffixes | length' "$VOCAB_DIR/affix_vocabulary.json")
    echo -e "${GREEN}Affix vocabulary: $PREFIX_COUNT prefixes, $SUFFIX_COUNT suffixes${NC}"
fi
echo ""

# Step 2: Train embeddings
echo -e "${BLUE}Step 2/2: Training compositional embeddings + Tree-LSTM...${NC}"
echo "This will take 5-8 hours total (1-1.5 hours per epoch)"
echo ""
echo -e "${YELLOW}Training configuration (FAST MODE):${NC}"
echo "  - Epochs: 5 (vs 10 standard)"
echo "  - Max samples: 200K (vs 4.2M full corpus)"
echo "  - Batch size: 64 (larger for speed)"
echo "  - Learning rate: 2e-4 (higher for faster convergence)"
echo "  - Embedding dim: 128"
echo "  - Hidden dim: 256"
echo ""
echo -e "${YELLOW}Monitor progress:${NC}"
echo "  tail -f $TRAIN_LOG"
echo ""

# Run training with both console and log output
python "$SCRIPT_DIR/train_compositional_tree_lstm.py" \
    --corpus "$CORPUS_FILE" \
    --vocab-dir "$VOCAB_DIR" \
    --output "$MODEL_DIR" \
    --epochs 5 \
    --batch-size 64 \
    --lr 2e-4 \
    --embed-dim 128 \
    --hidden-dim 256 \
    --max-samples 200000 \
    2>&1 | tee -a "$RUN_LOG" | tee -a "$TRAIN_LOG"

TRAIN_EXIT_CODE=${PIPESTATUS[0]}

# Record end time
END_TIME=$(date)
END_TIMESTAMP=$(date +%s)
DURATION=$((END_TIMESTAMP - START_TIMESTAMP))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo ""
echo -e "${BLUE}=====================================================================${NC}"

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}Embedding training completed successfully!${NC}"
    echo -e "${BLUE}=====================================================================${NC}"
    echo ""
    echo -e "${GREEN}Start time:${NC} $START_TIME"
    echo -e "${GREEN}End time:${NC} $END_TIME"
    echo -e "${GREEN}Duration:${NC} ${HOURS}h ${MINUTES}m ${SECONDS}s"
    echo ""

    # Show model files
    if [ -d "$MODEL_DIR" ]; then
        echo -e "${GREEN}Model files:${NC}"
        ls -lh "$MODEL_DIR"/*.pt 2>/dev/null || echo "  (No .pt files found)"
        echo ""
    fi

    # Show vocabulary
    echo -e "${GREEN}Vocabulary:${NC}"
    echo "  - Root vocab: $VOCAB_DIR/root_vocabulary.json ($ROOT_COUNT roots)"
    echo "  - Affix vocab: $VOCAB_DIR/affix_vocabulary.json"
    echo ""

    # Show logs
    echo -e "${GREEN}Logs:${NC}"
    echo "  - Run log: $RUN_LOG"
    echo "  - Training log: $TRAIN_LOG"
    echo ""

    # Next steps
    echo -e "${YELLOW}Next steps:${NC}"
    echo "  1. Evaluate embeddings:"
    echo "     python scripts/evaluate_embeddings.py --model $MODEL_DIR/best_model.pt"
    echo "  2. Build FAISS index (Issue #35):"
    echo "     python scripts/index_corpus.py --corpus $CORPUS_FILE --output data/corpus_index_m1"
    echo "  3. Test retrieval:"
    echo "     python scripts/demo_rag.py --index data/corpus_index_m1 --interactive"
    echo ""

else
    echo -e "${RED}Embedding training failed with exit code $TRAIN_EXIT_CODE${NC}"
    echo -e "${BLUE}=====================================================================${NC}"
    echo ""
    echo -e "${RED}Start time:${NC} $START_TIME"
    echo -e "${RED}End time:${NC} $END_TIME"
    echo -e "${RED}Duration:${NC} ${HOURS}h ${MINUTES}m ${SECONDS}s"
    echo ""
    echo -e "${YELLOW}Check logs for errors:${NC}"
    echo "  - Run log: $RUN_LOG"
    echo "  - Training log: $TRAIN_LOG"
    echo ""
    echo -e "${YELLOW}Last 20 lines of run log:${NC}"
    tail -20 "$RUN_LOG"
    echo ""
fi

echo -e "${BLUE}=====================================================================${NC}"

# Deactivate virtual environment
deactivate

exit $TRAIN_EXIT_CODE
