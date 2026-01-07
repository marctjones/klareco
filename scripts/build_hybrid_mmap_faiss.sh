#!/bin/bash
# Build mmap arrays and FAISS index for slot_hybrid/ with 128d hybrid embeddings
#
# RESTARTABLE: Automatically resumes from checkpoint if interrupted
# MEMORY-SAFE: Uses disk-backed mmap arrays, streams through 32GB slot_index.jsonl
#
# Usage:
#   ./scripts/build_hybrid_mmap_faiss.sh           # Resume or start fresh
#   ./scripts/build_hybrid_mmap_faiss.sh --fresh   # Start from scratch
#   ./scripts/build_hybrid_mmap_faiss.sh --mmap-only   # Only build mmap arrays
#   ./scripts/build_hybrid_mmap_faiss.sh --faiss-only  # Only build FAISS (requires mmap)
#
# Output:
#   data/indexes/slot_hybrid/mmap/SUBJ.npy, VERB.npy, OBJ.npy, full.npy (~2GB each)
#   data/indexes/slot_hybrid/mmap/*_norms.npy (~17MB each)
#   data/indexes/slot_hybrid/faiss/full_embeddings.index (~2GB)
#
# Estimated time: 30-60 minutes for 4.4M documents
# Estimated disk: ~12GB for mmap + ~2GB for FAISS

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Configuration
INDEX_DIR="data/indexes/slot_hybrid"
LOG_DIR="logs"
LOG_FILE="$LOG_DIR/build_hybrid_indexes_$(date +%Y%m%d_%H%M%S).log"

# Activate venv
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Error: No virtual environment found"
    exit 1
fi

# Create log directory
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "Building Mmap + FAISS Indexes for Hybrid Embeddings"
echo "=========================================="
echo ""
echo "Index directory: $INDEX_DIR"
echo "Log file: $LOG_FILE"
echo ""
echo "This will create:"
echo "  - mmap/SUBJ.npy, VERB.npy, OBJ.npy, full.npy (128d embeddings)"
echo "  - mmap/*_norms.npy (precomputed norms)"
echo "  - faiss/full_embeddings.index"
echo ""
echo "Progress will be logged to: $LOG_FILE"
echo "Checkpoints saved every 100K documents to: $INDEX_DIR/build_indexes_checkpoint.json"
echo ""
echo "To monitor progress in another terminal:"
echo "  tail -f $LOG_FILE"
echo ""
echo "Starting in 3 seconds... (Ctrl+C to cancel)"
sleep 3

# Parse arguments
EXTRA_ARGS=""
if [ "$1" == "--fresh" ]; then
    EXTRA_ARGS="--fresh"
    echo "Starting fresh (ignoring checkpoint)"
elif [ "$1" == "--mmap-only" ]; then
    EXTRA_ARGS="--mmap-only"
    echo "Building mmap arrays only (skipping FAISS)"
elif [ "$1" == "--faiss-only" ]; then
    EXTRA_ARGS="--faiss-only"
    echo "Building FAISS index only (requires mmap to exist)"
fi

# Run the build script with logging
echo ""
echo "Build started at $(date)"
echo "=========================================="

python scripts/build_mmap_faiss_from_slot_index.py \
    --index-dir "$INDEX_DIR" \
    $EXTRA_ARGS \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "=========================================="
echo "Build completed at $(date)"
echo "Log saved to: $LOG_FILE"
echo ""

# Show what was created
echo "Created files:"
if [ -d "$INDEX_DIR/mmap" ]; then
    ls -lh "$INDEX_DIR/mmap/"
fi
if [ -d "$INDEX_DIR/faiss" ]; then
    ls -lh "$INDEX_DIR/faiss/"
fi

echo ""
echo "Next steps:"
echo "  - Test ASTAwareRetriever: python scripts/test_ast_retriever.py"
echo "  - Run benchmark: python scripts/benchmark_qa_enhanced.py"
