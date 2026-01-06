#!/bin/bash
# Build all slot-based indexes with hybrid embeddings (128d)
#
# This script:
#   1. Builds slot_index.jsonl with hybrid embeddings (128d)
#   2. Builds HNSW index (other indexes auto-build on first use)
#
# Usage:
#   ./scripts/build_hybrid_indexes.sh              # Full corpus
#   ./scripts/build_hybrid_indexes.sh --limit 1000 # Test with 1K sentences

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
    echo "Error: No virtual environment found"
    exit 1
fi

# Parse arguments
LIMIT_ARG=""
if [ "$1" == "--limit" ]; then
    LIMIT_ARG="--limit $2"
    echo "Testing mode: limiting to $2 sentences"
fi

echo "=========================================="
echo "Building Hybrid Embedding Indexes (128d)"
echo "=========================================="
echo ""

# Configuration
CORPUS_FILE="data/corpus/unified_corpus.jsonl"
LINGUISTIC_MODEL="models/root_embeddings/best_model.pt"
TOPICAL_MODEL="models/topical_embeddings/best_model.pt"
AFFIX_MODEL="models/affix_transforms_v2/best_model.pt"
SLOT_INDEX_DIR="data/indexes/slot_hybrid"

# Check prerequisites
if [ ! -f "$CORPUS_FILE" ]; then
    echo "Error: Corpus not found: $CORPUS_FILE"
    echo "Run ./scripts/parse_corpus.sh first"
    exit 1
fi

if [ ! -f "$LINGUISTIC_MODEL" ]; then
    echo "Error: Linguistic model not found: $LINGUISTIC_MODEL"
    echo "Run ./scripts/train_roots.sh first"
    exit 1
fi

if [ ! -f "$TOPICAL_MODEL" ]; then
    echo "Error: Topical model not found: $TOPICAL_MODEL"
    echo "Run ./scripts/train_topical_embeddings.sh first"
    exit 1
fi

if [ ! -f "$AFFIX_MODEL" ]; then
    echo "Error: Affix model not found: $AFFIX_MODEL"
    echo "Note: Affix transforms are skipped in hybrid mode (need 128d versions)"
fi

echo "Prerequisites verified:"
echo "  ✓ Corpus: $CORPUS_FILE"
echo "  ✓ Linguistic embeddings: $LINGUISTIC_MODEL"
echo "  ✓ Topical embeddings: $TOPICAL_MODEL"
echo ""

# Step 1: Build slot-based index with hybrid embeddings
echo "=========================================="
echo "Step 1: Building Slot-Based Index (128d)"
echo "=========================================="
echo ""

python scripts/index_slot_based.py \
    --corpus "$CORPUS_FILE" \
    --output "$SLOT_INDEX_DIR" \
    --root-model "$LINGUISTIC_MODEL" \
    --affix-model "$AFFIX_MODEL" \
    --topical-model "$TOPICAL_MODEL" \
    --hybrid \
    --resume \
    $LIMIT_ARG

echo ""
echo "✓ Slot index created: $SLOT_INDEX_DIR/slot_index.jsonl"
echo ""

# Step 2: Build HNSW index (others auto-build on first use)
echo "=========================================="
echo "Step 2: Building HNSW Index"
echo "=========================================="
echo ""
echo "Building HNSW index (FAISS, ScaNN, MemoryMapped auto-build on first use)..."
echo ""

./scripts/build_hnsw_index.sh "$SLOT_INDEX_DIR"

echo ""
echo "✓ HNSW index created"
echo ""

# Summary
echo "=========================================="
echo "Hybrid Indexes Built Successfully!"
echo "=========================================="
echo ""
echo "Created:"
echo "  ✓ Slot index:  $SLOT_INDEX_DIR/slot_index.jsonl (128d hybrid embeddings)"
echo "  ✓ HNSW index:  $SLOT_INDEX_DIR/hnsw/"
echo ""
echo "Auto-built on first use:"
echo "  • FAISS index:  $SLOT_INDEX_DIR/faiss/"
echo "  • ScaNN index:  $SLOT_INDEX_DIR/scann/"
echo "  • Mmap arrays:  $SLOT_INDEX_DIR/mmap/"
echo ""
echo "All indexes use 128d hybrid embeddings (64d linguistic + 64d topical)"
echo ""
echo "Next steps:"
echo "  - Test retrieval: python scripts/demo_slot_retrieval.py --index $SLOT_INDEX_DIR -i"
echo "  - Benchmark: ./scripts/benchmark_qa_all.sh"
echo "  - Compare hybrid vs linguistic-only performance"
echo ""
