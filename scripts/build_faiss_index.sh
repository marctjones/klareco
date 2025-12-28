#!/bin/bash
#
# Build FAISS index from embeddings
#
# Usage:
#   ./scripts/build_faiss_index.sh                    # Build compositional index
#   ./scripts/build_faiss_index.sh data/corpus_index_v3  # Build specific index
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Default to compositional index
INDEX_DIR="${1:-data/corpus_index_compositional}"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Build FAISS Index${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Activate venv
if [[ -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
fi

# Check embeddings exist
EMBEDDINGS_FILE="$INDEX_DIR/embeddings.npy"
if [[ ! -f "$EMBEDDINGS_FILE" ]]; then
    echo -e "${YELLOW}Error: Embeddings not found: $EMBEDDINGS_FILE${NC}"
    echo -e "${YELLOW}Run indexing first: ./scripts/run_compositional_indexing.sh${NC}"
    exit 1
fi

# Get embedding info
echo -e "${GREEN}Index directory:${NC} $INDEX_DIR"
echo -e "${GREEN}Embeddings file:${NC} $EMBEDDINGS_FILE"
echo -e "${GREEN}File size:${NC} $(du -h "$EMBEDDINGS_FILE" | cut -f1)"
echo ""

# Build index
echo -e "${GREEN}Building FAISS index...${NC}"
python3 << 'EOF'
import sys
import time

try:
    import faiss
except ImportError:
    print("Error: FAISS not installed")
    print("Install with: pip install faiss-cpu")
    sys.exit(1)

import numpy as np

index_dir = sys.argv[1] if len(sys.argv) > 1 else "data/corpus_index_compositional"
embeddings_path = f"{index_dir}/embeddings.npy"
output_path = f"{index_dir}/faiss_index.bin"

print(f"Loading embeddings from {embeddings_path}...")
start = time.time()
embeddings = np.load(embeddings_path).astype(np.float32)
print(f"  Shape: {embeddings.shape}")
print(f"  Loaded in {time.time() - start:.1f}s")

print("Normalizing embeddings for cosine similarity...")
faiss.normalize_L2(embeddings)

print("Building FAISS index...")
start = time.time()
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)  # Inner product = cosine for normalized vectors
index.add(embeddings)
print(f"  Built in {time.time() - start:.1f}s")

print(f"Saving index to {output_path}...")
faiss.write_index(index, output_path)

print(f"\nSuccess! Built FAISS index with {index.ntotal:,} vectors ({dim}d)")
EOF

if [[ $? -eq 0 ]]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  FAISS Index Built Successfully${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo -e "${GREEN}Output:${NC} $INDEX_DIR/faiss_index.bin"
    echo -e "${GREEN}Size:${NC} $(du -h "$INDEX_DIR/faiss_index.bin" | cut -f1)"
    echo ""
    echo -e "${BLUE}Next steps:${NC}"
    echo "  1. Validate: python scripts/validate_stage1.py --compare-old"
    echo "  2. Demo:     python scripts/demo_compositional_embeddings.py -i"
    echo ""
else
    echo ""
    echo -e "${YELLOW}Failed to build FAISS index${NC}"
    exit 1
fi
