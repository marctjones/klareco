#!/bin/bash
# Build HNSW index for slot-based retrieval
# This script installs hnswlib if needed and builds the HNSW index

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

# Default values
INDEX_PATH="data/indexes/slot_full"
FORCE_REINSTALL=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --index)
            INDEX_PATH="$2"
            shift 2
            ;;
        --force-reinstall)
            FORCE_REINSTALL=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [INDEX_PATH] [OPTIONS]"
            echo ""
            echo "Arguments:"
            echo "  INDEX_PATH             Path to slot index (default: data/indexes/slot_full)"
            echo ""
            echo "Options:"
            echo "  --index PATH           Path to slot index (alternative to positional arg)"
            echo "  --force-reinstall      Force reinstall hnswlib even if already installed"
            echo "  -h, --help             Show this help"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Build HNSW index for slot_full"
            echo "  $0 data/indexes/slot_verified        # Build for slot_verified (positional)"
            echo "  $0 --index data/indexes/slot_test    # Build for test index (flag)"
            echo "  $0 --force-reinstall                 # Reinstall hnswlib first"
            exit 0
            ;;
        -*)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
        *)
            # Positional argument (index path)
            INDEX_PATH="$1"
            shift
            ;;
    esac
done

echo "========================================"
echo "HNSW Index Builder"
echo "========================================"
echo "Index path: $INDEX_PATH"
echo "Project root: $PROJECT_ROOT"
echo "========================================"
echo ""

# Check if hnswlib is installed
echo "Checking for hnswlib..."
if python -c "import hnswlib" 2>/dev/null; then
    echo "✓ hnswlib is already installed"
    if [ "$FORCE_REINSTALL" = true ]; then
        echo "  Force reinstall requested, reinstalling..."
        pip install --upgrade --force-reinstall hnswlib
    else
        echo "  Use --force-reinstall to reinstall"
    fi
else
    echo "✗ hnswlib not found, installing..."
    pip install hnswlib
    echo "✓ hnswlib installed successfully"
fi

echo ""
echo "========================================"
echo "Building HNSW Index"
echo "========================================"
echo ""

# Check if slot_index.jsonl exists
SLOT_INDEX="$INDEX_PATH/slot_index.jsonl"
if [ ! -f "$SLOT_INDEX" ]; then
    echo "Error: Slot index not found at $SLOT_INDEX"
    echo ""
    echo "You need to build the slot index first:"
    echo "  python scripts/index_slot_based.py --corpus <corpus> --output $INDEX_PATH"
    exit 1
fi

# Check if mmap directory exists
MMAP_DIR="$INDEX_PATH/mmap"
if [ ! -d "$MMAP_DIR" ]; then
    echo "Error: Mmap directory not found at $MMAP_DIR"
    echo ""
    echo "You need to build the mmap arrays first:"
    echo "  python scripts/index_slot_based.py --corpus <corpus> --output $INDEX_PATH"
    exit 1
fi

# Count documents
NUM_DOCS=$(wc -l < "$SLOT_INDEX")
echo "Index contains $NUM_DOCS documents"
echo ""

# Build HNSW index using Python
echo "Building HNSW index (this may take several minutes for large indexes)..."
echo ""

# Create temporary Python script to build HNSW index
TEMP_SCRIPT=$(mktemp)
cat > "$TEMP_SCRIPT" << 'PYTHON_EOF'
import sys
import json
import logging
from pathlib import Path
import numpy as np
import hnswlib
import psutil

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

def build_hnsw_index(index_path, M=16, ef_construction=200):
    """Build HNSW index from slot_index.jsonl."""
    index_path = Path(index_path)
    index_file = index_path / "slot_index.jsonl"
    hnsw_dir = index_path / "hnsw"
    hnsw_dir.mkdir(exist_ok=True, parents=True)
    hnsw_file = hnsw_dir / "full_embeddings.hnsw"

    # Count documents and get embedding dimension
    logger.info("Loading embeddings from slot index...")
    with open(index_file) as f:
        first_doc = json.loads(f.readline())
        embedding_dim = len(first_doc['full_embedding'])

    with open(index_file) as f:
        num_docs = sum(1 for _ in f)

    logger.info(f"  {num_docs:,} documents")
    logger.info(f"  {embedding_dim}d embeddings")
    logger.info(f"  HNSW params: M={M}, ef_construction={ef_construction}")

    # Create HNSW index
    physical_cores = psutil.cpu_count(logical=False) or 8
    logger.info(f"  Using {physical_cores} threads")

    index = hnswlib.Index(space='cosine', dim=embedding_dim)
    index.init_index(
        max_elements=num_docs,
        ef_construction=ef_construction,
        M=M,
    )
    index.set_num_threads(physical_cores)

    # Load embeddings and add to index
    logger.info("Adding embeddings to HNSW index...")
    full_embeddings = []
    ids = []

    with open(index_file) as f:
        for i, line in enumerate(f):
            doc = json.loads(line)
            full_embeddings.append(doc['full_embedding'])
            ids.append(i)

            if (i + 1) % 100000 == 0:
                logger.info(f"  Loaded {i+1:,} embeddings...")

    # Convert to numpy array and add to index
    logger.info("Building HNSW graph...")
    embeddings_array = np.array(full_embeddings, dtype=np.float32)
    index.add_items(embeddings_array, ids)

    # Save index
    logger.info(f"Saving HNSW index to {hnsw_file}...")
    index.save_index(str(hnsw_file))

    # Verify
    logger.info("Verifying index...")
    verify_index = hnswlib.Index(space='cosine', dim=embedding_dim)
    verify_index.load_index(str(hnsw_file))
    assert verify_index.get_current_count() == num_docs

    logger.info(f"✓ HNSW index built successfully!")
    logger.info(f"  Saved to: {hnsw_file}")
    logger.info(f"  {verify_index.get_current_count():,} vectors indexed")

if __name__ == '__main__':
    index_path = sys.argv[1]
    build_hnsw_index(index_path)
PYTHON_EOF

# Run the Python script
python "$TEMP_SCRIPT" "$INDEX_PATH"

# Clean up
rm "$TEMP_SCRIPT"

echo ""
echo "========================================"
echo "HNSW Index Build Complete!"
echo "========================================"
echo ""
echo "You can now use the HNSW retriever:"
echo "  python scripts/demo_slot_retrieval.py --index $INDEX_PATH -i"
echo ""
echo "The HNSWSlotRetriever will be automatically selected for this index."
echo ""
