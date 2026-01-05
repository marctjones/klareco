#!/bin/bash
# Build ScaNN index for slot-based retrieval
# This script installs scann if needed and builds the ScaNN searcher

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
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --index PATH           Path to slot index (default: data/indexes/slot_full)"
            echo "  --force-reinstall      Force reinstall scann even if already installed"
            echo "  -h, --help             Show this help"
            echo ""
            echo "Example:"
            echo "  $0                                    # Build ScaNN index for slot_full"
            echo "  $0 --index data/indexes/slot_test    # Build for test index"
            echo "  $0 --force-reinstall                 # Reinstall scann first"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "ScaNN Index Builder"
echo "========================================"
echo "Index path: $INDEX_PATH"
echo "Project root: $PROJECT_ROOT"
echo "========================================"
echo ""

# Check if scann is installed
echo "Checking for scann..."
if python -c "import scann" 2>/dev/null; then
    echo "✓ scann is already installed"
    if [ "$FORCE_REINSTALL" = true ]; then
        echo "  Force reinstall requested, reinstalling..."
        pip install --upgrade --force-reinstall scann
    else
        echo "  Use --force-reinstall to reinstall"
    fi
else
    echo "✗ scann not found, installing..."
    echo ""
    echo "Note: ScaNN requires TensorFlow and specific Python version (3.8-3.11)"
    echo "      This may take several minutes..."
    echo ""
    pip install scann
    echo "✓ scann installed successfully"
fi

echo ""
echo "========================================"
echo "Building ScaNN Index"
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

# Build ScaNN searcher using Python
echo "Building ScaNN searcher (this may take 10-20 minutes for large indexes)..."
echo ""
echo "⚠️  IMPORTANT: ScaNN requires normalized vectors!"
echo "    All embeddings will be normalized before building the searcher."
echo ""

# Create temporary Python script to build ScaNN searcher
TEMP_SCRIPT=$(mktemp)
cat > "$TEMP_SCRIPT" << 'PYTHON_EOF'
import sys
import json
import logging
from pathlib import Path
import numpy as np
import scann

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

def build_scann_searcher(
    index_path,
    num_leaves=2000,
    num_leaves_to_search=100,
    training_sample_size=250000,
    dimensions_per_block=2,
    quantization_threshold=0.2,
    reorder_k=100,
):
    """Build ScaNN searcher from slot_index.jsonl."""
    index_path = Path(index_path).absolute()  # Use absolute path
    index_file = index_path / "slot_index.jsonl"
    scann_dir = index_path / "scann"
    scann_dir.mkdir(exist_ok=True, parents=True)

    # Count documents and get embedding dimension
    logger.info("Loading embeddings from slot index...")
    with open(index_file) as f:
        first_doc = json.loads(f.readline())
        embedding_dim = len(first_doc['full_embedding'])

    with open(index_file) as f:
        num_docs = sum(1 for _ in f)

    logger.info(f"  {num_docs:,} documents")
    logger.info(f"  {embedding_dim}d embeddings")
    logger.info(f"  ScaNN params: num_leaves={num_leaves}, num_leaves_to_search={num_leaves_to_search}")
    logger.info(f"                dimensions_per_block={dimensions_per_block}, reorder_k={reorder_k}")

    # Load embeddings
    logger.info("Loading all embeddings into memory...")
    full_embeddings = []

    with open(index_file) as f:
        for i, line in enumerate(f):
            doc = json.loads(line)
            full_embeddings.append(doc['full_embedding'])

            if (i + 1) % 100000 == 0:
                logger.info(f"  Loaded {i+1:,} embeddings...")

    # Convert to numpy and normalize (REQUIRED for ScaNN dot_product)
    logger.info("Normalizing embeddings (required for ScaNN dot_product metric)...")
    embeddings = np.array(full_embeddings, dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / norms

    # Verify normalization
    test_norms = np.linalg.norm(normalized_embeddings[:10], axis=1)
    logger.info(f"  Normalized: min_norm={test_norms.min():.6f}, max_norm={test_norms.max():.6f}")
    assert np.allclose(test_norms, 1.0, atol=1e-5), "Normalization failed!"

    # Build ScaNN searcher
    logger.info("Building ScaNN searcher...")
    logger.info("  This may take 10-20 minutes for 4.2M documents...")

    # Use smaller training sample if dataset is smaller
    actual_training_sample = min(training_sample_size, num_docs)

    searcher = scann.scann_ops_pybind.builder(
        normalized_embeddings,
        500,  # final_num_neighbors for search
        "dot_product"
    ).tree(
        num_leaves=num_leaves,
        num_leaves_to_search=num_leaves_to_search,
        training_sample_size=actual_training_sample
    ).score_ah(
        dimensions_per_block=dimensions_per_block,
        anisotropic_quantization_threshold=quantization_threshold
    ).reorder(reorder_k).build()

    # Save searcher
    logger.info(f"Saving ScaNN searcher to {scann_dir}...")
    searcher.serialize(str(scann_dir))

    # Verify
    logger.info("Verifying searcher...")
    verify_searcher = scann.scann_ops_pybind.load_searcher(str(scann_dir))

    # Test search
    test_query = normalized_embeddings[0]
    neighbors, distances = verify_searcher.search(test_query, final_num_neighbors=10)
    logger.info(f"  Test search successful: {len(neighbors)} neighbors found")

    logger.info(f"✓ ScaNN searcher built successfully!")
    logger.info(f"  Saved to: {scann_dir}")
    logger.info(f"  Algorithm: Tree partitioning + Anisotropic quantization + Reordering")

if __name__ == '__main__':
    index_path = sys.argv[1]
    build_scann_searcher(index_path)
PYTHON_EOF

# Run the Python script
python "$TEMP_SCRIPT" "$INDEX_PATH"

# Clean up
rm "$TEMP_SCRIPT"

echo ""
echo "========================================"
echo "ScaNN Index Build Complete!"
echo "========================================"
echo ""
echo "You can now use the ScaNN retriever:"
echo "  python scripts/demo_slot_retrieval.py --index $INDEX_PATH -i"
echo ""
echo "The ScaNNSlotRetriever will be automatically selected for this index."
echo ""
echo "Expected performance:"
echo "  - Recall: 90-95% (highest accuracy)"
echo "  - Latency: 3-5ms"
echo "  - Best for: Production systems requiring >90% recall"
echo ""
