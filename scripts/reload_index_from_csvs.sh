#!/bin/bash
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
    echo "Error: No venv found"
    exit 1
fi

INDEX_DIR="data/indexes/kuzu_index"
PROGRESS_FILE="$INDEX_DIR/build_progress.json"

echo "============================================================"
echo "Reloading Kuzu Index from Existing CSVs"
echo "============================================================"

# Check if CSVs exist
if [ ! -d "$INDEX_DIR/temp_csv" ]; then
    echo "Error: CSV directory not found: $INDEX_DIR/temp_csv"
    echo "CSVs must be generated first. Run: ./scripts/index_kuzu.sh --fresh"
    exit 1
fi

# Check if CSV files exist
if [ ! -f "$INDEX_DIR/temp_csv/has_root.csv" ]; then
    echo "Error: CSV files not found in $INDEX_DIR/temp_csv/"
    echo "CSVs must be generated first. Run: ./scripts/index_kuzu.sh --fresh"
    exit 1
fi

echo "Found existing CSVs:"
ls -lh "$INDEX_DIR/temp_csv/"*.csv | head -10
echo ""

# Delete old database but keep CSVs
echo "Deleting old database..."
rm -f "$INDEX_DIR/kuzu.db"*
rm -f "$INDEX_DIR/documents.jsonl"
rm -f "$INDEX_DIR/doc_offsets.npy"

# Create progress file indicating CSVs are ready
echo "Creating progress checkpoint..."
cat > "$PROGRESS_FILE" << 'EOF'
{
    "schema_created": false,
    "phase1_csvs_created": true,
    "phase1_sentences": 0,
    "phase1_documents": 0,
    "phase1_roots": 0
}
EOF

echo "Progress file created: $PROGRESS_FILE"
echo ""

# Generate log filename
LOG_FILE="logs/reload_index_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

echo "Starting index reload from CSVs..."
echo "This will skip CSV generation and go straight to bulk loading."
echo "Log: $LOG_FILE"
echo ""

# Run index builder (will skip CSV generation due to progress flag)
python scripts/index_kuzu.py \
    --corpus data/enhanced_corpus/corpus_with_metadata.jsonl \
    --output "$INDEX_DIR" \
    --revo data/raw/eo/dictionaries/revo/revo_semantic_relations.json \
    --curated data/semantic_relations/curated_synonyms.json \
    --conceptnet data/external/conceptnet/conceptnet-assertions-5.7.0.csv.gz \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "============================================================"
echo "Reload Complete!"
echo "============================================================"
echo "Log: $LOG_FILE"
