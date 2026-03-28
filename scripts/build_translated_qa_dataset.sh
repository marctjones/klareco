#!/bin/bash
#
# Build Translated QA Dataset
#
# Downloads TriviaQA, translates to Esperanto, filters for corpus coverage.
#
# Usage:
#   ./scripts/build_translated_qa_dataset.sh           # Default: 1000 questions
#   ./scripts/build_translated_qa_dataset.sh 5000      # Process 5000 questions
#   ./scripts/build_translated_qa_dataset.sh --skip-download  # Use existing download

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
    echo "No venv found"
    exit 1
fi

# Parse arguments
LIMIT=1000
SKIP_DOWNLOAD=""

for arg in "$@"; do
    if [[ "$arg" == "--skip-download" ]]; then
        SKIP_DOWNLOAD="--skip-download"
    elif [[ "$arg" =~ ^[0-9]+$ ]]; then
        LIMIT="$arg"
    fi
done

# Paths
SAMPLE_FILE="data/external/triviaqa_sample_${LIMIT}.jsonl"
OUTPUT_FILE="data/test_sets/translated_qa_diverse.jsonl"

echo "========================================"
echo "Translated QA Dataset Builder"
echo "========================================"
echo "Limit: $LIMIT questions"
echo "Sample: $SAMPLE_FILE"
echo "Output: $OUTPUT_FILE"
echo ""

# Set PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Step 1: Download TriviaQA sample
echo "Step 1/2: Downloading TriviaQA sample..."
python scripts/download_triviaqa_sample.py \
    --output "$SAMPLE_FILE" \
    --limit "$LIMIT" \
    $SKIP_DOWNLOAD

echo ""

# Step 2: Translate and filter
echo "Step 2/2: Translating and filtering..."
python scripts/translate_and_filter_qa.py \
    --input "$SAMPLE_FILE" \
    --output "$OUTPUT_FILE" \
    --limit "$LIMIT"

echo ""
echo "========================================"
echo "Complete!"
echo "========================================"
echo "Translated dataset saved to: $OUTPUT_FILE"
echo ""
echo "To evaluate on this dataset:"
echo "  python scripts/evaluate_extractive_qa.py --test-set $OUTPUT_FILE"
