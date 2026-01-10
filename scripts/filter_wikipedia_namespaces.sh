#!/bin/bash
# Filter out talk/discussion/meta pages from Wikipedia corpus
#
# This removes non-article pages like:
# - Vikipedio: (Wikipedia meta pages)
# - Diskutejo: (Discussion/talk pages)
# - Uzanto: (User pages)
# - Ŝablono: (Template pages)
# - Kategorio: (Category pages)
# etc.
#
# Usage:
#   ./scripts/filter_wikipedia_namespaces.sh           # Creates filtered copy
#   ./scripts/filter_wikipedia_namespaces.sh --in-place  # Replaces original (with backup)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "ERROR: No virtual environment found"
    exit 1
fi

echo "=============================================="
echo "Filtering Wikipedia Namespace Pages"
echo "=============================================="
echo ""

# Run the filter script with all arguments passed through
python scripts/filter_wikipedia_namespaces.py "$@"

echo ""
echo "=============================================="
echo "Done!"
echo "=============================================="
