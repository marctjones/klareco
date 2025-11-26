#!/bin/bash
#
# Main user-facing script to ask a question to the Klareco system.
#
# This script runs the full AST-Native RAG pipeline.

# Ensure we are in the script's directory
cd "$(dirname "$0")"/..

# Check if a question was provided
if [ -z "$1" ]; then
  echo "Usage: $0 \"<your question in Esperanto>\""
  exit 1
fi

# Activate conda environment if it exists
# This makes the script more portable
if [ -d "env" ]; then
    source env/bin/activate
elif command -v conda &> /dev/null; then
    # Attempt to activate a known conda env name if it exists
    # This is a fallback and might need user configuration
    conda activate klareco-env &> /dev/null
fi

# Run the main pipeline
python3 scripts/run_pipeline.py "$1"
