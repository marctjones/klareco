#!/bin/bash
# This script runs a given command inside the 'klareco-env' conda environment.
# Example usage: ./run.sh python3 scripts/clean_corpus.py

# Check if a command was provided
if [ -z "$1" ]; then
    echo "Usage: ./run.sh <command_to_run>"
    exit 1
fi

# Execute the command within the conda environment
conda run -n klareco-env "$@"
