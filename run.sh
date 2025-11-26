#!/bin/bash
#
# This script runs the dataset synthesis process for the new Graph2Seq model.
# It is designed to be run in the background and can be restarted. If the
# output file already contains data, it will resume generating from where it
# left off.

# Create the logs directory first, so we can log to it.
mkdir -p logs

# Redirect all output from this script to a log file, while also showing it in the terminal.
exec > >(tee -a logs/run_script.log) 2>&1

set -x # Print each command to stderr before executing.

# --- Script Body ---

echo "Script started."

# Ensure we are in the project root directory
cd "$(dirname "$0")"
echo "Changed directory to: $(pwd)"


# Activate conda environment (best effort)
echo "Attempting to activate conda environment..."
if [ -d "env" ]; then
    source env/bin/activate
elif command -v conda &> /dev/null && conda env list | grep -q 'klareco-env'; then
    # We add a check and an error message instead of relying on 'set -e'
    conda activate klareco-env || echo "WARNING: Conda activation failed with exit code $?. Proceeding anyway."
fi

echo "Conda environment activation attempted. Proceeding..."

# Define the log file for the python script
DATASET_LOG_FILE="logs/dataset_generation.log"

echo "Starting dataset generation for 50,000 examples..."
echo "Python script output will be logged to: $DATASET_LOG_FILE"

# Run the dataset generation script
# We redirect its specific output to its own log file.
python3 scripts/create_synthesis_dataset.py --max-examples 50000 > "$DATASET_LOG_FILE" 2>&1

echo "Dataset generation script finished."
