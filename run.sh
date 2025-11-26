#!/bin/bash
#
# This script runs the dataset synthesis process for the new Graph2Seq model.
# It is designed to be run in the background and can be restarted. If the
# output file already contains data, it will resume generating from where it
# left off.

# Ensure we are in the project root directory
cd "$(dirname "$0")"

# Create the logs directory if it doesn't exist
mkdir -p logs

LOG_FILE="logs/dataset_generation.log"

# Activate conda environment (best effort)
# This makes the script more portable for different setups.
if [ -d "env" ]; then
    source env/bin/activate
elif command -v conda &> /dev/null && conda env list | grep -q 'klareco-env'; then
    conda activate klareco-env
fi

echo "Starting dataset generation for 50,000 examples..."
echo "Progress will be logged to: $LOG_FILE"
echo "This script can be stopped and restarted without losing progress."

# Run the dataset generation script.
# The script will automatically resume if the output file already exists.
# All output (stdout and stderr) is redirected to the log file.
python3 scripts/create_synthesis_dataset.py --max-examples 50000 > "$LOG_FILE" 2>&1

echo "Dataset generation script finished. Check log file for details."
