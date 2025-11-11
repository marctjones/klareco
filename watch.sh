#!/bin/bash
# Watch script for monitoring Klareco logs in real-time
# Run this in a separate terminal window while run.sh is executing

LOG_FILE="klareco.log"

# Create log file if it doesn't exist
touch "$LOG_FILE"

# Display header
echo "=========================================="
echo "  Watching Klareco Logs (klareco.log)"
echo "=========================================="
echo "Press Ctrl+C to exit"
echo ""

# Use tail -f to follow the log file in real-time
# -n 50: Show last 50 lines initially
# -f: Follow mode (keep watching for new lines)
tail -n 50 -f "$LOG_FILE"
