#!/bin/bash
# ⚠️ EXTERNAL LLM - STOPGAP MEASURE ONLY
#
# This script uses EXTERNAL Claude Code LLM (interactive).
# This is a TEMPORARY FALLBACK. Use ./ask.sh (local model) instead.
#
# Usage:
#   ./ask-external.sh "Kiu estas Frodo?"              # Uses external LLM (interactive)
#   ./ask-external.sh "Kiu estas Frodo?" --translate  # With English translation
#   ./ask-external.sh "Kiu estas Frodo?" -k 10        # Use 10 context docs
#
# WARNING: This requires Claude Code to be running and you must respond interactively.
# PREFER: ./ask.sh (uses local QA Decoder, fully automatic)

python scripts/query_with_llm.py "$@"
