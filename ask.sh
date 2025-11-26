#!/bin/bash
# Ask questions using LOCAL QA Decoder (no external LLM needed)
#
# This is the PREFERRED method - uses your trained local model.
#
# Usage:
#   ./ask.sh "Kiu estas Frodo?"              # Fully automatic, local
#   ./ask.sh "Kiu estas Frodo?" --translate  # With English translation
#   ./ask.sh "Kiu estas Frodo?" -k 10        # Use 10 context docs

python scripts/query_with_local_model.py "$@"
