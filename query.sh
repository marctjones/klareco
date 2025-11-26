#!/bin/bash
# Simple wrapper for quick_query.py - Pure Esperanto focus
#
# Usage:
#   ./query.sh "Kiu estas Frodo?"              # Pure Esperanto
#   ./query.sh "Kiu estas Frodo?" --translate  # With English translations
#   ./query.sh "Kiu estas Frodo?" --show-stage1  # Show keyword filtering

python scripts/quick_query.py "$@"
