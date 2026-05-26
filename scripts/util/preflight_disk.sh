#!/bin/bash
#
# preflight_disk.sh — refuse to proceed if the filesystem doesn't have
# enough free space for the upcoming operation.
#
# VERSION: v1.0
# COMPATIBLE WITH: any caller that sources or invokes this
# STAGE: Utility / Preflight
#
# Usage:
#   # As a guard at the top of a long-running script:
#   ./scripts/util/preflight_disk.sh 50    # require >= 50 GB free
#   # Exits non-zero on insufficient space; prints what's there.
#
#   # Source it and use the function:
#   source scripts/util/preflight_disk.sh
#   require_disk_gb 50 "DuckDB rebuild needs working space"
#
# Why this exists:
#   Several big scripts (parse_corpus.sh, build_duckdb_resilient.sh,
#   add_aliaj_flag_columns.py, the Whoosh optimize) can wedge the
#   filesystem partway through. This guard makes them refuse to start
#   instead of running for an hour and crashing on ENOSPC.
#
# Last Updated: 2026-05-26
set -u

require_disk_gb() {
    local min_gb="$1"
    local reason="${2:-}"
    local avail_kb
    avail_kb=$(df -k / | awk 'NR==2 {print $4}')
    local avail_gb=$(( avail_kb / 1024 / 1024 ))

    if [ "$avail_gb" -lt "$min_gb" ]; then
        echo "preflight_disk: FAIL — only ${avail_gb} GB free, need ${min_gb} GB" >&2
        [ -n "$reason" ] && echo "  reason: $reason" >&2
        echo "  See scripts/util/cleanup_stale.sh for quick space recovery." >&2
        return 1
    fi
    echo "preflight_disk: ${avail_gb} GB free (need ${min_gb} GB) — OK"
    return 0
}

# When invoked as a script (not sourced), expect a single numeric arg
# and behave as a check.
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    if [ "$#" -lt 1 ]; then
        echo "usage: $0 <min_gb> [reason]" >&2
        exit 2
    fi
    require_disk_gb "$@" || exit 1
fi
