#!/bin/bash
#
# cleanup_stale.sh — single entry point for safe disk-reclamation.
#
# VERSION: v1.0
# COMPATIBLE WITH: project root at /home/marc/Projects/klareco
# STAGE: Utility / Maintenance
#
# Description:
#   Removes regenerable artifacts that accumulate during normal work
#   (per-run logs, applied staging files, old bench outputs). Safe to
#   run weekly via cron or before any large operation that needs disk
#   headroom.
#
#   Categories of deletion, each with a documented rationale:
#     - logs/  per-run log files older than --log-days (default 30)
#     - results/  per-run bench output JSON/JSONL older than --results-days (default 60)
#     - data/staging/  staging files whose work has been applied to DB
#     - tmp/  duck pid lockfiles, .tmp checkpoints
#
# What is NEVER deleted:
#   - data/raw/         (source data — hard to re-acquire)
#   - data/dictionaries (ReVo, etc.)
#   - data/cleaned/     (cleaned text — feeds extraction)
#   - data/extracted/   (extracted sentences — feeds parsing)
#   - data/corpus/      (parsed corpus — feeds indexing)
#   - data/enhanced_corpus/   (same)
#   - data/indexes/     (the live DB and indexes)
#   - models/           (trained checkpoints)
#   - data/vocabularies (root/affix vocab)
#   - data/proper_nouns_dynamic*.json   (v3 + fallbacks v2/v1)
#
# Usage:
#   ./scripts/util/cleanup_stale.sh                # dry-run, report what would be deleted
#   ./scripts/util/cleanup_stale.sh --apply        # actually delete
#   ./scripts/util/cleanup_stale.sh --apply --log-days 14 --results-days 30
#
# Last Updated: 2026-05-26
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

APPLY=0
LOG_DAYS=30
RESULTS_DAYS=60
for arg in "$@"; do
    case "$arg" in
        --apply)            APPLY=1 ;;
        --log-days)         shift; LOG_DAYS="$1" ;;
        --log-days=*)       LOG_DAYS="${arg#*=}" ;;
        --results-days)     shift; RESULTS_DAYS="$1" ;;
        --results-days=*)   RESULTS_DAYS="${arg#*=}" ;;
        -h|--help)
            sed -n '3,30p' "$0"; exit 0 ;;
    esac
done

if [ "$APPLY" -eq 0 ]; then
    echo "=== DRY RUN — no files deleted. Add --apply to actually delete. ==="
    RM="echo would-delete:"
    RMDIR="echo would-rmdir:"
else
    RM="rm -f"
    RMDIR="rm -rf"
fi
echo "Project: $PROJECT_ROOT"
df -h / | head -2
echo ""

count_size() {
    local n=0 bytes=0
    for f in "$@"; do
        [ -e "$f" ] || continue
        n=$((n + 1))
        bytes=$((bytes + $(stat -c %s "$f" 2>/dev/null || echo 0)))
    done
    local mb=$((bytes / 1024 / 1024))
    echo "    ($n files, ${mb} MB)"
}

# ---------- 1. Old per-run log files ----------
echo "[1] logs/ — *.log older than ${LOG_DAYS} days"
# ⚠️ NO `mapfile`. macOS ships bash 3.2 (2007) — `mapfile` is a bash 4 builtin, so
# every array here failed with "mapfile: command not found" followed by an unbound-
# variable abort. The MAINTENANCE TOOLKIT DID NOT RUN ON THE MACHINE IT WAS FOR,
# which is how the disk filled to zero during a rebuild with nobody able to clean up.
# `while read` is POSIX and works everywhere.
OLD_LOGS=()
while IFS= read -r f; do OLD_LOGS+=("$f"); done < <(find logs/ -type f -name '*.log' -mtime "+${LOG_DAYS}" 2>/dev/null)
count_size ${OLD_LOGS[@]+${OLD_LOGS[@]+"${OLD_LOGS[@]}"}}
for f in ${OLD_LOGS[@]+${OLD_LOGS[@]+"${OLD_LOGS[@]}"}}; do $RM "$f"; done
# Empty leftover dirs
if [ "$APPLY" -eq 1 ]; then
    find logs/ -type d -empty -delete 2>/dev/null || true
fi

# ---------- 2. Old per-run bench / eval results ----------
echo ""
echo "[2] results/ — *.json{,l} older than ${RESULTS_DAYS} days"
OLD_RES=()
while IFS= read -r f; do OLD_RES+=("$f"); done < <(find results/ -type f \( -name '*.json' -o -name '*.jsonl' \) -mtime "+${RESULTS_DAYS}" 2>/dev/null)
count_size ${OLD_RES[@]+${OLD_RES[@]+"${OLD_RES[@]}"}}
for f in ${OLD_RES[@]+${OLD_RES[@]+"${OLD_RES[@]}"}}; do $RM "$f"; done

# ---------- 3. Applied staging files ----------
# These are regenerable from the DB and the build scripts should clean
# up after themselves, but historically haven't. List them explicitly
# so a future contributor knows what's safe to drop.
echo ""
echo "[3] data/staging/ — applied build outputs"
APPLIED=(
    "data/staging/entity_postings.jsonl"
    "data/staging/subj_propranoma_kat_backfill.jsonl"
    # Add new applied-staging files here as they accumulate.
)
PRESENT=()
for f in ${APPLIED[@]+"${APPLIED[@]}"}; do
    [ -f "$f" ] && PRESENT+=("$f")
done
count_size ${PRESENT[@]+"${PRESENT[@]}"}
for f in ${PRESENT[@]+"${PRESENT[@]}"}; do $RM "$f"; done

# ---------- 4. Leftover .tmp checkpoints ----------
echo ""
echo "[4] *.tmp checkpoint files (orphaned atomic-write scratch)"
OLD_TMP=()
while IFS= read -r f; do OLD_TMP+=("$f"); done < <(find . -type f -name '*.tmp' -mmin +60 \
    -not -path './.git/*' -not -path './.venv/*' 2>/dev/null)
count_size ${OLD_TMP[@]+${OLD_TMP[@]+"${OLD_TMP[@]}"}}
for f in ${OLD_TMP[@]+${OLD_TMP[@]+"${OLD_TMP[@]}"}}; do $RM "$f"; done

# ---------- 5. /tmp claude-1000 buffer files for THIS project ----------
echo ""
echo "[5] "${TMPDIR:-/tmp}"/claude-*/-*klareco/* older than 7 days"
OLD_TASKS=()
while IFS= read -r f; do OLD_TASKS+=("$f"); done < <(find "${TMPDIR:-/tmp}"/claude-*/-*klareco \
    -type f -mtime +7 2>/dev/null)
count_size ${OLD_TASKS[@]+${OLD_TASKS[@]+"${OLD_TASKS[@]}"}}
for f in ${OLD_TASKS[@]+${OLD_TASKS[@]+"${OLD_TASKS[@]}"}}; do $RM "$f"; done

echo ""
echo "=== Done ==="
df -h / | head -2
