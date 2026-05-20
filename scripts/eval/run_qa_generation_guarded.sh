#!/usr/bin/env bash
#
# Guarded driver for build_synthetic_qa_test_set.py
#
# VERSION: v2.x (DuckDB)
# COMPATIBLE WITH: scripts/eval/build_synthetic_qa_test_set.py (post-2026-05-19 refactor)
# DEPENDENCIES: bash, tmux, nice, ionice (optional)
# STAGE: Evaluation
#
# Description:
#   Pre-flight-checks system resources, then runs the synthetic Q&A
#   generator for N question types in parallel (default 2 at a time) under
#   `nice -n 10` and `ionice -c 3`. Each run goes into its own tmux session
#   and logs to logs/. Heartbeats each log; flags wedged sessions where
#   the python process is gone but tmux is alive.
#
# Usage:
#   ./scripts/eval/run_qa_generation_guarded.sh                 # all 5 non-WHO
#   ./scripts/eval/run_qa_generation_guarded.sh kio kie         # subset
#   PARALLEL=1 ./scripts/eval/run_qa_generation_guarded.sh      # serial
#   MIN_FREE_GB=4 LOAD_MAX_FRAC=0.5 ./scripts/eval/run_qa_generation_guarded.sh
#
# Last Updated: 2026-05-19
# Author: Claude Code (with Marc Jones)

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

PARALLEL="${PARALLEL:-2}"
MIN_FREE_GB="${MIN_FREE_GB:-3}"      # Refuse to launch a slot if free RAM below this
LOAD_MAX_FRAC="${LOAD_MAX_FRAC:-0.7}" # Refuse if 1-min load avg > cores * this
TARGET_SIZE="${TARGET_SIZE:-50}"
SEED="${SEED:-42}"
HEARTBEAT_SEC="${HEARTBEAT_SEC:-30}"
SILENT_DEAD_SEC="${SILENT_DEAD_SEC:-120}"  # log silent + python gone => wedged
PROTECT_PIDS="${PROTECT_PIDS:-}"      # Comma-separated PIDs whose death aborts us

DEFAULT_TYPES=(kio kie kiam kial kiel)
if [[ $# -eq 0 ]]; then
    TYPES=("${DEFAULT_TYPES[@]}")
else
    TYPES=("$@")
fi

# -----------------------------------------------------------------------------
# Pre-flight
# -----------------------------------------------------------------------------
echo "=== Pre-flight ==="
CORES="$(nproc)"
LOAD_NOW="$(awk '{print $1}' /proc/loadavg)"
FREE_GB="$(free -g | awk '/^Mem:/ {print $7}')"
LOAD_MAX="$(awk -v c="${CORES}" -v f="${LOAD_MAX_FRAC}" 'BEGIN{printf "%.2f", c*f}')"

printf "  cores=%s  load_1m=%s  load_max=%s  free_ram=%sG  min_free=%sG  parallel=%s\n" \
    "${CORES}" "${LOAD_NOW}" "${LOAD_MAX}" "${FREE_GB}" "${MIN_FREE_GB}" "${PARALLEL}"

if awk -v a="${LOAD_NOW}" -v b="${LOAD_MAX}" 'BEGIN{exit !(a>b)}'; then
    echo "  ✗ refusing to launch: load_1m=${LOAD_NOW} > load_max=${LOAD_MAX}" >&2
    echo "    (override with LOAD_MAX_FRAC=<higher>, or wait for load to drop)" >&2
    exit 1
fi
if [[ "${FREE_GB}" -lt "${MIN_FREE_GB}" ]]; then
    echo "  ✗ refusing to launch: free_ram=${FREE_GB}G < min_free=${MIN_FREE_GB}G" >&2
    echo "    (override with MIN_FREE_GB=<lower>, or close other processes)" >&2
    exit 1
fi
echo "  ✓ resources OK"

if ! command -v tmux >/dev/null; then echo "tmux required"; exit 1; fi
if [[ ! -d ".venv" ]];  then echo "expected .venv/ at project root"; exit 1; fi
if [[ ! -f "scripts/eval/build_synthetic_qa_test_set.py" ]]; then
    echo "generator script missing"; exit 1
fi

mkdir -p logs

# -----------------------------------------------------------------------------
# Launch a single type in tmux
# -----------------------------------------------------------------------------
launch_one() {
    local type="$1"
    local ts; ts="$(date +%Y%m%d_%H%M%S)"
    local log="logs/gen_${type}_${ts}.log"
    local session="gen-${type}"

    # ionice may not be installed; fall back to nice-only
    local nice_prefix="nice -n 10"
    if command -v ionice >/dev/null; then
        nice_prefix="ionice -c 3 nice -n 10"
    fi

    tmux new-session -d -s "${session}" \
"source .venv/bin/activate && \
echo '=== ${session} started '\$(date)' ===' | tee '${log}' && \
${nice_prefix} python -u scripts/eval/build_synthetic_qa_test_set.py \
    --type '${type}' --target-size ${TARGET_SIZE} --seed ${SEED} 2>&1 \
  | tee -a '${log}'; \
EXITCODE=\${PIPESTATUS[0]}; \
echo '=== ${session} finished exit='\${EXITCODE}' '\$(date)' ===' | tee -a '${log}'; \
sleep 1"

    echo "${session}|${log}"
}

# -----------------------------------------------------------------------------
# Wait for one launched session to finish (tmux session terminates).
# Heartbeat the log; flag wedged if python is gone but tmux lingers.
# -----------------------------------------------------------------------------
wait_one() {
    local session="$1" log="$2"
    local last_size=0 silent_since=0
    while tmux has-session -t "${session}" 2>/dev/null; do
        sleep "${HEARTBEAT_SEC}"
        local size; size="$(stat -c%s "${log}" 2>/dev/null || echo 0)"
        local now; now="$(date +%s)"
        if [[ "${size}" -ne "${last_size}" ]]; then
            last_size="${size}"
            silent_since="${now}"
            printf "  [%s] %s growing (%s bytes)\n" "$(date +%H:%M:%S)" "${session}" "${size}"
        else
            local silent_for=$(( now - silent_since ))
            local py_alive=0
            if pgrep -f "build_synthetic_qa_test_set.py --type ${session#gen-}" >/dev/null; then
                py_alive=1
            fi
            if [[ "${py_alive}" -eq 0 && "${silent_for}" -gt "${SILENT_DEAD_SEC}" ]]; then
                printf "  [%s] %s WEDGED: log silent %ss + python process gone — killing tmux\n" \
                    "$(date +%H:%M:%S)" "${session}" "${silent_for}"
                tmux kill-session -t "${session}" 2>/dev/null || true
                return 1
            fi
        fi
    done
    return 0
}

# -----------------------------------------------------------------------------
# Run in batches of $PARALLEL
# -----------------------------------------------------------------------------
echo
echo "=== Generating ${#TYPES[@]} type(s): ${TYPES[*]} (parallel=${PARALLEL}) ==="

# Re-check pre-flight + protected PIDs between batches; abort if any breached.
check_safe_to_continue() {
    local free_now load_now
    free_now="$(free -g | awk '/^Mem:/ {print $7}')"
    load_now="$(awk '{print $1}' /proc/loadavg)"
    if [[ "${free_now}" -lt "${MIN_FREE_GB}" ]]; then
        echo "  ✗ ABORTING remaining batches: available RAM ${free_now}G < ${MIN_FREE_GB}G" >&2
        return 1
    fi
    if awk -v a="${load_now}" -v b="${LOAD_MAX}" 'BEGIN{exit !(a>b)}'; then
        echo "  ✗ ABORTING remaining batches: load_1m ${load_now} > ${LOAD_MAX}" >&2
        return 1
    fi
    if [[ -n "${PROTECT_PIDS}" ]]; then
        local IFS=','
        for pid in ${PROTECT_PIDS}; do
            if [[ -n "${pid}" ]] && ! kill -0 "${pid}" 2>/dev/null; then
                echo "  ✗ ABORTING remaining batches: protected PID ${pid} is gone" >&2
                return 1
            fi
        done
    fi
    return 0
}

declare -a OK FAIL SKIPPED
i=0
while [[ $i -lt ${#TYPES[@]} ]]; do
    if ! check_safe_to_continue; then
        for ((k=i; k<${#TYPES[@]}; k++)); do SKIPPED+=("gen-${TYPES[$k]}"); done
        break
    fi
    batch=()
    for ((j=0; j<PARALLEL && i<${#TYPES[@]}; j++, i++)); do
        info="$(launch_one "${TYPES[$i]}")"
        echo "launched ${info%%|*} -> ${info##*|}"
        batch+=("${info}")
    done

    for info in "${batch[@]}"; do
        session="${info%%|*}"
        log="${info##*|}"
        if wait_one "${session}" "${log}"; then
            OK+=("${session}")
        else
            FAIL+=("${session}")
        fi
    done
done

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo
echo "=== Done ==="
echo "  succeeded (tmux exited cleanly): ${OK[*]:-none}"
echo "  failed (wedged/killed):          ${FAIL[*]:-none}"
echo "  skipped (resource breach):       ${SKIPPED[*]:-none}"
echo
echo "Outputs:"
ls -la data/test_sets/synthetic_*_active.jsonl 2>/dev/null
echo
echo "Tail each log:"
for t in "${TYPES[@]}"; do
    L="$(ls -t logs/gen_${t}_*.log 2>/dev/null | head -1)"
    [[ -n "${L}" ]] && echo "  tail -50 ${L}"
done
