#!/bin/bash
#
# Build capability_100.jsonl — the frozen capability test set.
#
# VERSION: v2.x
# COMPATIBLE WITH: DuckDB store + Whoosh v2 index; R1-R17 quality standard
# DEPENDENCIES: build_synthetic_qa_test_set.py, audit_qa_pairs.py,
#               audit_discriminability.py, audit_language_quality.py
# STAGE: Eval
#
# This is the ruler. Nothing else in milestone #14 can be measured until it
# exists. It is built under the FULL gate stack, and in particular under the
# two constraints that the previous sets lacked:
#
#   R16 (#778) — the gold passage must NOT already be at BM25 rank 1.
#                Without this, BM25 has already won and no reranker can move
#                the number. synthetic_who_rebuild_17_cleanish was 58.8%
#                rank-1, which is why all nine rerankers tied.
#
#   R17 (#783) — every pair carries a `gold_answer_span`, so extraction is
#                scored against the exact answer rather than by substring
#                containment over a whole sentence. Without this, extraction
#                improvements cannot be attributed to extraction — and
#                decomposable attribution is the thesis.
#
# Long-running (streams the 5.4M-sentence store per question type). Run it in
# its own terminal; do not run it from an agent session.
#
# Usage:
#     ./scripts/eval/build_capability_100.sh
#     ./scripts/eval/build_capability_100.sh --per-type 40   # bigger pool
#
# Output:
#     data/test_sets/synthetic_<type>_active.jsonl   (one per question type)
#     logs/build_capability_100_<timestamp>.log
#
# After it finishes, merge the per-type files into capability_100.jsonl under
# the R15 distribution (KIO 25 / KIU 25 / KIE 15 / KIAM 15 / rest 20), then run
# the Stage 1-2 audits printed at the end.
#
# Last Updated: 2026-07-13
# Related Issues: #778, #783, #737
# See Also: docs/QA_TEST_SET_QUALITY_STANDARD.md
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "No .venv found — create one first (see README)."; exit 1
fi

PER_TYPE=30
if [ "$1" == "--per-type" ]; then
    PER_TYPE="$2"
fi

mkdir -p logs data/test_sets
LOG="logs/build_capability_100_$(date +%Y%m%d_%H%M%S).log"

# R15 target distribution for a 100-pair capability set. We over-generate per
# type (the merge step samples down), because the R16 ceiling now rejects the
# easy pairs and yield per type will be lower than it used to be.
TYPES="kiu kio kie kiam"

echo "Building capability set under the full R1-R17 gate stack." | tee -a "$LOG"
echo "  R16 ceiling ENFORCED: pairs whose gold passage is already at BM25" | tee -a "$LOG"
echo "  rank 1 are DROPPED as trivial. Expect a lower yield than before —" | tee -a "$LOG"
echo "  that is the point. Those pairs could never measure reranking." | tee -a "$LOG"
echo "" | tee -a "$LOG"

for T in $TYPES; do
    echo "=== $T ===" | tee -a "$LOG"
    python scripts/eval/build_synthetic_qa_test_set.py \
        --type "$T" \
        --target-size "$PER_TYPE" \
        --gate-top-k 50 \
        2>&1 | tee -a "$LOG"
    echo "" | tee -a "$LOG"
done

echo "==========================================================" | tee -a "$LOG"
echo "Generation done. Now VERIFY headroom before trusting the set:" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "  python scripts/eval/audit_discriminability.py \\" | tee -a "$LOG"
echo "      data/test_sets/synthetic_*_active.jsonl \\" | tee -a "$LOG"
echo "      --top-k 50 --rank-histogram" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "  Every set must report '✅ Has headroom'. A set that reports" | tee -a "$LOG"
echo "  '❌ SATURATED' cannot measure reranking and must be regenerated." | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "  python scripts/eval/audit_qa_pairs.py --in <set>.jsonl --strict" | tee -a "$LOG"
echo "  python scripts/eval/audit_language_quality.py --in <set>.jsonl --strict" | tee -a "$LOG"
echo "==========================================================" | tee -a "$LOG"
echo "Log: $LOG"
