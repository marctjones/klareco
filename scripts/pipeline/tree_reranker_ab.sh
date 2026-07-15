#!/bin/bash
# Steps 1+2 of the "both trees" reranker task, self-chaining (#713/#836).
#
#   1. FINISH the dependency-arc index (the interrupted build left it at ~47%,
#      no indexes). build_dependency_arcs DROPs and rebuilds from scratch — so
#      this is a clean full build, not a fragile resume.
#   2. A/B: run the reranker bench with J_tree_aware (both trees) against
#      I_clause_aware and the BM25 baseline, on:
#        - clause_discriminating_qa.jsonl  (PRIMARY — every answer subordinate,
#          median BM25 gold rank 15, zero at rank 1)
#        - synthetic_who_rebuild_50.jsonl  (CONTROL — main-clause answers; a
#          tree win here would mean the gain is generic, not clause/arc-specific)
#      Both --append-history to data/perf/bench_history.jsonl per THE MERGE GATE.
#
# The A/B number is PUNCTUATION-INDEPENDENT (arcs exclude punct, clauses have no
# punct frames), so measuring on the current pre-punctuation store is valid and
# the step-3 rebuild will not change it.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_ROOT"
if [ -d ".venv" ]; then source .venv/bin/activate
elif [ -d "venv" ]; then source venv/bin/activate
else echo "No venv found"; exit 1; fi

export KLARECO_ALLOW_DEGRADED=1
HIST="data/perf/bench_history.jsonl"
PRIMARY="data/test_sets/clause_discriminating_qa.jsonl"
CONTROL="data/test_sets/synthetic_who_rebuild_50.jsonl"
mkdir -p results

echo "════════════ 1/3  BUILD dependency_arcs (resume-safe) ════════════"
# --resume continues after the last sid already indexed, so a kill costs nothing.
# smaller --batch keeps each per-batch DataFrame ~4x smaller (OOM insurance).
python scripts/index/build_dependency_arcs.py --apply --resume --batch 50000

echo "════════════ 2/3  A/B — PRIMARY (subordinate answers) ════════════"
python scripts/eval/multi_reranker_bench.py \
  --test-set "$PRIMARY" \
  --output-summary results/ab_tree_primary.json \
  --append-history "$HIST"

echo "════════════ 3/3  A/B — CONTROL (main-clause answers) ════════════"
python scripts/eval/multi_reranker_bench.py \
  --test-set "$CONTROL" \
  --output-summary results/ab_tree_control.json \
  --append-history "$HIST"

echo
echo "  ✓ DONE. J_tree_aware vs I_clause_aware:"
echo "    primary: results/ab_tree_primary.json"
echo "    control: results/ab_tree_control.json"
