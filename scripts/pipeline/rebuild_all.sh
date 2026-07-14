#!/bin/bash
# THE REBUILD (#835). One command, in the right order, with a HARD verification gate.
#
# The store has been untouched since 2026-06-15 — before ANY of the parser work.
# Every fix lives in klareco/ and NONE of it lives in the data:
#
#     redirect stubs still present   151,353  (2.81%)
#     propra_nomo SUBJECTS         2,254,494  (41.82%)   <- implausible
#     clause table                 does not exist
#     ontology_nodes / edges       0 rows
#     verb_klaso populated         0.0%
#
# THE PARSE IS ~15 MINUTES, NOT 5-6 HOURS. Measured 2026-07-14: 7,384 sentences/sec
# on one core, zero crashes. The old figure predates the Kuzu removal
# (KuzuASTReconstructor took ~17,000 ms PER AST). The wall clock here is I/O and
# INDEXING — writing a 20 GB JSONL and building Whoosh over ~4.6M documents.
#
# Usage:
#   ./scripts/pipeline/rebuild_all.sh              # everything
#   ./scripts/pipeline/rebuild_all.sh --skip-parse # reuse the existing corpus JSONL
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_ROOT"

if [ -d ".venv" ]; then source .venv/bin/activate
elif [ -d "venv" ]; then source venv/bin/activate
else echo "No venv found"; exit 1; fi

mkdir -p logs
LOG="logs/rebuild_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG") 2>&1

SKIP_PARSE=0
[[ "$1" == "--skip-parse" ]] && SKIP_PARSE=1

step() { echo; echo "════════════════════════════════════════════════════════════"; echo "  $*"; echo "════════════════════════════════════════════════════════════"; }

step "0/7  PREFLIGHT — fail loudly, never degrade"
python -c "
from klareco.preflight import run_preflight
run_preflight()
print('  preflight OK')
" || { echo "  PREFLIGHT FAILED — refusing to rebuild on missing artifacts."; exit 1; }

step "1/7  ACQUIRE the curated lexicons (ReVo / voko-akrido)"
python scripts/acquire/acquire_voko_akrido.py
python scripts/acquire/acquire_revo_ontology.py

step "2/7  SURFACE LEXICAL FACTS — protected_roots + capitalization_ratio"
# Reads RAW TEXT only. Must precede the parse, or the parser runs on sampled artifacts.
python scripts/index/build_surface_lexical_facts.py

step "3/7  ROOT LEXICON — ReVo-first, tiered, name roots held out"
python scripts/index/build_root_lexicon.py

if [ "$SKIP_PARSE" -eq 0 ]; then
step "4/7  PARSE — ~15 min of CPU; the wall clock is writing a 20 GB JSONL"
  ./scripts/parse/parse_corpus.sh --fresh
else
  step "4/7  PARSE — SKIPPED (--skip-parse)"
fi

step "5/7  STORE — the QUALITY GATE runs here and REPORTS its rejections"
# ~14.5% dropped as junk (redirects, English, wiki markup). A gate that drops rows
# silently is how 151,353 redirect stubs went unnoticed.
python scripts/index/build_duckdb_store.py

step "6/7  CLAUSE TABLE + ONTOLOGY — from ast_json, no reparse"
python scripts/index/build_clause_table.py
python scripts/index/load_ontology.py

step "7/7  WHOOSH — BM25 over the CLEAN store"
python scripts/index/rebuild_whoosh_from_duckdb.py

step "VERIFY — a column can be 100% populated and carry ZERO information"
python scripts/validate/validate_rebuild.py || {
  echo
  echo "  ✗ VERIFICATION FAILED. DO NOT SHIP THIS STORE."
  exit 1
}

echo
echo "  ✓ REBUILD COMPLETE AND VERIFIED.  log: $LOG"
echo
echo "  Next: the reranker A/B (#713) has been blocked since June. It is now unblocked."
echo "        python scripts/eval/multi_reranker_bench.py --test-set data/test_sets/..."
