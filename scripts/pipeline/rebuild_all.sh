#!/bin/bash
# THE REBUILD (#835). One command, in the right order, with a HARD verification gate.
#
# The store has been untouched since 2026-06-15 — before ANY of the parser work.
# Every fix lives in klareco/ and NONE of it lives in the data:
#
#     redirect stubs present       151,353  (2.81%)
#     propra_nomo SUBJECTS       2,254,494  (41.82%)   <- implausible
#     clause table               does not exist
#     ontology_nodes / edges     0 rows
#     verb_klaso populated       0.0%
#     stored AST vs fresh parse  53.4% agreement       <- the store and the code
#                                                          are different systems
#
# THE PARSE IS ~15 MINUTES, NOT 5-6 HOURS. Measured 2026-07-14: 7,384 sentences/sec
# on one core, zero crashes. The old figure predates the Kuzu removal
# (KuzuASTReconstructor took ~17,000 ms PER AST). The wall clock here is I/O and
# INDEXING — a 20 GB JSONL and a Whoosh index over ~4.6M documents.
#
# DISK. This is the constraint, and the ORDER below exists because of it:
#   store 32 GB + whoosh 2.8 GB, and only ~52 GB free. Building the new store
#   BESIDE the old one does not fit. So we PARSE FIRST — producing the JSONL,
#   which is the store's only input — and only THEN drop the old store. If the
#   build fails afterwards, the JSONL is still there and the retry is cheap.
#
#   This is safe precisely BECAUSE the parse is 15 minutes. CLAUDE.md marks the
#   corpus JSONL "never delete — re-parse is ~5-6 hours"; that rule was written
#   against the stale figure and it is what made the store feel un-rebuildable.
#
# Usage:
#   ./scripts/pipeline/rebuild_all.sh
#   ./scripts/pipeline/rebuild_all.sh --skip-parse   # reuse the existing JSONL
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

CORPUS="data/enhanced_corpus/corpus_with_metadata.jsonl"
STORE="data/indexes/duckdb_store.db"
WHOOSH="data/indexes/whoosh_v2"

step() { echo; echo "════════════════════════════════════════════════════════"; echo "  $*"; echo "════════════════════════════════════════════════════════"; }
free_gb() { df -g . | tail -1 | awk '{print $4}'; }

step "0/8  DISK — refuse to start without headroom"
echo "  free: $(free_gb) GB"
if [ "$(free_gb)" -lt 40 ]; then
  echo "  ✗ need >= 40 GB free. A half-written store is worse than a stale one."
  exit 1
fi

step "1/8  ACQUIRE the curated lexicons (ReVo / voko-akrido)"
python scripts/acquire/acquire_voko_akrido.py
python scripts/acquire/acquire_revo_ontology.py

step "2/8  SURFACE LEXICAL FACTS — protected_roots + capitalization_ratio"
# Reads RAW TEXT only. MUST precede the parse, or the parser runs on artifacts
# built from a sample.
python scripts/index/build_surface_lexical_facts.py

step "3/8  ROOT LEXICON — ReVo-first, tiered, NAME roots held out"
python scripts/index/build_root_lexicon.py

step "4/8  PREFLIGHT — fail loudly, never degrade"
# The June migration cost weeks because missing artifacts logged a warning and
# carried on. Runs AFTER the artifacts are built, obviously.
# ARTIFACTS ONLY. Running the full preflight here would be backwards — it
# validates the very store we are about to replace, and so fails on exactly the
# things this rebuild exists to fix. The store's gate is validate_rebuild.py,
# and it runs at the END.
python -c "
from klareco.preflight import preflight_artifacts
preflight_artifacts()
print('  parser artifacts OK')
"

if [ "$SKIP_PARSE" -eq 0 ]; then
  step "5/8  PARSE — ~15 min of CPU; the wall clock is writing a 20 GB JSONL"
  ./scripts/parse/parse_corpus.sh --fresh
else
  step "5/8  PARSE — SKIPPED (--skip-parse)"
fi
[ -s "$CORPUS" ] || { echo "  ✗ no corpus at $CORPUS — cannot build the store."; exit 1; }
echo "  corpus: $(du -h "$CORPUS" | cut -f1)"

step "6/8  DROP THE OLD STORE — the JSONL above is its only input"
# We do this AFTER the parse, so a failed build can be retried from the JSONL
# without re-parsing. Disk does not allow building the new store beside the old.
echo "  freeing $(du -sh "$STORE" 2>/dev/null | cut -f1) + $(du -sh "$WHOOSH" 2>/dev/null | cut -f1)"
rm -rf "$STORE" "$WHOOSH"
echo "  free now: $(free_gb) GB"

step "7/8  STORE — the QUALITY GATE runs here and REPORTS its rejections"
# ~14.5% dropped as junk (redirects, English, wiki markup). A gate that drops rows
# SILENTLY is how 151,353 redirect stubs went unnoticed for a month.
python scripts/index/build_duckdb_store.py --skip-whoosh

step "7b   CLAUSE TABLE + ONTOLOGY — read ast_json, NO reparse"
python scripts/index/build_clause_table.py
python scripts/index/load_ontology.py

step "7c   DEPENDENCY ARCS — index the TREE ITSELF (#713/#836)"
# The store DROP above takes dependency_arcs with it, and nothing else rebuilds
# it — leaving J_tree_aware silently reading an empty table. Rebuild it here so
# the store is internally consistent: every ast_json has its arcs indexed.
python scripts/index/build_dependency_arcs.py --apply

step "8/8  WHOOSH — BM25 over the CLEAN store"
python scripts/index/rebuild_whoosh_from_duckdb.py

step "VERIFY — a column can be 100% POPULATED and carry ZERO information"
python scripts/validate/validate_rebuild.py || {
  echo
  echo "  ✗ VERIFICATION FAILED. DO NOT SHIP THIS STORE."
  echo "    The corpus JSONL survives — fix the cause and re-run with --skip-parse."
  exit 1
}

echo
echo "  ✓ REBUILD COMPLETE AND VERIFIED.   log: $LOG"
echo "  free: $(free_gb) GB"
echo
echo "  The reranker A/B (#713) has been blocked since June. It is now unblocked."
