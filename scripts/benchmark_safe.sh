#!/bin/bash
#
# Safe benchmark runner - tests only memory-efficient retrievers
#
# This script runs benchmarks on slot-based retrievers WITHOUT the baseline
# retriever that loads the entire index into RAM.
#
# Tested retrievers (memory-safe):
#   - mmap:       Memory-mapped retriever (~500MB RAM)
#   - faiss:      FAISS-accelerated (~3-5GB RAM)
#   - multifaiss: Multi-index FAISS (~4-6GB RAM)
#   - sqlite:     SQLite backend (~1-2GB RAM)
#
# SKIPPED (to prevent system freeze):
#   - baseline:   Loads entire index into RAM (~30GB for 4.2M docs)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║            🛡️  SAFE SLOT RETRIEVER BENCHMARK 🛡️                    ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "This benchmark tests 4 MEMORY-EFFICIENT retrievers:"
echo ""
echo "  1️⃣  mmap       - Memory-mapped retriever"
echo "                  Memory: ~500MB  | Speed: Medium"
echo ""
echo "  2️⃣  faiss      - FAISS-accelerated retrieval"
echo "                  Memory: ~3-5GB  | Speed: Fast"
echo ""
echo "  3️⃣  multifaiss - Multi-FAISS (separate index per slot)"
echo "                  Memory: ~4-6GB  | Speed: Fastest"
echo ""
echo "  4️⃣  sqlite     - SQLite database backend"
echo "                  Memory: ~1-2GB  | Speed: Medium"
echo ""
echo "────────────────────────────────────────────────────────────────────"
echo "⚠️  BASELINE RETRIEVER IS SKIPPED"
echo "────────────────────────────────────────────────────────────────────"
echo "The baseline loads the entire index into RAM (~30GB for 4.2M docs)"
echo "and will freeze most systems. It is NOT included in this safe run."
echo ""
echo "════════════════════════════════════════════════════════════════════"
echo ""

# Pass all arguments to the main benchmark script
exec ./scripts/benchmark_all_retrievers.sh "$@"
