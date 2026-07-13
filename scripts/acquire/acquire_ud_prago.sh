#!/bin/bash
#
# Acquire UD_Esperanto-Prago — the parser's only EXTERNAL gold ruler.
#
# VERSION: v1.0
# COMPATIBLE WITH: scripts/eval/eval_ud_prago.py
# DEPENDENCIES: curl
# STAGE: Acquire
#
# WHY THIS SCRIPT EXISTS
# ----------------------
# This treebank was lost in the June 2026 laptop migration and stayed lost for a
# month — even though it is public, CC-BY-SA 4.0, and a single curl away.
#
# It was lost because it had no acquire script. Every other corpus in this
# project has one; this did not, so when `data/` was wiped there was nothing to
# re-run and nobody noticed the parser's only independent ruler was gone. The
# 80.3% / 93.3% figures quoted in DESIGN.md sat there for a month, unreproducible.
#
# The rule this establishes (#817):
#
#     If it lives under data/ and the pipeline needs it, it needs a script that
#     regenerates it from a source we do not control. Otherwise it is not a
#     dependency — it is a hostage.
#
# WHY IT MATTERS
# --------------
# Every other parser signal we have is CIRCULAR: the test-set generator is
# downstream of the parser (failure mode F13), so a parser bug silently produces
# test pairs built around that bug, and the pipeline is then scored against its
# own error. UD-Prago is linguist-curated, external, and touches none of our
# code. It is the only parser measurement we own that cannot lie to us.
#
# Usage:
#     ./scripts/acquire/acquire_ud_prago.sh
#     python scripts/eval/eval_ud_prago.py
#
# Outputs:
#     data/external/ud_esperanto_prago/eo_prago-ud-test.conllu  (131 gold sentences)
#
# Licence: CC-BY-SA 4.0 — https://github.com/UniversalDependencies/UD_Esperanto-Prago
#
# Last Updated: 2026-07-13
# Related Issues: #809, #817
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

DEST="data/external/ud_esperanto_prago"
BASE="https://raw.githubusercontent.com/UniversalDependencies/UD_Esperanto-Prago/master"
FILES="eo_prago-ud-test.conllu"

mkdir -p "$DEST"

for f in $FILES; do
    echo "Fetching $f ..."
    curl -sfL -o "$DEST/$f" "$BASE/$f"
    lines=$(wc -l < "$DEST/$f" | tr -d ' ')
    if [ "$lines" -lt 100 ]; then
        echo "ERROR: $f is suspiciously small ($lines lines) — refusing a bad download." >&2
        exit 1
    fi
    echo "  -> $DEST/$f  ($lines lines)"
done

echo
echo "Done. The parser's external ruler is restored. Verify with:"
echo "    python scripts/eval/eval_ud_prago.py"
