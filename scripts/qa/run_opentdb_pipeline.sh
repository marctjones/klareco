#!/bin/bash
# The OpenTDB gold pipeline, fully batched. One-time exhaustive download, then
# batched translate -> deterministic gate -> BATCHED+parallel answerability -> assemble.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$(dirname "$SCRIPT_DIR")")"
[ -d .venv ] && source .venv/bin/activate

RAW=data/staging/opentdb_raw.jsonl
EO=data/staging/opentdb_eo.jsonl
BANK=data/staging/trivia_bank.jsonl
GOLD=data/staging/opentdb_gold.jsonl

echo "=== 1/5 download ALL OpenTDB (once; ~10 min, token-paginated) ==="
[ -f "$RAW" ] || python scripts/qa/qa_source_opentdb.py --download-all --raw-out "$RAW"
echo "   raw: $(wc -l < "$RAW") questions"

echo "=== 2/5 translate corpus-friendly (batched) ==="
python scripts/qa/qa_source_opentdb.py --from-raw "$RAW" --out "$EO"

echo "=== 3/5 gate: parser + corpus coverage (deterministic) ==="
python scripts/qa/qa_gate.py --input "$EO"

echo "=== 4/5 answerability judge (BATCHED, parallel workers) ==="
python scripts/qa/qa_answerability.py --input "$BANK" --out "$GOLD" --batch 8 --workers 3

echo "=== 5/5 assemble -> versioned gold ==="
python scripts/qa/qa_build_assemble.py --inputs "$GOLD" --out data/test_sets/qa_gold_v1.jsonl

echo "=== DONE ==="; wc -l data/test_sets/qa_gold_v1.jsonl
