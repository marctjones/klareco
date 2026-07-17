#!/bin/bash
# Regenerate the FIXED smoke subset from the canonical gold set. Stable + committed,
# so "did we break it?" runs are fast (~1 min) and comparable across changes.
# Targeted slices (by band/type/category/source) are made on demand with qa_subset.py.
set -e
cd "$(dirname "$(dirname "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")")"
[ -d .venv ] && source .venv/bin/activate
GOLD=${1:-data/test_sets/qa_gold_v2.jsonl}
# 30 pairs balanced across difficulty bands (trivial/rerankable/deep)
python scripts/qa/qa_subset.py --input "$GOLD" --stratify difficulty_band --sample 30 \
    --out data/test_sets/smoke.jsonl
echo "  smoke set -> data/test_sets/smoke.jsonl"
echo "  targeted examples:"
echo "    qa_subset.py --band deep    --out /tmp/deep.jsonl     # hard-retrieval class"
echo "    qa_subset.py --type KIO     --out /tmp/kio.jsonl      # definition questions"
echo "    qa_subset.py --category Geography --out /tmp/geo.jsonl"
