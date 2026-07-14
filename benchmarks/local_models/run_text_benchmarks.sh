#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${LOCAL_MODEL_BENCH_OUT_DIR:-/tmp/klareco-local-model-bench}"
MAX_TOKENS="${LOCAL_MODEL_MAX_TOKENS:-160}"

mkdir -p "$OUT_DIR"

MLX_OUT="$OUT_DIR/mlx-results.jsonl"
RUST_OUT="$OUT_DIR/rust-gguf-results.jsonl"
DOTNET_OUT="$OUT_DIR/dotnet-gguf-results.jsonl"
SUMMARY_JSON="$OUT_DIR/summary.json"
SUMMARY_MD="$OUT_DIR/summary.md"

python_bin="${PYTHON:-$ROOT/.venv/bin/python}"

"$python_bin" "$ROOT/benchmarks/local_models/python/bench_mlx.py" \
  --models "$ROOT/benchmarks/local_models/models.json" \
  --prompts "$ROOT/benchmarks/local_models/prompts.json" \
  --output "$MLX_OUT" \
  --max-tokens "$MAX_TOKENS"

cargo run --release --manifest-path "$ROOT/benchmarks/local_models/rust/Cargo.toml" -- \
  --models "$ROOT/benchmarks/local_models/models.json" \
  --prompts "$ROOT/benchmarks/local_models/prompts.json" \
  --output "$RUST_OUT" \
  --max-tokens "$MAX_TOKENS"

dotnet run --project "$ROOT/benchmarks/local_models/dotnet/LocalModelBench" -- \
  --models "$ROOT/benchmarks/local_models/models.json" \
  --prompts "$ROOT/benchmarks/local_models/prompts.json" \
  --output "$DOTNET_OUT" \
  --max-tokens "$MAX_TOKENS"

"$python_bin" "$ROOT/benchmarks/local_models/python/summarize_results.py" \
  "$MLX_OUT" "$RUST_OUT" "$DOTNET_OUT" \
  --json-output "$SUMMARY_JSON" \
  --markdown-output "$SUMMARY_MD"

printf 'Wrote %s\n' "$SUMMARY_MD"
