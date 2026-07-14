# Local Model Benchmarks

This folder contains small benchmark programs for comparing local models on an
Apple Silicon M5-class Mac from native libraries in Python, Rust, and .NET.

For the practical model choices by situation, including separate Chinese and
non-Chinese recommendations, see `recommendations.md`.

## What to run first

Python MLX is the fastest path to a useful baseline:

```bash
python -m venv .venv
.venv/bin/pip install mlx-lm psutil
.venv/bin/python benchmarks/local_models/python/bench_mlx.py \
  --models benchmarks/local_models/models.json \
  --prompts benchmarks/local_models/prompts.json \
  --output /tmp/klareco-mlx-results.jsonl
```

Rust and .NET use GGUF files. Set env vars from `models.json` to point to local
GGUF files before running:

```bash
export QWEN3_4B_GGUF=/path/to/qwen3-4b-q4_k_m.gguf
export QWEN36_35B_A3B_GGUF=/path/to/qwen3.6-35b-a3b-q4_k_m.gguf
export LLAMA32_3B_GGUF=/path/to/llama-3.2-3b-instruct-q4_k_m.gguf
```

Rust:

```bash
cargo run --release --manifest-path benchmarks/local_models/rust/Cargo.toml -- \
  --models benchmarks/local_models/models.json \
  --prompts benchmarks/local_models/prompts.json \
  --output /tmp/klareco-rust-results.jsonl
```

.NET:

```bash
dotnet run --project benchmarks/local_models/dotnet/LocalModelBench -- \
  --models benchmarks/local_models/models.json \
  --prompts benchmarks/local_models/prompts.json \
  --output /tmp/klareco-dotnet-results.jsonl
```

The .NET runner uses the current `LLamaSharp.Backend.Cpu` package. NuGet still
lists `LLamaSharp.Backend.MacMetal` only at an old `0.7.0` release, so treat
.NET results as a native API compatibility test first and a CPU performance
baseline second unless LLamaSharp publishes a current Metal backend.

Image generation:

```bash
.venv/bin/pip install diffusers torch transformers accelerate sentencepiece protobuf
.venv/bin/python benchmarks/local_models/python/bench_diffusers.py \
  --models benchmarks/local_models/models.json \
  --output /tmp/klareco-image-results.jsonl \
  --image-dir /tmp/klareco-image-results
```

## Metrics

Each runner emits JSONL with:

- `model_id`
- `prompt_id`
- `runtime`
- `load_seconds`
- `generation_seconds`
- `tokens_generated`
- `tokens_per_second`
- `peak_rss_mb` when available
- `output`

Score and rank result files:

```bash
.venv/bin/python benchmarks/local_models/python/summarize_results.py \
  /tmp/klareco-mlx-results.jsonl /tmp/klareco-rust-results.jsonl \
  --json-output /tmp/klareco-model-summary.json \
  --markdown-output /tmp/klareco-model-summary.md
```

The summarizer applies task-specific checks for the included text prompts:

- `reasoning`: expects the pen-cost answer `0.05`.
- `tool_json`: requires parseable JSON with `action` and `arguments`.
- `chat`: rewards practical local-model benefits such as privacy, latency, offline use, cost, or speed.
- `long_context`: rewards key decision-record terms such as MLX, GGUF, diffusers, load, memory, and tokens.
- `adult_flexibility`: checks whether the model accepts lawful consensual adult creative writing without a broad refusal.

Run the full text harness:

```bash
LOCAL_MODEL_BENCH_OUT_DIR=/tmp/klareco-local-model-bench \
LOCAL_MODEL_MAX_TOKENS=160 \
benchmarks/local_models/run_text_benchmarks.sh
```

The script runs Python MLX, Rust GGUF, .NET GGUF, then writes `summary.md`.
GGUF entries whose `path_env` variables are unset are skipped.

Embedding benchmarks are split into English and Esperanto suites:

```bash
.venv/bin/pip install sentence-transformers scikit-learn
.venv/bin/python benchmarks/local_models/python/bench_embeddings.py \
  --models benchmarks/local_models/models.json \
  --tasks benchmarks/local_models/embedding_tasks.json \
  --output /tmp/klareco-embedding-results.jsonl
```

The GGUF runners only use local files referenced by environment variables. The
MLX, embedding, and diffusers runners use their native Hugging Face loaders, so
they can download missing models unless the cache already contains them.

Adult-safety benchmark:

```bash
.venv/bin/python benchmarks/local_models/python/bench_adult_safety_mlx.py \
  --models benchmarks/local_models/models.json \
  --prompts benchmarks/local_models/adult_safety_prompts.json \
  --output /tmp/klareco-adult-safety-results.jsonl
.venv/bin/python benchmarks/local_models/python/summarize_adult_safety.py \
  /tmp/klareco-adult-safety-results.jsonl \
  --json-output /tmp/klareco-adult-safety-summary.json \
  --markdown-output /tmp/klareco-adult-safety-summary.md
```

The adult-safety benchmark measures willingness on lawful consensual adult
creative requests and refusal on unsafe sexual-content requests. It does not
store explicit generated scenes as test fixtures.

Adult willingness probe:

```bash
.venv/bin/python benchmarks/local_models/python/bench_adult_safety_mlx.py \
  --models benchmarks/local_models/models.json \
  --prompts benchmarks/local_models/adult_willingness_probe_prompts.json \
  --output /tmp/klareco-adult-willingness-probe.jsonl
.venv/bin/python benchmarks/local_models/python/summarize_adult_willingness_probe.py \
  /tmp/klareco-adult-willingness-probe.jsonl \
  --json-output /tmp/klareco-adult-willingness-probe-summary.json \
  --markdown-output /tmp/klareco-adult-willingness-probe-summary.md
```

The willingness probe is the better tool for comparing how willing models say
they are to engage with lawful adult content or unsafe sexual requests. It uses
JSON-only decision prompts and does not ask models to generate unsafe content.

Adult image benchmark:

```bash
.venv/bin/python benchmarks/local_models/python/bench_adult_images_diffusers.py \
  --models /tmp/adult-image-models.json \
  --prompts benchmarks/local_models/adult_image_prompts.json \
  --output /tmp/klareco-adult-image-results.jsonl \
  --image-dir /tmp/klareco-adult-image-results \
  --steps 6 --width 512 --height 512
.venv/bin/python benchmarks/local_models/python/summarize_adult_images.py \
  /tmp/klareco-adult-image-results.jsonl \
  --json-output /tmp/klareco-adult-image-summary.json \
  --markdown-output /tmp/klareco-adult-image-summary.md
```

The adult image benchmark uses non-explicit adult/suggestive prompts plus a safe
control prompt. It measures local run success, generation time, memory, and
safety-checker behavior when the pipeline exposes it.
