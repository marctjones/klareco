#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import resource
import time
from pathlib import Path
from typing import Any

from mlx_lm import generate, load


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports KiB.
    if usage > 10_000_000:
        return usage / (1024 * 1024)
    return usage / 1024


def count_tokens(tokenizer: Any, text: str) -> int:
    encoded = tokenizer.encode(text)
    return len(encoded)


def format_prompt(tokenizer: Any, prompt: str) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    return prompt


def run_model(model_entry: dict[str, Any], prompts: list[dict[str, str]], max_tokens: int) -> list[dict[str, Any]]:
    process_started = time.process_time()
    started_load = time.perf_counter()
    model, tokenizer = load(model_entry["model"])
    load_seconds = time.perf_counter() - started_load

    rows: list[dict[str, Any]] = []
    for prompt in prompts:
        model_prompt = format_prompt(tokenizer, prompt["prompt"])
        prompt_cpu_started = time.process_time()
        started = time.perf_counter()
        output = generate(
            model,
            tokenizer,
            prompt=model_prompt,
            max_tokens=max_tokens,
            verbose=False,
        )
        generation_seconds = time.perf_counter() - started
        tokens_generated = count_tokens(tokenizer, output)
        rows.append(
            {
                "runtime": "python_mlx",
                "model_id": model_entry["id"],
                "model": model_entry["model"],
                "prompt_id": prompt["id"],
                "purpose": prompt["purpose"],
                "load_seconds": load_seconds,
                "generation_seconds": generation_seconds,
                "cpu_seconds": time.process_time() - prompt_cpu_started,
                "total_cpu_seconds": time.process_time() - process_started,
                "tokens_generated": tokens_generated,
                "tokens_per_second": tokens_generated / generation_seconds if generation_seconds else None,
                "peak_rss_mb": rss_mb(),
                "output": output,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark MLX local text models.")
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("LOCAL_MODEL_MAX_TOKENS", "160")))
    args = parser.parse_args()

    models = read_json(args.models)["python_mlx"]
    prompts = read_json(args.prompts)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        for model_entry in models:
            for row in run_model(model_entry, prompts, args.max_tokens):
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
