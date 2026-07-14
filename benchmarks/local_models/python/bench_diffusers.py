#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path
from typing import Any


IMAGE_PROMPTS = [
    {
        "id": "product_scene",
        "purpose": "image_generation_quality",
        "prompt": "A clean studio product photo of a compact open-source local AI workstation on a walnut desk, natural window light, sharp detail.",
    },
    {
        "id": "text_rendering",
        "purpose": "prompt_fidelity",
        "prompt": "A minimal poster with the exact words LOCAL MODELS, crisp typography, white background, black and red ink.",
    },
]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if usage > 10_000_000:
        return usage / (1024 * 1024)
    return usage / 1024


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark local image models through diffusers.")
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    import torch
    from diffusers import AutoPipelineForText2Image

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    models = read_json(args.models)["python_diffusers"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.image_dir.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as out:
        for model_entry in models:
            load_start = time.perf_counter()
            pipe = AutoPipelineForText2Image.from_pretrained(
                model_entry["model"],
                torch_dtype=torch.bfloat16 if device == "mps" else torch.float32,
            )
            pipe = pipe.to(device)
            load_seconds = time.perf_counter() - load_start

            for prompt in IMAGE_PROMPTS:
                generator = torch.Generator(device=device).manual_seed(args.seed)
                generation_start = time.perf_counter()
                image = pipe(
                    prompt=prompt["prompt"],
                    num_inference_steps=args.steps,
                    width=args.width,
                    height=args.height,
                    generator=generator,
                ).images[0]
                generation_seconds = time.perf_counter() - generation_start

                image_path = args.image_dir / f"{model_entry['id']}-{prompt['id']}.png"
                image.save(image_path)
                row = {
                    "runtime": "python_diffusers",
                    "model_id": model_entry["id"],
                    "model": model_entry["model"],
                    "prompt_id": prompt["id"],
                    "purpose": prompt["purpose"],
                    "device": device,
                    "load_seconds": load_seconds,
                    "generation_seconds": generation_seconds,
                    "steps": args.steps,
                    "width": args.width,
                    "height": args.height,
                    "peak_rss_mb": rss_mb(),
                    "image_path": str(image_path),
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()

            del pipe
            if device == "mps":
                torch.mps.empty_cache()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
