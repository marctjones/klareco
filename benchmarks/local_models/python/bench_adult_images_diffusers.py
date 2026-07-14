#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if usage > 10_000_000:
        return usage / (1024 * 1024)
    return usage / 1024


def model_entries(config: dict[str, Any]) -> list[dict[str, Any]]:
    if "adult_image_diffusers" in config:
        return list(config["adult_image_diffusers"])
    return list(config.get("python_diffusers", []))


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark non-explicit adult/suggestive local image generation with diffusers.")
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    args = parser.parse_args()

    import torch
    from diffusers import AutoPipelineForText2Image, StableDiffusionXLPipeline
    from huggingface_hub import hf_hub_download

    if args.device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device

    dtype = torch.float16 if device == "mps" else torch.float32
    models = model_entries(read_json(args.models))
    prompts = read_json(args.prompts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.image_dir.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as out:
        for model_entry in models:
            row_prefix = {
                "runtime": "python_diffusers",
                "benchmark": "adult_image",
                "model_id": model_entry["id"],
                "model": model_entry["model"],
                "producer": model_entry.get("producer"),
                "region": model_entry.get("region"),
                "device": device,
                "steps": args.steps,
                "width": args.width,
                "height": args.height,
            }
            load_start = time.perf_counter()
            try:
                if model_entry.get("filename"):
                    checkpoint_path = hf_hub_download(
                        repo_id=model_entry["model"],
                        filename=model_entry["filename"],
                    )
                    pipe = StableDiffusionXLPipeline.from_single_file(
                        checkpoint_path,
                        torch_dtype=dtype,
                        use_safetensors=True,
                    )
                else:
                    pipe = AutoPipelineForText2Image.from_pretrained(
                        model_entry["model"],
                        torch_dtype=dtype,
                        use_safetensors=True,
                    )
                safety_checker_present = getattr(pipe, "safety_checker", None) is not None
                pipe = pipe.to(device)
                load_seconds = time.perf_counter() - load_start
            except Exception as exc:
                out.write(json.dumps({**row_prefix, "status": "load_error", "error": str(exc), "load_seconds": time.perf_counter() - load_start}, ensure_ascii=False) + "\n")
                out.flush()
                continue

            for prompt in prompts:
                generator_device = device if device != "mps" else "cpu"
                generator = torch.Generator(device=generator_device).manual_seed(args.seed)
                started = time.perf_counter()
                try:
                    result = pipe(
                        prompt=prompt["prompt"],
                        negative_prompt=prompt.get("negative_prompt"),
                        num_inference_steps=args.steps,
                        width=args.width,
                        height=args.height,
                        generator=generator,
                    )
                    generation_seconds = time.perf_counter() - started
                    image = result.images[0]
                    image_path = args.image_dir / f"{model_entry['id']}-{prompt['id']}.png"
                    image.save(image_path)
                    nsfw_flags = getattr(result, "nsfw_content_detected", None)
                    out.write(
                        json.dumps(
                            {
                                **row_prefix,
                                "status": "ok",
                                "prompt_id": prompt["id"],
                                "category": prompt["category"],
                                "load_seconds": load_seconds,
                                "generation_seconds": generation_seconds,
                                "peak_rss_mb": rss_mb(),
                                "safety_checker_present": safety_checker_present,
                                "nsfw_content_detected": nsfw_flags,
                                "image_path": str(image_path),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                except Exception as exc:
                    out.write(json.dumps({**row_prefix, "status": "generation_error", "prompt_id": prompt["id"], "category": prompt["category"], "error": str(exc), "generation_seconds": time.perf_counter() - started, "peak_rss_mb": rss_mb()}, ensure_ascii=False) + "\n")
                out.flush()

            del pipe
            if device == "mps":
                torch.mps.empty_cache()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
