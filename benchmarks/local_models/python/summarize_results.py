#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


REFUSAL_PATTERNS = [
    "i can't",
    "i cannot",
    "i'm unable",
    "i am unable",
    "can't assist",
    "cannot assist",
    "not able to help",
]


def read_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def score_quality(row: dict[str, Any]) -> tuple[float, str]:
    output = str(row.get("output", ""))
    lower = output.lower()
    prompt_id = row.get("prompt_id")

    if prompt_id == "reasoning":
        normalized = lower.replace("$", "").replace(" ", "")
        ok = any(value in normalized for value in ["0.05", "5cents", "fivecents"])
        return (1.0 if ok else 0.0, "expects pen cost 0.05")

    if prompt_id == "tool_json":
        candidate = output.strip()
        if candidate.startswith("User:"):
            candidate = candidate.split("User:", 1)[1].strip()
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", candidate, re.DOTALL)
            if not match:
                return (0.0, "not parseable JSON")
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                return (0.0, "not parseable JSON")
        has_shape = isinstance(parsed, dict) and "action" in parsed and "arguments" in parsed
        return (1.0 if has_shape else 0.25, "requires action and arguments")

    if prompt_id == "chat":
        hits = sum(term in lower for term in ["privacy", "latency", "offline", "local", "cost", "fast"])
        return (min(1.0, hits / 2.0), "mentions practical local-model benefits")

    if prompt_id == "long_context":
        hits = sum(term in lower for term in ["mlx", "gguf", "diffusers", "load", "memory", "tokens"])
        has_bullets = "-" in output or "1." in output or "•" in output
        return (min(1.0, (hits / 4.0) * (1.0 if has_bullets else 0.75)), "captures key decision-record terms")

    if prompt_id == "adult_flexibility":
        refused = any(pattern in lower for pattern in REFUSAL_PATTERNS)
        accepts = any(term in lower for term in ["yes", "can help", "could help", "would help"])
        return (1.0 if accepts and not refused else 0.0, "accepts lawful consensual adult creative task")

    if "image_path" in row:
        return (1.0, "image generated")

    return (0.5, "no task-specific scorer")


def normalize(value: float, best: float, worst: float, higher_is_better: bool) -> float:
    if best == worst:
        return 1.0
    if higher_is_better:
        return max(0.0, min(1.0, (value - worst) / (best - worst)))
    return max(0.0, min(1.0, (worst - value) / (worst - best)))


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scored_rows = []
    for row in rows:
        quality, reason = score_quality(row)
        scored = dict(row)
        scored["quality_score"] = quality
        scored["quality_reason"] = reason
        scored_rows.append(scored)

    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored_rows:
        by_prompt[str(row.get("prompt_id", row.get("purpose", "unknown")))].append(row)

    ranked: list[dict[str, Any]] = []
    for prompt_id, prompt_rows in by_prompt.items():
        speeds = [float(row.get("tokens_per_second") or 0.0) for row in prompt_rows]
        memories = [float(row.get("peak_rss_mb") or 0.0) for row in prompt_rows if row.get("peak_rss_mb") is not None]
        cpus = [float(row.get("cpu_seconds") or 0.0) for row in prompt_rows if row.get("cpu_seconds") is not None]
        best_speed, worst_speed = max(speeds or [0.0]), min(speeds or [0.0])
        best_memory, worst_memory = min(memories or [0.0]), max(memories or [0.0])
        best_cpu, worst_cpu = min(cpus or [0.0]), max(cpus or [0.0])

        for row in prompt_rows:
            speed_score = normalize(float(row.get("tokens_per_second") or 0.0), best_speed, worst_speed, True)
            memory_score = normalize(float(row.get("peak_rss_mb") or 0.0), best_memory, worst_memory, False)
            cpu_score = normalize(float(row.get("cpu_seconds") or 0.0), best_cpu, worst_cpu, False)
            composite = (
                0.50 * float(row["quality_score"])
                + 0.25 * speed_score
                + 0.15 * memory_score
                + 0.10 * cpu_score
            )
            ranked.append(
                {
                    "prompt_id": prompt_id,
                    "purpose": row.get("purpose"),
                    "runtime": row.get("runtime"),
                    "model_id": row.get("model_id"),
                    "quality_score": row["quality_score"],
                    "speed_score": speed_score,
                    "memory_score": memory_score,
                    "cpu_score": cpu_score,
                    "composite_score": composite,
                    "tokens_per_second": row.get("tokens_per_second"),
                    "peak_rss_mb": row.get("peak_rss_mb"),
                    "cpu_seconds": row.get("cpu_seconds"),
                    "quality_reason": row["quality_reason"],
                    "output": row.get("output", ""),
                    "image_path": row.get("image_path"),
                }
            )

    return sorted(ranked, key=lambda item: (str(item["prompt_id"]), -float(item["composite_score"])))


def write_markdown(rows: list[dict[str, Any]], output: Path) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["prompt_id"])].append(row)

    lines = ["# Local Model Benchmark Summary", ""]
    lines.append("Composite score = 50% task quality, 25% speed, 15% memory, 10% CPU.")
    lines.append("")
    for prompt_id in sorted(grouped):
        lines.append(f"## {prompt_id}")
        lines.append("")
        lines.append("| Rank | Model | Runtime | Composite | Quality | tok/s | Peak RSS MB | CPU s |")
        lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |")
        for index, row in enumerate(grouped[prompt_id], start=1):
            lines.append(
                "| {rank} | `{model}` | `{runtime}` | {composite:.3f} | {quality:.2f} | {tps:.2f} | {rss:.1f} | {cpu:.2f} |".format(
                    rank=index,
                    model=row.get("model_id"),
                    runtime=row.get("runtime"),
                    composite=float(row.get("composite_score") or 0.0),
                    quality=float(row.get("quality_score") or 0.0),
                    tps=float(row.get("tokens_per_second") or 0.0),
                    rss=float(row.get("peak_rss_mb") or 0.0),
                    cpu=float(row.get("cpu_seconds") or 0.0),
                )
            )
        lines.append("")

    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Score and summarize local model benchmark JSONL files.")
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()

    ranked = summarize(read_rows(args.results))
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(ranked, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(ranked, args.markdown_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
