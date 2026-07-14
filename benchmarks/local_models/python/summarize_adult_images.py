#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def read_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_model[str(row["model_id"])].append(row)

    summaries: list[dict[str, Any]] = []
    for model_id, model_rows in by_model.items():
        ok_rows = [row for row in model_rows if row.get("status") == "ok"]
        adult_rows = [row for row in ok_rows if row.get("category") == "adult_suggestive"]
        success_rate = len(ok_rows) / len(model_rows) if model_rows else 0.0
        adult_success_rate = len(adult_rows) / max(1, len([row for row in model_rows if row.get("category") == "adult_suggestive"]))
        avg_seconds = mean(float(row.get("generation_seconds") or 0.0) for row in ok_rows) if ok_rows else 0.0
        rss = max(float(row.get("peak_rss_mb") or 0.0) for row in model_rows)
        safety_checker_present = any(bool(row.get("safety_checker_present")) for row in ok_rows)
        nsfw_flags = [flag for row in ok_rows for flag in (row.get("nsfw_content_detected") or [])]
        summaries.append(
            {
                "model_id": model_id,
                "model": model_rows[0].get("model"),
                "producer": model_rows[0].get("producer"),
                "region": model_rows[0].get("region"),
                "success_rate": success_rate,
                "adult_suggestive_success_rate": adult_success_rate,
                "avg_generation_seconds": avg_seconds,
                "peak_rss_mb": rss,
                "safety_checker_present": safety_checker_present,
                "nsfw_flags_true": sum(1 for flag in nsfw_flags if flag),
                "ok_images": [row.get("image_path") for row in ok_rows],
                "errors": [row.get("error") for row in model_rows if row.get("status") != "ok"],
            }
        )
    return sorted(summaries, key=lambda row: (-float(row["adult_suggestive_success_rate"]), float(row["avg_generation_seconds"] or 999999), float(row["peak_rss_mb"])))


def write_markdown(rows: list[dict[str, Any]], output: Path) -> None:
    lines = [
        "# Adult Image Benchmark Summary",
        "",
        "This benchmark uses non-explicit adult/suggestive prompts plus a safe control prompt. It does not generate explicit sexual images.",
        "",
        "| Rank | Model | Success | Adult prompt success | avg sec/image | Peak RSS MB | Safety checker | NSFW flags | Errors |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | --- |",
    ]
    for index, row in enumerate(rows, start=1):
        lines.append(
            "| {rank} | `{model}` | {success:.2f} | {adult:.2f} | {seconds:.2f} | {rss:.1f} | {checker} | {flags} | {errors} |".format(
                rank=index,
                model=row["model_id"],
                success=float(row["success_rate"]),
                adult=float(row["adult_suggestive_success_rate"]),
                seconds=float(row["avg_generation_seconds"]),
                rss=float(row["peak_rss_mb"]),
                checker="yes" if row["safety_checker_present"] else "no",
                flags=int(row["nsfw_flags_true"]),
                errors="; ".join(str(err)[:120] for err in row["errors"]) if row["errors"] else "-",
            )
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize adult image benchmark JSONL results.")
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()

    summaries = summarize(read_rows(args.results))
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(summaries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(summaries, args.markdown_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
