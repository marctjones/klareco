#!/usr/bin/env python3
"""Measure parser behavior on unlabeled Esperanto text.

This is a robustness/coverage instrument, not an accuracy evaluator.  It
reports failures, structural round trips, and deterministic repeatability on
large natural-text samples without treating parser output as gold.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from klareco.parser import compact_ast, expand_ast, parse
from scripts.eval.build_parser_corpus_pilot import DEFAULT_INPUTS, read_rows, select


def evaluate(rows: list[dict], parser=parse) -> dict:
    failures = []
    by_source = Counter()
    by_kind = Counter()
    round_trip = 0
    deterministic = 0
    total_ms = 0.0
    lengths = []
    for row in rows:
        source = row["source"]
        kind = row.get("kind") or "unknown"
        by_source[source] += 1
        by_kind[kind] += 1
        lengths.append(len(row["text"].split()))
        start = perf_counter()
        try:
            if hasattr(parser, "cache_clear"):
                parser.cache_clear()
            first = parser(row["text"])
            packed = json.dumps(compact_ast(first), ensure_ascii=False)
            restored = expand_ast(json.loads(packed))
            round_trip += restored == first
            if hasattr(parser, "cache_clear"):
                parser.cache_clear()
            second = parser(row["text"])
            deterministic += second == first
        except Exception as exc:  # coverage report must inventory, not hide failures
            failures.append({
                "id": row.get("id"), "source": source,
                "source_title": row.get("source_title"),
                "text": row["text"], "error": f"{type(exc).__name__}: {exc}",
            })
        finally:
            total_ms += (perf_counter() - start) * 1000
    count = len(rows)
    return {
        "sentences": count,
        "parse_failures": len(failures),
        "parse_success_rate": (count - len(failures)) / count if count else 0.0,
        "complete_storage_round_trips": round_trip,
        "storage_round_trip_rate": round_trip / count if count else 0.0,
        "deterministic_reparses": deterministic,
        "determinism_rate": deterministic / count if count else 0.0,
        "total_parse_ms": total_ms,
        "median_word_count": sorted(lengths)[len(lengths) // 2] if lengths else 0,
        "by_source": dict(by_source),
        "by_kind": dict(by_kind),
        "failures": failures,
        "interpretation": (
            "Unlabeled natural-text coverage; no linguistic accuracy is implied. "
            "Use Prago/Cairo for UPOS/UAS/LAS accuracy."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--size", type=int, default=5000)
    ap.add_argument("--input", type=Path, action="append", dest="inputs")
    args = ap.parse_args()
    if args.size < 1:
        ap.error("--size must be positive")
    inputs = args.inputs or list(DEFAULT_INPUTS)
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise SystemExit("missing input corpus: " + ", ".join(missing))
    rows = select(read_rows(inputs), args.size)
    for index, row in enumerate(rows, 1):
        row["id"] = f"coverage-v1-{index:05d}"
    report = {
        "report_version": 1,
        "seed": "parser-corpus-pilot-v1",
        "inputs": [str(path) for path in inputs],
        "sample_size": args.size,
        "sample_sha256": hashlib.sha256(
            "\n".join(row["text_sha256"] for row in rows).encode()
        ).hexdigest(),
        "metrics": evaluate(rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
