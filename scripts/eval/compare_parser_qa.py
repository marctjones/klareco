#!/usr/bin/env python3
"""Paired downstream experiment with identical retrieved candidates.

Retrieval uses the original store and baseline question parse for BOTH arms.
Each arm reparses those same passages; only question/passage ASTs differ. This
isolates reranking/extraction effects and is not a full retrieval/store A/B.
"""

import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
import logging
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
logging.basicConfig(level=logging.ERROR)
from klareco.parser import parse
from klareco.orchestrator.factory import build_default_pipeline
from klareco.eval import evaluate_question, summarize
import klareco.orchestrator.stages.parse_question as question_stage


class FixedCandidates:
    def __init__(self, candidates, parser):
        self.candidates = candidates
        self.parser = parser

    def retrieve_with_ast_roles(self, question_ast, top_k):
        return [
            dict(row, ast=self.parser(row["text"])) for row in self.candidates[:top_k]
        ]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline-ref", default="46e85d1")
    ap.add_argument(
        "--test-set", type=Path, default=ROOT / "data/test_sets/rebaseline_500.jsonl"
    )
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--live-candidates",
        action="store_true",
        help="Use each arm to select candidates from the unchanged production store",
    )
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        ap.error("Refusing to overwrite an experiment")
    revision = subprocess.check_output(
        ["git", "rev-parse", args.baseline_ref], text=True
    ).strip()
    source = subprocess.check_output(["git", "show", f"{revision}:klareco/parser.py"])
    spec = importlib.util.spec_from_file_location(
        "klareco._baseline_parser", ROOT / "klareco/parser.py"
    )
    baseline = importlib.util.module_from_spec(spec)
    exec(compile(source, str(ROOT / "klareco/parser.py"), "exec"), baseline.__dict__)
    pipeline = build_default_pipeline(whoosh_index_dir=ROOT / "data/indexes/whoosh_v2")
    retrieve = next(stage for stage in pipeline.stages if stage.name == "retrieve")
    live_retriever = retrieve.retriever
    entries = [
        json.loads(line)
        for line in args.test_set.read_text().splitlines()
        if line.strip()
    ]
    if args.limit:
        entries = entries[: args.limit]
    result = {
        "experiment": __doc__,
        "baseline_revision": revision,
        "baseline_parser_sha256": hashlib.sha256(source).hexdigest(),
        "test_set": str(args.test_set),
        "test_sha256": hashlib.sha256(args.test_set.read_bytes()).hexdigest(),
        "limit": args.limit,
        "live_candidates": args.live_candidates,
        "code_hashes": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                ROOT / "klareco/parser.py",
                ROOT / "klareco/syntax_graph.py",
                ROOT / "klareco/rag/duckdb_retriever.py",
                ROOT / "klareco/rag/unified_extractor.py",
                ROOT / "klareco/orchestrator/stages/parse_question.py",
                Path(__file__),
            ]
        },
        "status": "running",
        "pairs": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.live_candidates:
        result["experiment"] = (
            "Each arm selects passages using its question AST against the unchanged production store, then reparses selected passages. Production shredded columns remain old; this is not a complete rebuilt-store A/B."
        )
    started = time.monotonic()
    try:
        for i, entry in enumerate(entries):
            candidates = (
                []
                if args.live_candidates
                else live_retriever.retrieve_with_ast_roles(
                    baseline.parse(entry["question"]), retrieve.top_k
                )
            )
            pair = {
                "id": entry.get("id"),
                "candidate_ids": [r["id"] for r in candidates],
            }
            # Alternate order to reduce cache/temperature bias in timings.
            arms = [("before", baseline.parse), ("after", parse)]
            if i % 2:
                arms.reverse()
            for name, parser in arms:
                if args.live_candidates:
                    candidates = live_retriever.retrieve_with_ast_roles(
                        parser(entry["question"]), retrieve.top_k
                    )
                    pair[name + "_candidate_ids"] = [r["id"] for r in candidates]
                question_stage.parse = parser
                retrieve.retriever = FixedCandidates(candidates, parser)
                pair[name] = evaluate_question(pipeline, entry)
            result["pairs"].append(pair)
            result["elapsed_seconds"] = time.monotonic() - started
            args.output.write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n"
            )
            print(
                f'{i + 1}/{len(entries)} {pair["before"]["answer_correct"]} -> {pair["after"]["answer_correct"]}',
                flush=True,
            )
        result["summary"] = {
            arm: summarize([pair[arm] for pair in result["pairs"]])
            for arm in ("before", "after")
        }
        result["status"] = "complete"
    except BaseException as exc:
        result.update(status="failed", error=str(exc))
        raise
    finally:
        question_stage.parse = parse
        retrieve.retriever = live_retriever
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
