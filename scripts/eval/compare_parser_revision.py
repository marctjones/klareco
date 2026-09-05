#!/usr/bin/env python3
"""Run the same evaluator against a historical parser and the working parser."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/eval"))
import parser_quality_report as quality
import klareco.conllu as conllu


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline-ref", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    revision = subprocess.check_output(
        ["git", "rev-parse", args.baseline_ref], text=True
    ).strip()
    source = subprocess.check_output(["git", "show", f"{revision}:klareco/parser.py"])
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "baseline_parser.py"
        path.write_bytes(source)
        spec = importlib.util.spec_from_file_location("klareco._baseline_parser", path)
        module = importlib.util.module_from_spec(spec)
        module.__file__ = str(ROOT / "klareco/parser.py")
        exec(compile(source, str(path), "exec"), module.__dict__)
        targets = [quality, quality.eval_ud_prago, quality.eval_ud_roles, conllu]
        original = [t.parse for t in targets]
        try:
            for target in targets:
                target.parse = module.parse
            before = quality.report()
            before["files"]["klareco/parser.py"] = hashlib.sha256(source).hexdigest()
            before["comparison_baseline"] = {
                "revision": revision,
                "parser_sha256": hashlib.sha256(source).hexdigest(),
                "scope": "historical parser.py with shared current lexicons, serializer, and evaluators",
            }
        finally:
            for target, parse in zip(targets, original):
                target.parse = parse
    after = quality.report()
    for name, result in [("before", before), ("after", after)]:
        (args.output / (name + ".json")).write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        )
    summary = {}
    for bank in before["treebanks"]:
        summary[bank] = {
            metric: [
                before["treebanks"][bank]["metrics"][metric],
                after["treebanks"][bank]["metrics"][metric],
            ]
            for metric in ("las_all", "uas_all", "coverage")
        }
        summary[bank]["pos_strict"] = [
            before["treebanks"][bank]["pos_strict"],
            after["treebanks"][bank]["pos_strict"],
        ]
        summary[bank]["roles"] = [
            before["treebanks"][bank]["roles"],
            after["treebanks"][bank]["roles"],
        ]
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
