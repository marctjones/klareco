#!/usr/bin/env python3
"""Compare a complete historical parser package with the working package.

VERSION: v2.0
COMPATIBLE WITH: parser packages in Git; current frozen UD evaluator
DEPENDENCIES: Git, standard library, parser lexical artifacts; no models
STAGE: Evaluation
Pipeline Position: immutable revision + working code -> common evaluator -> reports
Usage: python scripts/eval/compare_parser_revision.py --baseline-ref HEAD --output <new-dir>
Inputs: Git revision, working parser, shared lexicons, frozen UD fixtures
Outputs: before.json, after.json, summary.json, attachment_changes.json
Quality Checks: complete historical package isolation, fixture hashes, explicit regressions
Last Updated: 2026-09-05
"""

# CHANGELOG: 2026-09-05: Isolate the whole historical package, including syntax and storage.

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/eval"))
import parser_quality_report as quality

# A separate process prevents historical morphology caches, projections, readers,
# and serializers from importing the working tree's implementation by accident.
_BASELINE = r"""
import hashlib, json, pathlib, sys
checkout, root, revision, output = sys.argv[1:]
sys.path.insert(0, checkout)
import klareco
sys.path.insert(0, str(pathlib.Path(root) / 'scripts/eval'))
import parser_quality_report as quality
result = quality.report()
for key in list(result['files']):
    if key.startswith('klareco/'):
        path = pathlib.Path(checkout) / key
        if path.is_file():
            result['files'][key] = hashlib.sha256(path.read_bytes()).hexdigest()
        else:
            del result['files'][key]
result['commit'] = revision
result['comparison_baseline'] = {
    'revision': revision,
    'scope': 'complete historical klareco package; shared current evaluators and lexical artifacts',
}
pathlib.Path(output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline-ref", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        ap.error("Comparison output must be a new path")
    revision = subprocess.check_output(
        ["git", "rev-parse", "--verify", args.baseline_ref + "^{commit}"],
        cwd=ROOT,
        text=True,
    ).strip()
    archive = subprocess.check_output(["git", "archive", revision, "klareco"], cwd=ROOT)
    args.output.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="klareco-baseline-") as directory:
        checkout = Path(directory)
        with tarfile.open(fileobj=io.BytesIO(archive)) as contents:
            # Explicit extraction supports Python 3.10 without trusting archive paths.
            for member in contents:
                relative = PurePosixPath(member.name)
                if (
                    relative.is_absolute()
                    or ".." in relative.parts
                    or not relative.parts
                    or relative.parts[0] != "klareco"
                ):
                    raise ValueError(f"Invalid package archive path: {member.name}")
                destination = checkout.joinpath(*relative.parts)
                if member.isdir():
                    destination.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with contents.extractfile(member) as source:
                        destination.write_bytes(source.read())
                else:
                    raise ValueError(
                        f"Unsupported package archive entry: {member.name}"
                    )
        (checkout / "data").symlink_to(ROOT / "data", target_is_directory=True)
        subprocess.run(
            [
                sys.executable,
                "-c",
                _BASELINE,
                str(checkout),
                str(ROOT),
                revision,
                str((args.output / "before.json").resolve()),
            ],
            cwd=checkout,
            check=True,
        )
    before = json.loads((args.output / "before.json").read_text())
    after = quality.report()
    (args.output / "after.json").write_text(
        json.dumps(after, ensure_ascii=False, indent=2) + "\n"
    )
    summary, changes = {}, {}
    for bank, baseline in before["treebanks"].items():
        treatment = after["treebanks"][bank]
        if baseline["fixture_sha256"] != treatment["fixture_sha256"]:
            raise ValueError("Benchmark fixture changed during comparison")
        fields = ("las_all", "uas_all", "coverage", "gold_tokens", "crashed")
        summary[bank] = {
            m: [baseline["metrics"][m], treatment["metrics"][m]] for m in fields
        }
        summary[bank].update(
            {
                m: [baseline[m], treatment[m]]
                for m in ("pos_strict", "pos_adjusted", "roles", "storage")
            }
        )
        old = {(t["sentence"], t["gold"]["id"]): t for t in baseline["tokens"]}
        new = {(t["sentence"], t["gold"]["id"]): t for t in treatment["tokens"]}
        if old.keys() != new.keys():
            raise ValueError("Diagnostic gold token coverage changed")
        changed = {"gained": [], "lost": []}
        for key, a in old.items():
            b = new[key]
            was_correct = not (set(a["errors"]) - {"pos"})
            now_correct = not (set(b["errors"]) - {"pos"})
            if was_correct != now_correct:
                changed["gained" if now_correct else "lost"].append(
                    {"before": a, "after": b}
                )
        changes[bank] = changed
        summary[bank].update({key: len(value) for key, value in changed.items()})
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "attachment_changes.json").write_text(
        json.dumps(changes, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                bank: {k: v for k, v in stats.items() if k not in ("roles", "storage")}
                for bank, stats in summary.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
