#!/usr/bin/env python3
"""Validate independently reviewed parser annotations and freeze a gold export.

VERSION: v1.1
COMPATIBLE WITH: parser pilot v1, annotation layers v1, plain-token CoNLL-U
DEPENDENCIES: standard library; annotation contracts; no parser, model, or database
STAGE: Evaluation
Pipeline Position: human review records -> validated CoNLL-U and stand-off layers
Inputs: reviewed pilot JSONL
Outputs: frozen gold.conllu, annotations.jsonl, and hash manifest
Quality Checks: independent review record, token coverage, valid dependency tree
Last Updated: 2026-09-05

Input is a pilot JSONL, never parser predictions. This checks structure and
review records, not the truth of a human linguistic judgment.
"""

# CHANGELOG: 2026-09-05: Export source-bound gold annotation layers without invoking the parser.

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from klareco.ast_annotations import annotation_basis, validate_json, validate_layers


def validate(row):
    if row.get("annotation_status") != "reviewed":
        raise ValueError(f"{row.get('id')}: annotation is not reviewed")
    if (
        not isinstance(row.get("annotator"), str)
        or not row["annotator"].strip()
        or not isinstance(row.get("reviewer"), str)
        or not row["reviewer"].strip()
        or row["annotator"].strip().casefold() == row["reviewer"].strip().casefold()
    ):
        raise ValueError("Distinct annotator and reviewer identities are required")
    text = row["text"]
    if hashlib.sha256(text.encode()).hexdigest() != row["text_sha256"]:
        raise ValueError("Source text changed after selection")
    conllu = row.get("gold_conllu")
    if not isinstance(conllu, str) or not conllu.strip():
        raise ValueError("Missing gold CoNLL-U")
    words = []
    for line in conllu.splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) != 10:
            raise ValueError("CoNLL-U requires ten tab-separated fields")
        if "-" in fields[0] or "." in fields[0]:
            raise ValueError("Pilot export currently requires plain syntactic tokens")
        words.append(fields)
    ids = [int(w[0]) for w in words]
    if ids != list(range(1, len(words) + 1)) or not words:
        raise ValueError("Token IDs must be contiguous and ordered")
    heads = {int(w[0]): int(w[6]) for w in words}
    if sum(h == 0 for h in heads.values()) != 1:
        raise ValueError("Gold must have exactly one root")
    for w in words:
        i = int(w[0])
        if w[3] == "_" or w[7] == "_":
            raise ValueError("Gold POS and dependency labels are required")
        if (heads[i] == 0) != (w[7] == "root"):
            raise ValueError("Root head and relation disagree")
        seen = set()
        while i:
            if i in seen or i not in heads:
                raise ValueError("Cycle or dangling gold head")
            seen.add(i)
            i = heads[i]
    # Whitespace is formatting; all source characters must otherwise be covered.
    if "".join("".join(w[1].split()) for w in words) != "".join(text.split()):
        raise ValueError("Gold token forms do not cover the original source text")
    if not isinstance(row.get("phenomena"), list) or not row.get("review_notes"):
        raise ValueError("Phenomena list and review notes are required")
    return conllu.strip() + "\n\n"


def annotation_layer(row):
    """Gold token IDs remain in their own layer; targets use original text spans.

    This works even when a future parser chooses different token boundaries.
    The supplied human labels are never aligned against parser predictions.
    """
    conllu = validate(row)
    text = row["text"]
    positions = [i for i, char in enumerate(text) if not char.isspace()]
    cursor = 0
    records = []
    columns = (
        "id",
        "form",
        "lemma",
        "upos",
        "xpos",
        "feats",
        "head",
        "deprel",
        "deps",
        "misc",
    )
    for line in conllu.splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        length = len("".join(fields[1].split()))
        if not length:
            raise ValueError("Gold token cannot be whitespace only")
        span = [positions[cursor], positions[cursor + length - 1] + 1]
        cursor += length
        records.append(
            {
                "id": "token:" + fields[0],
                "target": {"kind": "span", "spans": [span]},
                "value": dict(zip(columns, fields)),
            }
        )
    ast = {"vortoj": [], "source": {"original": text, "normalized": text}}
    layer = {
        "version": 1,
        "id": "gold:" + str(row["id"]),
        "schema": "urn:klareco:gold-conllu:1",
        "status": "reviewed",
        "producer": {
            "name": row["annotator"],
            "version": "parser-pilot-v1",
            "method": "human",
            "artifacts": {"gold_conllu": hashlib.sha256(conllu.encode()).hexdigest()},
        },
        "basis": annotation_basis(ast, tokens=False),
        "annotations": records,
        "review": {
            "annotator": row["annotator"],
            "reviewer": row["reviewer"],
            "notes": row["review_notes"],
        },
    }
    ast["annotation_layers"] = [layer]
    validate_json(ast)
    validate_layers(ast)
    return layer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite frozen gold")
    rows = [
        json.loads(line) for line in args.input.read_text().splitlines() if line.strip()
    ]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Empty or duplicate annotation set")
    if len({row["split"] for row in rows}) != 1:
        raise ValueError("Export development and heldout separately")
    content = "".join(validate(row) for row in rows)
    annotations = "".join(
        json.dumps(
            {"id": row["id"], "text": row["text"], "layer": annotation_layer(row)},
            ensure_ascii=False,
        )
        + "\n"
        for row in rows
    )
    args.output.mkdir(parents=True)
    (args.output / "gold.conllu").write_text(content)
    (args.output / "annotations.jsonl").write_text(annotations, encoding="utf-8")
    manifest = {
        "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "gold_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "annotations_sha256": hashlib.sha256(annotations.encode()).hexdigest(),
        "sentences": len(rows),
        "split": rows[0]["split"],
        "documents": sorted({(r["source_name"], r["document_title"]) for r in rows}),
        "reviewers": sorted({r["reviewer"] for r in rows}),
        "status": "review records structurally validated; linguistic judgments supplied by annotators",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
