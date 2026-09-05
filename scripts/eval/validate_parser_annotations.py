#!/usr/bin/env python3
"""Validate independently reviewed parser annotations and freeze a gold export.

Input is a pilot JSONL, never parser predictions. This checks structure and
review records, not the truth of a human linguistic judgment.
"""

import argparse
import hashlib
import json
from pathlib import Path


def validate(row):
    if row.get("annotation_status") != "reviewed":
        raise ValueError(f"{row.get('id')}: annotation is not reviewed")
    if (
        not row.get("annotator")
        or not row.get("reviewer")
        or row["annotator"] == row["reviewer"]
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
    args.output.mkdir(parents=True)
    (args.output / "gold.conllu").write_text(content)
    manifest = {
        "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "gold_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "sentences": len(rows),
        "split": rows[0]["split"],
        "documents": sorted({(r["source_name"], r["document_title"]) for r in rows}),
        "reviewers": sorted({r["reviewer"] for r in rows}),
        "status": "review records structurally validated; linguistic judgments supplied by annotators",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
