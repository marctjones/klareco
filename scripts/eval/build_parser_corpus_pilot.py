#!/usr/bin/env python3
"""Build a source-diverse, unreviewed Esperanto parser annotation queue.

This deliberately reads extracted non-Wikipedia JSONL corpora instead of the
production sentence store.  It creates candidates for human annotation; it
never invokes a parser, attaches automatic labels, or calls the result gold.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import unicodedata

ROOT = Path(__file__).resolve().parents[2]
SEED = "parser-corpus-pilot-v1"
DEFAULT_INPUTS = (
    ROOT / "data/extracted/eo/free/gutenberg_sentences.jsonl",
    ROOT / "data/extracted/eo/free/libera_folio_sentences.jsonl",
    ROOT / "data/extracted/eo/free/vikifontaro_sentences.jsonl",
)
WORD_RE = re.compile(r"\S+")


def normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def sentence_key(text: str) -> str:
    return hashlib.sha256(normalized(text).encode()).hexdigest()


def document_split(source: str, title: str, heldout_denominator: int = 5) -> str:
    key = f"{SEED}\0{source}\0{title}".encode()
    value = int(hashlib.sha256(key).hexdigest(), 16) % heldout_denominator
    return "heldout" if value == 0 else "development"


def read_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                row = json.loads(line)
                text = str(row.get("sentence", "")).strip()
                source = str(row.get("source", "")).strip()
                title = str(row.get("source_title", "")).strip()
                if not text or not source or not title:
                    continue
                words = len(WORD_RE.findall(text))
                if words < 4 or words > 80:
                    continue
                item = dict(row)
                item.update({"sentence": text, "_path": str(path), "_line": line_number})
                rows.append(item)
    return rows


def select(rows: list[dict], size: int) -> list[dict]:
    """Select document-disjoint, source/kind-balanced candidates deterministically."""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["source"], row.get("kind") or "unknown")].append(row)
    for group in groups.values():
        group.sort(key=lambda row: hashlib.sha256(
            f"{SEED}\0{row.get('_path', '')}\0{row.get('_line', '')}\0{row['sentence']}".encode()
        ).hexdigest())

    chosen: list[dict] = []
    seen_texts: set[str] = set()
    documents: set[tuple[str, str]] = set()
    splits = {"heldout": size // 5, "development": size - size // 5}
    counts = Counter()
    # Round-robin strata, then documents, so Gutenberg cannot crowd out news
    # or Wikisource.  A document contributes at most 8 sentences.
    while any(groups.values()) and len(chosen) < size:
        progressed = False
        for key in sorted(groups):
            group = groups[key]
            while group:
                row = group.pop(0)
                text_key = sentence_key(row["sentence"])
                document = (row["source"], row["source_title"])
                split = document_split(*document)
                if text_key in seen_texts or counts[split] >= splits[split]:
                    continue
                per_document = sum(
                    1 for item in chosen
                    if (item["source"], item["source_title"]) == document
                )
                if per_document >= 8:
                    continue
                seen_texts.add(text_key)
                counts[split] += 1
                item = {
                    "id": f"{SEED}-{len(chosen) + 1:04d}",
                    "text": row["sentence"],
                    "text_sha256": hashlib.sha256(row["sentence"].encode()).hexdigest(),
                    "source": row["source"],
                    "source_title": row["source_title"],
                    "author": row.get("author"),
                    "kind": row.get("kind"),
                    "license": row.get("licence"),
                    "url": row.get("url"),
                    "source_record": {
                        "path": row.get("_path"),
                        "line": row.get("_line"),
                    },
                    "split": split,
                    "annotation_status": "unreviewed",
                    "gold_conllu": None,
                    "klareco_annotation": None,
                    "phenomena": [],
                    "reviewers": [],
                    "adjudication": None,
                }
                chosen.append(item)
                progressed = True
                break
        if not progressed:
            break
    if len(chosen) != size:
        raise ValueError(f"only {len(chosen)}/{size} eligible candidates")
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=1000)
    parser.add_argument("--input", type=Path, action="append", dest="inputs")
    args = parser.parse_args()
    if args.size < 10:
        parser.error("--size must be at least 10")
    inputs = args.inputs or list(DEFAULT_INPUTS)
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise SystemExit("missing input corpus: " + ", ".join(missing))
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    candidates = select(read_rows(inputs), args.size)
    args.output.mkdir(parents=True)
    files = {}
    for split in ("development", "heldout"):
        content = "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in candidates if row["split"] == split
        )
        path = args.output / f"{split}.jsonl"
        path.write_text(content, encoding="utf-8")
        files[path.name] = hashlib.sha256(content.encode()).hexdigest()
    manifest = {
        "version": 1,
        "seed": SEED,
        "status": "unreviewed annotation queue, not gold",
        "inputs": [str(path) for path in inputs],
        "size": len(candidates),
        "splits": dict(Counter(row["split"] for row in candidates)),
        "sources": dict(Counter(row["source"] for row in candidates)),
        "kinds": dict(Counter(row["kind"] or "unknown" for row in candidates)),
        "documents": len({(row["source"], row["source_title"]) for row in candidates}),
        "files": files,
        "selection": "source/kind round-robin, max eight sentences per document, normalized-text deduplication, no parser labels",
        "annotation_requirement": "two independent reviewers plus adjudication before promotion to gold",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
