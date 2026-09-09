#!/usr/bin/env python3
"""Targeted corner-case sentence search over the unindexed literary/news
corpora, with one sentence of document context on each side for review.

VERSION: v1.0
COMPATIBLE WITH: klareco.parser's live AST shape (vortoj[].{kapo,rolo});
  data/extracted/eo/free/*.jsonl schema
DEPENDENCIES: klareco.parser only
STAGE: Evaluation

Why this exists (2026-09-08): a 200-sentence gold-correction pass measured
relation-label frequency directly and found the long tail is thin --
csubj: 3 tokens, dep/flat: 1 each, discourse: 6, vocative: 8, fixed: 9,
aux: 16, nummod: 27, ccomp: 33 (out of 3,122 gold tokens). Random sampling
gives proportionally few examples of anything uncommon by construction;
klareco#920's own gate (docs/PARSER_EVALUATION_PROTOCOL.md) already calls
for "fragments, quotations, and creative morphology" as required strata,
not left to chance. This script finds candidate sentences where the
PARSER'S OWN OUTPUT already contains one of the thin categories, or a
known problem shape (multi-root malformed trees, #941), rather than
guessing from surface keywords -- the parser's classification is exactly
what a reviewer will be checking, so search on it directly.

A second, independent finding motivated the context columns: of the 200
sentences corrected in the prior pass, 3 of the 11 left as
"needs_human_review" turned out to be extraction artifacts, not
grammatical ambiguity -- a leading comma continuing a stripped subject
from the previous sentence, a bracketed editorial gloss fused into the
sentence, a stage-direction speaker label fused with dialogue. A reviewer
(human or LLM) can only tell "genuine elliptical construction" from "this
clause's subject is one sentence back" by seeing what is actually next to
it. So every candidate here carries prev_text/next_text from the same
document (by contiguous source_title run, matching
build_literary_discourse_pilot.py's document grouping) -- as verification
context, NOT as a second annotation target. This stays a sentence-level
pilot; passages remain discourse_pilot_v1's job.

Pipeline Position: data/extracted/eo/free/*.jsonl -> parsed for scanning
    (not stored) -> targeted candidate queue -> human/LLM correction pass
    (same pipeline as parser_pilot_v1)
Usage: python scripts/eval/build_targeted_corner_case_pilot.py \
    --output data/test_sets/parser_pilot_targeted_v1 --per-category 15
Inputs: data/extracted/eo/free/{gutenberg,vikifontaro,libera_folio}_sentences.jsonl,
    data/test_sets/parser_pilot_v1 + parser_gold_candidates_v1 (excluded, already used)
Outputs: one JSON file per targeted category found, plus a manifest
    summarizing counts; NOT a gold benchmark -- these are unreviewed,
    parser-labeled candidates for a correction pass to work from
Quality Checks: excludes text already used in parser_pilot_v1,
    parser_pilot_v2, and parser_gold_candidates_v1 (checked by normalized
    text, not by source file, since the same sentence can appear via
    different extraction runs); excludes UD Prago/Cairo fixture text;
    each candidate is deduplicated by normalized text within this run
Last Updated: 2026-09-08
Related Issues: #920, #941
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from klareco.parser import parse  # noqa: E402

SOURCES = ("gutenberg", "vikifontaro", "libera_folio")

# Rare relations found thin in the 200-sentence gold-correction measurement
# (2026-09-08): csubj=3, dep/flat=1 each, discourse=6, vocative=8, fixed=9,
# aux=16, nummod=27, ccomp=33 tokens out of 3,122. A sentence qualifies for
# a category the first time ANY of its words carries that relation.
TARGET_RELATIONS = ("vocative", "csubj", "discourse", "fixed", "ccomp", "aux", "nummod")

# klareco#941: verbless sentences producing more than one head=0 token --
# deliberately over-sample this known, root-caused bug to give the eventual
# fix real regression fixtures beyond the 9 already found.
CATEGORY_MULTIROOT = "multiroot"

# The flagship claimed residue (VISION.md): PP-attachment the parser itself
# could not resolve deterministically (klareco.parser's own alligo_ambigua
# flag, set only for the narrow en/al/per/kiel-type noun-or-verb case).
CATEGORY_PP_AMBIGUOUS = "pp_ambiguous"

ALL_CATEGORIES = list(TARGET_RELATIONS) + [CATEGORY_MULTIROOT, CATEGORY_PP_AMBIGUOUS]


def normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def load_excluded_texts() -> set[str]:
    """Normalized text of every sentence already used in an existing pilot
    or frozen UD fixture, so this run never re-surfaces the same sentence."""
    excluded = set()
    for name in ("prago", "cairo"):
        fixture = ROOT / f"tests/fixtures/ud/eo_{name}-ud-test.conllu"
        excluded.update(
            normalized(line.split("=", 1)[1])
            for line in fixture.read_text().splitlines()
            if line.startswith("# text =")
        )
    for pilot_dir in ("parser_pilot_v1", "parser_pilot_v2", "parser_gold_candidates_v1"):
        base = ROOT / f"data/test_sets/{pilot_dir}"
        if not base.is_dir():
            continue
        for split_file in base.glob("*.jsonl"):
            for line in split_file.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                text = row.get("text")
                if text:
                    excluded.add(normalized(text))
    return excluded


def load_documents_with_context(free_dir: Path, sources: tuple[str, ...]):
    """Every sentence, grouped into maximal contiguous source_title runs
    (same convention as build_literary_discourse_pilot.py), with each
    sentence knowing its previous/next sibling in the same run."""
    for source in sources:
        path = free_dir / f"{source}_sentences.jsonl"
        current_title = None
        run_index = 0
        run_rows: list[dict] = []

        def flush():
            for i, row in enumerate(run_rows):
                row["prev_text"] = run_rows[i - 1]["sentence"] if i > 0 else None
                row["next_text"] = run_rows[i + 1]["sentence"] if i + 1 < len(run_rows) else None
            yield from run_rows

        with path.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                title = row.get("source_title")
                if title != current_title:
                    yield from flush()
                    run_rows.clear()
                    current_title = title
                    run_index += 1
                row["source"] = source
                row["_run_id"] = f"{source}::{title}::{run_index}"
                run_rows.append(row)
            yield from flush()


def classify(ast: dict) -> set[str]:
    """Every target category this sentence's PARSED output qualifies for."""
    hits = set()
    roots = [w for w in ast.get("vortoj", []) if isinstance(w, dict) and w.get("kapo") == 0]
    if len(roots) > 1:
        hits.add(CATEGORY_MULTIROOT)
    for w in ast.get("vortoj", []):
        if not isinstance(w, dict):
            continue
        if w.get("rolo") in TARGET_RELATIONS:
            hits.add(w["rolo"])
        if w.get("alligo_ambigua"):
            hits.add(CATEGORY_PP_AMBIGUOUS)
    return hits


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--free-dir", type=Path, default=ROOT / "data/extracted/eo/free")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--per-category", type=int, default=15,
                     help="max candidates to keep per category")
    ap.add_argument("--max-scan", type=int, default=200_000,
                     help="stop scanning after this many sentences (speed cap)")
    ap.add_argument("--min-words", type=int, default=4)
    ap.add_argument("--max-words", type=int, default=30)
    args = ap.parse_args()
    if args.output.exists():
        ap.error("output directory already exists; preserve existing candidates")

    excluded = load_excluded_texts()
    print(f"excluding {len(excluded)} already-used/frozen sentences", file=sys.stderr)

    by_category: dict[str, list[dict]] = defaultdict(list)
    seen_texts: set[str] = set(excluded)
    scanned = 0
    parse_crashes = 0

    for row in load_documents_with_context(args.free_dir, SOURCES):
        if scanned >= args.max_scan and all(len(v) >= args.per_category for v in by_category.values()):
            break
        text = row.get("sentence", "").strip()
        if not text:
            continue
        key = normalized(text)
        if key in seen_texts:
            continue
        n_words = len(text.split())
        if not (args.min_words <= n_words <= args.max_words):
            continue
        scanned += 1
        if scanned > args.max_scan:
            break
        try:
            ast = parse(text)
        except Exception:
            parse_crashes += 1
            continue
        hits = classify(ast)
        if not hits:
            continue
        seen_texts.add(key)
        candidate = {
            "id": f"targeted-v1-{hashlib.sha256(key.encode()).hexdigest()[:12]}",
            "source": row["source"], "document_title": row.get("source_title"),
            "author": row.get("author"), "kind": row.get("kind"),
            "text": text, "prev_text": row.get("prev_text"), "next_text": row.get("next_text"),
            "matched_categories": sorted(hits),
        }
        for cat in hits:
            if len(by_category[cat]) < args.per_category:
                by_category[cat].append(candidate)

    print(f"scanned {scanned} sentences, {parse_crashes} parse crashes", file=sys.stderr)

    args.output.mkdir(parents=True)
    all_candidates = {}
    for cat, rows in by_category.items():
        for row in rows:
            all_candidates[row["id"]] = row
    ordered = list(all_candidates.values())
    (args.output / "candidates.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ordered), encoding="utf-8"
    )
    manifest = {
        "version": 1, "free_dir": str(args.free_dir.resolve()),
        "status": "unreviewed targeted candidate queue, not gold -- parser-labeled, not human-labeled",
        "sentences_scanned": scanned, "parse_crashes": parse_crashes,
        "per_category_cap": args.per_category,
        "counts_by_category": {cat: len(rows) for cat, rows in sorted(by_category.items())},
        "total_unique_candidates": len(ordered),
        "sources": dict(Counter(r["source"] for r in ordered)),
        "selection": "sentence qualifies for a category the first time the PARSER'S OWN "
                     "output assigns it that relation (or, for multiroot/pp_ambiguous, that "
                     "structural flag); prev_text/next_text are same-document neighbors for "
                     "review context only, not annotation targets",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
