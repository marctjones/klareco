#!/usr/bin/env python3
"""Measure which parser errors are candidate-generation versus selection errors.

This is a diagnostic upper-bound measurement, not a new parser score.  For every
aligned token it compares the selected dependency with the gold dependency and
checks whether the parser's explicit ``syntax.attachment_candidates`` already
contains the gold head.  A gold head present in the candidate set identifies a
selection/refinement opportunity; an absent head identifies missing structural
candidate generation (or an alignment/morphology failure).
"""

from __future__ import annotations

import argparse
import json
import sys
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from klareco.parser import parse
from eval_conllu import read_gold


def _surface(word: dict) -> str:
    return str(word.get("plena_vorto") or word.get("surface") or "")


def _candidate_heads(ast: dict, token_id: int) -> set[int]:
    heads: set[int] = set()
    for candidate in ast.get("syntax", {}).get("attachment_candidates", []):
        if candidate.get("token_id") != token_id:
            continue
        selected = candidate.get("selected") or {}
        if isinstance(selected.get("head_id"), int):
            heads.add(selected["head_id"])
        for option in candidate.get("options", []):
            if isinstance(option.get("head_id"), int):
                heads.add(option["head_id"])
    return heads


def _candidate_edges(ast: dict, token_id: int) -> set[tuple[int, str]]:
    edges: set[tuple[int, str]] = set()
    for candidate in ast.get("syntax", {}).get("attachment_candidates", []):
        if candidate.get("token_id") != token_id:
            continue
        for option in candidate.get("options", []):
            if isinstance(option.get("head_id"), int) and isinstance(option.get("relation"), str):
                edges.add((option["head_id"], option["relation"]))
    return edges


def measure(path: Path) -> dict:
    gold_sentences = read_gold(str(path))
    totals = {
        "gold_tokens": 0,
        "aligned": 0,
        "head_correct": 0,
        "las_correct": 0,
        "candidate_head_recall": 0,
        "candidate_edge_recall": 0,
        "candidate_head_recall_on_head_errors": 0,
        "head_errors": 0,
        "head_errors_with_gold_candidate": 0,
        "head_errors_without_gold_candidate": 0,
        "relation_errors_with_gold_head": 0,
        "unaligned": 0,
        "crashed_sentences": 0,
    }
    by_relation: dict[str, dict[str, int]] = {}

    totals["gold_tokens"] = sum(
        1 for sentence in gold_sentences
        for token in sentence["tokens"]
        if token["upos"] != "PUNCT"
    )

    for sentence_index, sentence in enumerate(gold_sentences, start=1):
        text = sentence["text"] or " ".join(t["form"] for t in sentence["tokens"])
        try:
            ast = parse(text)
        except Exception:
            totals["crashed_sentences"] += 1
            continue
        ours = [w for w in ast.get("vortoj", []) if isinstance(w, dict)]
        gold = sentence["tokens"]
        ours_forms = [_surface(w).lower() for w in ours]
        gold_forms = [t["form"].lower() for t in gold]
        ours_to_gold: dict[int, int] = {}
        for oi, gi, size in SequenceMatcher(
                a=ours_forms, b=gold_forms, autojunk=False).get_matching_blocks():
            for offset in range(size):
                ours_to_gold[ours[oi + offset]["id"]] = gold[gi + offset]["id"]
        by_gold = {
            ours_to_gold[w["id"]]: w
            for w in ours
            if w["id"] in ours_to_gold
        }
        for gold_token in gold:
            if gold_token["upos"] == "PUNCT":
                continue
            ours_token = by_gold.get(gold_token["id"])
            if ours_token is None:
                totals["unaligned"] += 1
                continue
            totals["aligned"] += 1
            predicted_head = (
                0 if ours_token.get("kapo") in (None, 0)
                else ours_to_gold.get(ours_token.get("kapo"), -1)
            )
            head_ok = predicted_head == gold_token["head"]
            relation_ok = ours_token.get("rolo") == gold_token["dep"]
            if head_ok:
                totals["head_correct"] += 1
                if relation_ok:
                    totals["las_correct"] += 1
                else:
                    totals["relation_errors_with_gold_head"] += 1
            else:
                totals["head_errors"] += 1

            candidate_heads = {
                0 if head == 0 else ours_to_gold.get(head, -1)
                for head in _candidate_heads(ast, ours_token["id"])
            }
            candidate_has_gold = gold_token["head"] in candidate_heads
            candidate_edges = {
                (0 if head == 0 else ours_to_gold.get(head, -1), relation)
                for head, relation in _candidate_edges(ast, ours_token["id"])
            }
            candidate_has_gold_edge = (gold_token["head"], gold_token["dep"]) in candidate_edges
            if candidate_has_gold:
                totals["candidate_head_recall"] += 1
            if candidate_has_gold_edge:
                totals["candidate_edge_recall"] += 1
            if not head_ok:
                if candidate_has_gold:
                    totals["candidate_head_recall_on_head_errors"] += 1
                    totals["head_errors_with_gold_candidate"] += 1
                else:
                    totals["head_errors_without_gold_candidate"] += 1
            bucket = by_relation.setdefault(gold_token["dep"], {
                "tokens": 0,
                "head_errors": 0,
                "head_errors_with_gold_candidate": 0,
            })
            bucket["tokens"] += 1
            if not head_ok:
                bucket["head_errors"] += 1
                if candidate_has_gold:
                    bucket["head_errors_with_gold_candidate"] += 1

    aligned = totals["aligned"]
    head_errors = totals["head_errors"]
    totals["uas"] = totals["head_correct"] / aligned if aligned else 0.0
    totals["las"] = totals["las_correct"] / aligned if aligned else 0.0
    totals["candidate_head_recall_rate"] = (
        totals["candidate_head_recall"] / aligned if aligned else 0.0
    )
    totals["candidate_edge_recall_rate"] = (
        totals["candidate_edge_recall"] / aligned if aligned else 0.0
    )
    totals["candidate_recall_on_head_errors_rate"] = (
        totals["candidate_head_recall_on_head_errors"] / head_errors
        if head_errors else 0.0
    )
    return {"path": str(path), "metrics": totals, "by_gold_relation": by_relation}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, default=[
        Path("data/external/ud_esperanto_prago/eo_prago-ud-test.conllu"),
        Path("data/external/ud_esperanto_cairo/eo_cairo-ud-test.conllu"),
    ])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {path.stem: measure(path) for path in args.paths}
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
