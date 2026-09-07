#!/usr/bin/env python3
"""Evaluate conservative local selectors without changing the runtime parser.

This spike deliberately limits itself to accusative nominal ``obj``/``obl``
alternatives.  It is a negative-control experiment: a local selector is useful
only if it improves LAS while preserving the rest of the selected tree.
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


MODES = ("nearest", "finite-nearest", "rightmost", "finite-rightmost")


def _score(token: dict, head: dict, relation: str, mode: str) -> tuple:
    distance = abs(head["id"] - token["id"])
    is_verb = head.get("vortspeco") == "verbo"
    is_finite = head.get("tempo") is not None or head.get("finaĵo") in {
        "as", "is", "os", "us", "u"
    }
    if mode == "nearest":
        return (is_verb, -distance)
    if mode == "finite-nearest":
        return (is_verb and is_finite, is_verb, -distance)
    if mode == "rightmost":
        return (is_verb, head["id"])
    return (is_verb and is_finite, is_verb, head["id"])


def _measure(path: Path, mode: str) -> dict:
    total = baseline = selected_las = selected = 0
    gold_sentences = read_gold(str(path))
    for sentence in gold_sentences:
        ast = parse(sentence["text"])
        words = [word for word in ast.get("vortoj", []) if isinstance(word, dict)]
        forms = [str(word.get("plena_vorto", "")).lower() for word in words]
        gold_forms = [token["form"].lower() for token in sentence["tokens"]]
        ours_to_gold: dict[int, int] = {}
        for oi, gi, size in SequenceMatcher(
            a=forms, b=gold_forms, autojunk=False
        ).get_matching_blocks():
            for offset in range(size):
                ours_to_gold[words[oi + offset]["id"]] = sentence["tokens"][gi + offset]["id"]
        by_gold = {
            ours_to_gold[word["id"]]: word
            for word in words
            if word["id"] in ours_to_gold
        }
        baseline_edges = {
            word["id"]: (word.get("kapo"), word.get("rolo")) for word in words
        }
        for word in words:
            if word.get("kazo") != "akuzativo":
                continue
            options = [
                option for option in word.get("alligo_opcioj", [])
                if option.get("rolo") in {"obj", "obl"}
            ]
            if not options:
                continue
            heads = {head["id"]: head for head in words}
            ranked = [
                (_score(word, heads[option["kapo"]], option["rolo"], mode), option)
                for option in options
                if option.get("kapo") in heads
            ]
            if ranked:
                _, option = max(ranked, key=lambda item: item[0])
                word["kapo"], word["rolo"] = option["kapo"], option["rolo"]
                selected += 1
        for token in sentence["tokens"]:
            if token["upos"] == "PUNCT" or token["id"] not in by_gold:
                continue
            word = by_gold[token["id"]]
            predicted_head = 0 if not word.get("kapo") else ours_to_gold.get(word["kapo"], -1)
            total += 1
            baseline_head = (
                0
                if not baseline_edges[word["id"]][0]
                else ours_to_gold.get(baseline_edges[word["id"]][0], -1)
            )
            if baseline_head == token["head"] and baseline_edges[word["id"]][1] == token["dep"]:
                baseline += 1
            if predicted_head == token["head"] and word.get("rolo") == token["dep"]:
                selected_las += 1
    return {
        "tokens": total,
        "baseline_las_correct": baseline,
        "baseline_las": baseline / total if total else 0.0,
        "selected_las_correct": selected_las,
        "selected_las": selected_las / total if total else 0.0,
        "las_delta": selected_las - baseline,
        "selected": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, default=[
        Path("data/external/ud_esperanto_prago/eo_prago-ud-test.conllu"),
        Path("data/external/ud_esperanto_cairo/eo_cairo-ud-test.conllu"),
    ])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {
        path.stem: {mode: _measure(path, mode) for mode in MODES}
        for path in args.paths
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
