#!/usr/bin/env python3
"""Offline constrained decoding over the parser's explicit candidate graph."""

from __future__ import annotations

import argparse
import json
import sys
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from klareco.parser import parse
from eval_conllu import read_gold


_APERTIUM_PATH = Path("data/raw/eo/dictionaries/apertium_lexicon.json")
if not _APERTIUM_PATH.exists():
    raise FileNotFoundError(f"Required lexical resource is missing: {_APERTIUM_PATH}")
_APERTIUM = json.loads(_APERTIUM_PATH.read_text(encoding="utf-8"))["entries"]
_VALENCY = {
    entry["stem"]: ("transitive" if "vbtr" in entry.get("par", "") else "intransitive")
    for entry in _APERTIUM.values()
    if isinstance(entry, dict) and entry.get("pos") == "verbo" and entry.get("stem")
}


def _cycle(assignments: dict[int, tuple[int, str]], token: int, head: int) -> bool:
    seen = {token}
    node = head
    while node:
        if node in seen:
            return True
        seen.add(node)
        node = assignments.get(node, (0, ""))[0]
    return False


def _score(token: dict, head: dict, option: dict, baseline: tuple[int, str], counts: dict[int, int]) -> float:
    relation = option["rolo"]
    score = 8.0 if option["kapo"] == baseline[0] and relation == baseline[1] else 0.0
    if token.get("kazo") == "akuzativo" and relation in {"obj", "obl"}:
        score += 3.0
    if token.get("kazo") == "nominativo" and relation == "nsubj":
        score += 3.0
    if head.get("vortspeco") == "verbo" and relation in {"nsubj", "obj", "obl", "xcomp", "ccomp"}:
        score += 1.5
    suffixes = {str(s).lower() for s in (head.get("sufiksoj") or [])}
    if relation in {"obj", "obl"} and "iĝ" in suffixes:
        score -= 3.0
    if relation == "obj" and "ig" in suffixes:
        score += 2.0
    valency = _VALENCY.get(head.get("tigo"))
    if relation == "obj" and valency == "transitive":
        score += 2.5
    if relation in {"obj", "obl"} and valency == "intransitive":
        score -= 2.5
    if relation == "obj" and counts.get(head["id"], 0):
        score -= 2.0
    score -= 0.05 * abs(token["id"] - head["id"])
    return score


def _decode(ast: dict) -> tuple[dict[int, tuple[int, str]], float]:
    words = [word for word in ast.get("vortoj", []) if isinstance(word, dict)]
    by_id = {word["id"]: word for word in words}
    baseline = {word["id"]: (word.get("kapo", 0), word.get("rolo", "dep")) for word in words}
    ambiguous = [word for word in words if word.get("alligo_opcioj")]
    ambiguous.sort(key=lambda word: (len(word["alligo_opcioj"]), word["id"]))
    states: list[tuple[float, dict[int, tuple[int, str]], dict[int, int]]] = [(0.0, {}, {})]
    for token in ambiguous:
        next_states = []
        options = list(token["alligo_opcioj"])
        baseline_option = {"kapo": baseline[token["id"]][0], "rolo": baseline[token["id"]][1]}
        if baseline_option not in options:
            options.append(baseline_option)
        for score, assignments, counts in states:
            for option in options:
                head_id = option["kapo"]
                if head_id not in by_id and head_id != 0 or head_id == token["id"]:
                    continue
                if _cycle(assignments, token["id"], head_id):
                    continue
                head = by_id.get(head_id, {"id": 0, "vortspeco": "radiko"})
                updated = dict(assignments)
                updated[token["id"]] = (head_id, option["rolo"])
                updated_counts = dict(counts)
                if option["rolo"] == "obj":
                    updated_counts[head_id] = updated_counts.get(head_id, 0) + 1
                next_states.append((
                    score + _score(token, head, option, baseline[token["id"]], counts),
                    updated,
                    updated_counts,
                ))
        next_states.sort(key=lambda item: item[0], reverse=True)
        states = next_states[:64]
    if not states:
        return baseline, 0.0
    return states[0][1], states[0][0]


def measure(path: Path) -> dict:
    totals = {"tokens": 0, "baseline_las": 0, "decoded_las": 0, "sentences": 0, "decode_failures": 0}
    for sentence in read_gold(str(path)):
        ast = parse(sentence["text"])
        words = [word for word in ast.get("vortoj", []) if isinstance(word, dict)]
        forms = [str(word.get("plena_vorto", "")).lower() for word in words]
        gold_forms = [token["form"].lower() for token in sentence["tokens"]]
        alignment: dict[int, int] = {}
        for oi, gi, size in SequenceMatcher(a=forms, b=gold_forms, autojunk=False).get_matching_blocks():
            for offset in range(size):
                alignment[words[oi + offset]["id"]] = sentence["tokens"][gi + offset]["id"]
        by_gold = {alignment[word["id"]]: word for word in words if word["id"] in alignment}
        baseline = {word["id"]: (word.get("kapo", 0), word.get("rolo", "dep")) for word in words}
        decoded, _ = _decode(ast)
        totals["sentences"] += 1
        for token in sentence["tokens"]:
            if token["upos"] == "PUNCT" or token["id"] not in by_gold:
                continue
            word = by_gold[token["id"]]
            totals["tokens"] += 1
            for edge, key in ((baseline[word["id"]], "baseline_las"), (decoded.get(word["id"], baseline[word["id"]]), "decoded_las")):
                head = 0 if not edge[0] else alignment.get(edge[0], -1)
                if head == token["head"] and edge[1] == token["dep"]:
                    totals[key] += 1
    totals["baseline_las_rate"] = totals["baseline_las"] / totals["tokens"] if totals["tokens"] else 0.0
    totals["decoded_las_rate"] = totals["decoded_las"] / totals["tokens"] if totals["tokens"] else 0.0
    totals["las_delta"] = totals["decoded_las"] - totals["baseline_las"]
    return totals


def main() -> None:
    parser = argparse.ArgumentParser()
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
