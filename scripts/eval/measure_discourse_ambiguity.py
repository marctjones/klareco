#!/usr/bin/env python3
"""Measure the cross-sentence ambiguity residues in klareco/discourse/ on
real corpus text.

Covers every resolver in the package: pronoun/demonstrative coreference
(klareco.discourse.coreference) and proper-noun classification consistency
(klareco.discourse.proper_noun_consistency). For each, reports how often
document-level context resolves the ambiguity to exactly one candidate,
how often it finds a genuine, honestly-flagged ambiguity, and how often it
finds nothing at all (residue this window/document size cannot reach).

This is a measurement instrument, not an accuracy evaluator: neither
resolver has a hand-labeled gold set, so this cannot report precision or
recall. What it reports is honest residue-sizing (VISION.md's own bar:
"how big is it?"), plus per-status example sentences a human can
spot-check.

Usage:
    python scripts/eval/measure_discourse_ambiguity.py \
        --documents 1000 --window 1 \
        --output data/perf/parser_research/discourse_ambiguity_measurement.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import duckdb

from klareco.discourse import coreference, proper_noun_consistency
from klareco.parser import parse


def _document_groups(con, n_documents: int, seed: int) -> list[list[tuple]]:
    """Sample N documents (grouped by article_id) with >= 3 sentences each,
    returning each as a list of (sid, section, text) IN STORE ORDER (sid is
    monotonic within a document -- verified by inspection, not assumed)."""
    # DuckDB's USING SAMPLE clause does not accept prepared-statement
    # placeholders for its arguments (n_documents/seed are argparse ints,
    # not user-supplied strings, so direct interpolation is safe); it also
    # returns nothing when applied directly to a GROUP BY/HAVING query, so
    # the qualifying set is materialized in a CTE first, then sampled.
    article_ids = [
        r[0] for r in con.execute(
            "WITH qualifying AS ("
            "  SELECT article_id FROM sentences WHERE article_id IS NOT NULL"
            "  GROUP BY article_id HAVING count(*) >= 3"
            ") "
            "SELECT article_id FROM qualifying "
            f"USING SAMPLE reservoir({int(n_documents)} ROWS) REPEATABLE ({int(seed)})"
        ).fetchall()
    ]
    groups = []
    for aid in article_ids:
        rows = con.execute(
            "SELECT sid, section, text FROM sentences "
            "WHERE article_id = ? ORDER BY sid", [aid],
        ).fetchall()
        groups.append(rows)
    return groups


class _Tally:
    def __init__(self):
        self.status_counts = Counter()
        self.type_status_counts = Counter()
        self.candidate_count_when_ambiguous = Counter()
        self.total_mentions = 0
        self.examples = {"resolved": [], "ambiguous": [], "unresolved": []}

    def add(self, mention_type: str, res: dict, sentence_text: str, surface: str):
        self.total_mentions += 1
        self.status_counts[res["status"]] += 1
        self.type_status_counts[(mention_type, res["status"])] += 1
        if res["status"] == "ambiguous":
            self.candidate_count_when_ambiguous[len(res["candidates"])] += 1
        if len(self.examples[res["status"]]) < 15:
            self.examples[res["status"]].append({
                "sentence": sentence_text,
                "mention": surface,
                "candidates": res["candidates"],
            })

    def summary(self, window: int) -> dict:
        n = self.total_mentions
        return {
            "total_mentions": n,
            "status_counts": dict(self.status_counts),
            "status_rate": {k: round(v / n, 4) if n else 0.0
                             for k, v in self.status_counts.items()},
            "by_type_and_status": {f"{t}:{s}": c
                                    for (t, s), c in self.type_status_counts.items()},
            "ambiguous_candidate_count_histogram": dict(self.candidate_count_when_ambiguous),
            "examples": self.examples,
            "window": window,
        }


def measure(con, n_documents: int, window: int, seed: int) -> dict:
    entity_types = coreference.build_entity_type_lookup(con)
    groups = _document_groups(con, n_documents, seed)

    coref_tally = _Tally()
    propn_tally = _Tally()
    parse_failures = 0

    for rows in groups:
        parsed: list[tuple] = []
        text_by_sid: dict[int, str] = {}
        for sid, section, text in rows:
            try:
                parsed.append((sid, parse(text)))
                text_by_sid[sid] = text
            except Exception:
                parse_failures += 1
                continue

        for m, res in coreference.resolve_document(parsed, entity_types, window=window):
            coref_tally.add(m["pronoun"], res, text_by_sid.get(m["sid"], ""), m["surface"])

        for m, res in proper_noun_consistency.resolve_document(parsed):
            propn_tally.add("propra_nomo", res, text_by_sid.get(m["sid"], ""), m["surface"])

    return {
        "documents_sampled": len(groups),
        "parse_failures": parse_failures,
        "coreference": coref_tally.summary(window),
        "proper_noun_classification": propn_tally.summary(window),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", type=Path, default=ROOT / "data/indexes/duckdb_store.db")
    ap.add_argument("--documents", type=int, default=1000)
    ap.add_argument("--window", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    con = duckdb.connect(str(args.store), read_only=True)
    result = measure(con, args.documents, args.window, args.seed)

    print(f"documents sampled: {result['documents_sampled']}")
    print(f"parse failures: {result['parse_failures']}")
    for section in ("coreference", "proper_noun_classification"):
        r = result[section]
        print(f"--- {section} ---")
        print(f"  total mentions: {r['total_mentions']}")
        print(f"  status counts: {r['status_counts']}")
        print(f"  status rate: {r['status_rate']}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
