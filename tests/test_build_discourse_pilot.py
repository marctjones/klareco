"""scripts/eval/build_discourse_pilot.py -- multi-sentence passage queue
selection logic (no store needed; pure functions over synthetic documents).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.eval.build_discourse_pilot import document_split, select


def _doc(article_id: str, n: int, source: str = "wikipedia", section: str = None):
    """A synthetic document: n sentences, sid = int(article_id)*100 + i."""
    base = int(article_id) * 100
    return article_id, [
        (base + i, f"Frazo {i} de dokumento {article_id}.", section, source)
        for i in range(n)
    ]


class TestDocumentSplit:
    def test_split_is_deterministic(self):
        assert document_split("12345") == document_split("12345")

    def test_split_only_ever_two_values(self):
        for aid in [str(i) for i in range(50)]:
            assert document_split(aid) in ("development", "heldout")

    def test_roughly_one_fifth_go_to_heldout(self):
        splits = [document_split(str(i)) for i in range(2000)]
        heldout_rate = splits.count("heldout") / len(splits)
        assert 0.15 < heldout_rate < 0.25


class TestSelect:
    def test_a_short_document_is_skipped(self):
        documents = dict([_doc("1", 3)])   # shorter than passage_length
        with __import__("pytest").raises(ValueError):
            select(documents, excluded=set(), size=1, passage_length=8)

    def test_only_the_first_passage_length_sentences_are_used(self):
        documents = dict([_doc(str(i), 12) for i in range(10)])
        chosen = select(documents, excluded=set(), size=5, passage_length=8)
        for row in chosen:
            assert len(row["sentences"]) == 8
            assert [s["sid"] for s in row["sentences"]] == sorted(
                s["sid"] for s in row["sentences"])

    def test_a_document_whose_opening_matches_an_excluded_fixture_is_skipped(self):
        documents = dict([_doc(str(i), 10) for i in range(20)])
        # Exclude document "2"'s exact opening sentence text.
        excluded = {"frazo 0 de dokumento 2."}
        chosen = select(documents, excluded=excluded, size=10, passage_length=8)
        assert all(row["article_id"] != "2" for row in chosen)

    def test_splits_are_document_disjoint(self):
        documents = dict([_doc(str(i), 10) for i in range(30)])
        chosen = select(documents, excluded=set(), size=20, passage_length=8)
        article_ids = [row["article_id"] for row in chosen]
        assert len(article_ids) == len(set(article_ids)), "one passage per document"
        for row in chosen:
            assert row["split"] == document_split(row["article_id"])

    def test_raises_if_not_enough_qualifying_documents(self):
        documents = dict([_doc(str(i), 10) for i in range(3)])
        with __import__("pytest").raises(ValueError, match="eligible"):
            select(documents, excluded=set(), size=10, passage_length=8)
