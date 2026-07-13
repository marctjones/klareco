"""Tests for the test-set construction gates (R7 floor + R16 ceiling).

These guard the generator's fail-closed write gate. If they regress, we start
producing saturated test sets again — sets on which nine different rerankers
tie because BM25 already won, and no A/B can ever show anything. See #778.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "eval"))

from build_synthetic_qa_test_set import passes_r7_r16  # noqa: E402


class TestR7Floor:
    """The gold passage must be findable at all, or the question is
    unanswerable and tells us nothing about the pipeline."""

    def test_not_in_topk_is_rejected(self):
        ok, why = passes_r7_r16(None, enforce_r16=True)
        assert not ok
        assert "r7_floor" in why

    def test_floor_applies_even_when_r16_is_off(self):
        """--allow-trivial relaxes the ceiling, never the floor. An
        unanswerable question is worthless in every kind of set."""
        ok, why = passes_r7_r16(None, enforce_r16=False)
        assert not ok
        assert "r7_floor" in why


class TestR16Ceiling:
    """The gold passage must NOT already be at BM25 rank 1.

    This is the check whose absence produced synthetic_who_rebuild_17_cleanish
    (58.8% rank-1, median gold rank 1) and, downstream, nine tied rerankers.
    """

    def test_rank_1_is_rejected_as_trivial(self):
        ok, why = passes_r7_r16(1, enforce_r16=True)
        assert not ok
        assert "r16_ceiling" in why

    def test_rank_2_is_the_measurable_band(self):
        ok, _ = passes_r7_r16(2, enforce_r16=True)
        assert ok

    def test_deep_rank_is_kept_hard_but_possible(self):
        """Rank 47 of 50 is exactly the target: retrievable from a generous
        pool, but nowhere near won by lexical overlap alone."""
        ok, _ = passes_r7_r16(47, enforce_r16=True)
        assert ok

    def test_rank_1_is_allowed_when_ceiling_disabled(self):
        """A regression set legitimately contains pairs we already answer —
        that is its whole purpose. The flag exists for that, and for nothing
        else."""
        ok, _ = passes_r7_r16(1, enforce_r16=False)
        assert ok


class TestBandSemantics:
    def test_the_band_is_findable_but_not_won(self):
        """Restating the contract as a property: a pair is measurable iff it
        is inside the pool and outside rank 1."""
        for rank in (2, 3, 10, 50, 200):
            assert passes_r7_r16(rank, enforce_r16=True)[0]
        for rank in (None, 1):
            assert not passes_r7_r16(rank, enforce_r16=True)[0]
