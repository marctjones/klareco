"""Paired bootstrap CI — the machinery that decides whether a reranker delta is
real (#726). Mirrors klareco/eval/bootstrap.py's self-test as pytest cases."""

from klareco.eval.bootstrap import paired_delta_ci, bootstrap_mrr_ci


def test_identical_arms_are_not_significant():
    # Same ranks question-by-question: paired delta is exactly 0, CI includes 0.
    ranks = [3, 1, 7, 2, 5, 1, 9, 4, 2, 6] * 5
    d = paired_delta_ci(ranks, ranks, seed=1)
    assert d['delta'] == 0.0
    assert not d['significant']
    assert d['lo'] <= 0 <= d['hi']


def test_tiny_gap_small_n_is_inside_the_noise():
    # This is today's J-vs-I shape: a ~0.01 MRR gap on 60 questions. Must NOT be
    # called significant — the whole point of the CI.
    base = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2] * 5   # N=60
    worse = base.copy()
    worse[0] = 9
    d = paired_delta_ci(base, worse, seed=2)
    assert not d['significant']


def test_large_consistent_gap_is_significant():
    good = [1] * 200
    bad = [10] * 200
    d = paired_delta_ci(good, bad, seed=3)
    assert d['significant']
    assert d['lo'] > 0        # CI excludes 0 on the correct side


def test_unequal_length_arms_rejected():
    import pytest
    with pytest.raises(ValueError):
        paired_delta_ci([1, 2, 3], [1, 2])


def test_single_arm_ci_degenerate():
    c = bootstrap_mrr_ci([1] * 100)
    assert c['mrr'] == 1.0 and c['lo'] == 1.0 and c['hi'] == 1.0


def test_none_ranks_count_as_zero_rr():
    # A never-retrieved answer (None) contributes 0 reciprocal rank, not a crash.
    c = bootstrap_mrr_ci([1, None, 2, None])
    assert 0.0 < c['mrr'] < 1.0
