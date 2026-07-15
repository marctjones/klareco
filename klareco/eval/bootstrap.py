"""
Paired bootstrap confidence intervals for reranker MRR — is the delta real?

VERSION: v1.0
COMPATIBLE WITH: any reranker A/B that produces a per-question rank list
DEPENDENCIES: numpy
STAGE: Evaluation

Description:
    THE #726 LESSON, MADE MECHANICAL. On 2026-07-14 we reported J_tree_aware
    (0.716 MRR) vs I_clause_aware (0.725) on 60 questions and treated the 0.009
    gap as if it meant something. It almost certainly does not: a 60-question set
    cannot resolve a delta that small. Point MRRs are not evidence — a confidence
    interval is.

    Two rerankers scored on the SAME questions are a PAIRED comparison. Resample
    the QUESTIONS (not the rerankers) with replacement, recompute BOTH rerankers'
    MRR on each resample, and look at the distribution of the DIFFERENCE. If that
    distribution's 95% interval straddles zero, the two rerankers are
    indistinguishable on this set — however far apart their point estimates look.

    This is what turns "J is 0.009 below I" into either "J is worse (CI [-0.03,
    -0.001], excludes 0)" or "indistinguishable (CI [-0.04, +0.02], includes 0)".
    Only the first kind of statement clears the merge gate.

Why paired (same indices for both arms):
    An UNPAIRED bootstrap resamples each arm independently and inflates the
    variance of the difference — it ignores that a question hard for reranker A is
    usually hard for B too. Pairing (one index vector, applied to both) cancels
    that shared per-question difficulty and is strictly more powerful. Use it
    whenever both arms saw the same questions, which for a reranker A/B they did.

Usage (library):
    from klareco.eval.bootstrap import paired_delta_ci, bootstrap_mrr_ci
    d = paired_delta_ci(ranks_J, ranks_I)          # dict: delta, lo, hi, p, significant
    c = bootstrap_mrr_ci(ranks_J)                   # dict: mrr, lo, hi

Usage (self-test):
    python -m klareco.eval.bootstrap        # fabricates known cases, asserts CI behaviour

Ranks:
    a per-question list where each entry is the 1-based rank of the first relevant
    result, or None if it never appeared. reciprocal rank = 1/rank, or 0 for None.

Last Updated: 2026-07-14
Related Issues: #713, #726, #736
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

Rank = Optional[int]


def _reciprocal_ranks(ranks: Sequence[Rank]) -> np.ndarray:
    """Map a rank list to reciprocal ranks: 1/rank, or 0.0 when the item was
    never retrieved (rank is None or <= 0)."""
    return np.array(
        [1.0 / r if (r is not None and r > 0) else 0.0 for r in ranks],
        dtype=float,
    )


def _resample_indices(n: int, n_boot: int, seed: int) -> np.ndarray:
    # ONE index matrix, reused across arms — that is what makes the test paired.
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_boot, n))


def bootstrap_mrr_ci(ranks: Sequence[Rank], n_boot: int = 10_000,
                     seed: int = 0, alpha: float = 0.05) -> dict:
    """Point MRR and a bootstrap (1-alpha) CI for a single reranker."""
    rr = _reciprocal_ranks(ranks)
    n = len(rr)
    if n == 0:
        return {'mrr': 0.0, 'lo': 0.0, 'hi': 0.0, 'n': 0}
    idx = _resample_indices(n, n_boot, seed)
    boot = rr[idx].mean(axis=1)
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return {'mrr': float(rr.mean()), 'lo': float(lo), 'hi': float(hi), 'n': n}


def paired_delta_ci(ranks_a: Sequence[Rank], ranks_b: Sequence[Rank],
                    n_boot: int = 10_000, seed: int = 0,
                    alpha: float = 0.05) -> dict:
    """Paired bootstrap CI for the MRR difference (arm A minus arm B).

    Both arms must be scored on the SAME questions in the SAME order, so index i
    is the same question in both lists. Returns:
        delta       point estimate  mean(RR_a) - mean(RR_b)
        lo, hi      (1-alpha) CI on the difference
        p           two-sided bootstrap p-value for delta != 0
        significant True iff the CI excludes 0  (i.e. the gate can be cleared)
        n           number of paired questions
    """
    a = _reciprocal_ranks(ranks_a)
    b = _reciprocal_ranks(ranks_b)
    if len(a) != len(b):
        raise ValueError(f'paired arms must be equal length: '
                         f'{len(a)} vs {len(b)} — same questions, same order')
    n = len(a)
    if n == 0:
        return {'delta': 0.0, 'lo': 0.0, 'hi': 0.0, 'p': 1.0,
                'significant': False, 'n': 0}
    idx = _resample_indices(n, n_boot, seed)
    deltas = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    lo, hi = np.quantile(deltas, [alpha / 2, 1 - alpha / 2])
    point = float(a.mean() - b.mean())
    # Two-sided bootstrap p: twice the smaller tail mass on the far side of 0.
    frac_le0 = float((deltas <= 0).mean())
    frac_ge0 = float((deltas >= 0).mean())
    p = min(1.0, 2.0 * min(frac_le0, frac_ge0))
    return {'delta': point, 'lo': float(lo), 'hi': float(hi), 'p': p,
            'significant': bool(lo > 0 or hi < 0), 'n': n}


def _selftest() -> None:
    """Fabricate cases with a KNOWN answer and assert the CI calls them right."""
    rng = np.random.default_rng(42)

    # Case 1: two arms that are IDENTICAL question-by-question -> delta must be 0,
    # CI must include 0, not significant. (Paired cancellation is exact here.)
    ranks = [int(x) for x in rng.integers(1, 11, size=200)]
    d = paired_delta_ci(ranks, ranks, seed=1)
    assert abs(d['delta']) < 1e-9, d
    assert not d['significant'], d
    assert d['lo'] <= 0 <= d['hi'], d
    print(f"  identical arms      : delta={d['delta']:+.4f} "
          f"CI[{d['lo']:+.4f},{d['hi']:+.4f}] sig={d['significant']}  (want not-sig)")

    # Case 2: a tiny gap on a small N -> must be declared INSIDE the noise.
    # This is exactly today's J-vs-I shape: ~0.01 MRR gap, 60 questions.
    base = [int(x) for x in rng.integers(1, 6, size=60)]
    worse = base.copy()
    worse[0] = 9  # perturb one question a little
    d2 = paired_delta_ci(base, worse, seed=2)
    print(f"  tiny gap, N=60      : delta={d2['delta']:+.4f} "
          f"CI[{d2['lo']:+.4f},{d2['hi']:+.4f}] p={d2['p']:.3f} sig={d2['significant']}"
          f"  (want not-sig)")
    assert not d2['significant'], d2

    # Case 3: a LARGE, consistent gap -> must be significant, CI excludes 0.
    good = [1] * 200
    bad = [10] * 200
    d3 = paired_delta_ci(good, bad, seed=3)
    assert d3['significant'] and d3['lo'] > 0, d3
    print(f"  large gap           : delta={d3['delta']:+.4f} "
          f"CI[{d3['lo']:+.4f},{d3['hi']:+.4f}] p={d3['p']:.3f} sig={d3['significant']}"
          f"  (want SIG)")

    # Case 4: single-arm CI sanity — all rank-1 -> mrr 1.0, zero-width CI.
    c = bootstrap_mrr_ci([1] * 100)
    assert abs(c['mrr'] - 1.0) < 1e-9 and c['lo'] == 1.0 == c['hi'], c
    print(f"  single-arm all-R1   : mrr={c['mrr']:.4f} CI[{c['lo']:.4f},{c['hi']:.4f}]")

    print("\n  ✓ all bootstrap self-tests passed")


if __name__ == '__main__':
    _selftest()
