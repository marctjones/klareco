"""Accuracy regression — did this change make the system WORSE?

This tier did not exist. `data/perf/bench_history.jsonl` was being WRITTEN by the
bench scripts and never READ by anything. A baseline nobody asserts against is a
diary, not a gate.

This is the tier the merge gate (#784) stands on:

    "No capability merges without a number that moved."

…which requires (a) a recorded number, and (b) something that notices when it
moves the wrong way. (a) existed. (b) is here.

What this tier does NOT do
--------------------------
It does not re-run the pipeline — that takes minutes and needs the 32 GB store.
It asserts on the RECORD: the numbers a bench run appended. The workflow is:

    1. make a change
    2. run the bench      -> appends to bench_history.jsonl
    3. pytest -m accuracy -> fails if you regressed against the best recorded run

Honest caveat, deliberately loud
--------------------------------
The current baselines were measured on a SATURATED test set
(synthetic_who_rebuild_17_cleanish: 58.8% of its pairs have the gold passage
already at BM25 rank 1 — it is the set on which all nine rerankers tied). Those
numbers are therefore not yet a meaningful quality signal, and this tier says so
rather than lending them false authority.

When the discriminating set lands (#778 + #783), re-baseline and these
assertions become real. Until then they still catch a CATASTROPHIC regression
(a change that halves recall), which is worth having.
"""

import json
from pathlib import Path

import pytest

BENCH_HISTORY = Path('data/perf/bench_history.jsonl')

# A regression this large is a bug, not noise — even on a saturated test set.
CATASTROPHIC_DROP = 0.20   # 20 percentage points


def _entries() -> list[dict]:
    if not BENCH_HISTORY.exists():
        pytest.skip(f'{BENCH_HISTORY} absent — run a bench first')
    rows = [json.loads(l) for l in BENCH_HISTORY.read_text().splitlines() if l.strip()]
    if not rows:
        pytest.skip('bench_history.jsonl is empty')
    return rows


class TestBaselineRecordIsUsable:
    """If the record itself is broken, every downstream assertion is theatre."""

    def test_history_exists_and_parses(self):
        rows = _entries()
        assert rows, 'no bench runs recorded'

    def test_every_run_is_attributable(self):
        """A benchmark number with no commit is unusable as evidence: you cannot
        say WHAT moved the number. The merge gate requires attribution."""
        for r in _entries():
            assert r.get('git_commit'), f'bench run at {r.get("timestamp")} has no git_commit'
            assert r.get('test_set'), f'bench run at {r.get("timestamp")} names no test set'
            assert r.get('n_questions'), 'bench run records no question count'

    def test_runs_record_a_metric(self):
        """bench_history holds several BENCHMARK TYPES, each with its own shape:

            reranker A/B  -> {'rerankers': {name: {...}}}
            retriever A/B -> {'retrievers': {name: {...}}}
            parser (UD)   -> {'metrics': {...}}          <- flat; one system

        A run with none of these recorded nothing.
        """
        for r in _entries():
            has = r.get('rerankers') or r.get('retrievers') or r.get('metrics')
            assert has, f'bench run at {r.get("timestamp")} recorded no metrics at all'

    def test_degraded_runs_say_so(self):
        """A number measured on a broken instrument must carry that fact FOREVER,
        or someone will cite it in six months.

        The UD-Prago baseline is stamped `degraded: true` with the reason (no
        proper-noun dictionary — which is exactly what its 27.6% proper-noun F1
        measures). Any run flagged degraded must explain why.
        """
        for r in _entries():
            if r.get('degraded'):
                assert r.get('degraded_reason'), (
                    f'run at {r.get("timestamp")} is flagged degraded but does not '
                    f'say WHY — an unexplained caveat is no caveat at all')


class TestNoCatastrophicRegression:
    """Compare the latest run against the best PRIOR run ON THE SAME TEST SET.

    Comparing across test sets is meaningless — and it is the exact error the
    merge gate must never make. `recall_at_5 = 17` on a 17-question set and
    `recall_at_5 = 41` on a 50-question set are numbers from two different
    instruments; a "regression" between them is an artifact of arithmetic, not a
    fact about the system.

    (My first version of this test made precisely that mistake and reported a
    48-point collapse that never happened. Which is a decent argument for why
    this tier needed to exist.)
    """

    def _best_score(self, run: dict, metric: str):
        """Best score any reranker achieved in this run, as a FRACTION."""
        metrics = run.get('rerankers') or run.get('retrievers') or {}
        vals = [m[metric] for m in metrics.values() if m.get(metric) is not None]
        if not vals:
            return None
        best = max(vals)
        # recall_at_k is recorded as a COUNT of questions; mrr as a fraction.
        if metric.startswith('recall'):
            n = run.get('n_questions') or 1
            return best / n
        return best

    @pytest.mark.parametrize('metric', ['recall_at_1', 'recall_at_5', 'mrr'])
    def test_latest_run_has_not_collapsed(self, metric):
        rows = _entries()
        latest = rows[-1]
        test_set = latest.get('test_set')

        latest_score = self._best_score(latest, metric)
        if latest_score is None:
            pytest.skip(f'{metric} not recorded in the latest run')

        # Only comparable runs: same test set, and BEFORE the latest one.
        priors = [r for r in rows[:-1] if r.get('test_set') == test_set]
        prior_scores = [s for s in (self._best_score(r, metric) for r in priors)
                        if s is not None]
        if not prior_scores:
            pytest.skip(
                f'no prior run on {Path(test_set).name} to compare against — '
                f'this run IS the baseline for {metric}')

        best_prior = max(prior_scores)
        drop = best_prior - latest_score
        assert drop < CATASTROPHIC_DROP, (
            f'{metric} collapsed on {Path(test_set).name}: best prior '
            f'{best_prior:.1%}, latest {latest_score:.1%} (drop {drop:.1%}). '
            f'This is a severe regression on the SAME test set — not a '
            f'measurement artifact.')


class TestTheBaselineIsHonest:
    """Guard against the failure that produced this whole milestone: trusting a
    number from an instrument that cannot measure."""

    def test_saturated_test_sets_are_flagged_not_trusted(self):
        """`synthetic_who_rebuild_17_cleanish` has 58.8% of its pairs at BM25
        rank 1. All nine rerankers tied on it. Any baseline drawn from it is not
        a quality signal, and must not be quietly treated as one.

        This test does not fail — it FLAGS. It exists so that the moment someone
        re-baselines on a discriminating set (#778), the warning disappears on
        its own.
        """
        saturated = {'synthetic_who_rebuild_17_cleanish.jsonl',
                     'synthetic_who_rebuild_50.jsonl'}
        latest = _entries()[-1]
        ts = Path(latest.get('test_set', '')).name
        if ts in saturated:
            pytest.skip(
                f'BASELINE IS NOT TRUSTWORTHY: the latest bench ran on {ts}, '
                f'which is SATURATED (BM25 already wins). Reranking cannot be '
                f'measured on it — see #778. Re-baseline on capability_100.')
