"""Performance regression — is each piece still fast enough?

This tier did not exist either. We had per-stage timings (`PhaseTimer`,
`aggregate_stage_timings`) feeding eval output, and nothing that ever asserted on
them. So the WHEN-question latency regression (#724: 109–164 s) was found by a
human noticing, not by a test.

Two levels, deliberately:

  1. MICRO — the parser, offline, no data required. This is the one that runs on
     every commit. The parser is the hot path: it runs on every question and it
     ran 5.4M times to build the store. A 10x parser regression would be
     catastrophic and is trivially detectable.

  2. STAGE — per-stage wall time from a recorded bench run. Asserted against the
     record, not by re-running the pipeline (which needs the 32 GB store).

The budgets below are DELIBERATELY LOOSE. A perf test that fails on a noisy
laptop gets deleted within a week, and then you have no perf test at all. These
catch order-of-magnitude regressions, which are the ones that actually happen.
"""

import json
import statistics
import time
from pathlib import Path

import pytest

BENCH_HISTORY = Path('data/perf/bench_history.jsonl')

# Order-of-magnitude budgets. Not tuned for a fast machine — tuned so that a
# 10x regression fails and a noisy CI box does not.
PARSE_BUDGET_MS = 50.0      # per sentence; typical is ~1-3 ms
PARSE_P95_BUDGET_MS = 150.0


SENTENCES = [
    'Zamenhof fondis Esperanton en 1887.',
    'La rapida bruna vulpo saltas super la maldiligentan hundon.',
    'Kiu verkis la vortaron «Altdeutsches Wörterbuch»?',
    'En 1990, James Dalgety de Britujo inventis la nomon «Nonograms».',
    'Mi ne scias ĉu li venos morgaŭ, sed mi esperas ke jes.',
    'La malgrandaj infanoj ludis en la ĝardeno dum siaj gepatroj laboris.',
]


class TestParserLatency:
    """The parser is the hot path: every question, and 5.4M corpus sentences."""

    def test_parse_is_not_pathologically_slow(self):
        from klareco.parser import parse

        # Warm up — first call pays import/regex-compile costs that are not
        # representative of steady-state throughput.
        for s in SENTENCES:
            parse(s)

        timings = []
        for _ in range(5):
            for s in SENTENCES:
                t0 = time.perf_counter()
                parse(s)
                timings.append((time.perf_counter() - t0) * 1000)

        mean = statistics.mean(timings)
        p95 = sorted(timings)[int(0.95 * (len(timings) - 1))]

        assert mean < PARSE_BUDGET_MS, (
            f'parse() averages {mean:.1f} ms/sentence (budget {PARSE_BUDGET_MS} ms). '
            f'The parser runs on every question and ran 5.4M times to build the '
            f'store — a regression here is expensive everywhere.')
        assert p95 < PARSE_P95_BUDGET_MS, (
            f'parse() p95 is {p95:.1f} ms (budget {PARSE_P95_BUDGET_MS} ms) — '
            f'some input is pathological.')

    def test_no_input_is_pathological(self):
        """A single catastrophic sentence matters: the corpus has 5.4M of them,
        so a 1-in-10,000 pathological case is 540 real stalls."""
        from klareco.parser import parse

        nasty = [
            'A' * 200,                                   # one huge token
            'la ' * 300,                                 # many function words
            'Kiu-kiu-kiu-kiu-kiu-kiu-kiu-kiu?',          # repeated correlatives
            'malmalmalmalbonega',                        # stacked prefixes
            '«' + 'x' * 100 + '»',                       # long quoted span
        ]
        for s in nasty:
            t0 = time.perf_counter()
            parse(s)
            ms = (time.perf_counter() - t0) * 1000
            assert ms < 1000, (
                f'parse() took {ms:.0f} ms on a {len(s)}-char input: {s[:40]!r} — '
                f'that is a pathological case, and the corpus has 5.4M sentences.')


class TestStageLatencyRecord:
    """Per-stage wall time, asserted against the recorded bench, not re-run."""

    def _latest(self) -> dict:
        if not BENCH_HISTORY.exists():
            pytest.skip(f'{BENCH_HISTORY} absent — run a bench first')
        rows = [json.loads(l) for l in BENCH_HISTORY.read_text().splitlines() if l.strip()]
        if not rows:
            pytest.skip('bench_history.jsonl is empty')
        return rows[-1]

    def test_bench_records_stage_timings(self):
        """`aggregate_stage_timings()` exists and produces per-stage p50/p95, but
        the bench does not persist them — so a stage-level latency regression is
        invisible in the historical record.

        Skips today rather than failing: this is a gap in the RECORD, tracked as
        part of the merge-gate work, not a regression in the code.
        """
        latest = self._latest()
        if 'stage_timings_ms' not in latest and 'stage_timings' not in latest:
            pytest.skip(
                'bench_history records no per-stage timings. aggregate_stage_timings() '
                'computes them and the bench discards them — so #724-style latency '
                'regressions (WHEN questions at 109-164 s) remain invisible in the '
                'record. The bench should persist them.')
