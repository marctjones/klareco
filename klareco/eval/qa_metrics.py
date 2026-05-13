"""
Per-question evaluation and aggregation for extractive QA.

Two responsibilities:

1. evaluate_question(pipeline, entry) — run one question end-to-end and
   return a result dict that captures answer correctness, retrieval rank,
   *and* per-stage timing pulled straight from the orchestrator's StageTrace.

2. summarize(results) — aggregate a list of result dicts into a summary
   that includes per-stage latency stats (avg / p50 / p95 / max) so we can
   tell *which* stage is slow, not just "the pipeline is slow".

Used by:
  - scripts/evaluate_extractive_qa.py (local)
  - scripts/modal_eval.py (Modal-parallel)
"""
import time
from statistics import mean


def _percentile(values: list[float], pct: float) -> float:
    """Simple nearest-rank percentile (no interpolation)."""
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(pct / 100.0 * (len(s) - 1)))))
    return s[k]


def _keywords_in_text(keywords: list[str], text: str) -> list[str]:
    if not text:
        return []
    text_lc = text.lower()
    return [kw for kw in keywords if kw.lower() in text_lc]


def _first_relevant_rank(keywords: list[str], passages) -> int | None:
    for i, p in enumerate(passages, start=1):
        p_text = getattr(p, "text", "") or ""
        if _keywords_in_text(keywords, p_text):
            return i
    return None


# K thresholds at which we compute recall@K, precision@K, density@K.
# Picked to span the diagnostic range: rank-1 quality, extraction window,
# coarse-rank quality, full retrieval coverage.
RECALL_K_VALUES = (1, 5, 10, 25, 50, 100, 500, 1000)


def _passage_relevance(keywords: list[str], passages) -> list[bool]:
    """Return per-passage relevance flags (True if any keyword matches)."""
    out = []
    for p in passages:
        p_text = getattr(p, "text", "") or ""
        out.append(bool(_keywords_in_text(keywords, p_text)))
    return out


def _recall_curve(relevance: list[bool], k_values=RECALL_K_VALUES) -> dict:
    """recall@K = bool(any relevant passage in top-K). Returns {str(k): bool}."""
    if not relevance:
        return {str(k): False for k in k_values}
    return {str(k): any(relevance[:k]) for k in k_values}


def _density_curve(relevance: list[bool], k_values=RECALL_K_VALUES) -> dict:
    """density@K = (#relevant in top-K) / min(K, len(top)).

    When fewer than K passages were retrieved (e.g. top_k=10 retrieved 8),
    density@K uses the actual count as the denominator so it stays a
    'share-of-retrieved' rather than collapsing toward zero.
    """
    if not relevance:
        return {str(k): 0.0 for k in k_values}
    out = {}
    for k in k_values:
        slice_ = relevance[:k]
        if not slice_:
            out[str(k)] = 0.0
        else:
            out[str(k)] = sum(slice_) / len(slice_)
    return out


def _extract_stage_timings(trace) -> dict[str, float]:
    """Pull per-stage timing_ms from a list of StageTrace entries."""
    timings: dict[str, float] = {}
    for tr in trace:
        m = getattr(tr, "metrics", None)
        if m is not None:
            timings[tr.stage_name] = round(getattr(m, "timing_ms", 0.0), 2)
        elif getattr(tr, "skipped", False):
            timings[tr.stage_name] = 0.0
    return timings


def _extract_phase_timings(trace) -> dict[str, dict[str, float]]:
    """Pull per-stage sub-phase timings from StageMetrics.stage_specific.

    Returns ``{stage_name: {phase_name: ms}}``. Empty dict per stage if the
    stage didn't expose phase timings (most don't).
    """
    out: dict[str, dict[str, float]] = {}
    for tr in trace:
        m = getattr(tr, "metrics", None)
        if m is None:
            continue
        ss = getattr(m, "stage_specific", None) or {}
        phases = ss.get("phase_timings_ms") if isinstance(ss, dict) else None
        if phases:
            out[tr.stage_name] = {k: round(float(v), 2) for k, v in phases.items()}
    return out


def _extract_retrieved_passages(trace):
    for tr in trace:
        if tr.stage_name == "retrieve" and getattr(tr, "delta", None) is not None:
            return tr.ctx_after.symbolic.passage_asts
    return ()


def evaluate_question(pipeline, entry: dict) -> dict:
    """Run one test question through `pipeline` and return per-question metrics.

    The returned dict is JSON-serializable and includes:
      - answer correctness signals (matched_keywords, answer_correct)
      - retrieval signals (first_relevant_rank, retrieval_recall@k, mrr)
      - timing: total `latency_sec` and per-stage `stage_timings_ms`
    """
    question = entry["question"]
    expected_kw = entry.get("expected_keywords", [])

    t0 = time.perf_counter()
    try:
        result = pipeline.answer(question)
        elapsed = time.perf_counter() - t0
    except Exception as e:
        return {
            "id":                    entry.get("id"),
            "question":              question,
            "question_type":         entry.get("question_type"),
            "expected_keywords":     expected_kw,
            "error":                 f"{type(e).__name__}: {e}",
            "answer":                "",
            "matched_keywords":      [],
            "answer_correct":        False,
            "retrieved_count":       0,
            "first_relevant_rank":   None,
            "retrieval_recall@k":    False,
            "mrr":                   0.0,
            "latency_sec":           round(time.perf_counter() - t0, 2),
            "stage_timings_ms":      {},
            "stage_phase_timings_ms": {},
        }

    passages = _extract_retrieved_passages(result.trace)
    relevance = _passage_relevance(expected_kw, passages)
    rank = _first_relevant_rank(expected_kw, passages)
    matched = _keywords_in_text(expected_kw, result.text or "")
    stage_timings = _extract_stage_timings(result.trace)
    phase_timings = _extract_phase_timings(result.trace)

    return {
        "id":                    entry.get("id"),
        "question":              question,
        "question_type":         entry.get("question_type"),
        "expected_keywords":     expected_kw,
        "answer":                (result.text or "").strip(),
        "matched_keywords":      matched,
        "answer_correct":        bool(matched),
        "retrieved_count":       len(passages),
        "first_relevant_rank":   rank,
        "retrieval_recall@k":    rank is not None,
        "recall_at_k":           _recall_curve(relevance),
        "density_at_k":          _density_curve(relevance),
        "n_relevant":            sum(relevance),
        "mrr":                   (1.0 / rank) if rank else 0.0,
        "latency_sec":           round(elapsed, 2),
        "stage_timings_ms":      stage_timings,
        "stage_phase_timings_ms": phase_timings,
    }


def aggregate_phase_timings(results: list[dict]) -> dict[str, dict[str, dict]]:
    """Per-stage sub-phase latency aggregates.

    Returns ``{stage_name: {phase_name: {avg_ms, p50_ms, p95_ms, max_ms,
    total_ms, share_pct}}}`` where ``share_pct`` is the phase's share of
    its parent stage's total time. Stages without sub-phase instrumentation
    are absent from the result.
    """
    per_stage_phase: dict[str, dict[str, list[float]]] = {}
    for r in results:
        for stage, phases in (r.get("stage_phase_timings_ms") or {}).items():
            stage_phases = per_stage_phase.setdefault(stage, {})
            for phase, ms in phases.items():
                stage_phases.setdefault(phase, []).append(float(ms))

    out: dict[str, dict[str, dict]] = {}
    for stage, phases in per_stage_phase.items():
        stage_total = sum(sum(v) for v in phases.values()) or 1.0
        out[stage] = {}
        for phase, vals in phases.items():
            total = sum(vals)
            out[stage][phase] = {
                "n":         len(vals),
                "avg_ms":    round(mean(vals), 2) if vals else 0.0,
                "p50_ms":    round(_percentile(vals, 50), 2),
                "p95_ms":    round(_percentile(vals, 95), 2),
                "max_ms":    round(max(vals) if vals else 0.0, 2),
                "total_ms":  round(total, 2),
                "share_pct": round(100.0 * total / stage_total, 1),
            }
    return out


def aggregate_stage_timings(results: list[dict]) -> dict[str, dict]:
    """Per-stage latency aggregates across all results.

    Returns a dict like::

        {
            "parse_question": {"n": 30, "avg_ms": 1.2, "p50_ms": 1.0,
                               "p95_ms": 2.5, "max_ms": 3.1, "total_ms": 36.0,
                               "share_pct": 0.1},
            "retrieve":       {...},
            ...
        }

    `share_pct` is the stage's contribution to total wall time across the
    whole eval set — useful for spotting the dominant bottleneck.
    """
    per_stage: dict[str, list[float]] = {}
    for r in results:
        for stage, ms in (r.get("stage_timings_ms") or {}).items():
            per_stage.setdefault(stage, []).append(float(ms))

    total_all = sum(sum(v) for v in per_stage.values()) or 1.0

    out: dict[str, dict] = {}
    for stage, vals in per_stage.items():
        total = sum(vals)
        out[stage] = {
            "n":         len(vals),
            "avg_ms":    round(mean(vals), 2) if vals else 0.0,
            "p50_ms":    round(_percentile(vals, 50), 2),
            "p95_ms":    round(_percentile(vals, 95), 2),
            "max_ms":    round(max(vals) if vals else 0.0, 2),
            "total_ms":  round(total, 2),
            "share_pct": round(100.0 * total / total_all, 1),
        }
    return out


def summarize(results: list[dict]) -> dict:
    """Aggregate a list of per-question result dicts into one summary dict."""
    n = len(results)
    if n == 0:
        return {}
    n_correct = sum(1 for r in results if r["answer_correct"])
    n_recall = sum(1 for r in results if r["retrieval_recall@k"])
    mrr = sum(r["mrr"] for r in results) / n

    latencies = [r["latency_sec"] for r in results]
    avg_latency = mean(latencies)

    rank_buckets = {"1": 0, "2-3": 0, "4-10": 0, "11+": 0, "none": 0}
    for r in results:
        rk = r["first_relevant_rank"]
        if rk is None:
            rank_buckets["none"] += 1
        elif rk == 1:
            rank_buckets["1"] += 1
        elif rk <= 3:
            rank_buckets["2-3"] += 1
        elif rk <= 10:
            rank_buckets["4-10"] += 1
        else:
            rank_buckets["11+"] += 1

    # Recall@K curve and mean density@K curve aggregated across questions.
    # Per-question entries may be absent if the eval was run with an older
    # qa_metrics version — guard with .get().
    recall_curve = {}
    density_curve = {}
    for k in RECALL_K_VALUES:
        sk = str(k)
        n_recall_k = sum(1 for r in results
                         if (r.get("recall_at_k") or {}).get(sk))
        recall_curve[sk] = n_recall_k / n
        d_sum = sum((r.get("density_at_k") or {}).get(sk, 0.0) for r in results)
        density_curve[sk] = d_sum / n

    return {
        "n":                  n,
        "answer_accuracy":    n_correct / n,
        "retrieval_recall":   n_recall / n,
        "mrr":                mrr,
        "recall_at_k":        recall_curve,
        "density_at_k":       density_curve,
        "avg_latency_sec":    round(avg_latency, 2),
        "p50_latency_sec":    round(_percentile(latencies, 50), 2),
        "p95_latency_sec":    round(_percentile(latencies, 95), 2),
        "max_latency_sec":    round(max(latencies) if latencies else 0.0, 2),
        "rank_distribution":  rank_buckets,
        "stage_timings":      aggregate_stage_timings(results),
        "stage_phase_timings": aggregate_phase_timings(results),
    }


def print_summary(summary: dict, by_type: dict | None = None) -> None:
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)
    print(f"  Questions evaluated:   {summary['n']}")
    print(f"  Answer accuracy:       {summary['answer_accuracy']:.1%}")
    print(f"  Retrieval recall@k:    {summary['retrieval_recall']:.1%}")
    print(f"  Mean Reciprocal Rank:  {summary['mrr']:.3f}")
    print(f"  Latency  avg/p50/p95/max:  "
          f"{summary['avg_latency_sec']:.1f}s / "
          f"{summary['p50_latency_sec']:.1f}s / "
          f"{summary['p95_latency_sec']:.1f}s / "
          f"{summary['max_latency_sec']:.1f}s")

    rd = summary["rank_distribution"]
    print("  First-relevant-rank distribution:")
    print(f"    rank 1     {rd['1']:>4d}")
    print(f"    rank 2-3   {rd['2-3']:>4d}")
    print(f"    rank 4-10  {rd['4-10']:>4d}")
    print(f"    rank 11+   {rd['11+']:>4d}")
    print(f"    none       {rd['none']:>4d}")

    recall_curve = summary.get("recall_at_k") or {}
    density_curve = summary.get("density_at_k") or {}
    if recall_curve:
        print()
        print("  Recall@K (any expected keyword in top-K):")
        ks = sorted(recall_curve.keys(), key=int)
        print("    " + "  ".join(f"K={k:<5s}" for k in ks))
        print("    " + "  ".join(f"{recall_curve[k]:7.1%}" for k in ks))
    if density_curve:
        print()
        print("  Mean density@K (share of top-K matching any keyword):")
        ks = sorted(density_curve.keys(), key=int)
        print("    " + "  ".join(f"K={k:<5s}" for k in ks))
        print("    " + "  ".join(f"{density_curve[k]:7.1%}" for k in ks))

    stage_timings = summary.get("stage_timings") or {}
    phase_timings = summary.get("stage_phase_timings") or {}
    if stage_timings:
        print()
        print("  Per-stage latency (ms):")
        print(f"    {'stage':24s} {'avg':>8s} {'p50':>8s} {'p95':>8s} "
              f"{'max':>8s} {'share%':>8s}")
        ordered = sorted(stage_timings.items(),
                         key=lambda kv: kv[1]["total_ms"], reverse=True)
        for stage, s in ordered:
            print(f"    {stage:24s} "
                  f"{s['avg_ms']:>8.1f} {s['p50_ms']:>8.1f} "
                  f"{s['p95_ms']:>8.1f} {s['max_ms']:>8.1f} "
                  f"{s['share_pct']:>7.1f}%")
            sub = phase_timings.get(stage)
            if sub:
                sub_ordered = sorted(sub.items(),
                                     key=lambda kv: kv[1]["total_ms"],
                                     reverse=True)
                for phase, ps in sub_ordered:
                    print(f"      └─ {phase:19s} "
                          f"{ps['avg_ms']:>8.1f} {ps['p50_ms']:>8.1f} "
                          f"{ps['p95_ms']:>8.1f} {ps['max_ms']:>8.1f} "
                          f"{ps['share_pct']:>7.1f}%  (of stage)")

    if by_type:
        print()
        print("  By question type:")
        for qt, sub in sorted(by_type.items()):
            if not sub:
                continue
            print(f"    {qt:8s}  n={sub['n']:>3d}  "
                  f"answer={sub['answer_accuracy']:.0%}  "
                  f"recall={sub['retrieval_recall']:.0%}  "
                  f"mrr={sub['mrr']:.2f}  "
                  f"avg={sub['avg_latency_sec']:.1f}s")
