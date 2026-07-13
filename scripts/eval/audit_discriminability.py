#!/usr/bin/env python3
"""
Audit discriminability of existing Q&A test sets.

VERSION: v2.x (DuckDB)
COMPATIBLE WITH: DuckDB store (sentences: shredded cols + ast_json blob), Whoosh index v2
DEPENDENCIES: duckdb, Whoosh, klareco.parser
STAGE: Evaluation

Description:
    Validates existing test-set pairs against the empirical discriminability
    gate (BM25 retrievability) used in build_synthetic_who_test_set.py.

    Recovery paths:
    1. STALE-ID RECOVERY: When source_sentence_id doesn't exist in DuckDB but
       source_sentence_text is present, searches Whoosh for a text match and
       uses the new ID. Falls back to keyword proxy if no text match.
    2. KEYWORD-FALLBACK PATH: When no source_sentence_id is present (older test
       sets), validates that BM25 query of question terms surfaces a passage
       containing all content tokens from the answer (or expected_keywords).

    Categorizes failures by type: recovered by text, recovered by keywords,
    not in top-K (strict), not in top-K (keyword proxy), or no signal.

Pipeline Position:
    Whoosh BM25 index + test sets + DuckDB store → [THIS SCRIPT] → markdown audit report

Usage:
    python scripts/eval/audit_discriminability.py \
        data/test_sets/qa_test_set_50.jsonl \
        data/test_sets/qa_test_diverse_30.jsonl \
        --top-k 200 [--strict-source-only]

Inputs:
    - Test set JSONL files (one per line: question, expected_answer, source_sentence_id, etc.)
    - Whoosh index at data/indexes/whoosh_v2
    - DuckDB store at data/indexes/duckdb_store.db

Outputs:
    - Markdown audit report at data/test_sets/discriminability_audit_<date>.md
    - Summary table to stdout with STRICT vs PROXY columns

Quality Checks:
    - Per-set survival rate (kept / total)
    - Per-question-type survival rate (separate strict and proxy)
    - Failure patterns: recovered_by_text, recovered_by_keywords, not_in_topk_strict, not_in_topk_proxy, no_signal

Last Updated: 2026-05-19 (enhanced with recovery paths)
Author: Claude Code

Related Issues: Gold anchor 50 autopsy
See Also: scripts/eval/build_synthetic_who_test_set.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import duckdb
from whoosh.index import open_dir
from whoosh.qparser import OrGroup, QueryParser


# --- Function word exclusion (same as build_synthetic_who_test_set.py) -----
_GATE_STOP = set('kiu kio kie kiam kiom kial kiel kiuj kion estas estis '
                  'estos la de en al el ĉu por kaj aŭ ke ne je da'.split())


def _q_terms(q: str) -> list[str]:
    """Extract content-word query terms from a question."""
    toks = re.findall(r"[\wĉĝĥĵŝŭĈĜĤĴŜŬ-]+", q.lower())
    return [t for t in toks if t not in _GATE_STOP and len(t) > 2]


def _answer_terms(answer: str) -> list[str]:
    """Extract content-word terms from an answer."""
    toks = re.findall(r"[\wĉĝĥĵŝŭĈĜĤĴŜŬ-]+", answer.lower())
    return [t for t in toks if len(t) > 2]


def is_discriminating(searcher, qp, question: str,
                      source_sid: int | None, top_k: int) -> bool:
    """A pair is discriminating iff BM25 query surfaces source sentence in top-K."""
    terms = _q_terms(question)
    if not terms:
        return False
    q = qp.parse(' OR '.join(terms))
    for h in searcher.search(q, limit=top_k):
        try:
            if source_sid is not None and int(h['id']) == int(source_sid):
                return True
        except (KeyError, ValueError):
            continue
    return False


# ---------------------------------------------------------------------------
# R16 — non-triviality. The CEILING that R7 lacks. See #778.
#
# R7 asks "is the gold passage findable at all?" (rank <= top_k). It never asks
# "is it ALREADY at rank 1?" — and that omission is why the 17-question set
# saturated at recall@5 = 17/17. BM25 alone put the gold passage in the top 5
# for every question, so a *perfect* reranker could not have moved the number.
# Empirically, all nine rerankers then tied at recall@1 = 11.
#
# A test set on which the thing you are changing cannot possibly show up is not
# a measurement instrument. It is a formality.
#
# The measurable band is: NOT rank 1, but WITHIN top_k.
# ---------------------------------------------------------------------------

# Share of rank-1 pairs above which a reranking-capability set is considered
# saturated and is REJECTED.
R16_MAX_RANK1_SHARE = 0.20


def gold_rank(searcher, qp, question: str,
              source_sid: int | None, top_k: int) -> int | None:
    """1-based BM25 rank of the gold passage for `question`, or None if absent.

    This is `is_discriminating` with the answer it should always have returned:
    a rank, not a yes/no. A yes/no cannot express saturation.
    """
    if source_sid is None:
        return None
    terms = _q_terms(question)
    if not terms:
        return None
    q = qp.parse(' OR '.join(terms))
    for i, h in enumerate(searcher.search(q, limit=top_k), start=1):
        try:
            if int(h['id']) == int(source_sid):
                return i
        except (KeyError, ValueError):
            continue
    return None


def r16_report(pairs_ranks: list[tuple[str, int | None]],
               top_k: int,
               gate: bool = True) -> dict:
    """Gold-rank histogram + the R16 saturation gate.

    `pairs_ranks` is [(pair_id, rank_or_None), ...].
    """
    n = len(pairs_ranks)
    ranks = [r for _, r in pairs_ranks]
    buckets = {'1': 0, '2-5': 0, '6-20': 0, f'21-{top_k}': 0, 'not found': 0}
    for r in ranks:
        if r is None:
            buckets['not found'] += 1
        elif r == 1:
            buckets['1'] += 1
        elif r <= 5:
            buckets['2-5'] += 1
        elif r <= 20:
            buckets['6-20'] += 1
        else:
            buckets[f'21-{top_k}'] += 1

    found = [r for r in ranks if r is not None]
    n_rank1 = buckets['1']
    rank1_share = (n_rank1 / n) if n else 0.0
    # The measurable band: findable, but not already won by BM25.
    measurable = sum(1 for r in found if r > 1)
    measurable_share = (measurable / n) if n else 0.0

    median_rank = None
    if found:
        s = sorted(found)
        median_rank = s[len(s) // 2]

    saturated = rank1_share > R16_MAX_RANK1_SHARE

    print()
    print("=" * 70)
    print("R16 — NON-TRIVIALITY (the ceiling R7 lacks)")
    print("=" * 70)
    print(f"  pairs:                   {n}")
    print("  gold-passage BM25 rank distribution:")
    for label, count in buckets.items():
        share = (count / n * 100) if n else 0.0
        bar = "█" * int(share / 2)
        flag = "  <- BM25 already wins; uninformative for ranking" if label == '1' and count else ""
        print(f"    {label:>10}  {count:>4d}  {share:5.1f}%  {bar}{flag}")
    print(f"  median gold rank:        {median_rank}")
    print(f"  rank-1 share:            {rank1_share:.1%}  "
          f"(R16 ceiling: {R16_MAX_RANK1_SHARE:.0%})")
    print(f"  MEASURABLE band (2..{top_k}): {measurable_share:.1%}  "
          f"— the only pairs on which reranking can show up at all")

    if saturated:
        print()
        print("  ❌ SATURATED — this set cannot measure reranking.")
        print("     BM25 already places the gold passage first too often, so a")
        print("     perfect reranker could not move the number. Any A/B run on")
        print("     this set will report a tie regardless of what you changed.")
        print("     Regenerate under R16 (see #778).")
    else:
        print()
        print("  ✅ Has headroom — reranking is observable on this set.")
    print("=" * 70)

    return {
        'n': n,
        'rank_buckets': buckets,
        'median_gold_rank': median_rank,
        'rank1_share': round(rank1_share, 4),
        'measurable_share': round(measurable_share, 4),
        'r16_saturated': saturated,
        'r16_pass': (not saturated) if gate else None,
    }


def answer_match_in_results(searcher, qp, question: str,
                             answer: str, top_k: int) -> bool:
    """Fallback: check if the answer appears in the index at all via BM25.
    Used when source_sentence_id is missing from test set. We search for
    the answer itself; if it's not retrievable via its own terms, the pair
    can't be validated."""
    answer_words = _answer_terms(answer)
    if not answer_words:
        return False
    # Search for answer terms (e.g., "Zamenhof")
    try:
        q = qp.parse(' OR '.join(answer_words))
        results = list(searcher.search(q, limit=top_k))
        return len(results) > 0
    except Exception:
        return False


def question_type(q: str) -> str:
    """Classify question by leading interrogative."""
    q_lower = q.lower().strip()
    for qtype in ['kiu', 'kio', 'kie', 'kiam', 'kiom', 'kial', 'kiel', 'kiuj', 'kion', 'ĉu']:
        if q_lower.startswith(qtype):
            return qtype
    return 'other'


def _recover_stale_id(source_text: str, searcher, qp, duckdb_conn) -> int | None:
    """Attempt to recover a stale sentence ID by text match in Whoosh.

    Strategy:
    1. Try exact substring match in Whoosh (Levenshtein-like via BM25 on first 20 words)
    2. Return the sentence ID if high-confidence match found
    3. Return None if recovery fails
    """
    if not source_text or len(source_text) < 10:
        return None

    # Extract first ~10 words as search terms (heuristic for fast recovery)
    text_toks = re.findall(r"[\wĉĝĥĵŝŭĈĜĤĴŜŬ-]+", source_text.lower())[:10]
    if len(text_toks) < 3:
        return None

    try:
        q = qp.parse(' AND '.join(text_toks))
        results = list(searcher.search(q, limit=5))

        # Check for exact or near-exact match (simple heuristic: overlap > 70% of tokens)
        source_set = set(re.findall(r"[\wĉĝĥĵŝŭĈĜĤĴŜŬ-]+", source_text.lower()))
        for hit in results:
            hit_text = hit.get('text', '').lower()
            hit_set = set(re.findall(r"[\wĉĝĥĵŝŭĈĜĤĴŜŬ-]+", hit_text))
            overlap = len(source_set & hit_set) / max(len(source_set), 1)
            if overlap > 0.7:
                return int(hit['id'])
    except Exception:
        pass

    return None


def _answer_content_tokens(answer: str, expected_keywords: list | None) -> list[str]:
    """Extract answer content tokens from either expected_answer or expected_keywords."""
    if answer:
        return _answer_terms(answer)
    if expected_keywords:
        # Join keywords and extract terms
        keywords_text = ' '.join(expected_keywords)
        return _answer_terms(keywords_text)
    return []


def _answer_in_passage(passage_text: str, answer_tokens: list[str]) -> bool:
    """Check if all answer content tokens appear in a passage (case-insensitive)."""
    if not answer_tokens:
        return False
    passage_lower = passage_text.lower()
    # Simple check: all answer tokens must appear somewhere in the passage
    return all(tok in passage_lower for tok in answer_tokens)


def audit_test_set(path: str, searcher, qp: QueryParser, top_k: int,
                   duckdb_conn=None, strict_source_only: bool = False) -> dict:
    """Audit a single test set. Returns {kept, dropped, ...}.

    Strategy:
      - For pairs WITH source_sentence_id:
        1. Try exact match by ID in top-K
        2. If ID missing: try to recover by source_sentence_text match (RECOVERED_BY_TEXT)
        3. If recovery fails: try keyword-fallback path
      - For pairs WITHOUT source_sentence_id (older test sets):
        1. Extract answer from expected_answer or expected_keywords
        2. Run BM25 query on question terms
        3. Check if top-K contains passage with all answer content tokens (RECOVERED_BY_KEYWORDS)
      - With --strict-source-only: disable fallback paths, only trust source_sentence_id
    """
    fp = Path(path)
    if not fp.exists():
        return None

    kept = []
    dropped = defaultdict(list)  # {reason: [{'q': ..., 'ans': ..., 'why': ...}, ...]}
    by_type = defaultdict(lambda: {'kept': 0, 'dropped': 0})
    total = 0
    recovery_stats = {
        'recovered_by_text': 0,
        'recovered_by_keywords': 0,
        'not_in_topk_strict': 0,
        'not_in_topk_proxy': 0,
        'no_signal': 0,
    }

    for line in open(fp):
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        total += 1
        q = entry.get('question') or ''
        answer = entry.get('expected_answer') or entry.get('answer') or ''
        expected_keywords = entry.get('expected_keywords')
        source_id = entry.get('source_sentence_id')
        source_text = entry.get('source_sentence_text') or ''
        qtype = question_type(q)

        kept_this_pair = False
        reason = None

        # PATH 1: Pair has source_sentence_id
        if source_id is not None:
            # Try exact ID match
            is_disc = is_discriminating(searcher, qp, q, source_id, top_k)
            if is_disc:
                kept_this_pair = True
            else:
                # ID failed; try recovery by text match
                if not strict_source_only and source_text:
                    recovered_id = _recover_stale_id(source_text, searcher, qp, duckdb_conn)
                    if recovered_id:
                        # Re-check with recovered ID
                        is_disc = is_discriminating(searcher, qp, q, recovered_id, top_k)
                        if is_disc:
                            kept_this_pair = True
                            recovery_stats['recovered_by_text'] += 1
                            reason = 'recovered_by_text'

                # If still not found and we have an answer, try keyword fallback
                if not kept_this_pair and not strict_source_only:
                    answer_tokens = _answer_content_tokens(answer, expected_keywords)
                    if answer_tokens:
                        # Search and check if answer appears in top-K
                        terms = _q_terms(q)
                        if terms:
                            try:
                                q_obj = qp.parse(' OR '.join(terms))
                                for hit in searcher.search(q_obj, limit=top_k):
                                    hit_text = hit.get('text', '')
                                    if _answer_in_passage(hit_text, answer_tokens):
                                        kept_this_pair = True
                                        recovery_stats['recovered_by_keywords'] += 1
                                        reason = 'recovered_by_keywords'
                                        break
                            except Exception:
                                pass

                # Still not found: strict failure
                if not kept_this_pair:
                    recovery_stats['not_in_topk_strict'] += 1
                    reason = 'not_in_topk_strict'
                    by_type[qtype]['dropped'] += 1
                    dropped[reason].append({
                        'q': q, 'ans': answer,
                        'why': f'source_id={source_id} not in top-{top_k} (after recovery attempts)'
                    })

        # PATH 2: No source_sentence_id (older test sets)
        else:
            if strict_source_only:
                # Strict mode: no source_id = fail
                recovery_stats['no_signal'] += 1
                by_type[qtype]['dropped'] += 1
                dropped['no_signal'].append({
                    'q': q, 'ans': answer,
                    'why': 'missing source_sentence_id (strict mode)'
                })
            else:
                # Keyword-fallback path
                answer_tokens = _answer_content_tokens(answer, expected_keywords)
                if not answer_tokens:
                    recovery_stats['no_signal'] += 1
                    by_type[qtype]['dropped'] += 1
                    dropped['no_signal'].append({
                        'q': q, 'ans': answer,
                        'why': 'missing expected_answer and expected_keywords'
                    })
                else:
                    # Run BM25 query and check for answer in top-K
                    terms = _q_terms(q)
                    if not terms:
                        recovery_stats['no_signal'] += 1
                        by_type[qtype]['dropped'] += 1
                        dropped['no_signal'].append({
                            'q': q, 'ans': answer,
                            'why': 'question has no content terms'
                        })
                    else:
                        try:
                            q_obj = qp.parse(' OR '.join(terms))
                            for hit in searcher.search(q_obj, limit=top_k):
                                hit_text = hit.get('text', '')
                                if _answer_in_passage(hit_text, answer_tokens):
                                    kept_this_pair = True
                                    recovery_stats['recovered_by_keywords'] += 1
                                    reason = 'recovered_by_keywords'
                                    break
                        except Exception:
                            pass

                        if not kept_this_pair:
                            recovery_stats['not_in_topk_proxy'] += 1
                            by_type[qtype]['dropped'] += 1
                            dropped['not_in_topk_proxy'].append({
                                'q': q, 'ans': answer,
                                'why': f'answer tokens not found in top-{top_k} (proxy check)'
                            })

        # Finalize
        if kept_this_pair:
            by_type[qtype]['kept'] += 1
            kept.append(entry)

    return {
        'file': str(fp.name),
        'total': total,
        'kept': len(kept),
        'dropped': dict(dropped),
        'by_type': dict(by_type),
        'kept_entries': kept,
        'recovery_stats': recovery_stats,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Audit discriminability of test sets against empirical BM25 gate')
    parser.add_argument('test_sets', nargs='+', help='JSONL test set paths')
    parser.add_argument('--top-k', type=int, default=200,
                        help='Whoosh result limit (default 200)')
    parser.add_argument('--output', type=str,
                        default='data/test_sets/discriminability_audit_2026-05-19.md',
                        help='Output markdown report path')
    parser.add_argument('--strict-source-only', action='store_true',
                        help='Disable fallback paths; only trust source_sentence_id')
    parser.add_argument('--rank-histogram', action='store_true',
                        help='R16: print the gold-rank histogram and apply the '
                             'non-triviality gate (a set whose gold passage is '
                             'already at BM25 rank 1 too often cannot measure '
                             'reranking — see #778)')
    args = parser.parse_args()

    whoosh_idx_path = Path('data/indexes/whoosh_v2')
    if not whoosh_idx_path.exists():
        print(f"ERROR: Whoosh index not found at {whoosh_idx_path}")
        sys.exit(1)

    # Open Whoosh index ONCE
    try:
        ix = open_dir(str(whoosh_idx_path))
        searcher = ix.searcher()
        qp = QueryParser('text', ix.schema, group=OrGroup)
    except Exception as e:
        print(f"ERROR opening Whoosh index: {e}")
        sys.exit(1)

    # Try to open DuckDB (optional; for future enrichment)
    duckdb_conn = None
    try:
        duckdb_conn = duckdb.connect('data/indexes/duckdb_store.db', read_only=True)
    except Exception:
        pass

    # Audit each test set
    results = {}
    r16_results = {}
    r16_failed = False
    for ts_path in args.test_sets:
        print(f"Auditing {ts_path}...", file=sys.stderr)
        r = audit_test_set(ts_path, searcher, qp, args.top_k, duckdb_conn,
                          strict_source_only=args.strict_source_only)
        if r:
            results[ts_path] = r

        # R16 — the ceiling. Run only on request, since it is meaningful for
        # reranking-capability sets and reporting-only for the honest-ceiling
        # and regression sets.
        if args.rank_histogram:
            pairs_ranks: list[tuple[str, int | None]] = []
            with open(ts_path, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    e = json.loads(line)
                    pairs_ranks.append((
                        e.get('id', '?'),
                        gold_rank(searcher, qp, e.get('question', ''),
                                  e.get('source_sentence_id'), args.top_k),
                    ))
            print(f"\n### {ts_path}")
            rep = r16_report(pairs_ranks, args.top_k)
            r16_results[ts_path] = rep
            if rep['r16_saturated']:
                r16_failed = True

    searcher.close()
    if duckdb_conn:
        duckdb_conn.close()

    # Aggregate by question type across all sets
    agg_by_type = defaultdict(lambda: {'kept': 0, 'dropped': 0})
    for res in results.values():
        for qtype, counts in res.get('by_type', {}).items():
            agg_by_type[qtype]['kept'] += counts.get('kept', 0)
            agg_by_type[qtype]['dropped'] += counts.get('dropped', 0)

    # Aggregate recovery stats across all sets
    agg_recovery = {
        'recovered_by_text': 0,
        'recovered_by_keywords': 0,
        'not_in_topk_strict': 0,
        'not_in_topk_proxy': 0,
        'no_signal': 0,
    }
    for res in results.values():
        for key, val in res.get('recovery_stats', {}).items():
            agg_recovery[key] += val

    # Generate markdown report
    mode_label = "STRICT (source_id required)" if args.strict_source_only else "WITH FALLBACKS"
    report_lines = [
        "# Discriminability Audit Report",
        f"Generated: {datetime.now().isoformat()}",
        f"Top-K: {args.top_k}",
        f"Mode: {mode_label}",
        "",
        "## Recovery Summary",
        "",
        f"- **Recovered by Text Match:** {agg_recovery['recovered_by_text']} pairs",
        f"  (stale IDs matched to current sentences via source_sentence_text)",
        f"- **Recovered by Keywords:** {agg_recovery['recovered_by_keywords']} pairs",
        f"  (answer content tokens found in BM25 top-K results)",
        f"- **Not in Top-K (Strict):** {agg_recovery['not_in_topk_strict']} pairs",
        f"  (genuine retrievability failures after recovery attempts)",
        f"- **Not in Top-K (Proxy):** {agg_recovery['not_in_topk_proxy']} pairs",
        f"  (answer tokens never co-occurred in top-K)",
        f"- **No Signal:** {agg_recovery['no_signal']} pairs",
        f"  (missing source ID, answer, and keywords)",
        "",
        "## Summary by Test Set",
        "",
        "| Test Set | Total | Kept | Dropped | Survival % |",
        "|----------|-------|------|---------|-----------|",
    ]

    grand_total_kept = 0
    grand_total_all = 0

    for ts_path, res in results.items():
        total = res['total']
        kept = res['kept']
        dropped = total - kept
        survival = (kept / total * 100) if total > 0 else 0
        grand_total_kept += kept
        grand_total_all += total

        report_lines.append(
            f"| {res['file']} | {total} | {kept} | {dropped} | {survival:.1f}% |"
        )

    if grand_total_all > 0:
        grand_survival = grand_total_kept / grand_total_all * 100
    else:
        grand_survival = 0
    report_lines.append(
        f"| **TOTAL** | **{grand_total_all}** | **{grand_total_kept}** | "
        f"**{grand_total_all - grand_total_kept}** | **{grand_survival:.1f}%** |"
    )

    # Question type breakdown
    report_lines.extend([
        "",
        "## Survival Rate by Question Type (across all sets)",
        "",
        "| Question Type | Kept | Dropped | Survival % |",
        "|---|---|---|---|",
    ])

    for qtype in sorted(agg_by_type.keys()):
        kept = agg_by_type[qtype]['kept']
        dropped = agg_by_type[qtype]['dropped']
        total = kept + dropped
        survival = (kept / total * 100) if total > 0 else 0
        report_lines.append(f"| {qtype:12s} | {kept:3d} | {dropped:3d} | {survival:6.1f}% |")

    # Failure patterns
    report_lines.extend([
        "",
        "## Failure Patterns (Top Examples per Reason)",
        "",
    ])

    failure_buckets = defaultdict(list)
    for res in results.values():
        for reason, examples in res.get('dropped', {}).items():
            failure_buckets[reason].extend(examples)

    for reason in sorted(failure_buckets.keys()):
        examples = failure_buckets[reason][:5]  # Top 5 per reason
        report_lines.append(f"### {reason.upper()} ({len(failure_buckets[reason])} total)")
        report_lines.append("")
        for ex in examples:
            report_lines.append(f"- **Q:** {ex.get('q', '')}")
            report_lines.append(f"  **A:** {ex.get('ans', '')}")
            report_lines.append(f"  **Why:** {ex.get('why', '')}")
        report_lines.append("")

    # Recommendations
    report_lines.extend([
        "## Recommendations",
        "",
    ])

    report_lines.append("**Overall survival (with fallbacks):** {:.1f}% ({} / {} pairs)".format(
        grand_survival, grand_total_kept, grand_total_all))

    report_lines.append("")
    report_lines.append("### Per-Set Guidance")
    report_lines.append("")

    for ts_path, res in results.items():
        fname = res['file']
        total = res['total']
        kept = res['kept']
        survival = (kept / total * 100) if total > 0 else 0
        recovery = res.get('recovery_stats', {})
        recovered_count = recovery.get('recovered_by_text', 0) + recovery.get('recovered_by_keywords', 0)

        if survival > 80:
            guidance = "✓ **SALVAGEABLE.** Use these pairs directly. Recovery worked well."
        elif survival > 50:
            guidance = "◐ **MARGINAL.** Mix salvageable + new generation. Review recovered pairs for quality."
        else:
            guidance = "✗ **BROKEN.** Full regeneration required (recovery failed across most pairs)."

        report_lines.append(f"- **{fname}:** {survival:.1f}% ({kept}/{total}), {recovered_count} recovered. {guidance}")

    report_lines.append("")
    report_lines.append("### Per-Type Guidance")
    report_lines.append("")

    for qtype in sorted(agg_by_type.keys()):
        kept = agg_by_type[qtype]['kept']
        dropped = agg_by_type[qtype]['dropped']
        total = kept + dropped
        survival = (kept / total * 100) if total > 0 else 0

        if survival > 80:
            guidance = "✓ **Usable.** Salvage existing pairs into new ruler."
        elif survival > 50:
            guidance = "◐ **Marginal.** Mix of salvageable + regenerate."
        else:
            guidance = "✗ **Broken.** Full regeneration required."

        report_lines.append(f"- **{qtype}:** {survival:.1f}% ({kept}/{total}). {guidance}")

    report_lines.append("")

    # Write report
    report_path = Path(args.output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_text = '\n'.join(report_lines)
    report_path.write_text(report_text)

    print(f"\nReport written to {report_path}")
    print("\n" + '=' * 70)
    print("SUMMARY (MODE: WITH FALLBACKS)" if not args.strict_source_only else "SUMMARY (MODE: STRICT)")
    print('=' * 70)
    print(f"Total pairs: {grand_total_all}")
    print(f"Kept: {grand_total_kept} ({grand_survival:.1f}%)")
    print(f"Dropped: {grand_total_all - grand_total_kept}")
    print()
    print("Recovery breakdown:")
    print(f"  Recovered by text match: {agg_recovery['recovered_by_text']}")
    print(f"  Recovered by keywords:   {agg_recovery['recovered_by_keywords']}")
    print(f"  Not in top-K (strict):   {agg_recovery['not_in_topk_strict']}")
    print(f"  Not in top-K (proxy):    {agg_recovery['not_in_topk_proxy']}")
    print(f"  No signal:               {agg_recovery['no_signal']}")
    print()
    print("By question type:")
    for qtype in sorted(agg_by_type.keys()):
        kept = agg_by_type[qtype]['kept']
        dropped = agg_by_type[qtype]['dropped']
        total = kept + dropped
        survival = (kept / total * 100) if total > 0 else 0
        print(f"  {qtype:10s}: {survival:6.1f}% ({kept}/{total})")


if __name__ == '__main__':
    main()
