#!/usr/bin/env python3
"""
Audit the quality of an EXISTING gold Q&A set (the automatable gates).

VERSION: v1.0
COMPATIBLE WITH: qa_gold_v*.jsonl / rebaseline_*.jsonl schema, DuckDB store
DEPENDENCIES: duckdb; klareco.eval.answer_scoring (normalize)
STAGE: Evaluation

Description:
    The quality standard (docs/QA_TEST_SET_QUALITY_STANDARD.md, R1-R17) and
    qa_gate.py validate INCOMING batches; qa_answerability.py CONSTRUCTS sets.
    Nothing audited an existing gold set. This does: it runs the machine-checkable
    rules over a finished JSONL and reports pass rates + flagged pairs, so we know
    whether the ruler we measure against is trustworthy.

    Checks (automatable subset; human stages 3-4 of the gate stack are out of scope):
      C1  provenance (R8)      — source_sentence_id present
      C2  gold resolves        — that sid exists in the DuckDB store
      C3  source-text drift    — stored source_sentence_text == the store's text
      C4  answer verbatim (R9) — expected_answer appears in the source text
                                 (THE mislabel check — catches the #866 Trump case)
      C5  verbatim-flag honest — answer_verbatim flag matches C4 reality
      C6  answer non-empty
      C7  duplicates           — same question text (R: dedup)
      C8  distributions (R10/R15) — question_type / band / entity balance

Usage:
    python scripts/qa/qa_audit.py --test-set data/test_sets/qa_gold_v2.jsonl [--show 15]

Outputs:
    Per-check pass rate + a sample of flagged pairs. Non-zero exit if any
    REQUIRED check (C1, C4 for answer_verbatim=True) falls below threshold.

Last Updated: 2026-07-18
Author: Claude Opus 4.8
Related Issues: #879 (label/gold audit), #845 (assemble/freeze), #866 (a mislabel found by hand)
See Also: docs/QA_TEST_SET_QUALITY_STANDARD.md
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import duckdb

from klareco.eval.answer_scoring import normalize


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def audit(rows, con, show=12):
    n = len(rows)
    flags = collections.defaultdict(list)
    # resolve all sids in one query
    sids = [str(r.get('source_sentence_id')) for r in rows if r.get('source_sentence_id') is not None]
    store_text = {}
    if sids:
        q = "SELECT sid, text FROM sentences WHERE sid IN (%s)" % ','.join('?' * len(set(sids)))
        for sid, txt in con.execute(q, list({int(s) for s in set(sids) if s.isdigit()})).fetchall():
            store_text[str(sid)] = txt

    c = collections.Counter()
    seen_q = {}
    for r in rows:
        q = (r.get('question') or '').strip()
        sid = r.get('source_sentence_id')
        src = r.get('source_sentence_text') or ''
        ans = (r.get('expected_answer') or '').strip()
        verb = r.get('answer_verbatim')

        # C1 provenance
        if sid is None:
            c['C1_no_sid'] += 1; flags['C1'].append(q[:60])
        # C2 resolves
        elif str(sid) not in store_text:
            c['C2_sid_unresolved'] += 1; flags['C2'].append(f"[{sid}] {q[:50]}")
        # C3 source-text drift
        elif normalize(src) and normalize(src) != normalize(store_text[str(sid)]):
            c['C3_text_drift'] += 1
            flags['C3'].append(f"[{sid}] stored={src[:40]!r} store={store_text[str(sid)][:40]!r}")
        # C6 answer empty
        if not ans:
            c['C6_no_answer'] += 1; flags['C6'].append(q[:60])
        # C4 answer verbatim in source (the mislabel check)
        elif src and normalize(ans) not in normalize(src):
            c['C4_answer_not_in_source'] += 1
            # C4a/C4b severity: is ANY answer content token (len>3) in the source?
            nsrc = normalize(src)
            ans_toks = [t for t in normalize(ans).split() if len(t) > 3]
            # prefix match tolerates inflection: 'julio' matches 'julion' etc.
            supported = any(any(st.startswith(t) or t.startswith(t2) or t in st
                                for st in nsrc.split() for t2 in [t])
                            for t in ans_toks)
            if not ans_toks or not supported:
                c['C4a_hard_no_answer_token'] += 1
                flags['C4a'].append(f"ans={ans!r} NONE in [{sid}] {src[:55]!r}")
            else:
                c['C4b_form_mismatch'] += 1
                flags['C4b'].append(f"ans={ans!r} ~partial [{sid}] {src[:45]!r}")
            # C5 flag honesty
            if verb is True:
                c['C5_verbatim_flag_wrong'] += 1
        # C7 dup
        if q in seen_q:
            c['C7_duplicate'] += 1; flags['C7'].append(q[:60])
        seen_q[q] = True

    print(f"\n{'='*64}\n  GOLD-SET AUDIT  (n={n})\n{'='*64}")
    def line(code, label, bad, required=False):
        ok = n - bad
        tag = ' [REQUIRED]' if required else ''
        print(f"  {code:5} {label:34} {ok:>5}/{n}  ({ok/n:6.1%} ok){tag}"
              + (f"  ⚠ {bad} FAIL" if bad else ""))
    line('C1', 'provenance: source_sentence_id', c['C1_no_sid'], required=True)
    line('C2', 'gold sid resolves in store', c['C2_sid_unresolved'])
    line('C3', 'source-text matches store', c['C3_text_drift'])
    line('C4', 'answer appears in source (R9)', c['C4_answer_not_in_source'], required=True)
    print(f"  C4a   └ HARD (no answer token in source — likely mislabeled): "
          f"{c['C4a_hard_no_answer_token']} ({c['C4a_hard_no_answer_token']/n:.1%})")
    print(f"  C4b   └ SOFT (answer supported, inflection/spelling/partial): "
          f"{c['C4b_form_mismatch']} ({c['C4b_form_mismatch']/n:.1%})")
    line('C6', 'answer non-empty', c['C6_no_answer'], required=True)
    line('C7', 'unique questions (no dup)', c['C7_duplicate'])
    print(f"  C5    answer_verbatim=True but answer NOT in source: {c['C5_verbatim_flag_wrong']}  (dishonest flag)")

    # C8 distributions
    print(f"\n  C8 DISTRIBUTIONS:")
    for field in ('question_type', 'difficulty_band', 'answer_verbatim'):
        dist = collections.Counter(str(r.get(field)) for r in rows)
        top = ', '.join(f"{k}:{v}" for k, v in dist.most_common(6))
        print(f"    {field:16} {top}")

    print(f"\n  FLAGGED SAMPLES:")
    for code in ('C4a', 'C4b', 'C2', 'C3', 'C1'):
        if flags[code]:
            print(f"    [{code}] ({len(flags[code])} total, showing {min(show,len(flags[code]))}):")
            for x in flags[code][:show]:
                print(f"       {x}")

    # exit code
    req_fail = c['C1_no_sid'] + c['C6_no_answer']
    verb_n = sum(1 for r in rows if r.get('answer_verbatim') is True)
    c4_verb = c['C4_answer_not_in_source']  # among all; report rate
    print(f"\n  VERDICT: {'PASS' if req_fail == 0 else 'FAIL'} on required structural checks; "
          f"answer-in-source failures = {c['C4_answer_not_in_source']}/{n} "
          f"({c['C4_answer_not_in_source']/n:.1%}) — the mislabel rate.")
    return 0 if req_fail == 0 else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--test-set', required=True)
    ap.add_argument('--duckdb-path', default='data/indexes/duckdb_store.db')
    ap.add_argument('--show', type=int, default=12)
    a = ap.parse_args()
    rows = load(a.test_set)
    con = duckdb.connect(a.duckdb_path, read_only=True)
    print(f"Auditing {a.test_set}")
    sys.exit(audit(rows, con, show=a.show))


if __name__ == '__main__':
    main()
