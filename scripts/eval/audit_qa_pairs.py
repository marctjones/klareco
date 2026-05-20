#!/usr/bin/env python3
"""
Per-pair quality audit for synthetic Q&A test sets.

VERSION: v2.x (DuckDB)
COMPATIBLE WITH: synthetic_*_active.jsonl / synthetic_who_trivia_v2.jsonl
                 produced by build_synthetic_qa_test_set.py and
                 build_synthetic_who_test_set.py
DEPENDENCIES: duckdb
STAGE: Evaluation

Description:
    Runs three quality checks on every Q&A pair in the supplied test sets
    and emits a per-pair verdict (PASS / FAIL) with the failing check names.

    Checks:
      1. Question coherence (shape) — starts with a valid Esperanto question
         word, ends with `?`, plausible token count, references the source
         sentence's anchor (proper noun or quoted work).
      2. Retrievability — source_sentence_id must resolve to a row in the
         current DuckDB store, and the stored text must match the JSONL
         snapshot (drift detector).
      3. Specific trivia — the question must carry a rigid designator
         («quoted work» or a capitalised proper-noun token), not a generic
         common noun. Generic ones have no unique answer and aren't trivia.

    Reports aggregate + per-set + per-failure-type breakdowns to stdout
    and writes a per-pair JSONL audit log via --output.

Pipeline Position:
    synthetic_*_active.jsonl → [THIS SCRIPT] → audit report
                            → (regenerate / filter / accept the set)

Usage:
    python scripts/eval/audit_qa_pairs.py \\
        --test-sets data/test_sets/synthetic_who_trivia_v2.jsonl \\
                    data/test_sets/synthetic_kie_active.jsonl \\
        --output data/test_sets/audit_2026-05-20.jsonl

Inputs:
    --test-sets   one or more JSONL files with {id, question, expected_answer,
                  source_sentence_id, source_sentence_text, question_type, …}
    --duckdb-path data/indexes/duckdb_store.db

Outputs:
    Per-pair JSONL audit log (one JSON per pair) with verdict + reasons.
    Aggregate summary printed to stdout.

Quality Checks:
    See the function docstrings of `check_question_shape`,
    `check_retrievable`, `check_trivia_specificity` for the exact rules.

Last Updated: 2026-05-20
Author: Claude Code (with Marc Jones)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import duckdb

from klareco.parser import parse


_QUESTION_WORDS = ('Kiu', 'Kion', 'Kio', 'Kie', 'Kiam', 'Kial', 'Kiel')
# Match an initial uppercase letter from the Latin alphabet + Esperanto
# supersigned letters + common accented Latin (é, á, ñ, …) used in proper
# names borrowed from other languages.
_PROPER_NOUN_TOKEN_RE = re.compile(
    r'^[A-ZÀ-ÞĈĜĤĴŜŬ][\wÀ-ſĉĝĥĵŝŭĈĜĤĴŜŬ-]{2,}$'
)
_QUOTED_WORK_RE = re.compile(r'[«"„]\s*[^\s«»"][^«»"]*[»"]')
_YEAR_RE = re.compile(r'\b(1[0-9]{3}|20[0-9]{2})\b')

# Esperanto common words that are sentence-initial-capitalised in some
# corpora and may be mis-tagged as propra_nomo by the parser model. If one
# of these is the "proper-noun designator" in a question, the pair is
# almost certainly a parser false-positive, not real trivia.
_COMMON_WORDS_AS_PROPER = {
    # Conjunctions / discourse
    'Kaj', 'Sed', 'Aŭ', 'Do', 'Tamen', 'Tial', 'Ke', 'Ankaŭ',
    # Function words / prepositions
    'Anstataŭ', 'Krom', 'Malgraŭ', 'Sen', 'Por', 'Antaŭ', 'Post',
    'Dum', 'Tra', 'Trans', 'Apud', 'Pri', 'Pro',
    # Adverbs sometimes flagged
    'Tiam', 'Tiun', 'Tiu', 'Ĉi', 'Jen', 'Nun', 'Hodiaŭ',
    # Common nouns frequently capitalised in headings
    'Estro', 'Estis', 'Konsilio', 'Organizaĵo',
}


def check_question_shape(question: str) -> tuple[bool, str]:
    """Question must start with an Esperanto interrogative, end with `?`,
    be 3–15 tokens, and contain no obvious garbage markers."""
    if not question:
        return False, 'empty'
    first = question.split(' ', 1)[0]
    if first not in _QUESTION_WORDS:
        return False, f'bad question word: {first!r}'
    if not question.rstrip().endswith('?'):
        return False, 'no `?` terminator'
    n_tok = len(question.split())
    if n_tok < 3:
        return False, f'too short ({n_tok} tokens)'
    if n_tok > 20:
        return False, f'too long ({n_tok} tokens)'
    if any(m in question for m in ('[', ']', 'REDIRECT', 'ALIDIREKTU')):
        return False, 'junk marker in question'
    return True, 'ok'


def check_answer_shape(answer: str, question_type: str) -> tuple[bool, str]:
    """Answer is non-empty and roughly the right shape for the question type."""
    if not answer:
        return False, 'empty answer'
    if len(answer) > 250:
        return False, f'answer too long ({len(answer)} chars)'
    qt = (question_type or '').upper()
    if qt == 'KIAM':
        if not _YEAR_RE.search(answer) and not re.search(r'\d', answer):
            return False, 'KIAM answer has no year/digit'
    if qt in ('WHO', 'KIU', 'KIE'):
        if not answer[:1].isupper():
            return False, f'expected proper-noun answer, got {answer!r}'
    return True, 'ok'


def check_retrievable(conn, sid, expected_text: str) -> tuple[bool, str]:
    """Source sentence ID resolves in DuckDB and text agrees with JSONL."""
    if sid is None:
        return False, 'no source_sentence_id'
    try:
        sid_i = int(sid)
    except (TypeError, ValueError):
        return False, f'sid not int: {sid!r}'
    try:
        row = conn.execute(
            'SELECT text FROM sentences WHERE sid = ?', [sid_i]
        ).fetchone()
    except Exception as e:
        return False, f'duckdb error: {e}'
    if not row:
        return False, f'sid {sid_i} not in sentences table'
    db_text = (row[0] or '').strip()
    snap = (expected_text or '').strip()
    if not db_text:
        return False, 'DB text empty'
    if db_text == snap:
        return True, 'exact match'
    # Tolerate whitespace/quote normalization drift; require first 60 chars to align.
    if db_text[:60] == snap[:60]:
        return True, 'prefix match'
    return False, 'text drift between JSONL and DB'


def check_trivia_specificity(question: str) -> tuple[bool, str]:
    """The question must carry a RIGID DESIGNATOR — either a quoted work or a
    capitalised proper-noun token. Generic-common-noun questions
    («Kiu fondis firmaon?») have no unique answer and aren't trivia.

    Rejects the documented parser failure class where an Esperanto common
    word (Anstataŭ, Kaj, Tamen, …) is mis-tagged as propra_nomo and ends
    up as the question's designator — those questions are nonsense.
    """
    if not question:
        return False, 'empty question'
    if _QUOTED_WORK_RE.search(question):
        return True, 'quoted-work designator'
    # Look for proper-noun tokens in the question (skip the leading question word)
    tokens = [t.strip('?,.;:') for t in question.split()[1:]]
    proper = [t for t in tokens if _PROPER_NOUN_TOKEN_RE.match(t)]
    if not proper:
        return False, 'no rigid designator (no quoted work, no proper-noun token)'
    # Reject the parser-false-positive class.
    fp = [t for t in proper if t in _COMMON_WORDS_AS_PROPER]
    if fp and len(proper) == len(fp):
        return False, f'designator is an Esperanto common word: {fp[0]!r}'
    real = [t for t in proper if t not in _COMMON_WORDS_AS_PROPER]
    return True, f'proper-noun designator: {real[0]}'


_LOCATIVE_PREPS_RE = re.compile(
    r'(?:^|[\s,;:.\(])(en|ĉe|sur|apud|sub|super|tra|trans|kun|de|al|el|por|antaŭ|post|pri|pro)\s+',
    re.IGNORECASE,
)


def check_answer_role_alignment(answer: str, source_text: str,
                                question_type: str) -> tuple[bool, str]:
    """Re-parse the source and verify the answer is in the correct semantic
    role for the question type.

    Catches the dominant remaining failure class: the parser picked a name
    that happens to be in a prepositional phrase (e.g. `En "Głos" publikigis…`
    yields `Głos` as the "agent" of `publikigis`, but it's actually the
    venue). For WHO we require the answer to NOT be immediately preceded by
    a preposition in the surface text. For KIE we require the answer TO be
    preceded by a locative preposition. KIAM answers (years) are
    surface-position-agnostic.
    """
    if not answer or not source_text:
        return True, 'skipped (empty)'
    qt = (question_type or '').upper()
    a = answer.strip()
    # Find the answer span in the source.
    pos = source_text.find(a)
    if pos < 0:
        return True, 'skipped (answer not in source)'
    # Wider window so we can skip past intervening quotes/punctuation.
    pre = source_text[max(0, pos - 40):pos]
    # Last alphabetic-only token preceding the answer, regardless of
    # intervening `"`, `«`, parentheses, commas, etc.
    m = re.search(r'([\wĉĝĥĵŝŭĈĜĤĴŜŬ-]+)[^\wĉĝĥĵŝŭĈĜĤĴŜŬ-]*$', pre)
    last_word = m.group(1).lower() if m else ''
    locative = {'en', 'ĉe', 'sur', 'apud', 'sub', 'super', 'tra', 'trans'}
    governing_preps = {
        'en', 'ĉe', 'sur', 'apud', 'sub', 'super', 'tra', 'trans',
        'kun', 'de', 'al', 'el', 'por', 'antaŭ', 'post', 'pri', 'pro',
        'kontraŭ', 'inter', 'sub',
    }
    if qt in ('WHO', 'KIU'):
        if last_word in governing_preps:
            return False, f'WHO answer is governed by preposition {last_word!r}'
        return True, 'WHO answer not in PP'
    if qt == 'KIE':
        if last_word in locative:
            return True, f'KIE answer governed by locative prep {last_word!r}'
        return False, f'KIE answer not governed by locative prep (got {last_word!r})'
    return True, f'no role check for {qt!r}'


def check_answer_in_source(answer: str, source_text: str,
                           question_type: str) -> tuple[bool, str]:
    """The expected answer must appear verbatim (or as a clear prefix) inside
    the source sentence. This is the strongest signal that the answer is
    actually anchored in the source — a generator that hallucinated outside
    the source would fail here.

    For KIAL/KIEL the answer is a multi-token span starting with the marker
    (`ĉar …`, `per …`, or an -e adverb) — substring-match is appropriate.
    For KIU/KIO/KIE/KIAM the answer is a name or year — same.
    """
    if not answer or not source_text:
        return False, 'empty'
    a = answer.strip()
    s = source_text.strip()
    if a in s:
        return True, 'verbatim'
    # Try a fold that strips diacritics for a fuzzy fallback on noisy text.
    import unicodedata
    def _fold(x: str) -> str:
        return ''.join(
            c for c in unicodedata.normalize('NFKD', x)
            if not unicodedata.combining(c)
        ).lower()
    if _fold(a) in _fold(s):
        return True, 'diacritic-fold match'
    return False, 'answer not found in source sentence'


def audit_pair(pair: dict, conn) -> dict:
    question = pair.get('question') or ''
    answer = pair.get('expected_answer') or ''
    qt = pair.get('question_type') or ''
    sid = pair.get('source_sentence_id')
    sentence_text = pair.get('source_sentence_text') or ''

    checks = {
        'question_shape':    check_question_shape(question),
        'answer_shape':      check_answer_shape(answer, qt),
        'retrievable':       check_retrievable(conn, sid, sentence_text),
        'trivia_specific':   check_trivia_specificity(question),
        'answer_in_source':  check_answer_in_source(answer, sentence_text, qt),
        'role_alignment':    check_answer_role_alignment(answer, sentence_text, qt),
    }
    failed = [k for k, (ok, _) in checks.items() if not ok]
    return {
        'id':                  pair.get('id'),
        'question':            question,
        'expected_answer':     answer,
        'question_type':       qt,
        'source_sentence_id':  sid,
        'source_sentence_text': sentence_text,
        'verdict':             'PASS' if not failed else 'FAIL',
        'failed_checks':       failed,
        'reasons':             {k: r for k, (_, r) in checks.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--test-sets', nargs='+', required=True,
                    help='One or more JSONL test-set files.')
    ap.add_argument('--duckdb-path', default='data/indexes/duckdb_store.db')
    ap.add_argument('--output', default=None,
                    help='Per-pair JSONL audit log path.')
    ap.add_argument('--show-failures', type=int, default=10,
                    help='Print at most N failure examples per failure-type.')
    args = ap.parse_args()

    conn = duckdb.connect(args.duckdb_path, read_only=True)

    all_results: list[dict] = []
    per_set_counts: dict[str, dict[str, int]] = {}

    for ts_path in args.test_sets:
        path = Path(ts_path)
        if not path.exists():
            print(f"SKIP: {path} not found", file=sys.stderr)
            continue
        pairs: list[dict] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    pairs.append(json.loads(line))

        per_set_counts[path.name] = {'pass': 0, 'fail': 0}
        for p in pairs:
            r = audit_pair(p, conn)
            r['source_set'] = path.name
            all_results.append(r)
            per_set_counts[path.name][r['verdict'].lower()] += 1

    n = len(all_results)
    n_pass = sum(1 for r in all_results if r['verdict'] == 'PASS')
    n_fail = n - n_pass

    print(f"\nAudited {n} pairs across {len(args.test_sets)} set(s).")
    print(f"  PASS: {n_pass} ({100*n_pass/n:.1f}%)" if n else "  PASS: 0")
    print(f"  FAIL: {n_fail} ({100*n_fail/n:.1f}%)" if n else "  FAIL: 0")

    print("\nPer-set breakdown:")
    for s, c in per_set_counts.items():
        tot = c['pass'] + c['fail']
        pct = (c['pass'] / tot * 100) if tot else 0
        print(f"  {s:<48s}  {c['pass']:>3}/{tot:<3}  {pct:5.1f}% pass")

    # Failure-type counts
    fail_counts: dict[str, int] = {}
    for r in all_results:
        for chk in r['failed_checks']:
            fail_counts[chk] = fail_counts.get(chk, 0) + 1
    if fail_counts:
        print("\nFailure breakdown by check:")
        for chk, k in sorted(fail_counts.items(), key=lambda kv: -kv[1]):
            print(f"  {chk:<22s}  {k:>3}")

        # Examples per failure type
        per_chk_examples: dict[str, list[dict]] = {chk: [] for chk in fail_counts}
        for r in all_results:
            for chk in r['failed_checks']:
                if len(per_chk_examples[chk]) < args.show_failures:
                    per_chk_examples[chk].append(r)
        print("\nExamples:")
        for chk, examples in per_chk_examples.items():
            print(f"\n  [{chk}]")
            for r in examples:
                reason = r['reasons'].get(chk, '?')
                print(f"    - {r['id']:<22s} {r['question'][:70]}")
                print(f"      reason: {reason}")
                if chk in ('answer_shape', 'retrievable'):
                    print(f"      A: {r['expected_answer'][:80]}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            for r in all_results:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        print(f"\nFull audit written to {out_path}")


if __name__ == '__main__':
    main()
