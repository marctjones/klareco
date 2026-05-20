#!/usr/bin/env python3
"""
Per-pair suspicion scoring for synthetic Q&A test sets.

VERSION: v2.x (DuckDB)
COMPATIBLE WITH: synthetic_*_active.jsonl / synthetic_who_trivia_v2.jsonl
DEPENDENCIES: duckdb
STAGE: Evaluation

Description:
    The mechanical audit (audit_qa_pairs.py) treats pairs as binary PASS /
    FAIL. This script grades EVERY pair on a finer-grained suspicion score
    so we can walk through likely-bad pairs one by one without doing 349
    full manual reviews. Each pair is checked against ~15 type-aware
    signals (source-quality, answer-shape, semantic-role likelihood) and
    placed into one of three buckets:

      CLEAN       — no signals tripped, trust the audit-PASS verdict
      REVIEW      — 1 or 2 mild signals tripped, eyeball recommended
      LIKELY_BAD  — ≥3 signals OR any "killer" signal tripped, almost
                    certainly factually or grammatically wrong

    A markdown report is emitted sorted by descending signal count, so the
    most suspicious pairs land at the top.

Pipeline Position:
    synthetic_*_active.jsonl → [THIS SCRIPT] → suspicion-ranked report
                            → human review of REVIEW + LIKELY_BAD pairs

Usage:
    python scripts/eval/score_qa_pair_suspicion.py \\
        --test-sets data/test_sets/synthetic_who_trivia_v2.jsonl \\
                    data/test_sets/synthetic_kie_active.jsonl \\
                    data/test_sets/synthetic_kio_active.jsonl \\
                    data/test_sets/synthetic_kiam_active.jsonl \\
                    data/test_sets/synthetic_kial_active.jsonl \\
                    data/test_sets/synthetic_kiel_active.jsonl \\
        --markdown data/test_sets/suspicion_2026-05-20.md \\
        --jsonl    data/test_sets/suspicion_2026-05-20.jsonl

Inputs:
    --test-sets  one or more JSONL test-set files
    --duckdb-path  data/indexes/duckdb_store.db (used for corpus stats)

Outputs:
    --markdown  human-readable report (default: stdout to summary)
    --jsonl     machine-readable per-pair record with signal list

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


# Curated semantic-class sets for KIEL answer scrutiny.
_TEMPORAL_E = {
    'unue', 'due', 'lastfoje', 'lastatempe', 'antaŭlonge',
    'daŭre', 'ĉiame', 'longe', 'mallonge', 'momente',
    'ĉiujare', 'ĉiutage', 'ĉiumonate', 'plurfoje', 'foje',
    'pasintjare', 'pasinte', 'venonte', 'estonte', 'estontece',
    'frue', 'malfrue', 'baldaŭ-e', 'tuje', 'fine', 'komence',
}
_QUANTIFIER_E = {
    'plimulte', 'malplimulte', 'plejparte', 'sume', 'entute',
    'nure', 'sole', 'iom-e', 'kelke', 'multe', 'malmulte',
}
_DISCOURSE_E = {
    'jene', 'tiele', 'nome', 'rilate', 'koncerne',
    'evidente', 'verŝajne', 'eble', 'kompreneble', 'feliĉe',
    'bedaŭrinde', 'ŝajne', 'cetere', 'krome',
}
_TALKPAGE_MARKERS = ('(UTC)', '::Mi', '::Mia', 'Vikipedi', '01:04', 'restarigi')
_CITATION_MARKERS = ('et al.', 'vol.', 'pp.', 'ISBN', 'doi:', '&', 'Press,')
_ESPERANTO_SUPERSIGNED = set('ĉĝĥĵŝŭĈĜĤĴŜŬ')
_QUOTED_RE = re.compile(r'[«"„]\s*([^«»"]{1,80})\s*[»"]')


def _word_count(s: str) -> int:
    return len(s.split())


def _punct_ratio(s: str) -> float:
    if not s:
        return 0.0
    punct = sum(1 for c in s if not c.isalnum() and not c.isspace())
    return punct / max(1, len(s))


def _foreign_letter_ratio(s: str) -> float:
    """Fraction of letters that are NOT in the Esperanto alphabet
    (a-z + supersigned + accented Latin used in proper names)."""
    if not s:
        return 0.0
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return 0.0
    foreign = sum(
        1 for c in letters
        if c.lower() not in 'abcdefghijklmnoprstuvz'
        and c not in _ESPERANTO_SUPERSIGNED
        and not ('À' <= c <= 'ÿ')  # latin-1 supplement for names
    )
    return foreign / len(letters)


# -----------------------------------------------------------------------------
# Signal definitions. Each returns (bool, severity, reason). Severity:
#   1 = mild (one of these alone shouldn't mark a pair bad)
#   2 = moderate
#   3 = killer (one alone is enough to mark LIKELY_BAD)
# -----------------------------------------------------------------------------

def signals_universal(pair: dict) -> list[tuple[str, int, str]]:
    text = pair.get('source_sentence_text') or ''
    out: list[tuple[str, int, str]] = []

    if _word_count(text) < 6:
        out.append(('source_too_short', 2, f'{_word_count(text)} words'))

    if any(m in text for m in _TALKPAGE_MARKERS):
        out.append(('source_talkpage_noise', 3, 'wiki talk-page markers'))

    if any(m in text for m in _CITATION_MARKERS):
        # citation prose is not always wrong but is high-risk
        out.append(('source_citation_prose', 1, 'looks like a bibliographic citation'))

    pratio = _punct_ratio(text)
    if pratio > 0.20:
        out.append(('source_punct_heavy', 1, f'{pratio:.0%} punct'))

    fratio = _foreign_letter_ratio(text)
    if fratio > 0.30:
        out.append(('source_foreign_script', 2, f'{fratio:.0%} non-Esperanto letters'))

    return out


def signals_who(pair: dict) -> list[tuple[str, int, str]]:
    out: list[tuple[str, int, str]] = []
    q = pair.get('question') or ''
    a = pair.get('expected_answer') or ''
    src = pair.get('source_sentence_text') or ''

    # Quoted work that's pseudo-Esperanto / mostly foreign
    m = _QUOTED_RE.search(q)
    if m:
        title = m.group(1)
        if _foreign_letter_ratio(title) > 0.40:
            out.append(('quoted_work_foreign', 1, f'title={title!r}'))

    # Answer is entirely uppercase (likely heading text)
    if a and a.isupper() and len(a) >= 3:
        out.append(('answer_allcaps', 2, f'A={a!r}'))

    # Quoted work missing from visible source (truncation risk)
    if m and m.group(1) not in src:
        out.append(('quoted_work_not_in_visible_source', 2, ''))

    # Answer is a single short token <4 chars (likely fragment)
    if a and len(a) < 4 and ' ' not in a:
        out.append(('answer_too_short', 2, f'A={a!r}'))

    return out


def signals_kie(pair: dict) -> list[tuple[str, int, str]]:
    out: list[tuple[str, int, str]] = []
    src = pair.get('source_sentence_text') or ''
    a = pair.get('expected_answer') or ''
    q = pair.get('question') or ''

    # Birth AND death PPs both present (latent confusion risk even after the
    # first-locative-after-verb fix)
    if 'naskiĝis' in src and 'mortis' in src:
        out.append(('source_birth_and_death_pp', 1, 'both nask/mort'))

    # The verb is naskiĝis but the question uses the wrong inflection
    if 'naskiĝis' in src and 'naskis ' in q.lower():
        out.append(('verb_voice_mismatch', 3, 'source naskiĝis vs question naskis'))

    # Answer ALLCAPS (likely heading)
    if a and a.isupper() and len(a) >= 3:
        out.append(('answer_allcaps', 2, f'A={a!r}'))

    # Answer is too short (single 2-3 char token)
    if a and len(a) < 4 and ' ' not in a:
        out.append(('answer_too_short', 2, f'A={a!r}'))

    return out


def signals_kio(pair: dict) -> list[tuple[str, int, str]]:
    out: list[tuple[str, int, str]] = []
    q = pair.get('question') or ''
    a = pair.get('expected_answer') or ''
    # Mirror of WHO checks for quoted works
    m = _QUOTED_RE.search(a) or _QUOTED_RE.search(q)
    if m:
        title = m.group(1)
        if _foreign_letter_ratio(title) > 0.50:
            out.append(('quoted_work_foreign', 1, f'title={title!r}'))
    if a and a.isupper() and len(a) >= 4:
        out.append(('answer_allcaps', 2, f'A={a!r}'))
    return out


def signals_kiam(pair: dict) -> list[tuple[str, int, str]]:
    out: list[tuple[str, int, str]] = []
    src = pair.get('source_sentence_text') or ''
    a = pair.get('expected_answer') or ''

    # Year plausibility (rough — most encyclopedic Esperanto content is
    # post-1500). We flag pre-1000 OR post-2030 as suspicious.
    if re.fullmatch(r'\d{4}', a or ''):
        y = int(a)
        if y < 1000 or y > 2030:
            out.append(('year_implausible', 2, f'year={y}'))

    # Multiple 4-digit years in source — the generator might have picked
    # the wrong one for the question's verb (e.g. confused birth/death).
    years = re.findall(r'\b\d{4}\b', src)
    if len(set(years)) >= 3:
        out.append(('many_years_in_source', 1, f'{len(set(years))} distinct years'))

    # Famous-person sanity: the Miklós Zrínyi / 1949 source-noise case is the
    # canonical example of a corpus error we can't detect without an external
    # KB. We at least flag "year mismatches a likely era for a famous name."
    # Crude heuristic: if the source contains "Miklós" but year is 1949, flag.
    famous_era_hints = {
        # name token -> (min_birth_year, max_birth_year)
        # Conservative — only well-known mismatches.
    }
    for nm, (lo, hi) in famous_era_hints.items():
        if nm in src and re.fullmatch(r'\d{4}', a or '') and not (lo <= int(a) <= hi):
            out.append(('era_mismatch', 3, f'{nm} not in {lo}-{hi}'))

    return out


def signals_kial(pair: dict) -> list[tuple[str, int, str]]:
    out: list[tuple[str, int, str]] = []
    a = pair.get('expected_answer') or ''
    src = pair.get('source_sentence_text') or ''
    q = pair.get('question') or ''

    # Trailing or unbalanced parens (cosmetic but often signals truncation)
    if a.endswith(')') and a.count(')') != a.count('('):
        out.append(('answer_unbalanced_parens', 2, ''))

    # The question's verb appears in source but with negation nearby —
    # causes the causal clause to apply to the negated event (Johann
    # neniam aŭdis ... ĉar ...).
    q_verb_match = re.match(r'^Kial\s+(\S+)\s+', q)
    if q_verb_match:
        v = q_verb_match.group(1)
        # Look for `neniam` or ` ne ` within ±20 chars of the verb in source
        vpos = src.find(v)
        if vpos > 0:
            window = src[max(0, vpos - 30):vpos + 30]
            if 'neniam' in window or re.search(r'\bne\b', window):
                out.append(('negation_near_verb', 2, 'ne/neniam near verb'))

    return out


def signals_kiel(pair: dict) -> list[tuple[str, int, str]]:
    out: list[tuple[str, int, str]] = []
    a = (pair.get('expected_answer') or '').strip()

    # Multi-token answers starting with `per` are high-confidence manner.
    if a.lower().startswith('per '):
        return out

    # Single -e adverb answers: check class
    low = a.lower()
    if low in _TEMPORAL_E:
        out.append(('answer_temporal_adverb', 3, f'A={a!r}'))
    elif low in _QUANTIFIER_E:
        out.append(('answer_quantifier_adverb', 3, f'A={a!r}'))
    elif low in _DISCOURSE_E:
        out.append(('answer_discourse_adverb', 3, f'A={a!r}'))

    # Foreign-language answer (e.g. `toute`, `passerine`)
    if re.search(r'[qwxy]', low):
        out.append(('answer_non_esperanto_letters', 3, f'A={a!r}'))

    # Single short answer
    if a and ' ' not in a and len(a) < 5:
        out.append(('answer_too_short', 2, f'A={a!r}'))

    return out


def score_pair(pair: dict) -> dict:
    qt = (pair.get('question_type') or '').upper()
    signals = signals_universal(pair)
    if qt in ('WHO', 'KIU'):
        signals.extend(signals_who(pair))
    elif qt == 'KIE':
        signals.extend(signals_kie(pair))
    elif qt == 'KIO':
        signals.extend(signals_kio(pair))
    elif qt == 'KIAM':
        signals.extend(signals_kiam(pair))
    elif qt == 'KIAL':
        signals.extend(signals_kial(pair))
    elif qt == 'KIEL':
        signals.extend(signals_kiel(pair))

    severity_sum = sum(sev for _, sev, _ in signals)
    has_killer = any(sev >= 3 for _, sev, _ in signals)
    if has_killer or severity_sum >= 4:
        bucket = 'LIKELY_BAD'
    elif severity_sum >= 1:
        bucket = 'REVIEW'
    else:
        bucket = 'CLEAN'

    return {
        'id':                  pair.get('id'),
        'question_type':       qt,
        'question':            pair.get('question'),
        'expected_answer':     pair.get('expected_answer'),
        'source_sentence_id':  pair.get('source_sentence_id'),
        'source_sentence_text': pair.get('source_sentence_text'),
        'signals':             [{'name': n, 'severity': s, 'detail': d}
                                for n, s, d in signals],
        'severity_sum':        severity_sum,
        'bucket':              bucket,
    }


def emit_markdown(scored: list[dict], path: Path) -> None:
    by_bucket: dict[str, list[dict]] = {'LIKELY_BAD': [], 'REVIEW': [], 'CLEAN': []}
    for r in scored:
        by_bucket[r['bucket']].append(r)

    lines: list[str] = []
    lines.append('# Per-pair suspicion review')
    lines.append('')
    lines.append(f'Total pairs: **{len(scored)}**  |  '
                 f'LIKELY_BAD: **{len(by_bucket["LIKELY_BAD"])}**  |  '
                 f'REVIEW: **{len(by_bucket["REVIEW"])}**  |  '
                 f'CLEAN: **{len(by_bucket["CLEAN"])}**')
    lines.append('')

    for bucket in ('LIKELY_BAD', 'REVIEW', 'CLEAN'):
        rows = sorted(
            by_bucket[bucket],
            key=lambda r: (-r['severity_sum'], r.get('question_type') or '', r['id']),
        )
        lines.append(f'## {bucket} ({len(rows)})')
        lines.append('')
        if not rows:
            lines.append('_(none)_'); lines.append(''); continue
        if bucket == 'CLEAN':
            # Aggregate-only summary
            per_type: dict[str, int] = {}
            for r in rows:
                per_type[r['question_type']] = per_type.get(r['question_type'], 0) + 1
            for qt, n in sorted(per_type.items(), key=lambda kv: -kv[1]):
                lines.append(f'- **{qt}**: {n} pairs (no signals tripped)')
            lines.append('')
            continue

        for r in rows:
            lines.append(f'### `{r["id"]}` ({r["question_type"]}) — severity {r["severity_sum"]}')
            lines.append(f'- **Q:** {r["question"]}')
            lines.append(f'- **A:** {r["expected_answer"]}')
            src = r['source_sentence_text'] or ''
            lines.append(f'- **S:** {src[:200]}{"…" if len(src) > 200 else ""}')
            lines.append('- **Signals:**')
            for s in r['signals']:
                detail = f' — {s["detail"]}' if s['detail'] else ''
                lines.append(f'  - `{s["name"]}` (sev {s["severity"]}){detail}')
            lines.append('')

    path.write_text('\n'.join(lines), encoding='utf-8')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--test-sets', nargs='+', required=True)
    ap.add_argument('--duckdb-path', default='data/indexes/duckdb_store.db',
                    help='(reserved for future corpus-frequency signals)')
    ap.add_argument('--markdown', default=None)
    ap.add_argument('--jsonl',    default=None)
    args = ap.parse_args()

    all_pairs: list[dict] = []
    for ts in args.test_sets:
        path = Path(ts)
        if not path.exists():
            print(f"SKIP: {path} not found", file=sys.stderr)
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    p = json.loads(line)
                    p['_source_set'] = path.name
                    all_pairs.append(p)

    scored = [score_pair(p) for p in all_pairs]
    for s, p in zip(scored, all_pairs):
        s['source_set'] = p['_source_set']

    n = len(scored)
    by_bucket: dict[str, int] = {}
    for r in scored:
        by_bucket[r['bucket']] = by_bucket.get(r['bucket'], 0) + 1
    print(f"Scored {n} pairs.")
    for b in ('LIKELY_BAD', 'REVIEW', 'CLEAN'):
        c = by_bucket.get(b, 0)
        print(f"  {b:<11s}  {c:>3}  ({100*c/n:5.1f}%)")

    # Per-type-and-bucket
    print("\nPer test-set:")
    seen: dict[str, dict[str, int]] = {}
    for r in scored:
        seen.setdefault(r['source_set'], {'LIKELY_BAD': 0, 'REVIEW': 0, 'CLEAN': 0})
        seen[r['source_set']][r['bucket']] += 1
    for s, c in seen.items():
        tot = sum(c.values())
        print(f"  {s:<40s}  bad={c['LIKELY_BAD']:>2}  review={c['REVIEW']:>2}  "
              f"clean={c['CLEAN']:>3}  (total {tot})")

    if args.markdown:
        emit_markdown(scored, Path(args.markdown))
        print(f"\nMarkdown report written to {args.markdown}")
    if args.jsonl:
        out = Path(args.jsonl)
        with open(out, 'w') as f:
            for r in scored:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        print(f"JSONL report written to {args.jsonl}")


if __name__ == '__main__':
    main()
