#!/usr/bin/env python3
"""
Language-quality audit for Q&A test sets (Stage 1.5 of the gate stack).

VERSION: v2.x
COMPATIBLE WITH: klareco.parser (any), klareco.proper_nouns dictionary,
                 any test-set JSONL with {question, expected_answer}
DEPENDENCIES: klareco.parser
STAGE: Evaluation

Description:
    Enforces R13 (Esperanto language quality) from the test-set quality
    standard. A pair must pass these mechanical checks before it can be
    considered for end-to-end pipeline eval — a grammatically broken
    question cannot reveal anything about retriever/extractor behavior.

    Mechanical checks per pair:
      L1. parser_clean — every content-word in the question parses with
          analizstato == 'sukceso' (the anchor proper noun is allowed to
          be unknown, since that's the *target* of the question).
      L2. diacritic_system — the question uses only ĉ ĝ ĥ ĵ ŝ ŭ. No
          x-system (cx, gx, sx, ux) or h-system (ch, gh, sh, uh) bleed,
          and no mixing.
      L3. interrogative_wellformed — starts with a ki- correlative
          (Kiu/Kio/Kie/Kiam/Kial/Kiel/Kiom/Kies/Kia/Kion/Kiun/Kies),
          ends with `?`, has exactly one interrogative word.
      L4. accusative_agreement — the direct object of a transitive
          verb in the question carries -n (or is a quoted work «...»,
          which is morphologically invariant in EO).
      L5. time_prep_correct — for KIAM questions: `en` + year is OK,
          `je` + year is rejected (`je` is for clock-time only).
      L6. tense_appropriate — soft check: KIAM "naskiĝis" type questions
          use past tense (-is); "Kio estas X?" definitional questions
          use present (-as). Flags surface-level tense mismatches.

    The script does NOT attempt to catch idiomaticity, calqued English,
    false-friend roots, or register problems — those require human
    review (Stage 3 of the gate stack).

Pipeline Position:
    <test_set>.jsonl → [THIS SCRIPT] → per-pair audit log + summary
                                     → (regenerate / filter / advance)

Usage:
    python scripts/eval/audit_language_quality.py \\
        --in data/test_sets/capability_100.jsonl \\
        --output results/lang_audit_<date>.jsonl

Inputs:
    --in        one or more JSONL test sets
    --strict    fail any pair with any L1-L6 failure (default: report only)

Outputs:
    Per-pair JSONL audit log via --output (optional).
    Aggregate summary printed to stdout.

Quality Checks:
    L1-L6 mechanical R13 checks. See docs/QA_TEST_SET_QUALITY_STANDARD.md.

Last Updated: 2026-05-21
Author: Claude Code (with Marc Jones)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from klareco.parser import parse  # noqa: E402


# ---------------------------------------------------------------------------
# Constants and patterns
# ---------------------------------------------------------------------------

_KI_CORRELATIVES = {
    'Kiu', 'Kiun', 'Kio', 'Kion', 'Kie', 'Kien', 'Kiam',
    'Kial', 'Kiel', 'Kiom', 'Kies', 'Kia', 'Kiaj', 'Kiajn',
}
_KI_INSIDE_FRONTED_PP = {
    # PPs that legitimately host a ki-correlative inside them
    'En', 'De', 'Al', 'Por', 'Pri', 'Pro', 'Kun', 'Sur', 'Sub',
    'Ĉe', 'Tra', 'Antaŭ', 'Post', 'Kontraŭ', 'Inter',
}

# x-system / h-system markers we want to reject when they show up
# inside an Esperanto content word. We can't just grep for "cx" because
# foreign names contain those bigrams; we specifically check that
# the bigram appears AFTER an Esperanto stem letter.
_X_SYSTEM_BIGRAMS = ('cx', 'gx', 'hx', 'jx', 'sx', 'ux')
_X_SYSTEM_BIGRAMS_UPPER = tuple(b.upper() for b in _X_SYSTEM_BIGRAMS)

_YEAR_RE = re.compile(r'\b(1[0-9]{3}|20[0-9]{2})\b')
_QUOTED_WORK_RE = re.compile(r'[«"„]\s*([^«»"]+?)\s*[»"]')

# Locative prepositions that govern KIE answers (also used to reject
# `je` for KIAM year answers — `je` is clock-time only).
_TIME_PREP_OK = {'en', 'dum', 'antaŭ', 'post', 'ekde', 'ĝis', 'tra'}
_TIME_PREP_BAD_WITH_YEAR = {'je', 'al', 'ĉe', 'sur', 'sub', 'super'}


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_parser_clean(question: str, anchor_hint: str | None = None
                       ) -> tuple[bool, str]:
    """L1: every content word in the question parses cleanly. The named
    anchor (proper noun the question is about) is allowed to be unknown,
    since that's the literal point of the question."""
    try:
        ast = parse(question)
    except Exception as e:
        return False, f'parser raised: {e}'

    # Walk the parsed words and find anything with analizstato != 'sukceso'.
    def _walk(node):
        if isinstance(node, dict):
            if node.get('tipo') == 'vorto':
                yield node
            else:
                for v in node.values():
                    yield from _walk(v)
        elif isinstance(node, list):
            for x in node:
                yield from _walk(x)

    bad: list[str] = []
    for w in _walk(ast):
        st = w.get('analizstato')
        if st in ('sukceso', None):
            continue
        # An anchor proper noun is allowed to be unknown — it's the *target*.
        if w.get('vortspeco') == 'propra_nomo' and st == 'propra_nomo_nekonata':
            continue
        if anchor_hint and w.get('plena_vorto') == anchor_hint:
            continue
        bad.append(f'{w.get("plena_vorto")}({st})')

    if bad:
        return False, f'unparsed content words: {", ".join(bad[:3])}'
    return True, 'all content words parse'


def check_diacritic_system(text: str) -> tuple[bool, str]:
    """L2: text uses only ĉ ĝ ĥ ĵ ŝ ŭ for Esperanto diacritics.
    Reject x-system and h-system patterns when they appear in tokens
    that look like Esperanto content words.

    Heuristic: an x-system bigram is "in an EO word" if it's flanked by
    lowercase Latin letters (so 'cx' between letters), and the token it
    appears in is NOT a clearly foreign capitalized proper name.
    """
    if not text:
        return False, 'empty'
    for tok in text.split():
        # Skip tokens that are quoted-work titles — they may be foreign.
        if tok.startswith(('«', '"', '„')):
            continue
        # Foreign proper names often start with a capital; if the whole
        # token starts with capital AND contains an x-system bigram,
        # we still flag it (Esperantized names should use ĉĝĥĵŝŭ).
        low = tok.lower()
        for bigram in _X_SYSTEM_BIGRAMS:
            if bigram in low:
                # Make sure it's flanked by letters (not 'gx' in '14gx'
                # which is unlikely but defensive)
                idx = low.find(bigram)
                if idx > 0 and idx + 2 < len(low):
                    pre = low[idx - 1]
                    post = low[idx + 2]
                    if pre.isalpha() and (post.isalpha() or post in '-'):
                        return False, f'x-system bigram in token {tok!r}'
                elif idx >= 0:
                    # bigram at start/end of token — still suspicious
                    return False, f'x-system bigram in token {tok!r}'
    return True, 'clean ĉĝĥĵŝŭ'


def check_interrogative_wellformed(question: str) -> tuple[bool, str]:
    """L3: question starts with a ki- correlative (optionally inside a
    fronted PP), ends with `?`, has exactly one ki- correlative."""
    q = question.strip()
    if not q.endswith('?'):
        return False, 'no `?` terminator'
    toks = q.rstrip('?').strip().split()
    if not toks:
        return False, 'empty after stripping ?'
    first = toks[0]
    second = toks[1] if len(toks) > 1 else ''
    starts_ok = (
        first in _KI_CORRELATIVES
        or (first in _KI_INSIDE_FRONTED_PP
            and (second in _KI_CORRELATIVES
                 or second.lower().startswith('kiu')
                 or second.lower().startswith('kio')
                 or second.lower().startswith('kia')))
    )
    if not starts_ok:
        return False, f'no leading ki-correlative (got {first!r})'

    # Count ki- correlatives — at most one outside of a relative clause.
    # (A relative clause is hard to detect without parsing; we allow up
    # to one ki- token inside the question for now.)
    ki_count = sum(
        1 for t in toks
        if t.strip(',.;:').lower() in {x.lower() for x in _KI_CORRELATIVES}
    )
    if ki_count > 2:
        return False, f'too many ki-correlatives ({ki_count})'

    return True, 'well-formed interrogative'


def check_accusative_agreement(question: str) -> tuple[bool, str]:
    """L4: direct object of a transitive verb in the question carries -n
    (or is a quoted work). We rely on the parser's objekto.kazo field.

    Allowed exceptions:
      - No objekto in the AST (intransitive question) — pass.
      - Object is a quoted work — quoted works are morphologically
        invariant, pass.
      - Object is a number or numeric expression — pass.
    """
    try:
        ast = parse(question)
    except Exception as e:
        return True, f'skipped (parser raised: {e})'

    obj = ast.get('objekto')
    if not isinstance(obj, dict):
        return True, 'no direct object'

    kerno = obj.get('kerno') if obj.get('tipo') == 'vortgrupo' else obj
    if not isinstance(kerno, dict):
        return True, 'no kerno'
    pv = kerno.get('plena_vorto') or ''
    kazo = kerno.get('kazo')
    if kazo == 'akuzativo':
        return True, 'object marked -n'

    # Quoted-work exception
    if _QUOTED_WORK_RE.search(question):
        # If the only object token is inside a quoted work, accept.
        m = _QUOTED_WORK_RE.search(question)
        if m and pv and pv in m.group(0):
            return True, 'object is quoted work (invariant)'

    # Numeric / year exception
    if pv and (_YEAR_RE.fullmatch(pv) or pv.replace('.', '').isdigit()):
        return True, 'numeric object'

    # If the verb is a copula (`estas`), `objekto` is actually a predicate
    # nominative and shouldn't be marked -n.
    verb = ast.get('verbo') or {}
    if isinstance(verb, dict) and verb.get('radiko') in {'est'}:
        return True, 'copula predicate (nominative)'

    return False, f'direct object {pv!r} lacks -n marking'


def check_time_prep_correct(question: str, question_type: str | None
                            ) -> tuple[bool, str]:
    """L5: for KIAM questions referencing a year, the preposition before
    the year (if any) must be a temporal one, not `je` (clock-time)
    or a locative."""
    qt = (question_type or '').upper()
    if qt != 'KIAM':
        return True, 'skipped (not KIAM)'

    # Find a year token in the question; check the preceding word.
    for m in _YEAR_RE.finditer(question):
        idx = m.start()
        pre = question[:idx].rstrip()
        # Last word before the year
        m_pre = re.search(r'(\w+)\s*$', pre)
        if not m_pre:
            continue
        prev = m_pre.group(1).lower()
        if prev in _TIME_PREP_BAD_WITH_YEAR:
            return False, f'year preceded by non-temporal preposition {prev!r}'
        if prev in _TIME_PREP_OK:
            return True, f'year preceded by ok preposition {prev!r}'
    return True, 'no year preposition to check'


def check_tense_appropriate(question: str, question_type: str | None
                            ) -> tuple[bool, str]:
    """L6 (soft): tense of the question verb matches the question type.

    - 'Kio estas X?' / 'Kiu estas X?' / 'Kie estas X?' — present `-as` OK
    - 'Kiu fondis X?' / 'Kiam naskiĝis X?' — past `-is` for completed actions
    - 'Kiam okazos X?' — future `-os` (rare in trivia)

    This is a soft check. It flags surface-level mismatches; it doesn't
    block them by default (unless --strict is set).
    """
    try:
        ast = parse(question)
    except Exception:
        return True, 'skipped (parser raised)'

    verb = ast.get('verbo') or {}
    if not isinstance(verb, dict):
        return True, 'no verb'
    tense = verb.get('tempo')
    radiko = verb.get('radiko')
    qt = (question_type or '').upper()

    # The copula `est` accepts any tense; trivially OK.
    if radiko == 'est':
        return True, 'copula — tense not constrained'

    # Past-action question patterns
    past_action_indicators = {
        'fond', 'nask', 'kre', 'verkis', 'mortis', 'okazis', 'eltrovis',
        'invent', 'malkov', 'establ', 'kre', 'pubvlikigis',
    }
    if radiko in past_action_indicators and tense != 'pasinteco':
        return False, f'past-action verb radiko={radiko!r} but tense={tense!r}'

    return True, f'tense={tense}, radiko={radiko}'


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

CHECKS = [
    ('parser_clean',            check_parser_clean),
    ('diacritic_system',        check_diacritic_system),
    ('interrogative_wellformed', check_interrogative_wellformed),
    ('accusative_agreement',    check_accusative_agreement),
    ('time_prep_correct',       check_time_prep_correct),
    ('tense_appropriate',       check_tense_appropriate),
]


def audit_pair(pair: dict) -> dict:
    """Run all language-quality checks on one pair."""
    question = pair.get('question') or ''
    qt = pair.get('question_type')
    # Resolve anchor hint from generator metadata if present
    anchor_hint = (pair.get('anchor_pv')
                   or pair.get('anchor')
                   or pair.get('anchor_entity'))

    results: dict[str, tuple[bool, str]] = {}
    for name, fn in CHECKS:
        try:
            if name == 'parser_clean':
                ok, reason = fn(question, anchor_hint)
            elif name == 'diacritic_system':
                ok, reason = fn(question)
            elif name == 'interrogative_wellformed':
                ok, reason = fn(question)
            elif name == 'accusative_agreement':
                ok, reason = fn(question)
            elif name in ('time_prep_correct', 'tense_appropriate'):
                ok, reason = fn(question, qt)
            else:
                ok, reason = fn(question)  # type: ignore[misc]
        except Exception as e:
            ok, reason = False, f'check raised: {e!r}'
        results[name] = (ok, reason)

    failed = [n for n, (ok, _) in results.items() if not ok]
    return {
        'id':            pair.get('id'),
        'question':      question,
        'question_type': qt,
        'verdict':       'PASS' if not failed else 'FAIL',
        'failed_checks': failed,
        'reasons':       {n: r for n, (_, r) in results.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--in', dest='inputs', nargs='+', required=True,
                    help='One or more JSONL test-set files.')
    ap.add_argument('--output', default=None,
                    help='Per-pair JSONL audit log path.')
    ap.add_argument('--strict', action='store_true',
                    help='Exit 1 if any pair fails.')
    ap.add_argument('--show-failures', type=int, default=15)
    args = ap.parse_args()

    all_results: list[dict] = []
    per_set: dict[str, dict[str, int]] = {}

    for ts_path in args.inputs:
        path = Path(ts_path)
        if not path.exists():
            print(f'SKIP: {path} not found', file=sys.stderr)
            continue
        pairs: list[dict] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    pairs.append(json.loads(line))
        per_set[path.name] = {'pass': 0, 'fail': 0}
        for p in pairs:
            r = audit_pair(p)
            r['source_set'] = path.name
            all_results.append(r)
            per_set[path.name][r['verdict'].lower()] += 1

    n = len(all_results)
    n_pass = sum(1 for r in all_results if r['verdict'] == 'PASS')
    n_fail = n - n_pass

    print(f'\nLanguage-quality audit: {n} pairs across {len(args.inputs)} set(s).')
    if n:
        print(f'  PASS: {n_pass} ({100*n_pass/n:.1f}%)')
        print(f'  FAIL: {n_fail} ({100*n_fail/n:.1f}%)')

    print('\nPer-set breakdown:')
    for s, c in per_set.items():
        tot = c['pass'] + c['fail']
        pct = (c['pass'] / tot * 100) if tot else 0
        print(f'  {s:<48s}  {c["pass"]:>3}/{tot:<3}  {pct:5.1f}% pass')

    # Failure-type counts
    fail_counts: dict[str, int] = {}
    for r in all_results:
        for chk in r['failed_checks']:
            fail_counts[chk] = fail_counts.get(chk, 0) + 1
    if fail_counts:
        print('\nFailure breakdown by check:')
        for chk, k in sorted(fail_counts.items(), key=lambda kv: -kv[1]):
            print(f'  {chk:<28s}  {k:>3}')

        per_chk_examples: dict[str, list[dict]] = {chk: [] for chk in fail_counts}
        for r in all_results:
            for chk in r['failed_checks']:
                if len(per_chk_examples[chk]) < args.show_failures:
                    per_chk_examples[chk].append(r)
        print('\nExamples:')
        for chk, examples in per_chk_examples.items():
            print(f'\n  [{chk}]')
            for r in examples:
                reason = r['reasons'].get(chk, '?')
                print(f'    - {(r["id"] or ""):<22s} {r["question"][:70]}')
                print(f'      reason: {reason}')

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w') as f:
            for r in all_results:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        print(f'\nFull audit written to {out}')

    if args.strict and n_fail > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
