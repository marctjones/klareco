#!/usr/bin/env python3
"""
The MISSING upstream: OpenTriviaDB -> Claude-translated Esperanto candidates. (#737)

VERSION: v1.0
COMPATIBLE WITH: qa_gate.py (produces its --input JSONL)
DEPENDENCIES: network (opentdb.com), the `claude` CLI (headless -p)
STAGE: Evaluation / test-set construction

Description:
    qa_gate.py was written to VALIDATE translated trivia against the
    parser + DuckDB store, but the step that FETCHES English trivia and TRANSLATES
    it to Esperanto never existed as a script. This is that step. It is the
    circularity-free track: the questions originate OUTSIDE the parser's frame, so
    unlike corpus-generated questions they are not answerable-by-construction via
    the same structures a reranker uses.

    Flow:
      1. FETCH English trivia from OpenTriviaDB, restricted to corpus-friendly
         categories (History / Geography / Science / General / Politics) — the
         Anglo pop-culture categories have ~zero overlap with an Esperanto
         literary+encyclopedic corpus and would all be rejected downstream.
      2. TRANSLATE each {question, correct_answer} to natural Esperanto via the
         Claude CLI, rephrasing the question into a proper interrogative
         (Kiu/Kio/Kie/Kiam/Kiom…). Batched.
      3. WRITE the candidate JSONL that qa_gate.py then validates
         (parse quality + corpus coverage). Yield is expected LOW — that is the
         point: only questions the corpus can actually answer survive.

    HONEST CAVEAT: translation is GENERATION (Claude writes the Esperanto), so the
    surface is fluent/canonical, not corpus-shaped. That is acceptable here because
    (a) the question is NOT derived from the answer sentence, so there is no
    parser-circularity, and (b) downstream validation is deterministic. The bias we
    remove (circularity) is worse than the bias we add (translationese).

Usage:
    python scripts/qa/qa_source_opentdb.py --amount 30 --out data/staging/opentdb_eo.jsonl
    # then:
    python scripts/qa/qa_gate.py --input data/staging/opentdb_eo.jsonl

Last Updated: 2026-07-17
Related Issues: #736, #737
"""

from __future__ import annotations

import argparse
import html
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# OpenTDB category ids that plausibly overlap an Esperanto encyclopedic corpus.
_CATEGORIES = {
    9:  'General Knowledge', 17: 'Science & Nature', 19: 'Science: Mathematics',
    22: 'Geography', 23: 'History', 24: 'Politics', 25: 'Art',
}


# Multiple-choice framing that has no standalone answer (#844). "Which of these
# is NOT a berry?" needs the options to make sense, and no corpus sentence asserts
# a negative — these ALWAYS fail the answerability check downstream, so drop them
# at the source to spend translation budget only on answerable questions.
_MC_MARKERS = ('which of the following', 'which of these', 'which one of',
               'of the following', 'of these', ' not ', ' not?', ' except',
               'all of the', 'none of the')


def _is_mc_framed(en_question: str) -> bool:
    q = ' ' + en_question.lower()
    return any(m in q for m in _MC_MARKERS)


# OpenTDB rate limit: each IP may hit the API once per 5 seconds. We pace to a
# small margin above that and back off hard on a rate-limit response_code.
_API = 'https://opentdb.com/api.php'
_TOKEN_API = 'https://opentdb.com/api_token.php'
_MIN_INTERVAL = 5.5   # seconds between requests (documented minimum is 5)
_RATE_BACKOFF = 15    # seconds to wait if the API reports response_code 5


def _get(url: str, timeout: int = 30) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def request_token() -> str | None:
    """A session token makes OpenTDB never re-serve a question until exhausted —
    avoids duplicates when accumulating across runs."""
    try:
        d = _get(f'{_TOKEN_API}?command=request')
        return d.get('token') if d.get('response_code') == 0 else None
    except Exception:
        return None


def fetch_opentdb(amount: int, category: int | None = None,
                  difficulty: str | None = None,
                  token: str | None = None) -> tuple[list[dict], int]:
    """Returns (results, response_code). Does NOT sleep — the caller paces requests.
    response_code: 0 ok · 1 no results · 3/4 token issue · 5 rate-limited."""
    url = f'{_API}?amount={min(amount, 50)}&type=multiple'
    if category:
        url += f'&category={category}'
    if difficulty:
        url += f'&difficulty={difficulty}'
    if token:
        url += f'&token={token}'
    data = _get(url)
    code = data.get('response_code', -1)
    out = []
    if code == 0:
        for q in data.get('results', []):
            enq = html.unescape(q['question'])
            if _is_mc_framed(enq):    # drop unanswerable multiple-choice framing
                continue
            out.append({
                'en_question': enq,
                'en_answer': html.unescape(q['correct_answer']),
                'category': q.get('category'), 'difficulty': q.get('difficulty'),
            })
    return out, code


_TR_PROMPT = """Translate these English trivia items into natural, grammatical, PURE Esperanto.
For each: rephrase the QUESTION as a proper Esperanto interrogative starting with
Kiu/Kio/Kies/Kiun/Kie/Kiam/Kiom/Kial/Kiel, and translate the ANSWER as the exact
Esperanto term (a proper noun stays a proper noun). No English words in the output.

Reply with ONLY a JSON array, one object per item, no prose:
[{"id":"...","eo_question":"...","eo_answer":"..."}]

ITEMS:
"""


def translate_batch(items: list[dict], model: str | None = None,
                    timeout: int = 240) -> dict:
    prompt = _TR_PROMPT + '\n'.join(
        json.dumps({'id': str(i), 'question': it['en_question'],
                    'answer': it['en_answer']}, ensure_ascii=False)
        for i, it in enumerate(items))
    cmd = ['claude', '-p'] + (['--model', model] if model else [])
    try:
        r = subprocess.run(cmd, input=prompt, text=True,
                           capture_output=True, timeout=timeout)
        i, j = r.stdout.find('['), r.stdout.rfind(']')
        arr = json.loads(r.stdout[i:j + 1]) if i != -1 and j > i else []
    except Exception:
        arr = []
    return {str(v['id']): v for v in arr if isinstance(v, dict) and 'id' in v}


# OpenTDB has category ids 9..32. _CATEGORIES (above) is the corpus-friendly subset.
_ALL_CATEGORY_IDS = tuple(range(9, 33))
_FRIENDLY_IDS = frozenset(_CATEGORIES)


def _pacer(min_interval=_MIN_INTERVAL):
    last = [0.0]

    def pace():
        dt = time.monotonic() - last[0]
        if dt < min_interval:
            time.sleep(min_interval - dt)
        last[0] = time.monotonic()
    return pace


def download_all(out_path: str, category_ids=_ALL_CATEGORY_IDS) -> list:
    """Exhaustively pull EVERY question OpenTDB will serve, using a session token so
    nothing repeats. MC-filtered. Rate-limit paced with backoff. Steps the request
    amount down near a category's tail (OpenTDB returns code 1 if it can't fill the
    full 50), so we don't miss the last few. Writes a raw EN dump; returns the rows."""
    token = request_token()
    pace = _pacer()
    rows, seen = [], set()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    # write incrementally so a kill during the ~10-min rate-limited fetch does not
    # throw away progress (the cache is valid at any point).
    with open(out_path, 'w', encoding='utf-8') as f:
        for cid in category_ids:
            for amount in (50, 10, 1):      # step down to sweep the category tail
                while True:
                    pace()
                    try:
                        got, code = fetch_opentdb(amount, category=cid, token=token)
                    except Exception as e:
                        print(f'  cat {cid} error: {e}', flush=True); code, got = 1, []
                    if code == 5:                       # rate-limited
                        print(f'  rate-limited; backoff {_RATE_BACKOFF}s', flush=True)
                        time.sleep(_RATE_BACKOFF); continue
                    if code == 3:                       # token lost
                        token = request_token(); continue
                    if code in (1, 4) or not got:       # exhausted at this amount
                        break
                    new = 0
                    for it in got:
                        it['category_id'] = cid
                        if it['en_question'] in seen:
                            continue
                        seen.add(it['en_question']); rows.append(it); new += 1
                        f.write(json.dumps(it, ensure_ascii=False) + '\n')
                    if new:
                        f.flush()
                        print(f'  cat {cid}: +{new}  (total {len(rows)})', flush=True)
    print(f'\n  ✓ downloaded {len(rows)} questions -> {out_path}')
    return rows


def translate_all(rows: list, out_path: str, batch: int, model=None) -> int:
    """Batch-translate EN rows to Esperanto candidates (batched claude calls)."""
    out = []
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        tr = translate_batch(chunk, model=model)
        for k, it in enumerate(chunk):
            v = tr.get(str(k))
            if not v or not v.get('eo_question') or not v.get('eo_answer'):
                continue
            out.append({
                'source': 'opentdb.com', 'translator': 'claude-cli',
                'category': it.get('category'), 'difficulty': it.get('difficulty'),
                'en_question': it['en_question'], 'en_answer': it['en_answer'],
                'eo_question': v['eo_question'].strip(),
                'eo_answer': v['eo_answer'].strip(),
                'question_type': 'KIU',
            })
        print(f'  translated {min(i+batch,len(rows)):>4d}/{len(rows)}  '
              f'usable {len(out)}', flush=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'\n  ✓ {len(out)} translated candidates -> {out_path}')
    return len(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--download-all', action='store_true',
                    help='exhaustively pull EVERY OpenTDB question to --raw-out, then stop')
    ap.add_argument('--raw-out', default='data/staging/opentdb_raw.jsonl',
                    help='where --download-all writes / --from-raw reads the EN dump')
    ap.add_argument('--from-raw', metavar='FILE',
                    help='translate from a cached raw dump instead of fetching')
    # Default: translate ALL categories and let the corpus-coverage + answerability
    # gates decide empirically. (A category pre-filter was a wrong assumption —
    # measured, the corpus has thousands of video-game/anime/cartoon sentences.)
    ap.add_argument('--friendly-only', action='store_true', default=False,
                    help='opt in to translate only corpus-friendly categories '
                         '(faster/cheaper; may miss pop-culture gold the corpus has)')
    ap.add_argument('--limit', type=int, default=None, help='cap rows to translate')
    ap.add_argument('--batch', type=int, default=15, help='items per translate call')
    ap.add_argument('--model', default=None)
    ap.add_argument('--out', default='data/staging/opentdb_eo.jsonl')
    args = ap.parse_args()

    if args.download_all:
        download_all(args.raw_out)
        print(f'  next: python scripts/qa/qa_source_opentdb.py --from-raw {args.raw_out}')
        return 0

    # get the EN rows: from a cached dump, else a one-shot exhaustive download
    if args.from_raw:
        rows = [json.loads(l) for l in open(args.from_raw, encoding='utf-8') if l.strip()]
        print(f'  loaded {len(rows)} rows from {args.from_raw}')
    else:
        rows = download_all(args.raw_out)

    if args.friendly_only:
        rows = [r for r in rows if r.get('category_id', -1) in _FRIENDLY_IDS
                or (r.get('category') in _CATEGORIES.values())]
        print(f'  corpus-friendly categories only: {len(rows)} rows')
    if args.limit:
        rows = rows[:args.limit]

    print(f'  translating {len(rows)} via claude CLI (batched, {args.batch}/call)…')
    translate_all(rows, args.out, args.batch, model=args.model)
    print(f'  next: python scripts/qa/qa_gate.py --input {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
