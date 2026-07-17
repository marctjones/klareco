#!/usr/bin/env python3
"""
The MISSING upstream: OpenTriviaDB -> Claude-translated Esperanto candidates. (#737)

VERSION: v1.0
COMPATIBLE WITH: build_trivia_bank.py (produces its --input JSONL)
DEPENDENCIES: network (opentdb.com), the `claude` CLI (headless -p)
STAGE: Evaluation / test-set construction

Description:
    build_trivia_bank.py was written to VALIDATE translated trivia against the
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
      3. WRITE the candidate JSONL that build_trivia_bank.py then validates
         (parse quality + corpus coverage). Yield is expected LOW — that is the
         point: only questions the corpus can actually answer survive.

    HONEST CAVEAT: translation is GENERATION (Claude writes the Esperanto), so the
    surface is fluent/canonical, not corpus-shaped. That is acceptable here because
    (a) the question is NOT derived from the answer sentence, so there is no
    parser-circularity, and (b) downstream validation is deterministic. The bias we
    remove (circularity) is worse than the bias we add (translationese).

Usage:
    python scripts/eval/build_trivia_from_opentdb.py --amount 30 --out data/staging/opentdb_eo.jsonl
    # then:
    python scripts/eval/build_trivia_bank.py --input data/staging/opentdb_eo.jsonl

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


def fetch_opentdb(amount: int, category: int | None = None,
                  difficulty: str | None = None) -> list[dict]:
    url = f'https://opentdb.com/api.php?amount={min(amount, 50)}&type=multiple'
    if category:
        url += f'&category={category}'
    if difficulty:
        url += f'&difficulty={difficulty}'
    with urllib.request.urlopen(url, timeout=30) as r:
        data = json.loads(r.read().decode())
    out = []
    for q in data.get('results', []):
        out.append({
            'en_question': html.unescape(q['question']),
            'en_answer': html.unescape(q['correct_answer']),
            'category': q.get('category'), 'difficulty': q.get('difficulty'),
        })
    return out


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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--amount', type=int, default=30, help='total EN trivia to fetch')
    ap.add_argument('--batch', type=int, default=10, help='items per translate call')
    ap.add_argument('--model', default=None)
    ap.add_argument('--out', default='data/staging/opentdb_eo.jsonl')
    args = ap.parse_args()

    # fetch across categories (OpenTDB caps 50/request + rate-limits ~1/5s)
    fetched, per_cat = [], max(1, args.amount // len(_CATEGORIES))
    for cid, cname in _CATEGORIES.items():
        if len(fetched) >= args.amount:
            break
        try:
            got = fetch_opentdb(per_cat, category=cid)
            fetched.extend(got)
            print(f'  fetched {len(got):2d} from {cname}', flush=True)
        except Exception as e:
            print(f'  fetch failed for {cname}: {e}', flush=True)
        time.sleep(5)  # respect OpenTDB rate limit
    fetched = fetched[:args.amount]
    print(f'\n  {len(fetched)} English trivia fetched\n  translating via claude CLI…')

    out = []
    for i in range(0, len(fetched), args.batch):
        chunk = fetched[i:i + args.batch]
        tr = translate_batch(chunk, model=args.model)
        for k, it in enumerate(chunk):
            v = tr.get(str(k))
            if not v or not v.get('eo_question') or not v.get('eo_answer'):
                continue
            out.append({
                'source': 'opentdb.com', 'translator': 'claude-cli',
                'category': it['category'], 'difficulty': it['difficulty'],
                'en_question': it['en_question'], 'en_answer': it['en_answer'],
                'eo_question': v['eo_question'].strip(),
                'eo_answer': v['eo_answer'].strip(),
                'question_type': 'KIU',
            })
        print(f'  translated {min(i+args.batch,len(fetched)):>3d}/{len(fetched)}  '
              f'usable {len(out)}', flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'\n  ✓ {len(out)} translated candidates -> {args.out}')
    print(f'  next: python scripts/eval/build_trivia_bank.py --input {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
