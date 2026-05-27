#!/usr/bin/env python3
"""
Fetch fresh trivia questions, translate to Esperanto, test both systems.

VERSION: v1.0
COMPATIBLE WITH: Klareco orchestrator + local Ollama LLM
DEPENDENCIES: stdlib (urllib, html, json); Ollama running locally
STAGE: Evaluation

Description:
    Pulls N trivia questions from Open Trivia Database (OpenTDB —
    https://opentdb.com/api.php), translates each into Esperanto using
    the local LLM, runs both Klareco and the LLM on the Esperanto
    question, and reports per-category / per-question-type accuracy.

    Optionally appends successful pairs to a corpus JSONL for use as
    future test data.

Pipeline:
    OpenTDB API → fetch → HTML-decode → translate to EO → ask Klareco
                                                       → ask LLM
                                                       → score
    Aggregate by category and question_type (KIU/KIO/KIE/KIAM/KIOM/Other)

Usage:
    # Test 20 fresh questions
    python scripts/eval/test_new_trivia.py --n 20

    # Filter by category (numeric ID — see OpenTDB docs)
    python scripts/eval/test_new_trivia.py --n 30 --category 9    # General Knowledge
    python scripts/eval/test_new_trivia.py --n 30 --category 17   # Science & Nature

    # Filter by difficulty
    python scripts/eval/test_new_trivia.py --n 30 --difficulty easy

    # Append translated questions to a corpus JSONL for reuse
    python scripts/eval/test_new_trivia.py --n 30 \
        --append-to data/test_sets/trivia_external_fresh.jsonl

    # Use a different LLM for translation + comparison
    python scripts/eval/test_new_trivia.py --n 20 --model qwen3:latest

OpenTDB categories (subset):
    9  = General Knowledge       17 = Science & Nature
    18 = Computers                19 = Mathematics
    22 = Geography                23 = History
    24 = Politics                 25 = Art
    26 = Celebrities              27 = Animals
    28 = Vehicles                 21 = Sports

Last Updated: 2026-05-27
"""
from __future__ import annotations

import argparse
import html
import json
import re
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

OPENTDB_URL = 'https://opentdb.com/api.php'
OLLAMA_URL = 'http://localhost:11434/api/chat'


# ---------------------------------------------------------------------------
# OpenTDB client
# ---------------------------------------------------------------------------

def fetch_trivia(n: int, category: Optional[int] = None,
                  difficulty: Optional[str] = None) -> list[dict]:
    """Fetch N trivia questions from OpenTDB. Returns the 'results' list.

    Decodes HTML entities. Returns up to N — OpenTDB caps at 50 per
    request; we loop if needed."""
    out: list[dict] = []
    remaining = n
    while remaining > 0:
        batch_size = min(remaining, 50)
        params = {'amount': batch_size, 'type': 'multiple'}
        if category is not None:
            params['category'] = category
        if difficulty:
            params['difficulty'] = difficulty
        url = OPENTDB_URL + '?' + urllib.parse.urlencode(params)
        with urllib.request.urlopen(url, timeout=30) as resp:
            payload = json.loads(resp.read().decode('utf-8'))
        if payload.get('response_code') != 0:
            print(f'WARN: OpenTDB returned code {payload.get("response_code")}',
                  file=sys.stderr)
            break
        for q in payload.get('results', []):
            out.append({
                'question_en':       html.unescape(q['question']),
                'correct_answer_en': html.unescape(q['correct_answer']),
                'category':          q.get('category', ''),
                'difficulty':        q.get('difficulty', ''),
                'type':              q.get('type', ''),
            })
        remaining = n - len(out)
        if remaining > 0:
            # OpenTDB rate-limits per IP; ~5 req/sec is safe
            time.sleep(1.5)
    return out[:n]


# ---------------------------------------------------------------------------
# Ollama client (mini-version, no shared module dependency)
# ---------------------------------------------------------------------------

def ollama_chat(model: str, user_message: str, system: str = '',
                 timeout: int = 60) -> tuple[str, float]:
    """Single-turn Ollama chat. Returns (response, latency_seconds)."""
    msgs = []
    if system:
        msgs.append({'role': 'system', 'content': system})
    msgs.append({'role': 'user', 'content': user_message})
    payload = {
        'model': model,
        'messages': msgs,
        'stream': False,
        'options': {'temperature': 0.1, 'num_predict': 200},
    }
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(OLLAMA_URL, data=data,
                                  headers={'Content-Type': 'application/json'})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode('utf-8'))
    except Exception as e:
        return f'[OLLAMA_ERROR: {e}]', time.time() - t0
    return (body.get('message', {}).get('content') or '').strip(), time.time() - t0


def check_ollama_alive(model: str) -> bool:
    try:
        with urllib.request.urlopen('http://localhost:11434/api/tags',
                                     timeout=5) as resp:
            tags = json.loads(resp.read().decode('utf-8'))
            return model in {m['name'] for m in tags.get('models', [])}
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------

_TRANSLATE_QUESTION_SYS = (
    "You are a translator. Translate the user's English trivia question "
    "into Esperanto. Critical rules:\n"
    "1. Output ONLY the Esperanto translation — nothing else.\n"
    "2. Translate the QUESTION as-is. Do NOT answer it. Do NOT change "
    "the topic. Do NOT add facts.\n"
    "3. Keep proper nouns (people, places, works, song titles, etc.) in "
    "their original spelling.\n"
    "4. Keep years and numbers as digits.\n"
    "5. End with '?' if the original is a question.\n"
    "\n"
    "Examples:\n"
    "EN: Who painted the Mona Lisa?\n"
    "EO: Kiu pentris la Mona Lisa?\n"
    "\n"
    "EN: In what year did World War II end?\n"
    "EO: En kiu jaro finiĝis la Dua Mondmilito?\n"
    "\n"
    "EN: What is the capital of Australia?\n"
    "EO: Kio estas la ĉefurbo de Aŭstralio?\n"
    "\n"
    "EN: Who released the song 'Photograph' in 2005?\n"
    "EO: Kiu eldonis la kanzonon 'Photograph' en 2005?"
)

_TRANSLATE_ANSWER_SYS = (
    "You translate a single English word, name, or short phrase into "
    "Esperanto. Output ONLY the Esperanto form, no preamble.\n"
    "Rules:\n"
    "- Person names: keep original spelling (George R. R. Martin → George R. R. Martin).\n"
    "- Place names: use the Esperanto form if standard (Australia → Aŭstralio, "
    "London → Londono); otherwise keep original.\n"
    "- Common nouns: translate to Esperanto (clown → klaŭno, cheese → fromaĝo).\n"
    "- Numbers and years: as digits."
)


def translate_to_esperanto(model: str, text: str,
                            is_answer: bool = False) -> str:
    """Translate text to Esperanto via the LLM. Returns cleaned text.

    Validates the output isn't obviously wrong (e.g. just the answer,
    or empty, or excessively long)."""
    sys_prompt = _TRANSLATE_ANSWER_SYS if is_answer else _TRANSLATE_QUESTION_SYS
    result, _ = ollama_chat(model, text, system=sys_prompt, timeout=30)
    # Strip any 'Esperanto:' / 'EO:' / 'Translation:' prefix
    result = re.sub(r'^\s*(?:Esperanto|EO|Translation|Traduko)\s*:\s*',
                    '', result, flags=re.IGNORECASE)
    result = result.strip().strip('"').strip("'").strip()
    # Question translations should END with '?'; if not, the model
    # probably gave an answer or commentary.
    if not is_answer and text.rstrip().endswith('?') and '?' not in result:
        # Retry once with a more explicit nudge
        retry_msg = (f"Translate ONLY this English question to Esperanto. "
                     f"Output the Esperanto question ending in '?'. "
                     f"Do NOT answer it.\n\nEnglish: {text}")
        result, _ = ollama_chat(model, retry_msg, system=sys_prompt, timeout=30)
        result = re.sub(r'^\s*(?:Esperanto|EO|Translation|Traduko)\s*:\s*',
                        '', result, flags=re.IGNORECASE)
        result = result.strip().strip('"').strip("'").strip()
    return result


# ---------------------------------------------------------------------------
# Question-type detection
# ---------------------------------------------------------------------------

_TYPE_PATTERNS = [
    (r'^\s*who\b',          'KIU'),
    (r'^\s*what\s+is\b',    'KIO'),
    (r'^\s*what\s+was\b',   'KIO'),
    (r'^\s*what\b',         'KIO'),
    (r'^\s*which\b',        'KIO'),
    (r'^\s*where\b',        'KIE'),
    (r'^\s*when\b',         'KIAM'),
    (r'^\s*how\s+many\b',   'KIOM'),
    (r'^\s*how\s+much\b',   'KIOM'),
    (r'^\s*how\s+old\b',    'KIOM'),
    (r'^\s*how\b',          'KIEL'),
    (r'^\s*why\b',          'KIAL'),
    (r'^\s*in\s+which\s+year\b', 'KIAM'),
    (r'^\s*in\s+what\s+year\b',  'KIAM'),
]


def detect_question_type(question_en: str) -> str:
    q = question_en.strip().lower()
    for pat, qt in _TYPE_PATTERNS:
        if re.match(pat, q):
            return qt
    return 'Other'


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _fold(s: str) -> str:
    import unicodedata
    return ''.join(c for c in unicodedata.normalize('NFKD', s)
                   if not unicodedata.combining(c)).lower()


def is_correct(answer: str, expected_keywords: list[str]) -> bool:
    if not answer or not expected_keywords:
        return False
    a = _fold(answer)
    return any(_fold(k) in a for k in expected_keywords if k)


# ---------------------------------------------------------------------------
# Klareco (lazy)
# ---------------------------------------------------------------------------

_PIPELINE = None


def klareco_answer(question_eo: str) -> tuple[str, float]:
    global _PIPELINE
    if _PIPELINE is None:
        from klareco.orchestrator.factory import build_default_pipeline
        _PIPELINE = build_default_pipeline(
            whoosh_index_dir='data/indexes/whoosh_v2',
            duckdb_path='data/indexes/duckdb_store.db',
            top_k=10,
        )
    t0 = time.time()
    try:
        result = _PIPELINE.answer(question_eo)
    except Exception as e:
        return f'[KLARECO_ERROR: {e}]', time.time() - t0
    return result.text, time.time() - t0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n', type=int, required=True,
                    help='Number of fresh trivia questions to fetch and test')
    ap.add_argument('--model', default='llama3.2:latest',
                    help='Ollama model for the answer-comparison run '
                         '(small/fast OK, e.g. llama3.2:latest)')
    ap.add_argument('--translate-model', default=None,
                    help='Ollama model for EN→EO translation. Defaults '
                         'to --model. Use a larger model here (e.g. '
                         'qwen3:latest) if --model is too small to '
                         'translate trivia questions accurately')
    ap.add_argument('--category', type=int, default=None,
                    help='OpenTDB category ID (see docstring)')
    ap.add_argument('--difficulty', choices=['easy', 'medium', 'hard'],
                    default=None)
    ap.add_argument('--append-to',
                    help='Optional: path to JSONL to append translated '
                         'questions for reuse as test data')
    ap.add_argument('--output-jsonl',
                    help='Optional: full per-question detail output')
    ap.add_argument('--no-klareco', action='store_true',
                    help='Skip Klareco — useful when only verifying translations')
    ap.add_argument('--no-llm-answer', action='store_true',
                    help='Skip the LLM answer column (translation still uses LLM)')
    ap.add_argument('--seed', type=int, default=None,
                    help='Reserved; OpenTDB doesn\'t take seeds, but reserves '
                         'for future reproducibility')
    args = ap.parse_args()

    translate_model = args.translate_model or args.model
    needed = {args.model, translate_model}
    for m in needed:
        if not check_ollama_alive(m):
            print(f'ERROR: model {m!r} not available via Ollama. '
                  f'Run: ollama pull {m}', file=sys.stderr)
            sys.exit(2)
    if translate_model != args.model:
        print(f'Using {translate_model!r} for EN→EO translation, '
              f'{args.model!r} for answer comparison.', flush=True)

    # 1. Fetch
    print(f'Fetching {args.n} trivia questions from OpenTDB…', flush=True)
    trivia = fetch_trivia(args.n, args.category, args.difficulty)
    print(f'  got {len(trivia)} questions', flush=True)
    if not trivia:
        sys.exit(1)

    # 2. Translate + test
    rows: list[dict] = []
    for i, t in enumerate(trivia, 1):
        q_en = t['question_en']
        a_en = t['correct_answer_en']
        qtype = detect_question_type(q_en)

        print(f'\n[{i:>3}/{len(trivia)}] {qtype} · {t["category"]}')
        print(f'  EN-Q: {q_en}')
        print(f'  EN-A: {a_en}')

        # Translate question
        q_eo = translate_to_esperanto(translate_model, q_en, is_answer=False)
        print(f'  EO-Q: {q_eo}')
        # Translate answer
        a_eo = translate_to_esperanto(translate_model, a_en, is_answer=True)
        print(f'  EO-A: {a_eo}')

        # Build expected keyword list: include both EN and EO forms
        expected = list({a_en, a_eo} - {''})

        # Ask Klareco
        k_ans, k_lat = ('', 0.0)
        k_correct = False
        if not args.no_klareco:
            k_ans, k_lat = klareco_answer(q_eo)
            k_correct = is_correct(k_ans, expected)
            print(f'  Klareco ({k_lat:.1f}s): {"✓" if k_correct else "·"} '
                  f'{k_ans[:120]}')

        # Ask LLM (in Esperanto, like Klareco)
        l_ans, l_lat = ('', 0.0)
        l_correct = False
        if not args.no_llm_answer:
            l_ans, l_lat = ollama_chat(
                args.model, q_eo,
                system='Vi estas asistanto kiu respondas en Esperanto. '
                       'Respondu mallonge (ne pli ol 2 frazoj). '
                       'Se vi ne scias, diru "Mi ne scias".',
            )
            l_correct = is_correct(l_ans, expected)
            print(f'  LLM ({l_lat:.1f}s):     {"✓" if l_correct else "·"} '
                  f'{l_ans[:120]}')

        rows.append({
            'category':          t['category'],
            'difficulty':        t['difficulty'],
            'question_type':     qtype,
            'question_en':       q_en,
            'question_eo':       q_eo,
            'expected_answer_en': a_en,
            'expected_answer_eo': a_eo,
            'expected_keywords': expected,
            'klareco_answer':    k_ans,
            'klareco_correct':   k_correct,
            'klareco_latency_s': round(k_lat, 3),
            'llm_answer':        l_ans,
            'llm_correct':       l_correct,
            'llm_latency_s':     round(l_lat, 3),
        })

    # 3. Aggregate report
    print(f'\n{"="*72}\nAGGREGATE — {len(rows)} questions\n{"="*72}')
    n = max(1, len(rows))
    if not args.no_klareco:
        k_c = sum(1 for r in rows if r['klareco_correct'])
        k_lat = statistics.mean(r['klareco_latency_s'] for r in rows)
        print(f'  Klareco:           {k_c}/{n}  {100*k_c/n:5.1f}%   '
              f'avg_lat={k_lat:.2f}s')
    if not args.no_llm_answer:
        l_c = sum(1 for r in rows if r['llm_correct'])
        l_lat = statistics.mean(r['llm_latency_s'] for r in rows)
        print(f'  {args.model:<18s} {l_c}/{n}  {100*l_c/n:5.1f}%   '
              f'avg_lat={l_lat:.2f}s')

    if not args.no_klareco and not args.no_llm_answer:
        both = sum(1 for r in rows if r['klareco_correct'] and r['llm_correct'])
        k_only = sum(1 for r in rows if r['klareco_correct'] and not r['llm_correct'])
        l_only = sum(1 for r in rows if r['llm_correct'] and not r['klareco_correct'])
        none = sum(1 for r in rows if not r['klareco_correct'] and not r['llm_correct'])
        print(f'\n  Head-to-head:')
        print(f'    both correct:      {both:>4d}')
        print(f'    klareco only:      {k_only:>4d}')
        print(f'    {args.model} only: {l_only:>4d}')
        print(f'    neither:           {none:>4d}')

    # By category
    by_cat: dict[str, dict] = defaultdict(
        lambda: {'n': 0, 'k_correct': 0, 'l_correct': 0})
    for r in rows:
        c = by_cat[r['category']]
        c['n'] += 1
        if r['klareco_correct']: c['k_correct'] += 1
        if r['llm_correct']:     c['l_correct'] += 1
    print(f'\n  By category:')
    print(f'    {"category":<28s} {"n":>4s}  {"Klareco":>10s}  {"LLM":>10s}')
    for cat, c in sorted(by_cat.items(), key=lambda kv: -kv[1]['n']):
        kp = f'{100*c["k_correct"]/c["n"]:.0f}%' if c['n'] else '-'
        lp = f'{100*c["l_correct"]/c["n"]:.0f}%' if c['n'] else '-'
        print(f'    {cat[:28]:<28s} {c["n"]:>4d}  '
              f'{c["k_correct"]:>2d}/{c["n"]:<2d} {kp:>5s}  '
              f'{c["l_correct"]:>2d}/{c["n"]:<2d} {lp:>5s}')

    # By question type
    by_type: dict[str, dict] = defaultdict(
        lambda: {'n': 0, 'k_correct': 0, 'l_correct': 0})
    for r in rows:
        t = by_type[r['question_type']]
        t['n'] += 1
        if r['klareco_correct']: t['k_correct'] += 1
        if r['llm_correct']:     t['l_correct'] += 1
    print(f'\n  By question type (KIU/KIO/KIE/KIAM/KIOM/KIAL/KIEL/Other):')
    print(f'    {"type":<8s} {"n":>4s}  {"Klareco":>10s}  {"LLM":>10s}')
    for qt, c in sorted(by_type.items(), key=lambda kv: -kv[1]['n']):
        kp = f'{100*c["k_correct"]/c["n"]:.0f}%' if c['n'] else '-'
        lp = f'{100*c["l_correct"]/c["n"]:.0f}%' if c['n'] else '-'
        print(f'    {qt:<8s} {c["n"]:>4d}  '
              f'{c["k_correct"]:>2d}/{c["n"]:<2d} {kp:>5s}  '
              f'{c["l_correct"]:>2d}/{c["n"]:<2d} {lp:>5s}')

    # By difficulty
    by_diff: dict[str, dict] = defaultdict(
        lambda: {'n': 0, 'k_correct': 0, 'l_correct': 0})
    for r in rows:
        d = by_diff[r['difficulty']]
        d['n'] += 1
        if r['klareco_correct']: d['k_correct'] += 1
        if r['llm_correct']:     d['l_correct'] += 1
    print(f'\n  By difficulty:')
    print(f'    {"diff":<8s} {"n":>4s}  {"Klareco":>10s}  {"LLM":>10s}')
    for diff in ['easy', 'medium', 'hard']:
        c = by_diff.get(diff)
        if not c or c['n'] == 0:
            continue
        kp = f'{100*c["k_correct"]/c["n"]:.0f}%'
        lp = f'{100*c["l_correct"]/c["n"]:.0f}%'
        print(f'    {diff:<8s} {c["n"]:>4d}  '
              f'{c["k_correct"]:>2d}/{c["n"]:<2d} {kp:>5s}  '
              f'{c["l_correct"]:>2d}/{c["n"]:<2d} {lp:>5s}')

    # 4. Outputs
    if args.output_jsonl:
        Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_jsonl, 'w') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        print(f'\nPer-question JSONL: {args.output_jsonl}')

    if args.append_to:
        Path(args.append_to).parent.mkdir(parents=True, exist_ok=True)
        with open(args.append_to, 'a') as f:
            for r in rows:
                pair = {
                    'id':                 f'opentdb_{int(time.time())}_{rows.index(r)}',
                    'topic':              r['category'].lower().replace(' ', '_'),
                    'question_type':      r['question_type'],
                    'question':           r['question_eo'],
                    'expected_answer':    r['expected_answer_eo'],
                    'expected_keywords':  r['expected_keywords'],
                    'source':             'opentdb',
                    'difficulty':         r['difficulty'],
                    'original_en':        r['question_en'],
                }
                f.write(json.dumps(pair, ensure_ascii=False) + '\n')
        print(f'Appended {len(rows)} pairs to: {args.append_to}')


if __name__ == '__main__':
    main()
