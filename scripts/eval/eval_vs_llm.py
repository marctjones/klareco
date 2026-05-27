#!/usr/bin/env python3
"""
Head-to-head: Klareco vs a local LLM via Ollama.

VERSION: v1.0
COMPATIBLE WITH: post-Stage-3 Klareco pipeline + Ollama with at least
                 one Esperanto-capable model pulled
DEPENDENCIES: stdlib only (urllib for Ollama API)
STAGE: Evaluation

Description:
    For each question in a test set, runs both:
      1. Klareco's full orchestrator pipeline (the system we built)
      2. A local LLM via Ollama's HTTP API

    Records: answer text, latency, hallucination indicator, correctness
    (substring match against expected_keywords).

    Aggregates: per-system accuracy, mean/median latency, hallucination
    rate, percentage of questions where each system "wins" (got it right
    when the other didn't).

Pipeline Position:
    Test set JSONL → [THIS SCRIPT] → side-by-side report
                                  → markdown trade-off table

Usage:
    # Trivia subset, default model:
    python scripts/eval/eval_vs_llm.py \
        --test-set data/test_sets/capability_candidates_v1.jsonl \
        --model llama3.2:latest \
        --limit 30

    # Compare two LLMs at once:
    python scripts/eval/eval_vs_llm.py \
        --test-set data/test_sets/capability_candidates_v1.jsonl \
        --model llama3.2:latest --model phi3:latest \
        --limit 30

    # Math subset only:
    python scripts/eval/eval_vs_llm.py --math-suite --model llama3.2:latest

Outputs:
    Stdout: per-question table + aggregate report
    --output-jsonl: per-question detail JSON
    --output-md:    markdown report

Last Updated: 2026-05-27
Author: Claude Code (with Marc Jones)
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


OLLAMA_URL = 'http://localhost:11434/api/chat'

# Standard system prompt — keep it neutral and ask for short answers.
_SYSTEM_PROMPT = (
    'Vi estas asistanto kiu respondas en Esperanto. '
    'Respondu mallonge (ne pli ol 2 frazoj). '
    'Se vi ne scias la respondon, diru "Mi ne scias".'
)


# ---------------------------------------------------------------------------
# Built-in math test suite (for --math-suite mode)
# ---------------------------------------------------------------------------

MATH_TESTS = [
    ("Kiom estas du plus tri?",           ["5", "kvin"]),
    ("Kiom estas dek minus tri?",         ["7", "sep"]),
    ("Kio estas dudek oble tri?",         ["60", "sesdek"]),
    ("Kiom estas cent dividite per kvar?", ["25", "dudek kvin"]),
    ("Kio estas la kvadrata radiko de 144?", ["12", "dek du"]),
    ("Kiom estas 1917 minus 1859?",       ["58", "kvindek ok"]),
    ("Kiom da jaroj inter 1859 kaj 1917?", ["58", "kvindek ok"]),
    ("Kiom estas 100 plus 50?",            ["150", "cent kvindek"]),
    ("Kiom estas dudek tri minus dek?",    ["13", "dek tri"]),
    ("Kio estas la kvadrata radiko de 81?", ["9", "naŭ", "nau"]),
]


# ---------------------------------------------------------------------------
# Ollama client
# ---------------------------------------------------------------------------

def ollama_chat(model: str, user_message: str,
                 system_prompt: str = _SYSTEM_PROMPT,
                 timeout: int = 120) -> tuple[str, float]:
    """Call Ollama's /api/chat endpoint with a single-turn message.
    Returns (response_text, latency_seconds)."""
    payload = {
        'model': model,
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user',   'content': user_message},
        ],
        'stream': False,
        'options': {
            'temperature': 0.1,   # near-deterministic for reproducibility
            'num_predict': 200,
        },
    }
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(
        OLLAMA_URL, data=data,
        headers={'Content-Type': 'application/json'},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode('utf-8'))
    except urllib.error.URLError as e:
        return f'[OLLAMA_ERROR: {e}]', time.time() - t0
    except Exception as e:
        return f'[ERROR: {e}]', time.time() - t0
    elapsed = time.time() - t0
    msg = body.get('message') or {}
    return (msg.get('content') or '').strip(), elapsed


def check_ollama_alive(model: str) -> bool:
    """Verify Ollama is reachable AND the model is pulled."""
    try:
        with urllib.request.urlopen('http://localhost:11434/api/tags',
                                     timeout=5) as resp:
            tags = json.loads(resp.read().decode('utf-8'))
            names = {m['name'] for m in tags.get('models', [])}
            return model in names
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Klareco client (lazy: import only when the eval starts)
# ---------------------------------------------------------------------------

_PIPELINE = None


def klareco_answer(question: str) -> tuple[str, float]:
    """Run Klareco's orchestrator. Returns (final_text, latency_s)."""
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
        result = _PIPELINE.answer(question)
    except Exception as e:
        return f'[KLARECO_ERROR: {e}]', time.time() - t0
    return result.text, time.time() - t0


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def is_correct(answer: str, expected_keywords: list[str]) -> bool:
    """Substring match (case-insensitive, diacritic-folded)."""
    if not answer or not expected_keywords:
        return False
    a = _fold(answer.lower())
    return any(_fold(k.lower()) in a for k in expected_keywords if k)


def _fold(s: str) -> str:
    """Diacritic fold for fuzzy matching."""
    import unicodedata
    return ''.join(c for c in unicodedata.normalize('NFKD', s)
                   if not unicodedata.combining(c))


_CITATION_RE = __import__('re').compile(
    r'\n\n\[\d+\][\s\S]*$', __import__('re').MULTILINE)


def _strip_citation_footer(answer: str) -> str:
    """Remove the '\\n\\n[1] Source — snippet…' citation footer that
    Klareco appends. Leaves the actual body for length-based checks."""
    return _CITATION_RE.sub('', answer or '').strip()


def detect_hallucination(answer: str, correct: bool) -> bool:
    """Heuristic: does the answer assert facts without grounding?

    Refined definition: an answer is 'hallucinating' if it is WRONG AND
    confidently-asserted (verbose, not hedged with 'mi ne scias').

      - Correct answers are never hallucinations (definitionally)
      - 'mi ne scias' / 'mi ne trovis' answers are honest refusals
      - For Klareco: strip the citation footer before length-check
        (the citations make every answer look long otherwise)
      - For both: any output ≥ 80 chars asserting facts is suspicious
        if it's wrong
    """
    if correct:
        return False
    body = _strip_citation_footer(answer)
    a = body.lower()
    if 'mi ne scias' in a or 'mi ne trovis' in a:
        return False
    return len(body) >= 80


# ---------------------------------------------------------------------------
# Test-set loader
# ---------------------------------------------------------------------------

def load_test_set(path: Path, limit: int = 0) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def normalise_row(row: dict) -> tuple[str, list[str]]:
    """Extract (question, expected_keywords) from a test-set row."""
    q = row.get('question') or row.get('eo_question') or ''
    kws = row.get('expected_keywords') or []
    ans = row.get('expected_answer') or row.get('eo_answer')
    if ans and ans not in kws:
        kws = list(kws) + [ans]
    return q, [k for k in kws if k]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--test-set',
                    help='JSONL test set path. If omitted, use --math-suite.')
    ap.add_argument('--math-suite', action='store_true',
                    help='Use the built-in 10-question math suite.')
    ap.add_argument('--model', action='append', default=[],
                    help='Ollama model tag(s). Repeat for multiple models. '
                         'Default: llama3.2:latest')
    ap.add_argument('--limit', type=int, default=0,
                    help='Cap the number of questions (0 = all)')
    ap.add_argument('--output-jsonl', help='Write per-question detail here')
    ap.add_argument('--output-md', help='Write a markdown report here')
    ap.add_argument('--no-klareco', action='store_true',
                    help='Run only the LLM(s), skip Klareco')
    ap.add_argument('--no-llm', action='store_true',
                    help='Run only Klareco, skip the LLM(s)')
    args = ap.parse_args()

    if not args.model:
        args.model = ['llama3.2:latest']

    # Verify models exist before any work
    if not args.no_llm:
        for m in args.model:
            if not check_ollama_alive(m):
                print(f'ERROR: model {m!r} not available via Ollama. '
                      f'Run: ollama pull {m}', file=sys.stderr)
                sys.exit(2)
        print(f'Ollama alive; models verified: {args.model}')

    # Load test set
    if args.math_suite:
        test = [{'question': q, 'expected_keywords': k, 'category': 'math'}
                for q, k in MATH_TESTS]
        print(f'Using built-in math suite: {len(test)} questions')
    elif args.test_set:
        test = load_test_set(Path(args.test_set), limit=args.limit)
        print(f'Loaded {len(test)} questions from {args.test_set}')
    else:
        print('Need --test-set or --math-suite', file=sys.stderr)
        sys.exit(2)

    if args.limit and not args.math_suite:
        test = test[:args.limit]
        print(f'  (limited to first {len(test)})')

    # Run
    rows = []
    for i, row in enumerate(test, 1):
        q, expected = normalise_row(row)
        if not q:
            continue
        record = {'id': row.get('id', f'q{i}'),
                  'question': q, 'expected_keywords': expected}

        # Klareco
        if not args.no_klareco:
            text, lat = klareco_answer(q)
            correct = is_correct(text, expected)
            record['klareco_answer'] = text
            record['klareco_latency_s'] = round(lat, 3)
            record['klareco_correct'] = correct
            record['klareco_hallucinated'] = detect_hallucination(text, correct)

        # LLMs
        for model in args.model:
            if args.no_llm:
                continue
            text, lat = ollama_chat(model, q)
            tag = model.replace(':', '_').replace('/', '_')
            correct = is_correct(text, expected)
            record[f'{tag}_answer'] = text
            record[f'{tag}_latency_s'] = round(lat, 3)
            record[f'{tag}_correct'] = correct
            record[f'{tag}_hallucinated'] = detect_hallucination(text, correct)

        rows.append(record)

        # Stream a one-line summary
        marks = []
        if not args.no_klareco:
            marks.append(f"K{'✓' if record['klareco_correct'] else '·'}")
        for model in args.model:
            tag = model.replace(':', '_').replace('/', '_')
            if not args.no_llm:
                marks.append(f"{tag[:8]}{'✓' if record[f'{tag}_correct'] else '·'}")
        print(f'  [{i:>3}/{len(test)}]  {" ".join(marks)}  {q[:60]}')

    # Aggregate report
    print('\n' + '=' * 72)
    print('AGGREGATE')
    print('=' * 72)
    print(f'{"system":<25s} {"correct":>9s} {"acc%":>6s} {"avg_lat_s":>10s} '
          f'{"med_lat_s":>10s} {"hall%":>7s}')
    print('-' * 72)
    systems: list[tuple[str, str]] = []  # (display_name, key_prefix)
    if not args.no_klareco:
        systems.append(('klareco', 'klareco'))
    for model in args.model:
        if not args.no_llm:
            tag = model.replace(':', '_').replace('/', '_')
            systems.append((model, tag))
    for name, prefix in systems:
        n_correct = sum(1 for r in rows if r.get(f'{prefix}_correct'))
        n_hall = sum(1 for r in rows if r.get(f'{prefix}_hallucinated'))
        lats = [r.get(f'{prefix}_latency_s', 0) for r in rows if r.get(f'{prefix}_latency_s')]
        avg_lat = statistics.mean(lats) if lats else 0
        med_lat = statistics.median(lats) if lats else 0
        n = max(1, len(rows))
        print(f'{name:<25s} {n_correct:>4d}/{n:<4d} '
              f'{100*n_correct/n:>5.1f}% {avg_lat:>9.2f} {med_lat:>9.2f} '
              f'{100*n_hall/n:>6.1f}%')

    # Where each system uniquely won
    if not args.no_klareco and not args.no_llm and len(args.model) == 1:
        m = args.model[0].replace(':', '_').replace('/', '_')
        k_only = sum(1 for r in rows
                     if r.get('klareco_correct') and not r.get(f'{m}_correct'))
        l_only = sum(1 for r in rows
                     if r.get(f'{m}_correct') and not r.get('klareco_correct'))
        both = sum(1 for r in rows
                   if r.get('klareco_correct') and r.get(f'{m}_correct'))
        neither = sum(1 for r in rows
                      if not r.get('klareco_correct') and not r.get(f'{m}_correct'))
        print('\nHead-to-head (Klareco vs', args.model[0] + '):')
        print(f'  both correct:        {both}')
        print(f'  klareco only:        {k_only}')
        print(f'  {args.model[0]} only: {l_only}')
        print(f'  neither:             {neither}')

    # Output files
    if args.output_jsonl:
        with open(args.output_jsonl, 'w') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        print(f'\nPer-question JSONL: {args.output_jsonl}')
    if args.output_md:
        write_markdown_report(args.output_md, rows, systems, args)
        print(f'Markdown report:    {args.output_md}')


def write_markdown_report(path: str, rows: list, systems: list, args) -> None:
    """Compact markdown summary."""
    lines = [f'# Klareco vs LLM bench — {len(rows)} questions',
             '', '## Aggregate', '',
             '| System | Correct | Accuracy | Avg latency (s) | Hallucination |',
             '|---|---:|---:|---:|---:|']
    for name, prefix in systems:
        n_correct = sum(1 for r in rows if r.get(f'{prefix}_correct'))
        n_hall = sum(1 for r in rows if r.get(f'{prefix}_hallucinated'))
        lats = [r.get(f'{prefix}_latency_s', 0) for r in rows
                if r.get(f'{prefix}_latency_s')]
        avg = statistics.mean(lats) if lats else 0
        n = max(1, len(rows))
        lines.append(f'| {name} | {n_correct}/{n} | {100*n_correct/n:.1f}% | '
                     f'{avg:.2f} | {100*n_hall/n:.1f}% |')
    lines.extend(['', '## Per-question detail', '',
                  '| # | Question | ' +
                  ' | '.join(s[0] for s in systems) + ' |',
                  '|---:|---|' + '|'.join(['---'] * len(systems)) + '|'])
    for i, r in enumerate(rows, 1):
        q = r['question'][:50].replace('|', '\\|')
        cells = [q]
        for _, prefix in systems:
            ok = '✓' if r.get(f'{prefix}_correct') else '·'
            ans = (r.get(f'{prefix}_answer') or '')[:60].replace('|', '\\|')
            cells.append(f'{ok} {ans}')
        lines.append(f'| {i} | ' + ' | '.join(cells) + ' |')
    Path(path).write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
