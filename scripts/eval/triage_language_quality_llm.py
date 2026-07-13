#!/usr/bin/env python3
"""
LLM grammaticality triage — rank the human review queue worst-first.

VERSION: v1.0
COMPATIBLE WITH: any Q&A test-set JSONL carrying `question` (+ optional
                 `gold_answer_span` / `expected_answer`, `source_sentence_text`)
DEPENDENCIES: stdlib (urllib) for the Ollama backend; `anthropic` SDK for the
              Claude backend. No runtime dependency on klareco.
STAGE: Eval

Description:
    Asks a language model to judge whether each question is GRAMMATICAL,
    NATURAL Esperanto, and writes its verdict back into the pair as a frozen
    field. The output is a review queue ordered WORST-FIRST, so a human's hour
    buys the most information.

    THIS IS A CONSTRUCTION-TIME TOOL. It is never called at scoring time.
    See docs/QA_TEST_SET_PIPELINE.md, "The bright line":

        ML may BUILD a test set. ML may never SCORE one.

    A scoring-time judge would make the merge gate (#784) unfalsifiable: a
    moved number in bench_history could mean the system improved, or that the
    judge drifted. Here, by contrast, the model's opinion is frozen into an
    artifact, audited by a human, and committed to git — after which scoring
    is deterministic string comparison forever.

    THE MODEL DOES NOT DECIDE. It scores and explains; the human adjudicates
    (#798). Its only job is to order the queue.

Why this exists:
    The hand-written deterministic auditor passed five broken questions 8/8 —
    including `Kiu venkis Rorke's Driftn?` — until #791 tightened it. It will
    have another hole: an auditor can only catch failure modes we have already
    thought of. A model that reads Esperanto flags them WITHOUT our having to
    enumerate them first. That is the entire value proposition, and it is why
    this does not replace the deterministic auditor — it complements it.

Pipeline Position:
    generator -> deterministic audit (R1-R17) -> [THIS] -> human review (#798)

Usage:
    # Claude (recommended for judgment — this is where a weak model fails quietly)
    export ANTHROPIC_API_KEY=...
    python scripts/eval/triage_language_quality_llm.py \
        --in  data/test_sets/gold_trivia_review_queue_v1.jsonl \
        --out data/test_sets/gold_trivia_review_queue_v1.triaged.jsonl

    # Local model via Ollama (cheap first pass; NOT the last word on grammar)
    python scripts/eval/triage_language_quality_llm.py --backend ollama \
        --model qwen3:latest --in <set>.jsonl --out <set>.triaged.jsonl

    # See the prompts without spending anything / without credentials
    python scripts/eval/triage_language_quality_llm.py --in <set>.jsonl --dry-run

Inputs:
    - JSONL test set. Each row needs `question`; `source_sentence_text` and
      `gold_answer_span` are used for context when present.

Outputs:
    - JSONL, sorted WORST-FIRST, each row gaining:
        llm_grammatical : bool   — is the question well-formed Esperanto?
        llm_score       : float  — 0.0 (broken) .. 1.0 (native-quality)
        llm_reason      : str    — WHY, in one sentence
        llm_issues      : list   — failure tags (e.g. foreign_inflection)
        llm_model       : str    — provenance: exactly which model judged it
        llm_backend     : str
        llm_triage_version : str — bump when the prompt changes
    - A summary to stdout, including the auditor-vs-LLM disagreement count,
      which is the health metric of our own deterministic gates.

Quality Checks:
    - Provenance is mandatory on every touched row (attribution must stay
      decomposable — the discipline the thesis demands of the pipeline applies
      to the instrument that measures it).
    - Deterministic invocation: no temperature/top_p (rejected by Opus 4.8
      anyway) and a fixed structured-output schema.
    - --dry-run works with no credentials, so the prompt is reviewable.

Last Updated: 2026-07-13
Author: klareco
Related Issues: #795, #792, #798, #791
See Also: docs/QA_TEST_SET_PIPELINE.md, docs/QA_TEST_SET_QUALITY_STANDARD.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# Bump when the prompt or schema changes — a verdict is only comparable to
# another verdict produced by the same triage version.
TRIAGE_VERSION = 'v1.0'

OLLAMA_URL = 'http://localhost:11434/api/chat'
DEFAULT_CLAUDE_MODEL = 'claude-opus-4-8'
DEFAULT_OLLAMA_MODEL = 'qwen3:latest'

# Backend defaults. `claude-cli` is the default because it needs NO credentials
# — it authenticates with the user's Claude subscription via `claude -p`.
DEFAULT_BACKEND = 'claude-cli'


# ---------------------------------------------------------------------------
# The prompt.
#
# It names the failure modes we have actually paid for, but explicitly invites
# the model to flag anything else — because the whole point of using a model
# here is to catch what our enumerated rules do not.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a meticulous Esperanto linguist reviewing questions for a benchmark \
test set. Esperanto is a fully regular language: 16 rules, no exceptions.

Judge ONLY whether the QUESTION is grammatical, natural Esperanto. Do NOT judge \
whether the answer is correct, whether the fact is interesting, or whether the \
question is answerable — other checks handle those.

Known failure modes in this generator (flag these, but do NOT limit yourself to \
them — anything ungrammatical or unidiomatic counts):

- foreign_inflection: the accusative -n glued onto an unassimilated foreign name
  or title. `Kiu venkis Rorke's Driftn?` and `Kiu reĝisoris Théâtre des
  Variétésn?` are broken. A foreign title should be quoted and left uninflected,
  with an Esperanto head noun carrying the case:
  `Kiu verkis la vortaron "Altdeutsches Wörterbuch"?` is correct.
- double_accusative: -n marked twice where once is right, e.g. an accusative
  noun inside a `de`-phrase (`Ĉambron de Arton`) — a de-phrase is genitive.
- foreign_fragment: a bare English/French/German clause dropped into the
  question unquoted (`Kiu gajnis The World Is Not Overn?`).
- calque: word-for-word translation from English that no Esperanto speaker would
  say (`Kio estas la fakto pri X?`).
- bad_correlative: malformed or misplaced ki-word; more than one interrogative.
- agreement: wrong case, number, or tense agreement.
- garbled: word salad, truncation, duplicated words.

Score 1.0 only if a fluent Esperantist would accept the sentence as written, \
without editing. Anything you would correct scores below 0.7. Be strict: this \
question will become a permanent benchmark item.

Answer with JSON only."""

USER_TEMPLATE = """\
Question (Esperanto):
{question}
{context}
Is this grammatical, natural Esperanto?"""

RESPONSE_SCHEMA: dict[str, Any] = {
    'type': 'object',
    'properties': {
        'grammatical': {
            'type': 'boolean',
            'description': 'True only if a fluent Esperantist would accept it as written.',
        },
        'score': {
            'type': 'number',
            'description': '0.0 = broken, 1.0 = native quality.',
        },
        'issues': {
            'type': 'array',
            'items': {'type': 'string'},
            'description': 'Failure tags. Empty if none. Use the listed tags where they fit; invent a snake_case tag if the defect is not covered.',
        },
        'reason': {
            'type': 'string',
            'description': 'One sentence explaining the verdict, naming the exact offending token where applicable.',
        },
    },
    'required': ['grammatical', 'score', 'issues', 'reason'],
    'additionalProperties': False,
}


def answer_span(row: dict) -> str:
    """The R17 gold span — the exact answer, NOT the sentence containing it.

    Field-name archaeology: `build_gold_review_queue.py` calls it `short_answer`
    and uses `expected_answer` for the WHOLE SOURCE SENTENCE. Reading
    `expected_answer` as the answer would hand the model (and later the
    extractor) a target that is a whole paragraph — exactly the conflation #783
    exists to eliminate. Prefer, in order: gold_answer_span, short_answer,
    expected_answer.
    """
    for key in ('gold_answer_span', 'short_answer', 'expected_answer'):
        v = (row.get(key) or '').strip()
        if v:
            return v
    return ''


def build_user_prompt(row: dict) -> str:
    ctx = ''
    src = (row.get('source_sentence_text') or '').strip()
    ans = answer_span(row)
    if ans:
        ctx += f'\nIntended answer: {ans}'
    if src:
        ctx += f'\nSource sentence the question was built from:\n{src[:400]}'
    if ctx:
        ctx += '\n'
    return USER_TEMPLATE.format(question=(row.get('question') or '').strip(),
                                context=ctx)


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

def judge_claude(row: dict, model: str) -> dict:
    """Claude backend. Structured outputs guarantee a parseable verdict.

    No temperature/top_p — Opus 4.7+ rejects sampling parameters outright, and
    we want determinism anyway.
    """
    try:
        import anthropic
    except ImportError:
        raise SystemExit(
            "The `anthropic` package is not installed.\n"
            "  pip install anthropic\n"
            "…or use the local backend:  --backend ollama")

    client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY / ant profile
    resp = client.messages.create(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        thinking={'type': 'adaptive'},
        output_config={
            'effort': 'medium',
            'format': {'type': 'json_schema', 'schema': RESPONSE_SCHEMA},
        },
        messages=[{'role': 'user', 'content': build_user_prompt(row)}],
    )
    if resp.stop_reason == 'refusal':
        return {'grammatical': False, 'score': 0.0, 'issues': ['model_refusal'],
                'reason': 'model declined to judge this item'}
    text = next(b.text for b in resp.content if b.type == 'text')
    return json.loads(text)


def _extract_json(text: str) -> dict:
    """Pull a JSON object out of a model's stdout.

    The CLI backend has no structured-output guarantee, so the model may wrap
    its answer in a markdown fence or add a sentence of prose. Be liberal in
    what we accept — but still parse strictly, so a malformed verdict surfaces
    as an error rather than as a silent PASS.
    """
    t = text.strip()
    if '```' in t:                       # strip a markdown fence
        parts = t.split('```')
        for p in parts:
            p = p.strip()
            if p.startswith('json'):
                p = p[4:].strip()
            if p.startswith('{'):
                t = p
                break
    start, end = t.find('{'), t.rfind('}')
    if start == -1 or end == -1:
        raise ValueError(f'no JSON object in model output: {text[:160]!r}')
    return json.loads(t[start:end + 1])


def _parse_line_protocol(text: str) -> dict:
    """Parse the line-based verdict format.

    We do NOT ask the CLI backend for JSON. It has no structured-output
    guarantee, and the `reason` field is free text in which the model naturally
    quotes the offending token — `the name "Rorke's Drift" ...` — whose inner
    double quote corrupts the JSON string. Five of eight pairs failed that way,
    and they were the interesting ones: the model had judged them CORRECTLY and
    the verdict was lost to a formatting accident.

    A line protocol cannot break on quotes. Be liberal in what you accept.
    """
    out: dict[str, Any] = {}
    for line in text.splitlines():
        line = line.strip()
        if ':' not in line:
            continue
        key, _, val = line.partition(':')
        key = key.strip().lower().lstrip('-* ').strip()
        val = val.strip()
        if key == 'grammatical':
            out['grammatical'] = val.lower().startswith(('yes', 'true'))
        elif key == 'score':
            try:
                out['score'] = float(val.split()[0])
            except (ValueError, IndexError):
                pass
        elif key == 'issues':
            v = val.strip('[]').strip()
            out['issues'] = ([] if v.lower() in ('', 'none', '-')
                             else [t.strip().strip('"\'') for t in v.split(',') if t.strip()])
        elif key == 'reason':
            out['reason'] = val
    if 'grammatical' not in out:
        raise ValueError(f'no GRAMMATICAL line in model output: {text[:160]!r}')
    return out


CLI_FORMAT_INSTRUCTIONS = """\
Answer in EXACTLY this line format and nothing else — no JSON, no markdown, no \
preamble:

GRAMMATICAL: yes|no
SCORE: <0.0-1.0>
ISSUES: <comma-separated snake_case tags, or none>
REASON: <one sentence; quote offending tokens with 'single quotes'>"""


def judge_claude_cli(row: dict, model: str, timeout: int = 180) -> dict:
    """Claude via the `claude` CLI in headless mode — uses the SUBSCRIPTION.

    No ANTHROPIC_API_KEY required: `claude -p` authenticates with the user's
    Claude subscription. This is the cheapest way to get Claude-quality
    judgment, which matters because Esperanto grammaticality is exactly where a
    weaker model fails *quietly*.

    Trade-offs, stated honestly:
      - One process per pair, so it is slow (seconds each). Fine at the scale
        that matters here — the gold queue is 63 rows, not 63,000.
      - No structured-output guarantee (hence _extract_json) and no temperature
        control, so it is not bit-for-bit reproducible.

    That last point is acceptable ONLY because this is construction-time
    tooling: the verdict is frozen into an artifact, audited by a human (#798),
    and committed. Scoring never calls a model. If this were a scoring-time
    judge, non-determinism would be disqualifying — see the epic (#792).

    Runs in a temp cwd so the CLI does not load this repo's CLAUDE.md as
    context: we want the model judging Esperanto, not reading our conventions.
    """
    import subprocess
    import tempfile

    prompt = (SYSTEM_PROMPT + '\n\n' + build_user_prompt(row)
              + '\n\n' + CLI_FORMAT_INSTRUCTIONS)
    with tempfile.TemporaryDirectory() as tmp:
        proc = subprocess.run(
            ['claude', '-p', prompt, '--model', model],
            capture_output=True, text=True, timeout=timeout, cwd=tmp,
        )
    if proc.returncode != 0:
        raise RuntimeError(
            f'claude CLI exited {proc.returncode}: {(proc.stderr or "").strip()[:200]}')
    return _parse_line_protocol(proc.stdout)


def judge_ollama(row: dict, model: str, timeout: int = 90) -> dict:
    """Local backend. Cheap first pass — NOT the last word on grammaticality.

    A weak model fails *quietly* on Esperanto, which is the exact failure this
    whole tool exists to prevent. Use it to pre-filter volume, not to decide.
    """
    payload = {
        'model': model,
        'messages': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': build_user_prompt(row)},
        ],
        'stream': False,
        'format': 'json',
        'options': {'temperature': 0.0},
    }
    req = urllib.request.Request(
        OLLAMA_URL, data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode('utf-8'))
    content = (body.get('message') or {}).get('content') or ''
    # Reasoning models emit <think>…</think> before the answer.
    if '</think>' in content:
        content = content.split('</think>', 1)[1]
    return json.loads(content.strip())


def normalize_verdict(raw: dict) -> dict:
    """Coerce a model verdict into the frozen field shape. Never trust it blindly."""
    try:
        score = float(raw.get('score', 0.0))
    except (TypeError, ValueError):
        score = 0.0
    score = max(0.0, min(1.0, score))
    issues = raw.get('issues') or []
    if not isinstance(issues, list):
        issues = [str(issues)]
    return {
        'llm_grammatical': bool(raw.get('grammatical', False)),
        'llm_score': round(score, 3),
        'llm_issues': [str(i) for i in issues],
        'llm_reason': str(raw.get('reason', ''))[:400],
    }


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--in', dest='in_path', required=True)
    ap.add_argument('--out', dest='out_path',
                    help='default: <in>.triaged.jsonl')
    ap.add_argument('--backend',
                    choices=('claude-cli', 'claude-api', 'ollama'),
                    default=DEFAULT_BACKEND,
                    help='claude-cli (default): headless `claude -p`, uses your '
                         'SUBSCRIPTION, no API key needed. claude-api: the '
                         'anthropic SDK, needs ANTHROPIC_API_KEY. ollama: local '
                         'model, cheap bulk pre-filter only.')
    ap.add_argument('--model', default=None,
                    help=f'default: {DEFAULT_CLAUDE_MODEL} / {DEFAULT_OLLAMA_MODEL}')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--dry-run', action='store_true',
                    help='print the first prompt and exit — no credentials needed')
    args = ap.parse_args()

    model = args.model or (DEFAULT_OLLAMA_MODEL if args.backend == 'ollama'
                           else DEFAULT_CLAUDE_MODEL)
    in_path = Path(args.in_path)
    out_path = Path(args.out_path) if args.out_path else in_path.with_suffix('.triaged.jsonl')

    rows = [json.loads(l) for l in in_path.read_text(encoding='utf-8').splitlines() if l.strip()]
    if args.limit:
        rows = rows[:args.limit]

    if args.dry_run:
        print('=== SYSTEM ===')
        print(SYSTEM_PROMPT)
        print('\n=== USER (first row) ===')
        print(build_user_prompt(rows[0]))
        print(f'\n=== would judge {len(rows)} rows via {args.backend}:{model} ===')
        return 0

    judge = {
        'claude-cli': judge_claude_cli,
        'claude-api': judge_claude,
        'ollama':     judge_ollama,
    }[args.backend]

    out: list[dict] = []
    errors = 0
    for i, row in enumerate(rows, 1):
        try:
            verdict = normalize_verdict(judge(row, model))
        except Exception as e:            # noqa: BLE001 — one bad row must not kill the run
            errors += 1
            verdict = {'llm_grammatical': False, 'llm_score': 0.0,
                       'llm_issues': ['triage_error'],
                       'llm_reason': f'{type(e).__name__}: {e}'[:400]}
        # Provenance is MANDATORY: a verdict without a model id is unauditable.
        verdict.update({'llm_model': model, 'llm_backend': args.backend,
                        'llm_triage_version': TRIAGE_VERSION})
        out.append({**row, **verdict})
        if i % 10 == 0 or i == len(rows):
            print(f'  … {i}/{len(rows)}', file=sys.stderr)

    # WORST FIRST — the point of the whole exercise is to spend the human's
    # attention where it buys the most information.
    out.sort(key=lambda r: (r['llm_score'], r.get('id') or ''))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    n = len(out)
    n_bad = sum(1 for r in out if not r['llm_grammatical'])
    print(f'\nTriaged {n} pairs via {args.backend}:{model}')
    print(f'  flagged ungrammatical : {n_bad} ({100 * n_bad / max(n, 1):.1f}%)')
    print(f'  triage errors         : {errors}')

    tags: dict[str, int] = {}
    for r in out:
        for t in r['llm_issues']:
            tags[t] = tags.get(t, 0) + 1
    if tags:
        print('  issue tags:')
        for t, c in sorted(tags.items(), key=lambda kv: -kv[1]):
            print(f'    {t:28s} {c}')

    # The health metric of our OWN gates. The quality standard expects a 5-9
    # point gap between mechanical PASS and human accept. When the deterministic
    # auditor reported 8/8 PASS on a batch containing five broken questions, the
    # gap was ~60 points. A large disagreement here means the deterministic
    # auditor has fallen behind the generator — and the remedy is to ADD A CHECK
    # (R11), never to hand-fix pairs.
    audited = [r for r in out if 'audit_verdict' in r]
    if audited:
        disagree = sum(1 for r in audited
                       if (r['audit_verdict'] == 'PASS') != r['llm_grammatical'])
        print(f'\n  auditor-vs-LLM disagreement: {disagree}/{len(audited)} '
              f'({100 * disagree / len(audited):.1f}%)')
        print('  ^ a large gap means the deterministic auditor is too permissive.')

    print(f'\nWrote {out_path} (worst-first).')
    print('NEXT: human adjudication (#798). The model ORDERS the queue; it does '
          'not decide.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
