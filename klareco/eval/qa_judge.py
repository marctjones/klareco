"""
Claude-CLI judge for QA pairs — the ONE check determinism can't make. (#736)

VERSION: v1.0
STAGE: Evaluation / test-set construction
DEPENDENCIES: the `claude` CLI on PATH (headless `-p` mode)

Description:
    Grammar, pureness, and corpus-presence are judged deterministically (the parser
    and SQL). The residue is SEMANTIC: given the SOURCE sentence, is the question's
    answer correct and unambiguous, and is the Esperanto natural? That is what an
    LLM judge is for — and ONLY that. The judge is a FILTER (reject bad items),
    never a generator (per the test-set-validity rule), and it is grounded in the
    source sentence so it judges the PAIR, not its own trivia knowledge.

    Batched: many pairs per `claude -p` call, JSON array back, aligned by id.

Usage:
    from klareco.eval.qa_judge import judge_batch
    verdicts = judge_batch([{ 'id':..., 'question':..., 'expected_answer':...,
                              'source_sentence_text':... }, ...])
    # each verdict: {id, grammatical, natural, answer_correct, keep, reason}

Self-test:
    python -m klareco.eval.qa_judge     # judges 4 hand pairs (2 good, 2 bad)

Last Updated: 2026-07-17
Related Issues: #736, #737
"""

from __future__ import annotations

import json
import subprocess
from typing import List, Dict, Optional

_PROMPT_HEAD = """You are a strict judge building an Esperanto question-answering TEST SET.
For EACH item you are given a SOURCE sentence (from the corpus), a QUESTION, and a
proposed ANSWER. Judge ONLY against the source sentence — do NOT use outside knowledge.

For each item decide:
  grammatical    : is the QUESTION well-formed, grammatical Esperanto?
  natural        : is it a question a person might actually ask (not robotic/garbled)?
  answer_correct : does the SOURCE sentence support this exact answer to this question,
                   unambiguously? (reject if negation/meaning was dropped, the answer
                   is only a fragment of the real entity, or the question is ambiguous)
  keep           : true ONLY if grammatical AND natural AND answer_correct.

Reply with ONLY a JSON array, one object per item, no prose:
[{"id":"...","grammatical":true,"natural":true,"answer_correct":true,"keep":true,"reason":"..."}]

ITEMS:
"""


def _build_prompt(pairs: List[Dict]) -> str:
    lines = [_PROMPT_HEAD]
    for p in pairs:
        lines.append(json.dumps({
            'id': str(p.get('id')),
            'source': p.get('source_sentence_text', ''),
            'question': p.get('question', ''),
            'answer': p.get('expected_answer', ''),
        }, ensure_ascii=False))
    return '\n'.join(lines)


def _extract_json_array(text: str):
    """Claude may wrap the array in prose or fences; pull the outermost [...]."""
    i, j = text.find('['), text.rfind(']')
    if i == -1 or j == -1 or j < i:
        return None
    try:
        return json.loads(text[i:j + 1])
    except Exception:
        return None


def judge_batch(pairs: List[Dict], model: Optional[str] = None,
                timeout: int = 240, retries: int = 1) -> List[Dict]:
    """Judge a batch of QA pairs via the claude CLI. Returns one verdict per input
    id (order preserved). On unrecoverable failure a pair defaults to keep=False so
    the judge can only ever SHRINK the set toward validity."""
    if not pairs:
        return []
    prompt = _build_prompt(pairs)
    cmd = ['claude', '-p']
    if model:
        cmd += ['--model', model]

    out = None
    for attempt in range(retries + 1):
        try:
            r = subprocess.run(cmd, input=prompt, text=True,
                               capture_output=True, timeout=timeout)
            out = _extract_json_array(r.stdout)
            if out is not None:
                break
        except subprocess.TimeoutExpired:
            out = None

    by_id = {}
    if out:
        for v in out:
            if isinstance(v, dict) and 'id' in v:
                by_id[str(v['id'])] = v

    verdicts = []
    for p in pairs:
        pid = str(p.get('id'))
        v = by_id.get(pid)
        if v is None:
            verdicts.append({'id': pid, 'keep': False,
                             'reason': 'judge returned no verdict (default reject)'})
        else:
            v['id'] = pid
            v['keep'] = bool(v.get('keep')) and bool(v.get('answer_correct')) \
                and bool(v.get('grammatical')) and bool(v.get('natural'))
            verdicts.append(v)
    return verdicts


def _selftest():
    pairs = [
        {'id': 'good1', 'source_sentence_text': 'Zamenhof verkis la libron de Petro en 1905.',
         'question': 'Kiu verkis la libron de Petro?', 'expected_answer': 'Zamenhof'},
        {'id': 'good2', 'source_sentence_text': 'La Manifesto de Prago en 1996 reasertis novajn celojn.',
         'question': 'Kio reasertis novajn celojn en 1996?', 'expected_answer': 'La Manifesto de Prago'},
        {'id': 'bad_negation', 'source_sentence_text': 'Proklo ne havigas ajnan fonton por siaj indikoj.',
         'question': 'Kiu havigas fonton por siaj indikoj?', 'expected_answer': 'Proklo'},
        {'id': 'bad_fragment', 'source_sentence_text': 'Esperanta PEN-Centro helpas videbligi la heredaĵon.',
         'question': 'Kiu helpas videbligi la heredaĵon?', 'expected_answer': 'Esperanta'},
    ]
    for v in judge_batch(pairs):
        print(f"  {v['id']:14s} keep={str(v.get('keep')):5s}  {v.get('reason','')[:90]}")


if __name__ == '__main__':
    _selftest()
