"""
The canonical gold Q&A record — one shape, validated at assemble time. (#842)

VERSION: v1.0
STAGE: Evaluation / test-set construction

Every gold pair, from either engine (OpenTDB or corpus), is this shape. A record is
GOLD only if it validates: no malformed/partial rows enter data/test_sets/qa_gold_v*.jsonl.

Fields
------
required:
  id                   stable unique id, e.g. "gold-<sid>" or "gold-<engine>-<n>"
  question             the Esperanto question (ends with '?')
  question_type        KIU | KIUN | KIO | KION | KIE | KIAM | KIOM | KIAL | KIES
  expected_answer      the answer string (verbatim in the source sentence)
  source_sentence_id   provenance into the DuckDB `sentences` table
  source_sentence_text the sentence that answers the question
  source               engine: "opentdb" | "corpus"
optional (added downstream):
  gap_shape            subject | object | oblique | definitional
  expected_keywords    [answer, aliases...]  (for substring scoring)
  difficulty_band      trivial | rerankable | deep | miss   (from BM25 gold rank)
  bm25_gold_rank       int or None
  category             topic tag
  verified             {grammar, pureness, answerability, judge}: bool
  en_question          original English (opentdb only)
  created              YYYY-MM-DD (stamp at assemble time; not here — no clock in libs)
"""

from __future__ import annotations

from typing import Tuple, List, Dict

REQUIRED = ('id', 'question', 'question_type', 'expected_answer',
            'source_sentence_id', 'source_sentence_text', 'source')

_QTYPES = {'KIU', 'KIUN', 'KIO', 'KION', 'KIE', 'KIAM', 'KIOM', 'KIAL', 'KIES'}
_GAPS = {'subject', 'object', 'oblique', 'definitional'}
_BANDS = {'trivial', 'rerankable', 'deep', 'miss'}
_SOURCES = {'opentdb', 'corpus'}


def validate(row: Dict) -> Tuple[bool, List[str]]:
    """Return (ok, errors). ok iff no errors."""
    errs: List[str] = []
    for f in REQUIRED:
        v = row.get(f)
        if v is None or (isinstance(v, str) and not v.strip()):
            errs.append(f'missing/blank required field: {f}')

    q = row.get('question') or ''
    if q and not q.strip().endswith('?'):
        errs.append('question does not end with "?"')

    qt = row.get('question_type')
    if qt is not None and qt not in _QTYPES:
        errs.append(f'question_type not in {sorted(_QTYPES)}: {qt!r}')

    if row.get('source') is not None and row['source'] not in _SOURCES:
        errs.append(f'source not in {sorted(_SOURCES)}: {row["source"]!r}')

    # NOTE: answer-verbatim-in-source is NOT a hard reject. A question the judge
    # confirmed is answerable stays valid even if the answer is phrased differently
    # in the sentence (spelling/translation variant, paraphrase). It is recorded as
    # the `answer_verbatim` flag (see answer_verbatim()) so extraction scoring can
    # treat the two cases appropriately; retrieval/reranking use every question.
    ans = (row.get('expected_answer') or '').lower()

    # the answer must not sit inside the QUESTION (would give it away) — hard reject.
    if ans and ans in (row.get('question') or '').lower():
        errs.append('expected_answer appears in the question')

    gap = row.get('gap_shape')
    if gap is not None and gap not in _GAPS:
        errs.append(f'gap_shape not in {sorted(_GAPS)}: {gap!r}')

    band = row.get('difficulty_band')
    if band is not None and band not in _BANDS:
        errs.append(f'difficulty_band not in {sorted(_BANDS)}: {band!r}')

    return (not errs), errs


def answer_verbatim(row) -> bool:
    """Does the expected answer appear literally in the source sentence? Extraction
    scoring can require this; retrieval/reranking do not (they only need the gold sid)."""
    ans = (row.get('expected_answer') or '').lower()
    src = (row.get('source_sentence_text') or '').lower()
    return bool(ans) and ans in src


def band_for(rank) -> str:
    """BM25 gold rank -> difficulty band."""
    if rank is None:
        return 'miss'
    if rank == 1:
        return 'trivial'
    if rank <= 50:
        return 'rerankable'
    return 'deep'


def _selftest():
    good = {'id': 'gold-1', 'question': 'Kiu verkis la libron?', 'question_type': 'KIU',
            'expected_answer': 'Zamenhof', 'source_sentence_id': '1',
            'source_sentence_text': 'Zamenhof verkis la libron.', 'source': 'corpus'}
    ok, e = validate(good); assert ok, e
    ok, e = validate({**good, 'question': 'Kiu verkis la libron'}); assert not ok  # no ?
    # non-verbatim answer is VALID (just flagged), not a reject:
    ok, e = validate({**good, 'expected_answer': 'Kabe'}); assert ok, e
    assert answer_verbatim(good) and not answer_verbatim({**good, 'expected_answer': 'Kabe'})
    ok, e = validate({**good, 'question': 'Kiu estas Zamenhof?'}); assert not ok   # answer in q
    ok, e = validate({**good, 'question_type': 'WHO'}); assert not ok              # bad type
    assert band_for(None) == 'miss' and band_for(1) == 'trivial' and band_for(10) == 'rerankable'
    print('  ✓ qa_schema self-test passed')


if __name__ == '__main__':
    _selftest()
