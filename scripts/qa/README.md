# `scripts/qa/` — building the gold Q&A set

The automated, LLM-judged pipeline that produces `data/test_sets/qa_gold_v*.jsonl`.
(Human adjudication was removed 2026-07-17; see EPIC #840.)

## Pipeline (run in order)

| stage | script | what it does |
|---|---|---|
| **source** | `qa_source_opentdb.py` | fetch English trivia from OpenTriviaDB, Claude-CLI translate Q+A to Esperanto → candidate JSONL |
| **gate** | `qa_gate.py` | deterministic gates: parser re-parse (grammar/pureness) + corpus coverage → accumulates `data/staging/trivia_bank.jsonl` |
| **answerability** | `qa_answerability.py` | retrieve corpus candidates + Claude-judge which sentence actually answers the question → attach `source_sentence_id` → `qa_gold_v*.jsonl` |

```bash
python scripts/qa/qa_source_opentdb.py --amount 200 --out data/staging/opentdb_eo.jsonl
python scripts/qa/qa_gate.py           --input data/staging/opentdb_eo.jsonl
python scripts/qa/qa_answerability.py  --input data/staging/trivia_bank.jsonl --out data/test_sets/qa_gold_v1.jsonl
```

## What "gold" means (all automated)
grammatical + pure Esperanto (parser) · correct answer (Claude judge) · a real corpus
sentence answers it (`source_sentence_id`) · not a duplicate · **not parser-circular**.

## Shared library — `klareco/eval/`
- `qa_judge.py` — the Claude-CLI judge (filter only, source-grounded)
- `bootstrap.py` — paired bootstrap CIs (a delta is real only if the CI excludes 0)

## Evaluators (live in `scripts/eval/`, run *against* the gold set)
`retrieval_bottleneck_diagnostic.py` (recall) · `multi_reranker_bench.py` (reranker A/B + CIs) ·
`ab_recall.py` (tokenization recall A/B) · `evaluate_extractive_qa.py` (end-to-end).

## Roadmap
Engine B (corpus→LLM question), schema/assemble, and the 300/500/1000 + reranker-stratum
targets are tracked in EPIC #840 (milestones #20–#23).
