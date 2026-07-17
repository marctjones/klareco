# `data/test_sets/` — gold-quality evaluation sets only

This directory holds **only gold-quality** sets, tracked in git. Low-quality /
biased / superseded sets were removed on 2026-07-17 (recoverable from git history:
`git show 7e0deb0:data/test_sets/legacy/<file>`).

| file | what it is | how it was built |
|---|---|---|
| `qa_gold_v1.jsonl` | **canonical QA set** — Esperanto question, answer, and the real corpus sentence that answers it (`source_sentence_id`) | OpenTDB trivia → Claude-CLI translate → parser+pureness gates → **answerability check** (a judge confirms a corpus sentence actually answers the question). Circularity-free: questions originate outside the parser's frame. |
| `treebank_sample.jsonl` | **parser** LAS/UAS gold (UD dependency annotations) — a *separate* deliverable, not QA | UD-Prago sample |

## Why the old sets were removed

The previous sets were generated *from the parser's own output* (clause/arc
extraction), so they were biased toward the AST/structural strategy — the questions
were answerable-by-construction via the same structures a reranker reads (F13
circularity). They were also too small (17–63) to resolve the reranker deltas that
matter (~185 pairs needed for a 0.03 MRR delta; see `bench_history.jsonl`).

## Growing the canonical set

The gold QA set is built by, in order:
1. `scripts/eval/build_trivia_from_opentdb.py` — fetch English trivia + Claude-translate to Esperanto.
2. `scripts/eval/build_trivia_bank.py` — parser + corpus-coverage gate (accumulates to `data/staging/trivia_bank.jsonl`).
3. `scripts/eval/finalize_trivia_gold.py` — the answerability check → writes `qa_gold_v1.jsonl`.

True yield is ~19% of fetched trivia, so reaching N gold pairs means fetching ~5N.
