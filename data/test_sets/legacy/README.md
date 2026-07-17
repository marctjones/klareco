# Legacy QA sets — ARCHIVED 2026-07-17, do not use as source of truth

These were generated FROM the parser's own output (clause/arc extraction), so they
are biased toward the AST/structural strategy (F13 circularity): the questions are
answerable-by-construction via the same structures a reranker reads. They are also
small (17–63) — far below the ~185/stratum needed to resolve a 0.03 MRR delta.

KEPT, NOT DELETED, because:
  - they remain useful as PARSER-REGRESSION fixtures,
  - `data/perf/bench_history.jsonl` entries reference them (reproducibility).

Canonical replacement: OpenTDB-sourced, Claude-translated, corpus-answerability-
checked trivia (circularity-free). See scripts/eval/build_trivia_from_opentdb.py.

NOT archived (different deliverables): treebank_sample.jsonl (PARSER LAS/UAS),
trivia_bank.jsonl (the new canonical set).
