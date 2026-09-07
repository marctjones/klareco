# Changelog

## Unreleased — parser stabilization (2026-09-07)

- Consolidated parser planning under GitHub milestone #35, with separate issues
  for candidate coverage (#914), clause ownership (#922), lexical and morphology
  resources (#919, #923–#925), and global constrained decoding (#926).
- Added bounded structural candidate generation and relation alternatives while
  preserving the selected runtime dependency tree.
- Added oracle decomposition reports that separate candidate-generation gaps
  from selection errors and record theoretical candidate ceilings.
- Added nominative `nsubj` candidates; Prago exact candidate-edge recall rose to
  76.45% and its measured candidate ceiling to 81.21% LAS.
- Added offline selector and decoder research spikes. Local proximity selectors
  were rejected after LAS regressions. The Apertium-backed global decoder is the
  first positive spike: +2 Prago LAS edges with no Cairo regression. It remains
  offline until broader clause and coordination constraints are validated.
- Updated parser handoff documentation in `README.md`, `AGENTS.md`, `CLAUDE.md`,
  `DESIGN.md`, and `docs/PARSER_RESOURCE_PLAN.md`.

Research spikes are recorded in `data/perf/parser_research/` and
`data/perf/bench_history.jsonl`. They do not imply a runtime parser change unless
the frozen LAS merge gate is satisfied.
