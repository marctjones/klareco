---
id: 29
title: 'Research: Evaluate retrieval semantics for different sentence types'
state: open
created: '2026-01-05T00:40:31.599432Z'
labels:
- research
- retrieval
priority: low
---
## Background

Parser detects sentence types via `fraztipo` field:
- `deklaro`: Statement/declaration
- `demando`: Question (✅ already using for Bug #2 fix)
- `ordono`: Command/imperative
- `kondiĉa`: Conditional (⚠️ detection broken - see #22)

Currently only using `demando` for question-aware partial bonus (0.8 vs 0.5).

## Research Questions

**What does "retrieval" mean for each sentence type?**

1. **Questions**: Retrieve answers to fill missing information ✅ (implemented)
2. **Statements**: Retrieve similar/related info, supporting evidence (current default)
3. **Commands**: Skip? (commands are for action, not information)
4. **Conditionals**: Retrieve hypothetical scenarios? (need detection fix first)

## Potential Improvements

- **Statements**: Could differentiate factual vs opinion statements
- **Commands**: Filter out from retrieval corpus? Or treat as statements?
- **Conditionals**: Special handling for "if X then Y" reasoning

## Next Steps

1. Fix conditional detection (#22)
2. Analyze corpus distribution of sentence types
3. Evaluate whether type-specific retrieval strategies improve results
4. Benchmark on diverse query types

## Related

- Depends on #22 (fix conditional detection)
- Builds on Bug #2 fix (question-aware bonus)
