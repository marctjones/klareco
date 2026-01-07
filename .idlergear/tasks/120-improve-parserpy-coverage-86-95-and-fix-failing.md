---
id: 120
title: "Improve parser.py coverage (86% \u2192 95%) and fix failing tests"
state: open
created: '2026-01-07T00:12:07.265142Z'
labels:
- testing
- bug
- parser
priority: high
---
## Goal

Bring parser coverage to 95% and fix currently failing parser tests.

## Current State

- Coverage: 86% (63 lines missing)
- 42 failing tests related to prefix/suffix protection

## Failing Tests

From test output:
- `test_bo_prefix_protection` - 5 failures (boa, boben, boj, bol, boraks)
- `test_dis_prefix_protection` - 2 failures (distil, distr)
- `test_ek_prefix_protection` - 2 failures (ekscit, ekspozici)
- `test_fi_prefix_protection` - 4 failures (fibr, fig, filtr, firm)
- `test_re_prefix_protection` - 7 failures (reĝ, rel, rem, ren, renkont, ret, rev)
- `test_ar_suffix_protection` - 1 failure (bazar)
- `test_er_suffix_protection` - 9 failures
- `test_et_suffix_protection` - 3 failures
- `test_il_suffix_protection` - 5 failures
- `test_ul_suffix_protection` - 1 failure (tabul)
- `test_dangero_not_er_suffix` - 1 failure

## Root Cause

Parser is incorrectly splitting words that look like they have prefixes/suffixes but are actually roots:
- "danĝero" → parsed as "danĝ" + "-er-" suffix (wrong!)
- "boraks" → parsed as "bo-" + "raks" (wrong!)

## Fix Needed

Add these roots to protected vocabulary or improve morpheme boundary detection.

## Acceptance Criteria

- [ ] All 42 failing tests pass
- [ ] Coverage: 86% → 95%
- [ ] No regressions in other tests

## Estimated Effort

~3-4 hours
