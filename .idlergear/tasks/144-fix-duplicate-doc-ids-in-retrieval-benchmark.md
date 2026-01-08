---
id: 144
title: Fix duplicate doc IDs in retrieval benchmark ground truth
state: open
created: '2026-01-08T15:28:35.804068Z'
labels:
- bug
- benchmark
priority: low
---
## Problem
Some expected_doc_ids in `data/benchmarks/retrieval_benchmark_v1.json` have duplicate entries.

## Examples
- t1_001: has 422114 listed three times
- t1_004: has 4037 listed twice

## Fix
Deduplicate the expected_doc_ids arrays while preserving order.

## File
`data/benchmarks/retrieval_benchmark_v1.json`
