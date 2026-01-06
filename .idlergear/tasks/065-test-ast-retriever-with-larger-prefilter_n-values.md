---
id: 65
title: Test AST retriever with larger prefilter_n values
state: open
created: '2026-01-05T22:18:57.744354Z'
labels:
- research
- 'priority: medium'
priority: medium
---
**Goal**: Determine optimal `prefilter_n` value to balance accuracy vs latency.

**Context**:
- Current default: `prefilter_n=500` (359ms latency, 12% accuracy)
- Testing showed answers exist deeper in rankings (e.g., rank 1144)
- Larger prefilter_n means more AST parsing (slower) but better recall

**Experiment Design**:

Test different `prefilter_n` values on full Q&A benchmark:

| prefilter_n | Expected Latency | Expected Accuracy |
|-------------|------------------|-------------------|
| 500 (baseline) | 359ms | 12% |
| 1000 | ~600ms | 15-20%? |
| 2000 | ~1.0s | 20-25%? |
| 5000 | ~2.5s | 25-35%? |

**Implementation**:
1. Run benchmark with `--prefilter-n` parameter for each value
2. Measure accuracy and latency trade-offs
3. Plot accuracy vs latency curve
4. Find "knee" of curve (optimal trade-off point)

**Commands**:
```bash
# Baseline (already done)
python scripts/benchmark_qa_enhanced.py --retriever ast --output results/ast_n500.jsonl

# Test larger values
python scripts/benchmark_qa_enhanced.py --retriever ast --prefilter-n 1000 --output results/ast_n1000.jsonl
python scripts/benchmark_qa_enhanced.py --retriever ast --prefilter-n 2000 --output results/ast_n2000.jsonl
python scripts/benchmark_qa_enhanced.py --retriever ast --prefilter-n 5000 --output results/ast_n5000.jsonl

# Analyze results
python scripts/analyze_prefilter_n_experiment.py results/ast_n*.jsonl
```

**Success Criteria**:
- Identify optimal prefilter_n for production use
- Document accuracy/latency trade-off curve
- If n=2000 gives 25%+ accuracy with <2s latency → use as new default

**Blockers**:
- Need to add `--prefilter-n` parameter to benchmark script
- May need to optimize parsing speed if latency too high

**Related**: Task #63 (parent - improve AST retrieval), Task #64 (semantic embeddings - long-term fix)
