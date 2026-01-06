---
id: 63
title: Improve embedding semantic quality for AST-aware retrieval
state: open
created: '2026-01-05T21:53:05.050065Z'
labels:
- enhancement
- training
- 'priority: high'
---
**Problem**: AST-aware retriever achieved only 12% accuracy (6/50) on Q&A benchmark, far below 60% target.

**Root Cause**: The pre-filter (HNSW) is returning irrelevant candidates because root embeddings lack semantic signal. AST pattern matching works correctly but can't find answers when pre-filter fails.

**Example Failure**:
- Query: "Kiam aperis la Fundamento de Esperanto?" (When did Fundamento appear?)
- Expected: "1905"
- Pre-filter top result: "Li proponis al la koĉero dividi kun li sandviĉon..." (He proposed to coachman to share sandwich...) - IRRELEVANT
- AST matching finds grammatically similar but semantically wrong documents

**Solution Options**:
1. **Train better root embeddings** with semantic similarity objectives (not just morphological)
2. **Add semantic expansion** in pre-filter using ReVo relations
3. **Combine multiple signals**: embeddings + keyword matching + entity matching
4. **Use larger prefilter_n**: Increase from 500 to 5000 candidates for AST matching

**Next Steps**:
1. Analyze embedding quality: Are "Fundamento" and "Esperanto" close to relevant concepts?
2. Test with larger prefilter_n to see if answer is in top 5000
3. Consider training embeddings with Q&A pairs as supervision
4. Implement keyword-based re-ranking as fallback

**Related**: Task #62 (AST-aware retriever integration - complete but low accuracy)
