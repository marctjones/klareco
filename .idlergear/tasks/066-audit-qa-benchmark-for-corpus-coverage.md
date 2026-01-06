---
id: 66
title: Audit Q&A benchmark for corpus coverage
state: open
created: '2026-01-05T22:19:16.416684Z'
labels:
- research
- 'priority: high'
- documentation
priority: high
---
**Goal**: Determine which benchmark questions are actually answerable with the current 4.3M document corpus.

**Problem**:
- AST retriever achieved 12% accuracy (6/50)
- Some questions may be unanswerable due to missing information in corpus
- Example discovered: "When did Fundamento appear?" expects "1905"
  - NO document in corpus mentions both "Fundamento" AND "1905"
  - This question is impossible to answer correctly

**Audit Process**:

For each of 50 benchmark questions:

1. **Extract key entities/facts from expected answer**
2. **Search corpus for documents containing those facts**
3. **Classify question as**:
   - ✅ **Answerable**: Document(s) contain the answer
   - ⚠️ **Partial**: Documents contain related info but not exact answer
   - ❌ **Unanswerable**: No relevant documents exist

4. **Document results in spreadsheet/JSON**

**Implementation**:
```bash
# Create audit script
python scripts/audit_qa_coverage.py \
  --benchmark data/benchmarks/datasets/qa_benchmark_v1.jsonl \
  --corpus-index data/indexes/slot_full \
  --output data/benchmarks/coverage_audit.json
```

**Script should**:
- For each question, extract keywords from acceptable_answers
- Search corpus for documents containing those keywords
- Check if any document actually answers the question
- Generate coverage report

**Expected Output**:
```json
{
  "total_questions": 50,
  "answerable": 25,
  "partial": 10,
  "unanswerable": 15,
  "questions": [
    {
      "id": "q002",
      "question": "Kiam aperis la Fundamento de Esperanto?",
      "expected": "1905",
      "status": "unanswerable",
      "reason": "No document mentions both 'Fundamento' and '1905'",
      "related_docs": []
    },
    // ...
  ]
}
```

**Use Audit Results To**:
1. Create "answerable subset" benchmark for fair evaluation
2. Identify corpus gaps to fill with targeted data acquisition
3. Adjust accuracy targets based on theoretical maximum

**Success Criteria**:
- Complete audit of all 50 questions
- Categorize each as answerable/partial/unanswerable
- Calculate realistic accuracy ceiling (e.g., if only 30/50 answerable, max = 60%)

**Related**: Task #63 (parent - improve AST retrieval accuracy)
