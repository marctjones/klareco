# Translation Pipeline: English QA → Esperanto

## Overview

This pipeline translates English QA datasets to Esperanto and filters for corpus coverage, creating diverse test sets that go beyond the narrow Esperanto-domain focus of the current `qa_test_set_50.jsonl`.

## Problem Statement

Current test set (`qa_test_set_50.jsonl`):
- 78% questions about Esperanto/Zamenhof
- "Zamenhof" is answer to 18% of all questions
- Measures domain-specific memorization, not general QA capability
- Would not generalize to other topics

## Solution

Translate diverse English QA datasets (TriviaQA, Natural Questions) to Esperanto:
1. Download diverse English QA dataset
2. Translate questions to Esperanto (GoogleTranslator via deep-translator)
3. Verify translation parses correctly (klareco.parser)
4. Check if answers exist in corpus (Whoosh full-text search)
5. Keep only questions with verifiable answers
6. Expected result: 300-500 diverse Q&A pairs from 1000 English questions

## Quick Start

```bash
# Build dataset with 1000 questions (default)
./scripts/build_translated_qa_dataset.sh

# Build with 5000 questions
./scripts/build_translated_qa_dataset.sh 5000

# Skip download if already exists
./scripts/build_translated_qa_dataset.sh --skip-download
```

Output: `data/test_sets/translated_qa_diverse.jsonl`

## Individual Scripts

### 1. Download TriviaQA Sample

```bash
python scripts/download_triviaqa_sample.py \
    --output data/external/triviaqa_sample_1000.jsonl \
    --limit 1000
```

Downloads TriviaQA dataset and extracts sample questions.

### 2. Translate and Filter

```bash
python scripts/translate_and_filter_qa.py \
    --input data/external/triviaqa_sample_1000.jsonl \
    --output data/test_sets/translated_qa_diverse.jsonl \
    --limit 1000
```

Translates questions, verifies parsing, checks corpus coverage.

## Expected Results

### Translation Success Rate

Based on Esperanto corpus coverage (~5% of English Wikipedia):

| Category | Expected Retention | Reason |
|----------|-------------------|--------|
| Famous people | 60-70% | Major figures covered |
| World capitals | 80%+ | Geography well-covered |
| Historical events | 50-60% | Major events present |
| Scientific terms | 40-50% | Basic science covered |
| Pop culture | 10-20% | Limited coverage |
| **Overall** | **40-50%** | Mixed topics |

### Quality Improvements

From 1000 English questions, expect:
- 400-500 successfully translated and verified
- 200-300 failed translation/parsing
- 300-400 answers not in corpus

**Quality improvements over current test set:**
- Answer diversity: 400+ unique answers (vs 9× "Zamenhof")
- Topic diversity: 50+ topics (vs 78% Esperanto)
- No single answer > 5% of total (vs 18% "Zamenhof")

## Output Format

```jsonl
{
  "question": "Kiu pentris la Mona Lizan?",
  "expected_keywords": ["Leonardo da Vinci"],
  "answer": "Leonardo da Vinci",
  "answer_variants": ["Leonardo da Vinci", "da Vinci", "Leonardo"],
  "source": "translated",
  "original_english": {
    "question": "Who painted the Mona Lisa?",
    "answer": ["Leonardo da Vinci", "da Vinci", "Leonardo"]
  }
}
```

## Pipeline Statistics

The script reports:
- Total questions processed
- Translation success rate
- Parse failure rate
- Corpus coverage rate
- Final dataset size

Example output:
```
============================================================
RESULTS
============================================================
Total processed: 1000
✓ Success: 430 (43.0%)
✗ Translation failed: 20
✗ Parse failed: 50
✗ Not in corpus: 500

Saved 430 questions to data/test_sets/translated_qa_diverse.jsonl
```

## Evaluation

Once the translated dataset is created, evaluate your QA system:

```bash
python scripts/evaluate_extractive_qa.py \
    --test-set data/test_sets/translated_qa_diverse.jsonl
```

This will show if your system can handle diverse topics beyond Esperanto-specific questions.

## Architecture

### Translation
- **Library**: deep-translator (GoogleTranslator)
- **Direction**: English → Esperanto
- **Fallback**: Can swap to LibreTranslate or Apertium if needed

### Validation
- **Parse Check**: Uses `klareco.parser` to ensure question is grammatically valid
- **Corpus Check**: Uses Whoosh full-text search to verify answer exists

### Filtering Strategy
Keep only questions where:
1. Translation succeeds
2. Question parses correctly (has verb, valid grammar)
3. Answer exists in 5.4M sentence corpus

## Why This Approach Works

1. **Diverse topics**: Not limited to Esperanto
2. **Verified answers**: Only keep questions where answers exist in corpus
3. **Automatic quality control**: Parse check + corpus check
4. **Scalable**: Can process thousands of questions
5. **Free**: No API costs (using free translation services)
6. **Reproducible**: Can regenerate with different datasets

## Alternatives to TriviaQA

If TriviaQA coverage is low, try:

1. **Natural Questions** (Google): Real user queries, diverse topics
2. **SimpleQuestions** (Facebook): Simpler factoid questions
3. **WebQuestions**: Real search queries

Modify `download_triviaqa_sample.py` to download from alternative sources.

## See Also

- `/tmp/practical_implementation_plan.md` - Complete implementation details
- `/tmp/qa_test_set_semantic_analysis.md` - Analysis of current test set bias
- `docs/WHOOSH_FINAL_SUMMARY.md` - Current 38% accuracy baseline
