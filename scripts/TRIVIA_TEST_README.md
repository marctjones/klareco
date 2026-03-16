# Trivia Questions Test Suite

Test the extractive QA system with 8 trivia questions covering different question types and topics.

## Questions

### Esperanto Questions (4)
1. **WHO**: Kiu kreis Esperanton? (Who created Esperanto?)
2. **WHEN**: Kiam estis publikigita la unua libro? (When was the first book published?)
3. **WHERE**: Kie okazis la unua Kongreso? (Where was the first Congress?)
4. **WHAT**: Kio estas la Fundamento? (What is the Fundamento?)

### Benjamin Franklin Questions (4)
5. **WHO**: Kiu estis Benjamin Franklin? (Who was Benjamin Franklin?)
6. **WHEN**: Kiam naskiĝis Benjamin Franklin? (When was Franklin born?)
7. **WHERE**: Kie naskiĝis Benjamin Franklin? (Where was Franklin born?)
8. **WHAT**: Kion inventis Benjamin Franklin? (What did Franklin invent?)

## Usage

### Interactive Mode (with pauses between questions)

```bash
# Basic run
./scripts/test_trivia_questions.sh

# With embedding expansion
./scripts/test_trivia_questions.sh --expand

# Save results to file
./scripts/test_trivia_questions.sh --save results.txt

# Both options
./scripts/test_trivia_questions.sh --expand --save results.txt
```

**Interactive mode:**
- Pauses between questions (press Enter to continue)
- Good for step-by-step review
- Can stop at any time (Ctrl+C)

### Batch Mode (runs all questions automatically)

```bash
# Run all 8 questions non-stop
./scripts/test_trivia_questions_batch.sh

# With expansion
./scripts/test_trivia_questions_batch.sh --expand

# Save to file
./scripts/test_trivia_questions_batch.sh > results.txt

# With expansion, save to file
./scripts/test_trivia_questions_batch.sh --expand > results.txt
```

**Batch mode:**
- Runs all questions automatically (no pauses)
- Good for full regression testing
- Cleaner output (suppresses stderr)
- Easy to redirect to file

## What to Look For

### Success Indicators
- ✅ Answer directly addresses the question
- ✅ Highest-scored fact is most relevant
- ✅ Facts are from correct entity (Esperanto vs. Franklin)
- ✅ Temporal/location modifiers extracted correctly

### Known Limitations
- ⚠️ Repetitive phrasing (no pronominalization)
- ⚠️ Long run-on sentences (no aggregation)
- ⚠️ May include off-topic facts (overly broad matching)
- ⚠️ WHERE questions might struggle with events vs. entities

### Analysis Questions
1. **Question Type Performance**: Which types work best? (WHAT, WHO, WHERE, WHEN)
2. **Topic Generalization**: Do Franklin questions work as well as Esperanto?
3. **Fact Ranking**: Are facts ranked correctly by importance?
4. **Coverage**: Do we have the necessary Wikipedia articles?

## Expected Results

### Should Work Well
- Esperanto Q1 (WHO): Zamenhof
- Esperanto Q2 (WHEN): 1887
- Franklin Q5 (WHO): Statesman, scientist, inventor
- Franklin Q6 (WHEN): 1706

### Might Struggle
- Esperanto Q3 (WHERE): Event location (Boulogne-sur-Mer)
- Franklin Q8 (WHAT): Multiple inventions (lightning rod, bifocals)

### Unknown (Depends on Wikipedia Coverage)
- Esperanto Q4: Fundamento definition
- Franklin Q7: Birth location (Boston)

## Example Output

```
======================================================================
QUESTION 1/8 [WHO]
======================================================================
Esperanto: Kiu kreis Esperanton?
English:   Who created Esperanto?
----------------------------------------------------------------------

Query: Kiu kreis Esperanton?
Question type: who
Query entity: esperant

Answer:
  Zamenhof kreis Esperanton en 1887. ...

Metadata:
  Facts extracted: 5
  Facts selected: 3

FACT SCORES
----------------------------------------------------------------------
1. Fact(esperant, CREATED-BY, args=[agent=zamenhof], mods=[time=1887])
   Score=0.95 [Q:1.00, D:0.80, E:1.00, C:1.00]
...
```

## Files

- `test_trivia_questions.sh` - Interactive version (pauses between questions)
- `test_trivia_questions_batch.sh` - Batch version (runs all automatically)
- `TRIVIA_TEST_README.md` - This file

## Tips

**For Quick Testing:**
```bash
./scripts/test_trivia_questions_batch.sh > /tmp/trivia_results.txt
grep "ANSWER" /tmp/trivia_results.txt -A 3
```

**For Detailed Analysis:**
```bash
./scripts/test_trivia_questions.sh --expand --save detailed_results.txt
# Review detailed_results.txt
```

**For Comparing With/Without Expansion:**
```bash
./scripts/test_trivia_questions_batch.sh > no_expansion.txt
./scripts/test_trivia_questions_batch.sh --expand > with_expansion.txt
diff no_expansion.txt with_expansion.txt
```
