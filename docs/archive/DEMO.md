# Klareco CLI Demo Guide

This guide demonstrates what Klareco can do right now and provides practical usage examples.

## 🎯 What Klareco Can Do Today

### Core Capabilities

1. **Deterministic Esperanto Parsing** - The main feature
2. **Multi-language Translation** - Via Opus-MT models
3. **Intent Classification** - Rule-based from AST structure
4. **Complete Traceability** - Every step logged with inputs/outputs
5. **Safety Monitoring** - Input length and AST complexity validation
6. **Symbolic Response Generation** - Currently basic, expandable

---

## 🚀 Quick Start Examples

### 1. Parse Esperanto Text

**What it does:** Breaks down Esperanto sentences into structured ASTs showing grammar structure.

```bash
python -m klareco parse "Mi amas la grandan hundon."
```

**Output:** Complete AST showing:
- **Subject**: Mi (pronoun, nominative)
- **Verb**: amas (present tense)
- **Object**: hundon (noun, accusative) with adjective "grandan" and article "la"

**Why it's useful:**
- Validates Esperanto grammar
- Extracts morphological features (case, number, tense)
- Shows adjective agreement
- Perfect for debugging grammar rules

---

### 2. Run Full Pipeline

**What it does:** Processes text through all 6 stages (Safety → FrontDoor → Parser → Safety → Intent → Responder).

```bash
# Simple text output
python -m klareco run "La hundo vidas la katon."
# Output: Vi diras, ke la hundo vidas la katon.

# Full trace with JSON
python -m klareco run "La hundo vidas la katon." --output-format trace > trace.json
```

**Why it's useful:**
- See complete execution flow
- Debug pipeline stages
- Understand intent classification
- Analyze performance (timing for each step)

---

### 3. Translate Text

**What it does:** Translates between Esperanto and other languages (auto-detects source language).

```bash
# English to Esperanto
python -m klareco translate "The dog sees the cat."
# Output: La hundo vidas la katon.

# Esperanto to English
python -m klareco translate "La hundo vidas la katon." --to en
# Output: The dog sees the cat.
```

**Why it's useful:**
- Test FrontDoor translation component
- Generate Esperanto for testing
- Validate translation quality

---

### 4. Debug Pipeline Stages

**What it does:** Stop pipeline at specific stage to inspect intermediate results.

```bash
# Stop after parsing to inspect AST
python -m klareco run "Mi vidas hundon." --stop-after Parser --output-format json

# Stop after language detection
python -m klareco run "Hello world" --stop-after FrontDoor --output-format json
```

**Available stages:**
- `SafetyMonitor` - Input length check
- `FrontDoor` - Language detection and translation
- `Parser` - Esperanto parsing
- `SafetyMonitor_AST` - AST complexity check
- `IntentClassifier` - Intent classification
- `Responder` - Response generation

**Why it's useful:**
- Debug specific pipeline stages
- Inspect AST structure
- Validate translation output
- Test safety limits

---

### 5. Test the System

**What it does:** Runs integration tests against test corpus.

```bash
# Run all tests
python -m klareco test

# Run limited number of sentences
python -m klareco test --num-sentences 5

# Use pytest for detailed output
python -m klareco test --pytest -v

# Run with debug logging
python -m klareco test --num-sentences 3 --debug
```

**Why it's useful:**
- Validate parser improvements
- Check test coverage
- Identify vocabulary gaps
- Monitor system health

---

### 6. System Information

**What it does:** Shows current system configuration and vocabulary size.

```bash
python -m klareco info
```

**Output:**
```
=== Klareco System Information ===

Python: 3.13.9
PyTorch: 2.9.0+cpu
CUDA available: False

Vocabulary:
  Roots: 8247
  Prefixes: 5
  Suffixes: 25

Test corpus: 20 sentences
Models: 2 files in models/
```

**Why it's useful:**
- Check installed models
- Verify CPU-only installation
- See current vocabulary size
- Confirm test corpus availability

---

## 📚 Current Vocabulary

**Massively expanded vocabulary (November 2025):**

**Roots: 8,247** (extracted from Gutenberg English-Esperanto Dictionary) including:
- **Verbs**: est (be), manĝ (eat), dorm (sleep), kur (run), labor (work), vol (want), help (help), paf (shoot), far (do), dir (say), ven (come), ir (go), don (give), pren (take), hav (have), pov (can), dev (must), sci (know), parol (speak), skrib (write), leg (read), pens (think), sent (feel), stud (study), lern (learn), instru (teach)
- **Nouns**: hom (person), vir (man), infan (child), patr (father), frat (brother), dom (house), urb (city), land (country), mond (world), temp (time), jar (year), tag (day), libr (book), tabl (table), seĝ (chair), akvo (water), pan (bread), arb (tree), flor (flower), sun (sun), lun (moon), amik (friend)
- **Adjectives**: bel (beautiful), rapid (fast), nov (new), jung (young), alt (tall), long (long), varm (warm), ver (true), feliĉ (happy), trist (sad), facil (easy), fort (strong), riĉ (rich), plen (full)
- **Colors**: ruĝ (red), blu (blue), verd (green), flav (yellow), nigr (black), blank (white), griz (gray)
- **Numbers**: unu (1), du (2), tri (3), kvar (4), kvin (5), ses (6), sep (7), ok (8), naŭ (9), dek (10), cent (100), mil (1000)

**Prepositions (29):**
- `en` (in), `sur` (on), `sub` (under), `super` (above), `kun` (with), `sen` (without), `de` (of/from), `al` (to), `ĝis` (until), `tra` (through), and 19 more

**Correlatives (45):** - Complete table
- `kiu/kio/kia` (who/what/which), `tiu/tio/tia` (that), `ĉiu/ĉio/ĉia` (every), `neniu/nenio/nenia` (no one/nothing), `iu/io/ia` (someone/something)

**Prefixes (5):**
- `mal-` (opposite), `re-` (again), `ge-` (both genders), `eks-` (former), `pra-` (primordial)

**Suffixes (25):**
- `-ul` (person), `-ej` (place), `-in` (feminine)
- `-et` (diminutive), `-eg` (augmentative), `-ad` (continuous action), `-ig` (make/cause), `-iĝ` (become)
- `-ist` (professional - e.g., programisto), `-ism` (doctrine)
- `-ar` (collection), `-aĵ` (concrete thing), `-aĉ` (pejorative)
- `-ebl` (possible), `-end` (must), `-ind` (worthy), `-em` (tendency)
- `-ec` (quality), `-er` (unit), `-estr` (leader), `-id` (offspring)
- `-il` (tool), `-ing` (holder), `-uj` (container/country), `-um` (indefinite)

**Conjunctions (10):**
- `kaj` (and), `aŭ` (or), `sed` (but), `nek` (neither/nor), `se` (if), `ĉar` (because), `kvankam` (although), `ke` (that), `tamen` (however), `do` (therefore)

**Particles (27):**
- `ne` (not), `jes` (yes), `ankaŭ` (also), `nur` (only), `tre` (very), `tro` (too much), `jam` (already), `nun` (now), `hodiaŭ` (today), and 18 more

**Grammar Endings:**
- Nouns: `-o`, Adjectives: `-a`, Adverbs: `-e`
- Verbs: `-as` (present), `-is` (past), `-os` (future), `-us` (conditional), `-u` (imperative), `-i` (infinitive)
- Case: `-n` (accusative), `-j` (plural)

**Pronouns (9):**
- `mi` (I), `vi` (you), `li` (he), `ŝi` (she), `ĝi` (it)
- `si` (self-reflexive), `ni` (we), `ili` (they), `oni` (one/people)

**Example sentences that work (verified 100% test coverage):**
```
✅ "La hundo vidas la katon."
✅ "Mi amas la grandan hundon."
✅ "La bona kato manĝas."  ← NOW WORKS!
✅ "Malgrandaj hundoj vidas la grandan katon."
✅ "La programisto laboras."
✅ "Sanaj katoj dormas."  ← NOW WORKS!
✅ "Mi vidas la katon."
✅ "La kato estas bona."  ← NOW WORKS!
✅ "Grandaj hundoj kuros."  ← NOW WORKS!
✅ "La hundo vidis la katon."
✅ "Mi amas katon."
✅ "Bonan tagon."  ← NOW WORKS!
✅ "La hundo kaj la kato."  ← NOW WORKS (conjunction)!
✅ "Estas bela tago."  ← NOW WORKS!
✅ "Mi volas manĝi."  ← NOW WORKS!
✅ "La programisto programas."
✅ "Bonaj amikoj helpas."  ← NOW WORKS!
✅ "La sana hundo kuras rapide."  ← NOW WORKS!
✅ "Mi vidis grandan katon."
✅ "La katoj manĝas."  ← NOW WORKS!
```

**Example sentences that DON'T work yet:**
```
❌ "Mi iras hejmen." (hejm = home - not in vocabulary)
❌ "La birdo kantas." (bird = birdo, kant = sing - not in vocabulary)
❌ "Li loĝas en la urbo." (en = in - preposition not added yet)
```

---

## 🎪 Practical Use Cases

### Use Case 1: Grammar Validation

**Goal:** Check if Esperanto sentence is grammatically correct.

```bash
# Valid sentence
python -m klareco parse "Mi vidas la hundon."
# ✅ Returns complete AST

# Invalid sentence (wrong case)
python -m klareco parse "Mi vidas la hundo."
# ❌ Parser completes but shows "hundo" in nominative (subject case) as object
```

### Use Case 2: Morpheme Analysis

**Goal:** Understand word structure (prefixes, roots, suffixes, endings).

```bash
python -m klareco parse "resanigos"
```

**Output shows:**
- Prefix: `re-` (again)
- Root: `san` (healthy)
- Suffix: `-ig` (make/cause)
- Ending: `-os` (future tense)
- **Meaning**: "will make healthy again" = "will cure/heal"

### Use Case 3: Pipeline Debugging

**Goal:** Debug why a sentence fails processing.

```bash
# Run with debug logging
python -m klareco run "La birdo kantas." --debug 2>&1 | grep ERROR

# Stop at each stage to find where it fails
python -m klareco run "La birdo kantas." --stop-after FrontDoor
python -m klareco run "La birdo kantas." --stop-after Parser
```

### Use Case 4: Testing New Grammar Rules

**Goal:** Validate parser changes with test corpus.

```bash
# Test specific sentences
python -m klareco test --num-sentences 5

# Full test suite
python -m klareco test --pytest

# Check coverage
python -m pytest tests/ --cov=klareco --cov-report=term-missing
```

### Use Case 5: Translation Quality Check

**Goal:** Test FrontDoor translation component.

```bash
# Translate and parse
python -m klareco translate "I love the big dog." > /tmp/eo.txt
cat /tmp/eo.txt
# La amas la grandan hundon. (missing "mi" - translation limitation)

python -m klareco parse "$(cat /tmp/eo.txt)"
```

---

## 🔬 Advanced Features

### Debug Mode

Full context logging with file:line numbers:

```bash
python -m klareco run "La hundo vidas la katon." --debug
```

**Shows:**
- Full input text
- Stack traces on errors
- AST state at each stage
- File names and line numbers

### Output Formats

**Text mode** (default):
```bash
python -m klareco run "La hundo vidas la katon."
# Vi diras, ke la hundo vidas la katon.
```

**JSON mode** (compact):
```bash
python -m klareco run "La hundo vidas la katon." --output-format json
```

**Trace mode** (full detail):
```bash
python -m klareco run "La hundo vidas la katon." --output-format trace
# Complete trace with all 6 stages, inputs, outputs, timestamps
```

### File Input

Process text from files:

```bash
echo "La hundo vidas la katon." > /tmp/input.txt
python -m klareco run --file /tmp/input.txt
```

### Stdin Input

Use in pipelines:

```bash
echo "La hundo vidas la katon." | python -m klareco run
```

---

## 🎯 What to Expect vs. Current Limitations

### ✅ What Works Well

1. **Parsing Known Vocabulary**
   - Perfect accuracy for the 9 roots
   - Correct case marking (nominative vs. accusative)
   - Adjective agreement validation
   - Pronoun handling (all 9 personal pronouns)

2. **Grammar Analysis**
   - Morpheme decomposition (prefix + root + suffix + ending)
   - Tense identification (present, past, future)
   - Case identification (nominative, accusative)
   - Number identification (singular, plural)

3. **Traceability**
   - Complete execution logs
   - Timestamps for each step
   - Input/output tracking
   - Error messages with context

4. **Testing Infrastructure**
   - 114 passing tests
   - 75% code coverage
   - E2E integration tests
   - Real-time log monitoring

### ⚠️ Current Limitations

1. **Vocabulary Coverage** ✅ MOSTLY SOLVED
   - Now has 8,247 roots from Gutenberg English-Esperanto Dictionary
   - Test corpus coverage: 91.7%
   - Can validate vocabulary with: `python scripts/validate_vocabulary.py`
   - **Remaining issue**: Some specialized/technical terms may still be missing

2. **Basic Response Generation**
   - Currently just echoes: "Vi diras, ke..." (You say that...)
   - No semantic understanding yet
   - No question answering
   - **Future**: Phase 4 will add neural decoder for responses

3. **Translation Quality**
   - Depends on Opus-MT models
   - Sometimes drops pronouns or articles
   - Best for simple sentences
   - **Workaround**: Test translations with `parse` command

4. **No Complex Syntax**
   - Doesn't handle subordinate clauses yet
   - No relative pronouns (kiu, kio, etc.)
   - No correlatives (ĉi-, ki-, etc.)
   - **Future**: Parser expansion in Phase 2.5

5. **Intent Classification Limited**
   - Only recognizes "SimpleStatement" intent
   - No question detection yet
   - No command recognition
   - **Future**: Phase 4 will expand intent types

---

## 🛠️ Vocabulary Management

### Validating Vocabulary Coverage

Check how well the vocabulary covers your texts:

```bash
# Generate comprehensive vocabulary report
python scripts/validate_vocabulary.py

# Save detailed report to JSON
python scripts/validate_vocabulary.py --output vocab_report.json
```

**What it checks:**
- Parser vocabulary size (8,247 roots + grammatical items)
- Test corpus coverage percentage
- Missing roots in corpus and other texts
- Comparison between different vocabulary sources

### Adding New Roots (if needed)

The parser now uses the Gutenberg dictionary automatically. For additional roots:

```python
# Edit klareco/parser.py
KNOWN_ROOTS = {
    # ... existing 8,247 roots ...
    "custom_root",  # your custom addition
}
```

### Testing New Vocabulary

```bash
# Test specific sentence
python -m klareco parse "Nova frazo por testi."

# Run full test suite
python -m klareco test --num-sentences 20

# Validate vocabulary coverage
python scripts/validate_vocabulary.py
```

---

## 📖 Learn More

- **Architecture**: See `DESIGN.md` for 9-phase roadmap
- **Grammar Rules**: See `16RULES.MD` for Esperanto specification
- **Developer Guide**: See `CLAUDE.md` for AI assistant guidance
- **Quality Standards**: See `QUALITY_STANDARDS.md` for testing requirements
- **Examples**: See `examples/` directory for code examples

---

## 🎉 Next Steps

**For Users:**
1. Try parsing your own Esperanto sentences
2. Test translation quality with different inputs
3. Explore the trace output to understand the pipeline
4. Report vocabulary gaps by running tests

**For Developers:**
1. Add missing vocabulary roots
2. Improve test coverage for edge cases
3. Expand intent classification rules
4. Prepare for Phase 3 (GNN Encoder for RAG)

---

**Last Updated**: 2025-11-11
**Version**: Phase 2 (Core Foundation & Traceability)
**Test Status**: 20/20 integration tests passing, 75% code coverage
**Vocabulary**: 8,397 items (8,247 roots + 150 grammatical words)
**Corpus Coverage**: 91.7%
