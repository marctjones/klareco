# Vocabulary Management Tools

This document describes the vocabulary management tools added on November 11, 2025.

## Overview

Three new tools for managing and improving vocabulary:

1. **Categorize Vocabulary** - Semantic categories and frequency analysis
2. **Continuous Validation** - Track quality metrics over time
3. **Enhanced Parser** - Use enriched vocabulary data in parsing

## Tools

### 1. Categorize Vocabulary (`scripts/categorize_vocabulary.py`)

**Purpose**: Analyze vocabulary to add semantic categories and frequency data.

**Features**:
- Infers semantic categories (verb/noun/adjective) from dictionary
- Calculates frequency from test corpus
- Calculates frequency from all Esperanto text files
- Generates enriched vocabulary JSON

**Usage**:
```bash
# Generate enriched vocabulary
python scripts/categorize_vocabulary.py

# Specify custom output path
python scripts/categorize_vocabulary.py --output custom_path.json
```

**Output**: `data/enriched_vocabulary.json`

**Structure**:
```json
{
  "hund": {
    "root": "hund",
    "category": "unknown",
    "corpus_frequency": 15,
    "text_frequency": 30,
    "total_frequency": 45
  }
}
```

**Statistics Generated**:
- Category distribution (verb/noun/adjective/unknown)
- Frequency tiers (common/moderate/rare/unused)
- Top 20 most frequent roots
- Roots by usage frequency

**What the categories mean**:
- **verb**: Root typically used as verb base (e.g., "manĝ" = eat)
- **noun**: Root typically used as noun base (e.g., "hund" = dog)
- **adjective**: Root typically used as adjective base (e.g., "bel" = beautiful)
- **unknown**: Category could not be determined (89% of roots)

**Frequency tiers**:
- **common**: Appears 10+ times (129 roots)
- **moderate**: Appears 5-9 times (103 roots)
- **rare**: Appears 1-4 times (332 roots)
- **unused**: Never appears (7,668 roots)

---

### 2. Continuous Validation (`scripts/continuous_validation.py`)

**Purpose**: Track vocabulary quality metrics over time.

**Features**:
- Monitors vocabulary size changes
- Tracks corpus coverage trends
- Records test pass rates
- Compares with previous reports
- Generates trend analysis

**Usage**:
```bash
# Run validation (creates report)
python scripts/continuous_validation.py

# View trend analysis
python scripts/continuous_validation.py --trends

# Custom history file
python scripts/continuous_validation.py --history custom_history.json
```

**Output**: `data/validation_history.json`

**What it tracks**:
- **Vocabulary statistics**: Total roots, prefixes, suffixes, etc.
- **Enriched vocabulary**: Category distribution, frequency tiers
- **Test results**: Pass/fail counts, pass rate
- **Corpus coverage**: Percentage of corpus roots recognized

**When to run**:
- After adding new vocabulary
- After modifying parser
- Before major releases
- Weekly/monthly for monitoring
- When debugging vocabulary issues

**Report format**:
```json
{
  "timestamp": "2025-11-11T12:39:52",
  "vocabulary": {
    "roots": 8247,
    "total": 8397
  },
  "enriched": {
    "total_enriched": 8232,
    "frequency_tiers": {
      "common": 129,
      "moderate": 103,
      "rare": 332,
      "unused": 7668
    }
  },
  "tests": {
    "passed": 40,
    "total": 40,
    "pass_rate": 100.0
  },
  "coverage": {
    "corpus_coverage": 91.7
  }
}
```

**Trend Analysis**:
Shows changes over time:
- Vocabulary size growth
- Coverage improvements
- Test pass rate trends
- Best/worst periods

---

### 3. Enhanced Parser (`scripts/update_parser_with_enriched_vocab.py`)

**Purpose**: Demonstrate using enriched vocabulary in parsing.

**Features**:
- Adds metadata to ASTs
- Includes semantic category hints
- Provides frequency information
- Marks common vs rare roots

**Usage**:
```bash
# Run demonstration
python scripts/update_parser_with_enriched_vocab.py
```

**In your code**:
```python
from scripts.update_parser_with_enriched_vocab import \
    load_enriched_vocabulary, enhance_ast_with_metadata
from klareco.parser import parse_word

# Load once at startup
enriched = load_enriched_vocabulary('data/enriched_vocabulary.json')

# Use when parsing
ast = parse_word('hundon')
ast = enhance_ast_with_metadata(ast, ast['radiko'], enriched)

# AST now contains metadata
print(ast['metadata'])
# {
#   'semantic_category': 'unknown',
#   'corpus_frequency': 15,
#   'text_frequency': 30,
#   'total_frequency': 45,
#   'frequency_tier': 'common'
# }
```

**Benefits**:
- Debugging: See if root is recognized
- Quality: Identify rare/unused roots
- Optimization: Prioritize common roots
- Analysis: Understand usage patterns

---

## Workflow

### Initial Setup (One Time)

```bash
# 1. Generate enriched vocabulary
python scripts/categorize_vocabulary.py

# 2. Create baseline validation report
python scripts/continuous_validation.py
```

### After Vocabulary Changes

```bash
# 1. Regenerate enriched vocabulary
python scripts/categorize_vocabulary.py

# 2. Run validation and compare with baseline
python scripts/continuous_validation.py

# 3. View trends
python scripts/continuous_validation.py --trends
```

### Regular Monitoring (Weekly/Monthly)

```bash
# Run validation to track metrics over time
python scripts/continuous_validation.py

# Generate trend report
python scripts/continuous_validation.py --trends
```

---

## Example: Adding New Vocabulary

**Scenario**: You want to add 100 new roots to the parser.

**Before**:
```bash
# 1. Create baseline
python scripts/continuous_validation.py
# Output: Coverage: 91.7%, Tests: 40/40 passing
```

**After adding roots** to `klareco/parser.py`:
```bash
# 2. Regenerate enriched vocabulary
python scripts/categorize_vocabulary.py
# Output: Shows new roots categorized

# 3. Run validation
python scripts/continuous_validation.py
# Output: Coverage: 93.2%, Tests: 40/40 passing
# Comparison: +1.5% coverage, +100 vocabulary items

# 4. View trends
python scripts/continuous_validation.py --trends
# Output: Shows vocabulary growth over time
```

---

## Current Metrics (Baseline - Nov 11, 2025)

**Vocabulary**:
- Total: 8,397 items
- Roots: 8,247
- Enriched with metadata: 8,232

**Categories**:
- Verb: 211 (2.6%)
- Noun: 25 (0.3%)
- Adjective: 598 (7.3%)
- Unknown: 7,389 (89.8%)

**Frequency**:
- Common (>=10): 129 roots
- Moderate (5-9): 103 roots
- Rare (1-4): 332 roots
- Unused (0): 7,668 roots

**Quality**:
- Corpus coverage: 91.7%
- Test pass rate: 100%
- Corpus roots covered: 22/24

---

## Integration with Existing Tools

These new tools complement existing vocabulary tools:

**Existing**:
- `scripts/extract_dictionary_roots.py` - Extract roots from dictionaries
- `scripts/validate_vocabulary.py` - One-time validation reports

**New**:
- `scripts/categorize_vocabulary.py` - Add semantic metadata
- `scripts/continuous_validation.py` - Track changes over time
- `scripts/update_parser_with_enriched_vocab.py` - Use metadata in parsing

**Complete Workflow**:
```bash
# 1. Extract vocabulary (one time or when dictionary updates)
python scripts/extract_dictionary_roots.py

# 2. Generate enriched vocabulary
python scripts/categorize_vocabulary.py

# 3. Validate coverage
python scripts/validate_vocabulary.py

# 4. Track over time
python scripts/continuous_validation.py

# 5. View trends
python scripts/continuous_validation.py --trends
```

---

## Future Enhancements

Possible improvements:

1. **Better Categorization**
   - Use machine learning to classify roots
   - Parse Plena Vortaro for better definitions
   - Manual curation of common roots

2. **Automated Monitoring**
   - CI/CD integration
   - Email/Slack alerts on regressions
   - Automatic reports on PR

3. **Visualization**
   - Plot trends over time
   - Coverage heatmaps
   - Category distribution charts

4. **Smart Recommendations**
   - Suggest roots to add for coverage
   - Identify unused roots to remove
   - Find categorization conflicts

---

## Troubleshooting

**Q: enriched_vocabulary.json not found**
```bash
# Run categorization first
python scripts/categorize_vocabulary.py
```

**Q: validation_history.json not found**
```bash
# Run validation to create history
python scripts/continuous_validation.py
```

**Q: No trends to show**
```bash
# Need at least 2 reports
python scripts/continuous_validation.py  # Run again after changes
python scripts/continuous_validation.py --trends
```

**Q: Category is "unknown" for most roots**
```bash
# This is expected (89.8%)
# Gutenberg dictionary format makes inference hard
# Future: manual curation or better sources
```

---

## Files Generated

```
data/
├── enriched_vocabulary.json      # Vocabulary with categories/frequencies
├── validation_history.json        # Historical validation reports
└── extracted_vocabulary.py        # Original extracted roots

scripts/
├── categorize_vocabulary.py       # Generate enriched vocab
├── continuous_validation.py       # Track metrics over time
└── update_parser_with_enriched_vocab.py  # Use enriched data
```

---

**Last Updated**: 2025-11-11
**Vocabulary Size**: 8,397 items
**Coverage**: 91.7%
**Test Pass Rate**: 100%
