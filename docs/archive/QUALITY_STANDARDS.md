# Quality Standards for Klareco Development

This document defines the quality standards that MUST be maintained throughout development.

## Test Coverage Standards

### Minimum Coverage Requirements

| Component Type | Minimum Coverage | Target |
|----------------|------------------|---------|
| **Core Logic** (parser, pipeline, etc.) | 80% | 90%+ |
| **Utilities** (helpers, formatters) | 70% | 80%+ |
| **Integration Points** (front door, safety) | 75% | 85%+ |
| **Overall Project** | 75% | 85%+ |

**Current Status**: 49% overall - BELOW MINIMUM ⚠️

### Types of Tests Required

**1. Unit Tests** (tests/test_*.py)
- ✅ MUST test each public function/method
- ✅ MUST test edge cases (empty input, invalid input, boundary conditions)
- ✅ MUST test error paths (exceptions, validation failures)
- ✅ MUST use descriptive test names: `test_<function>_<scenario>_<expected_result>`
- ✅ MUST include docstrings explaining what each test validates

**Example:**
```python
def test_parse_word_with_accusative_pronoun_returns_correct_case(self):
    """Tests that accusative pronoun 'min' is parsed with correct case marker."""
    ast = parse_word("min")
    self.assertEqual(ast['kazo'], 'akuzativo')
    self.assertEqual(ast['radiko'], 'mi')
```

**2. Integration Tests** (tests/test_integration_*.py)
- ✅ MUST test component interactions (parser → deparser, pipeline stages)
- ✅ MUST test realistic workflows (input → output through full pipeline)
- ✅ MUST test data flow between components
- ✅ MUST verify end-to-end correctness

**3. End-to-End Tests** (tests/test_e2e_*.py)
- ✅ MUST test complete user scenarios
- ✅ MUST test multi-language input → Esperanto → response
- ✅ MUST test error handling across the full stack
- ✅ MUST verify trace/logging output

### Test Quality Standards

- ✅ Tests MUST be deterministic (no flaky tests)
- ✅ Tests MUST run in isolation (no dependencies between tests)
- ✅ Tests MUST be fast (< 1 second per test, < 30 seconds total suite)
- ✅ Tests MUST have clear failure messages
- ✅ Test data MUST be in fixtures or constants (not hardcoded)

## Code Comment Standards

### Minimum Comment Requirements

**Comment Density**: 20-30% of lines should be comments/docstrings

**What MUST be commented:**

1. **Module-level docstrings** (every .py file)
   ```python
   """
   Module description in English (for human readers).
   Explains purpose, key concepts, usage.
   """
   ```

2. **Class docstrings**
   ```python
   class Parser:
       """
       Parses Esperanto text into morpheme-based ASTs.

       Uses the 16 Rules of Esperanto for deterministic parsing.
       """
   ```

3. **Function/method docstrings**
   ```python
   def parse_word(word: str) -> dict:
       """
       Parses a single Esperanto word into its morpheme components.

       Args:
           word: Esperanto word (e.g., "hundon", "mi", "vidas")

       Returns:
           AST dictionary with radiko, vortspeco, kazo, nombro, etc.

       Raises:
           ValueError: If word has invalid morphology
       """
   ```

4. **Complex algorithm explanations**
   ```python
   # Rule 6: Accusative case (-n) MUST be stripped before POS endings
   # Example: "hundon" → strip "n" → "hundo" → strip "o" → "hund" (root)
   if remaining_word.endswith('n'):
       ast["kazo"] = "akuzativo"
       remaining_word = remaining_word[:-1]
   ```

5. **Grammar references** (when implementing linguistic rules)
   ```python
   # Source: Wikipedia Esperanto Grammar, Rule 5 (Fundamento de Esperanto 1905)
   # "Personal pronouns take the accusative suffix -n as nouns do"
   KNOWN_PRONOUNS = {"mi", "vi", "li", "ŝi", "ĝi", "si", "ni", "ili", "oni"}
   ```

6. **Esperanto term translations** (for readability)
   ```python
   ast['radiko'] = stem  # radiko (Esperanto) = root (English)
   ast['vortspeco'] = 'substantivo'  # vortspeco = part of speech, substantivo = noun
   ```

### What NOT to comment

- ❌ Obvious operations: `i += 1  # increment i`
- ❌ Redundant restatements: `return True  # returns True`
- ❌ Outdated comments (delete or update, don't leave stale)

## Documentation Standards

### Required Documentation Files

**User-Facing:**
- ✅ README.md - Installation, quick start, usage examples
- ✅ examples/ - Runnable code examples with explanations
- ✅ API documentation (TODO: add Sphinx/MkDocs)

**Developer-Facing:**
- ✅ CLAUDE.md - AI assistant guidance (architecture, commands, philosophy)
- ✅ DESIGN.md - System architecture, roadmap
- ✅ TODO.md - Current priorities and tasks
- ✅ CONTRIBUTING.md (TODO: add contribution guidelines)

**Domain Knowledge:**
- ✅ 16RULES.MD - Esperanto grammar specification
- ✅ eHy.md - Esperanto-Hy integration vision
- ✅ DATA_AUDIT.md - Copyright compliance and data management

**Process:**
- ✅ QUALITY_STANDARDS.md (this file)

### Documentation Standards

1. **Keep docs synchronized with code**
   - Update docs in the same commit as code changes
   - Reference specific file:line numbers in docs
   - Update examples when APIs change

2. **Write for your audience**
   - User docs: Focus on "how to use"
   - Developer docs: Focus on "how it works"
   - Comments: Focus on "why this way"

3. **Use Esperanto appropriately**
   - AST field names: Esperanto (radiko, vortspeco)
   - Code comments: English (for human readability)
   - Variable names: English (Python convention)
   - Documentation: Translate Esperanto terms when first introduced

## Development Workflow Standards

### Before Starting New Work

1. ✅ Pull latest changes: `git pull`
2. ✅ Review TODO.md for current priorities
3. ✅ Check DESIGN.md for architectural guidance
4. ✅ Create todos: Use `TodoWrite` tool to plan work

### During Development

1. **Write tests FIRST** (TDD preferred)
   ```bash
   # 1. Write failing test
   pytest tests/test_myfeature.py::test_new_feature

   # 2. Implement feature
   # (edit code)

   # 3. Verify test passes
   pytest tests/test_myfeature.py::test_new_feature
   ```

2. **Add comments as you code**
   - Don't leave commenting for later
   - Explain complex decisions immediately
   - Reference grammar sources for linguistic rules

3. **Update documentation alongside code**
   - New feature? Add example to examples/
   - Changed API? Update README.md
   - New concept? Add to DESIGN.md

### Before Committing

**Run the quality checklist:**

```bash
# 1. Run full test suite
pytest tests/ -v

# 2. Check coverage
python -m coverage run --source=klareco -m pytest tests/
python -m coverage report -m
# Target: 75%+ overall, 80%+ for new code

# 3. Run integration tests
./run.sh

# 4. Check for obvious issues
python -m py_compile klareco/*.py  # Syntax check
grep -r "TODO\|FIXME\|XXX" klareco/  # Find markers

# 5. Verify documentation
ls examples/  # Examples present?
grep -l "$(date +%Y)" *.md  # Docs updated this year?
```

**Commit message standards:**
```
<type>: <short summary (50 chars max)>

<detailed explanation of what and why>

**Changes:**
- Component A: What changed and why
- Component B: What changed and why

**Tests:**
- Added test_xyz to cover scenario ABC
- Updated test_foo to handle new case

**Documentation:**
- Updated README.md with new example
- Added comments explaining algorithm X

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

### Types of Commits

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation only
- `test:` Test additions/changes
- `refactor:` Code restructuring (no behavior change)
- `perf:` Performance improvement
- `style:` Code style/formatting
- `chore:` Maintenance (deps, configs)

## My Commitment Going Forward

**As Claude Code, I commit to:**

### ✅ Always Do (Non-Negotiable)

1. **Write tests for every new feature/fix**
   - Unit tests for functions
   - Integration tests for components
   - E2E tests for workflows

2. **Maintain 75%+ test coverage**
   - Check coverage before committing
   - Flag any drops in coverage
   - Prioritize testing critical paths

3. **Add comprehensive comments**
   - Explain complex algorithms
   - Reference grammar sources
   - Translate Esperanto terms
   - Target 20-30% comment density

4. **Update documentation**
   - Keep README.md current
   - Update examples/ when APIs change
   - Maintain DESIGN.md alignment
   - Update TODO.md with progress

5. **Use TodoWrite tool**
   - Plan work before starting
   - Track progress during implementation
   - Mark completed when done

6. **Commit regularly with good messages**
   - Descriptive commit messages
   - Logical grouping of changes
   - Include test and doc updates in same commit

### ⚠️ Never Do

1. ❌ **Write code without tests**
2. ❌ **Commit code that breaks existing tests**
3. ❌ **Skip comments on complex logic**
4. ❌ **Leave documentation outdated**
5. ❌ **Make "quick fixes" without proper testing**
6. ❌ **Use English field names in ASTs** (keep Esperanto)

### 🎯 Quality Checklist

Before considering work "done":

- [ ] All tests pass (pytest tests/ -v)
- [ ] Coverage at target (pytest-cov)
- [ ] Integration tests pass (./run.sh)
- [ ] New code has tests (unit + integration)
- [ ] Complex code has comments (20%+ density)
- [ ] Documentation updated (README, examples, etc.)
- [ ] Grammar references cited (if linguistic code)
- [ ] Commit message is descriptive
- [ ] TodoWrite list is updated

## Dealing with Gaps in Existing Code

**Priority order for addressing current 49% coverage:**

1. **Critical (P0)**: Add tests for pipeline.py (0% → 80%+)
2. **High (P1)**: Add tests for logging_config.py (0% → 70%+)
3. **Medium (P2)**: Improve safety.py (44% → 75%+)
4. **Medium (P2)**: Improve intent_classifier.py (39% → 75%+)
5. **Low (P3)**: Improve other files to 75%+ gradually

**Approach**: Fix gaps incrementally, not all at once
- Add tests for critical paths first
- Improve coverage with each new feature
- Don't let new code reduce coverage

## Measuring Success

**Weekly checks:**
```bash
# Coverage trend
python -m coverage report -m | grep "TOTAL"

# Test count
pytest tests/ --collect-only | grep "tests collected"

# Documentation size
wc -w *.md examples/*.md | tail -1
```

**Monthly reviews:**
- Review TODO.md for progress
- Update DESIGN.md with learnings
- Audit test coverage gaps
- Update examples/ for new features

## Resources

- **Testing guide**: https://docs.pytest.org/
- **Coverage guide**: https://coverage.readthedocs.io/
- **Python docstring conventions**: PEP 257
- **Commit message guide**: Conventional Commits

---

**Last Updated**: 2025-11-11
**Status**: Established after achieving 47/47 tests passing
**Next Review**: When starting Phase 3 development
