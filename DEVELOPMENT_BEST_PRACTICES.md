# Development Best Practices for Klareco

This document outlines the best practices I should follow automatically when developing Klareco code. This ensures high code quality, maintainability, and professional standards.

---

## ✅ Checklist for Every Feature

When implementing a new feature, I should **automatically** complete ALL of these:

### 1. Code Quality
- [ ] **Docstrings**: All classes and functions have comprehensive docstrings
- [ ] **Type hints**: Use typing annotations where appropriate
- [ ] **Comments**: Add inline comments for complex logic
- [ ] **Error handling**: Proper try/except with meaningful error messages
- [ ] **Logging**: Add INFO logs for major steps, DEBUG for details

### 2. Testing
- [ ] **Unit tests**: Create `tests/test_<module>.py` with comprehensive test cases
- [ ] **Integration tests**: Create `scripts/test_<feature>.py` if needed
- [ ] **Test coverage**: Aim for >80% coverage on new code
- [ ] **Run tests**: Verify all tests pass before committing
- [ ] **Edge cases**: Test error conditions, empty inputs, boundary values

### 3. Documentation
- [ ] **Code docs**: Docstrings explain what, why, and how
- [ ] **User docs**: Update CLAUDE.md with new features/commands
- [ ] **Examples**: Add usage examples to docstrings or separate docs
- [ ] **API docs**: Document public interfaces clearly

### 4. Git Workflow
- [ ] **Atomic commits**: One logical change per commit
- [ ] **Descriptive messages**: Clear commit messages following convention
- [ ] **Commit frequently**: Don't accumulate large uncommitted changes
- [ ] **Attribution**: Include Claude Code attribution footer

### 5. Demo and Examples
- [ ] **Demo script**: Create or update demo showing feature in action
- [ ] **Usage examples**: Show how to use the feature
- [ ] **Error examples**: Show what happens with invalid input

---

## Detailed Guidelines

### Code Comments

**Good Examples**:
```python
def _extract_expression(self, ast: Dict[str, Any]) -> tuple:
    """
    Extract mathematical expression from AST.

    This traverses the AST recursively to find:
    1. Numbers (Esperanto words or digits)
    2. Mathematical operators (plus, minus, etc.)
    3. Operation context

    Returns:
        (expression_dict, operation_type)
        expression_dict has: {'operands': [num1, num2, ...], 'operator': '+'}

    Examples:
        >>> ast = parse("du plus tri")
        >>> expr, op = self._extract_expression(ast)
        >>> expr
        {'operands': [2, 3], 'operator': '+'}
    """
    words = self._extract_all_words(ast)

    # Use set to deduplicate numbers from recursive AST traversal
    seen = set()
    numbers = []

    for word in words:
        # Try numeric literal first (fastest check)
        if word.isdigit():
            num = int(word)
            if num not in seen:  # Avoid duplicates
                numbers.append(num)
                seen.add(num)
            continue

        # Try Esperanto number words
        # (more examples...)
```

**What to Comment**:
- **Why**, not what (code shows what, comments explain why)
- Complex algorithms or logic
- Non-obvious optimizations
- Workarounds for edge cases
- References to external resources/papers
- TODO items with context

**What NOT to Comment**:
- Obvious code (`i += 1  # increment i`)
- Redundant with docstring
- Commented-out code (delete it, it's in git history)

### Logging Best Practices

**Logging Levels**:
```python
# ERROR: Something failed that prevents operation
logging.error(f"Pipeline failed with error: {e}", exc_info=True)

# WARNING: Something unexpected but recoverable
logging.warning(f"No expert registered for intent '{intent}'")

# INFO: Major milestones in operation
logging.info("Step 5: Orchestrator - Routing to expert system.")
logging.info(f"Classified as '{intent}' (confidence: {confidence:.2f})")

# DEBUG: Detailed information for debugging
logging.debug(f"{expert.name} can handle (confidence: {confidence:.2f})")
logging.debug(f"Extracted operands: {numbers}, operator: {operator}")
```

**Guidelines**:
- Log at decision points (routing, selection, classification)
- Log inputs and outputs of major operations
- Include context (file names, IDs, counts)
- Use structured logging when possible
- Don't log sensitive data
- Don't log in tight loops (performance)

### Unit Test Structure

**Example**: `tests/test_math_expert.py`

```python
"""
Unit tests for MathExpert.

Tests the mathematical computation expert's ability to:
- Detect mathematical queries
- Extract numbers and operations from AST
- Perform symbolic computation
- Handle edge cases
"""

import pytest
from klareco.parser import parse
from klareco.experts.math_expert import MathExpert


class TestMathExpert:
    """Test suite for MathExpert."""

    def setup_method(self):
        """Initialize expert before each test."""
        self.expert = MathExpert()

    def test_can_handle_simple_addition(self):
        """Test detection of simple addition query."""
        ast = parse("Kiom estas du plus tri?")
        assert self.expert.can_handle(ast) is True

    def test_execute_addition(self):
        """Test execution of addition."""
        ast = parse("Kiom estas du plus tri?")
        result = self.expert.execute(ast)

        assert 'answer' in result
        assert 'result' in result
        assert result['result'] == 5
        assert result['confidence'] == 0.95
        assert result['expert'] == 'Math Tool Expert'

    # More tests...
```

**Test Organization**:
- One test class per component
- Clear test names: `test_<action>_<condition>_<expected_result>`
- Test happy path, edge cases, error conditions
- Use setup_method for common initialization
- Each test should test ONE thing
- Tests should be independent (no order dependency)

### Git Commit Messages

**Format**:
```
<type>: <short description>

<detailed description>
- Bullet points for changes
- Explain what and why
- Reference issues if applicable

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `test`: Adding tests
- `docs`: Documentation changes
- `refactor`: Code refactoring
- `chore`: Maintenance tasks
- `perf`: Performance improvements

**Examples**:

```
feat: Add checkpoint resumption for GNN training

- Save model state, optimizer state, and training history after each epoch
- Auto-discover latest checkpoint with --resume auto flag
- Prevents loss of progress from interruptions
- Enables safe overnight training runs

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

```
test: Add comprehensive unit tests for MathExpert

- 18 tests covering can_handle, confidence, execution
- Test number extraction (Esperanto words and digits)
- Test operation detection (plus, minus, foje, divid)
- Test edge cases and error handling
- All tests passing (18/18)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

### Documentation Standards

**CLAUDE.md Updates**:
When adding new features, update:
1. Current Implementation Status
2. Commands section (if new commands added)
3. Architecture section (if structure changes)
4. Examples section (show usage)

**Docstring Template**:
```python
def method_name(self, param1: Type1, param2: Type2) -> ReturnType:
    """
    One-line summary of what the method does.

    More detailed explanation of the method's purpose,
    algorithm, or approach. Include why this exists.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When param1 is invalid
        TypeError: When param2 has wrong type

    Examples:
        >>> obj.method_name("value", 42)
        "result"

    Notes:
        Additional important information, caveats, or
        references to related functionality.
    """
```

---

## What I Did Wrong This Session

Let me be honest about where I fell short:

### ❌ Missing Initially

1. **Unit Tests**: Didn't create unit tests until prompted
   - Should have created `tests/test_math_expert.py` automatically
   - Should have created `tests/test_orchestrator_unit.py` automatically
   - Integration tests were good, but unit tests are essential

2. **Git Commits**: Accumulated 6 commits worth of changes before committing
   - Should commit after each logical feature
   - Let ~2,000 lines of code go uncommitted
   - This is unprofessional and risky

3. **Inline Comments**: Sparse in expert code
   - MathExpert has good docstrings but few inline comments
   - Complex logic in `_extract_expression` needs more explanation
   - Should explain the deduplication logic explicitly

### ✅ What I Did Well

1. **Docstrings**: Comprehensive docstrings on all classes/methods
2. **Integration Tests**: Created test_experts.py and test_orchestrator.py
3. **Demo**: Excellent demo_klareco.py showing full pipeline
4. **Documentation**: Comprehensive EXPERT_INTEGRATION_SUMMARY.md
5. **Error Handling**: Good try/except with meaningful error messages
6. **Logging**: INFO-level logging at major pipeline steps

---

## Automated Checklist for Future Work

When I start writing a new component, I should:

1. ✅ Write the code with docstrings and type hints
2. ✅ Add logging at major decision points
3. ✅ Write unit tests in `tests/` directory
4. ✅ Write integration tests if needed in `scripts/`
5. ✅ Update CLAUDE.md with new features
6. ✅ Create or update demo script
7. ✅ Run all tests to verify they pass
8. ✅ **Commit to git immediately** after feature is working
9. ✅ Repeat for next feature (don't accumulate changes)

---

## Testing Standards

### Unit Test Coverage Goals

**Minimum Coverage**: 80% for new code

**What to Test**:
- All public methods
- Happy path (expected input → expected output)
- Edge cases (empty input, boundary values)
- Error conditions (invalid input → proper error)
- Integration points (how components interact)

**Test File Organization**:
```
tests/
  test_parser.py              # Existing
  test_front_door.py          # Existing
  test_math_expert.py         # NEW ✅
  test_date_expert.py         # TODO
  test_grammar_expert.py      # TODO
  test_orchestrator_unit.py   # NEW ✅
  test_gating_network.py      # TODO
  test_pipeline.py            # Should enhance
```

### Integration Test Standards

**Location**: `scripts/test_*.py`

**Purpose**: Test complete workflows end-to-end

**Examples**:
- `test_experts.py`: Test individual experts with sample queries
- `test_orchestrator.py`: Test orchestration and routing
- `run_integration_test.py`: Test full pipeline

---

## Demo Standards

**Every major feature should have a demo** showing:
1. How to use it
2. What inputs work
3. What outputs you get
4. Common patterns/workflows

**Good Demo Characteristics**:
- Clear, visual output (emojis, formatting)
- Shows multiple examples
- Includes success and failure cases
- Educational (explains what's happening)
- Runnable with no configuration

**Example**: `demo_klareco.py` is excellent
- Shows multi-language input
- Demonstrates all 3 experts
- Visual step-by-step output
- Explains the architecture
- Runs standalone

---

## Summary

Going forward, I will **automatically**:

1. ✅ Write comprehensive docstrings for all code
2. ✅ Add inline comments for complex logic
3. ✅ Create unit tests in `tests/` directory
4. ✅ Create integration tests in `scripts/` if needed
5. ✅ Update CLAUDE.md with new features
6. ✅ Create or update demo scripts
7. ✅ Add appropriate logging (INFO for milestones, DEBUG for details)
8. ✅ **Make git commits frequently** (after each logical feature)
9. ✅ Run all tests before committing
10. ✅ Follow commit message conventions

This ensures professional-quality code that is:
- **Testable**: Comprehensive test coverage
- **Maintainable**: Clear documentation and comments
- **Traceable**: Proper git history
- **Usable**: Demos and examples for users
- **Debuggable**: Good logging and error handling

---

## Current Git Status

As of this session:

✅ **8 commits made** covering:
1. GNN training infrastructure (checkpoint resumption)
2. Expert system core (3 experts + orchestrator)
3. Pipeline integration
4. Tests and demo (54 tests total)
5. Documentation (5 major docs)
6. Historical reports and scripts

✅ **All code committed** - working tree clean

✅ **Ready to push** - 8 commits ahead of origin/master

---

## Lessons Learned

**What I'll remember**:
- Test-driven development: Write tests alongside code, not after
- Commit early, commit often: Don't let changes accumulate
- Documentation is code: Update docs as you build
- Demos sell features: Show, don't just tell
- Comments explain why: Code shows what, comments show why

**Questions I should ask myself**:
- "Have I written unit tests for this?"
- "Is this a good place for a git commit?"
- "Would someone else understand this code?"
- "Is there a demo showing how to use this?"
- "What happens when this fails?"
