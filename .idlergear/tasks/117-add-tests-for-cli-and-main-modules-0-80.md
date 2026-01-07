---
id: 117
title: "Add tests for CLI and main modules (0% \u2192 80%)"
state: open
created: '2026-01-07T00:12:04.647888Z'
labels:
- testing
- coverage
priority: low
---
## Goal

Add tests for command-line interface modules.

## Files to Cover

| File | Current | Target | Lines |
|------|---------|--------|-------|
| `cli.py` | 0% | 80% | 176 |
| `__main__.py` | 0% | 80% | 3 |

## Test Categories

### CLI Command Tests
- `test_parse_command()`
- `test_translate_command()`
- `test_demo_command()`
- `test_help_command()`
- `test_version_command()`

### Argument Parsing Tests
- `test_parse_args_basic()`
- `test_parse_args_with_options()`
- `test_invalid_args()`

### Output Format Tests
- `test_json_output()`
- `test_text_output()`
- `test_verbose_output()`

### Integration Tests
- `test_cli_end_to_end()`
- `test_main_entry_point()`

## Mock Strategy

- Use `click.testing.CliRunner` for CLI tests
- Mock external dependencies (models, indexes)
- Capture stdout/stderr for validation

## Acceptance Criteria

- [ ] cli.py at 80%+ coverage
- [ ] __main__.py at 80%+ coverage
- [ ] All commands tested
- [ ] Error handling tested

## Estimated Effort

~3-4 hours
