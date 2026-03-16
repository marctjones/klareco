# Git Hooks for Klareco

This directory contains git hooks that enforce code quality and versioning standards.

## Installation

### Option 1: Configure Git to Use This Directory (Recommended)

```bash
git config core.hooksPath .githooks
```

This makes git use hooks from `.githooks/` instead of `.git/hooks/`.

**Benefit**: Hooks are version-controlled and automatically used by all contributors.

### Option 2: Copy Hooks Manually

```bash
cp .githooks/pre-commit .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

**Downside**: Must re-copy when hooks are updated.

## Available Hooks

### pre-commit

**Purpose**: Validates that all Python scripts have proper version information.

**What it checks**:
- All staged scripts in `scripts/` have module-level docstring
- Docstring contains required fields: VERSION, COMPATIBLE WITH, DEPENDENCIES, STAGE
- STAGE field has valid value (Data, Training, Evaluation, Inspection, Utility)
- VERSION field follows format (vX.Y)

**Example output**:
```
Running pre-commit checks...
Validating 2 script(s)...

Checking: scripts/train/roots_v3.py
  ✓ VALID

Checking: scripts/data/export_m1.py
  ✗ INVALID
  Missing required fields: VERSION, COMPATIBLE WITH

✗ Commit rejected: Scripts missing version information

Required fields in docstring:
  - VERSION: v2.1 | v3.0
  - COMPATIBLE WITH: v2.1 database schema, ...
  - DEPENDENCIES: Root Embeddings v3, ...
  - STAGE: Data | Training | Evaluation | Inspection | Utility
```

**To bypass (NOT RECOMMENDED)**:
```bash
git commit --no-verify
```

## Testing Hooks Locally

You can test the hooks without committing:

```bash
# Test pre-commit hook on specific file
python scripts/util/validate_script_versions.py scripts/your_script.py

# Test pre-commit hook on all staged files
python scripts/util/validate_script_versions.py --git-staged

# See template for missing docstring
python scripts/util/validate_script_versions.py scripts/your_script.py --template
```

## Why These Hooks?

**Problem**: Without enforcement, scripts accumulate without version information, making it unclear:
- What version of the system they're compatible with
- What dependencies they require
- Whether they're outdated or current

**Solution**: Git hooks enforce versioning standards automatically, preventing:
- Scripts without version info from being committed
- Version confusion as system evolves
- Incompatible scripts being used together

## Related Documentation

- `docs/CLI_ARCHITECTURE.md` - Complete versioning strategy and docstring template
- `docs/VERSION_COMPATIBILITY.md` - Version compatibility matrix
- `CLAUDE.md` - Script Versioning Policy (MANDATORY section)

## Troubleshooting

### Hook not running

**Check installation**:
```bash
git config core.hooksPath
# Should show: .githooks
```

If not set:
```bash
git config core.hooksPath .githooks
```

### Hook fails with "Python not found"

Ensure Python is in your PATH:
```bash
which python
python --version
```

If using virtual environment, activate it before committing.

### Want to see what would be checked

```bash
git diff --cached --name-only --diff-filter=ACM | grep "^scripts/.*\.py$"
```

## Future Hooks

Planned hooks:
- **pre-push**: Run fast tests before pushing
- **commit-msg**: Validate commit message format (e.g., link to issues)
- **post-merge**: Notify about dependency changes

## Feedback

If hooks cause issues or need improvements, create an issue:
- Too strict? We can adjust validation
- False positives? We can fix the validator
- Missing edge cases? We can add tests

The goal is to help, not hinder!
