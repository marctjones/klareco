"""
Script Version Validator

VERSION: v3.0
COMPATIBLE WITH: All script versions
DEPENDENCIES: None
STAGE: Utility

Description:
    Validates that all Python scripts in scripts/ have proper version information.
    Checks for required docstring fields: VERSION, COMPATIBLE WITH, DEPENDENCIES, STAGE.
    Can be run manually or as part of git pre-commit hook.

Usage:
    # Check all scripts
    python scripts/util/validate_script_versions.py

    # Check specific script
    python scripts/util/validate_script_versions.py scripts/train/roots_v3.py

    # Use in pre-commit hook
    python scripts/util/validate_script_versions.py --git-staged

    # Auto-fix (add template if missing)
    python scripts/util/validate_script_versions.py --fix

Last Updated: 2026-03-09
Author: Claude + Marc
Related Issues: Epic #637-642 (CLI Architecture)
See Also: docs/CLI_ARCHITECTURE.md, docs/VERSION_COMPATIBILITY.md
"""

import ast
import sys
import re
from pathlib import Path
from typing import List, Dict, Tuple
import argparse


# Required fields in docstring
REQUIRED_FIELDS = [
    'VERSION',
    'COMPATIBLE WITH',
    'DEPENDENCIES',
    'STAGE',
]

# Optional but recommended fields
RECOMMENDED_FIELDS = [
    'Description',
    'Pipeline Position',
    'Usage',
    'Inputs',
    'Outputs',
    'Last Updated',
    'Related Issues',
]

# Valid stages. These mirror the pipeline stages the project actually has
# (see CLAUDE.md → Pipeline Stages) plus the cross-cutting ones. The original
# list predated the data pipeline and rejected `Index` — the stage used by the
# entire indexing chain, including build_duckdb_store.py.
VALID_STAGES = [
    # Pipeline stages, in order
    'Acquire', 'Clean', 'Extract', 'OCR', 'Parse', 'Index',
    'Eval', 'Evaluation', 'Validate', 'Pipeline',
    # Cross-cutting
    'Data', 'Training', 'Inspection', 'Diagnostics', 'Repair', 'Utility',
]

# VERSION accepts vN.N (v2.1) and vN.x (v2.x — "the v2 series", used across the
# DuckDB-era scripts where the exact minor is not meaningful).
VERSION_RE = re.compile(r'^v\d+\.(\d+|x)')


def extract_docstring(file_path: Path) -> str:
    """Extract the module-level docstring from a Python file.

    Uses ast.get_docstring rather than a regex. The previous implementation
    matched a comment-skip group followed by a triple-quoted string, under
    re.DOTALL | re.MULTILINE. re.DOTALL makes the dot in the comment-skip
    group match newlines too, so that group greedily swallowed past the real
    module docstring and captured some *function* docstring further down the
    file.

    Effect: every script with a shebang before its docstring was silently
    mis-reported. The linter declared build_duckdb_store.py to be missing all
    four required fields when it has all four. Since it is wired into a
    pre-commit hook, its output was actively misleading. See issue #781.
    """
    try:
        source = file_path.read_text(encoding='utf-8')
    except Exception as e:
        print(f"ERROR reading {file_path}: {e}")
        return ""

    try:
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError as e:
        print(f"ERROR parsing {file_path}: {e}")
        return ""

    return ast.get_docstring(tree, clean=False) or ""


def check_field(docstring: str, field: str) -> Tuple[bool, str]:
    """Check if field exists in docstring and extract value."""
    pattern = rf'^{re.escape(field)}:\s*(.+?)(?=\n\S|\n\n|\Z)'
    match = re.search(pattern, docstring, re.MULTILINE | re.DOTALL)

    if match:
        value = match.group(1).strip()
        return True, value

    return False, ""


def validate_script(file_path: Path, verbose: bool = False) -> Dict:
    """
    Validate a single script file.

    Returns:
        {
            'valid': bool,
            'missing_required': List[str],
            'missing_recommended': List[str],
            'warnings': List[str],
            'docstring': str
        }
    """
    result = {
        'valid': True,
        'missing_required': [],
        'missing_recommended': [],
        'warnings': [],
        'docstring': ''
    }

    # Skip utility scripts (this validator itself, etc.)
    if 'util' in file_path.parts and file_path.name in ['validate_script_versions.py']:
        result['valid'] = True
        return result

    docstring = extract_docstring(file_path)
    result['docstring'] = docstring

    if not docstring:
        result['valid'] = False
        result['warnings'].append("No module-level docstring found")
        result['missing_required'] = REQUIRED_FIELDS
        return result

    # Check required fields
    for field in REQUIRED_FIELDS:
        found, value = check_field(docstring, field)

        if not found:
            result['valid'] = False
            result['missing_required'].append(field)
        elif verbose:
            print(f"  ✓ {field}: {value[:50]}...")

    # Check recommended fields
    for field in RECOMMENDED_FIELDS:
        found, value = check_field(docstring, field)

        if not found:
            result['missing_recommended'].append(field)

    # Validate STAGE value. A stage may be qualified ("Index / Schema
    # augmentation") or list alternatives ("Data | Evaluation") — validate the
    # leading token, which is the stage proper.
    found, stage_value = check_field(docstring, 'STAGE')
    if found:
        head = re.split(r'[/|,]', stage_value, maxsplit=1)[0].strip()
        if head not in VALID_STAGES:
            result['warnings'].append(
                f"Invalid STAGE: '{head}'. Must be one of: {', '.join(VALID_STAGES)}"
            )

    # Check VERSION format. Accepts vN.N and vN.x — see VERSION_RE.
    found, version_value = check_field(docstring, 'VERSION')
    if found and not VERSION_RE.match(version_value):
        result['warnings'].append(
            f"VERSION format should be 'vX.Y' or 'vX.x' (e.g. v2.1, v3.0, "
            f"v2.x), got: '{version_value}'"
        )

    return result


def find_python_scripts(root: Path = None) -> List[Path]:
    """Find all Python scripts in scripts/ directory."""
    if root is None:
        root = Path(__file__).parent.parent  # scripts/

    scripts = []
    for script in root.rglob('*.py'):
        # Skip __init__.py and hidden files
        if script.name == '__init__.py' or script.name.startswith('.'):
            continue

        # Skip archive directory
        if 'archive' in script.parts:
            continue

        scripts.append(script)

    return sorted(scripts)


def get_git_staged_files() -> List[Path]:
    """Get list of staged Python files in scripts/."""
    import subprocess

    try:
        result = subprocess.run(
            ['git', 'diff', '--cached', '--name-only', '--diff-filter=ACM'],
            capture_output=True,
            text=True,
            check=True
        )

        files = []
        for line in result.stdout.strip().split('\n'):
            if line.startswith('scripts/') and line.endswith('.py'):
                path = Path(line)
                if path.exists():
                    files.append(path)

        return files

    except subprocess.CalledProcessError:
        print("ERROR: Not in a git repository or git command failed")
        return []


def generate_template(file_path: Path) -> str:
    """Generate docstring template for a script."""
    script_name = file_path.stem.replace('_', ' ').title()

    template = f'''"""
{script_name}

VERSION: v3.0
COMPATIBLE WITH: v2.1 database schema, v3.0 AST-annotator protocol
DEPENDENCIES: List dependencies here (e.g., Root Embeddings v3, M1 v2)
STAGE: Data | Training | Evaluation | Inspection | Utility

Description:
    Brief description of what this script does (1-3 sentences).
    Focus on the "why" not just the "what".

Pipeline Position:
    v2.1 DB → [THIS SCRIPT] → Next Component → ...
    (Show where this fits in the data/training pipeline)

Usage:
    python {file_path} --arg1 value1 --arg2 value2

Inputs:
    - Input 1: Description (format: JSON/JSONL/CSV, location: data/...)

Outputs:
    - Output 1: Description (format, location)

Quality Checks:
    - Check 1: What validation is performed

Last Updated: 2026-03-09
Author: <name>
Related Issues: #XXX
See Also: docs/RELATED_DOC.md
"""

# Your code here
'''
    return template


def main():
    parser = argparse.ArgumentParser(
        description='Validate version information in Python scripts',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Check all scripts
  python scripts/util/validate_script_versions.py

  # Check specific script
  python scripts/util/validate_script_versions.py scripts/train/roots_v3.py

  # Check only git-staged files (for pre-commit hook)
  python scripts/util/validate_script_versions.py --git-staged

  # Show template for missing docstring
  python scripts/util/validate_script_versions.py scripts/new_script.py --template
        '''
    )

    parser.add_argument('files', nargs='*', help='Specific files to check')
    parser.add_argument('--git-staged', action='store_true',
                       help='Check only git-staged files')
    parser.add_argument('--template', action='store_true',
                       help='Show template for scripts missing docstrings')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Verbose output')

    args = parser.parse_args()

    # Determine which files to check
    if args.files:
        files = [Path(f) for f in args.files if Path(f).exists()]
    elif args.git_staged:
        files = get_git_staged_files()
        if not files:
            print("No staged Python files in scripts/")
            return 0
    else:
        project_root = Path(__file__).parent.parent.parent
        files = find_python_scripts(project_root / 'scripts')

    if not files:
        print("No Python scripts found to validate")
        return 0

    print(f"Validating {len(files)} script(s)...")
    print()

    all_valid = True
    invalid_files = []

    for file_path in files:
        print(f"Checking: {file_path}")

        result = validate_script(file_path, verbose=args.verbose)

        if not result['valid']:
            all_valid = False
            invalid_files.append((file_path, result))

            print(f"  ✗ INVALID")

            if result['missing_required']:
                print(f"  Missing required fields: {', '.join(result['missing_required'])}")

            if args.template and not result['docstring']:
                print("\n  Suggested template:")
                print("  " + "=" * 70)
                print(generate_template(file_path))
                print("  " + "=" * 70)

        else:
            print(f"  ✓ VALID")

            if result['missing_recommended']:
                print(f"  Missing recommended fields: {', '.join(result['missing_recommended'])}")

        if result['warnings']:
            for warning in result['warnings']:
                print(f"  ⚠ WARNING: {warning}")

        print()

    # Summary
    if all_valid:
        print(f"✓ All {len(files)} script(s) have proper version information")
        return 0
    else:
        print(f"✗ {len(invalid_files)} script(s) missing version information:")
        for file_path, result in invalid_files:
            print(f"  - {file_path}")
        print()
        print("Run with --template to see suggested docstring format")
        print("See: docs/CLI_ARCHITECTURE.md for versioning guidelines")
        return 1


if __name__ == '__main__':
    sys.exit(main())
