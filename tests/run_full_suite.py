#!/usr/bin/env python3
"""
Klareco Full Test Suite Runner

Runs all four test categories in order:
1. Code Tests - Unit tests for implementation correctness
2. Data Quality Tests - Validate training data quality
3. Model Quality Tests - Measure trained model performance
4. Integration Tests - End-to-end pipeline validation

Usage:
    python tests/run_full_suite.py                    # Run all tests
    python tests/run_full_suite.py --fast             # Skip slow tests
    python tests/run_full_suite.py --code-only        # Only code tests
    python tests/run_full_suite.py --models-only      # Only model quality tests
    python tests/run_full_suite.py --coverage         # Generate coverage report
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


class TestSuite:
    """Test suite manager."""

    def __init__(self, args):
        self.args = args
        self.project_root = Path(__file__).parent.parent
        self.failed_categories = []

    def run_category(self, name: str, pytest_args: List[str]) -> bool:
        """Run a test category and return True if passed."""
        print(f"\n{'=' * 80}")
        print(f"CATEGORY: {name}")
        print(f"{'=' * 80}\n")

        cmd = ['python', '-m', 'pytest'] + pytest_args

        if self.args.verbose:
            cmd.append('-v')

        if self.args.fast:
            cmd.extend(['-m', 'not slow'])

        if self.args.coverage and 'code' in name.lower():
            cmd.extend(['--cov=klareco', '--cov-report=html', '--cov-report=term'])

        print(f"Running: {' '.join(cmd)}\n")

        result = subprocess.run(cmd, cwd=self.project_root)

        if result.returncode != 0:
            self.failed_categories.append(name)
            print(f"\n❌ {name} FAILED")
            return False

        print(f"\n✅ {name} PASSED")
        return True

    def run_code_tests(self) -> bool:
        """Run code unit tests."""
        return self.run_category(
            "1. Code Tests (Unit Tests)",
            [
                'tests/test_parser.py',
                'tests/test_deparser.py',
                'tests/test_embeddings.py',
                'tests/test_ast_to_graph.py',
                '--tb=short'
            ]
        )

    def run_data_quality_tests(self) -> bool:
        """Run data quality tests."""
        data_quality_test = self.project_root / 'tests' / 'test_data_quality.py'

        if not data_quality_test.exists():
            print(f"\n⚠️  Skipping Data Quality Tests - {data_quality_test} not found")
            return True

        return self.run_category(
            "2. Data Quality Tests",
            [
                'tests/test_data_quality.py',
                '--tb=short'
            ]
        )

    def run_model_quality_tests(self) -> bool:
        """Run model quality tests."""
        return self.run_category(
            "3. Model Quality Tests",
            [
                'tests/test_stage1_model_quality.py',
                'tests/test_m1_model_quality.py',
                '--tb=short'
            ]
        )

    def run_integration_tests(self) -> bool:
        """Run integration tests."""
        integration_test = self.project_root / 'tests' / 'test_integration.py'

        if not integration_test.exists():
            print(f"\n⚠️  Skipping Integration Tests - {integration_test} not found")
            return True

        return self.run_category(
            "4. Integration Tests",
            [
                'tests/test_integration.py',
                '--tb=short'
            ]
        )

    def run_all(self):
        """Run all test categories in order."""
        print("\n" + "=" * 80)
        print("KLARECO FULL TEST SUITE")
        print("=" * 80)

        if self.args.code_only:
            self.run_code_tests()
        elif self.args.models_only:
            self.run_model_quality_tests()
        else:
            # Run all categories in order
            self.run_code_tests()

            if not self.args.skip_data:
                self.run_data_quality_tests()

            if not self.args.skip_models:
                self.run_model_quality_tests()

            if not self.args.skip_integration:
                self.run_integration_tests()

        # Summary
        print("\n" + "=" * 80)
        print("TEST SUITE SUMMARY")
        print("=" * 80)

        if self.failed_categories:
            print(f"\n❌ FAILED CATEGORIES: {', '.join(self.failed_categories)}")
            print(f"\nTotal failures: {len(self.failed_categories)}")
            return 1
        else:
            print("\n✅ ALL TESTS PASSED")
            return 0


def main():
    parser = argparse.ArgumentParser(description='Run Klareco full test suite')

    # Test selection
    parser.add_argument('--code-only', action='store_true',
                        help='Run only code unit tests')
    parser.add_argument('--models-only', action='store_true',
                        help='Run only model quality tests')
    parser.add_argument('--skip-data', action='store_true',
                        help='Skip data quality tests')
    parser.add_argument('--skip-models', action='store_true',
                        help='Skip model quality tests')
    parser.add_argument('--skip-integration', action='store_true',
                        help='Skip integration tests')

    # Test options
    parser.add_argument('--fast', action='store_true',
                        help='Skip slow tests')
    parser.add_argument('--coverage', action='store_true',
                        help='Generate coverage report for code tests')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose output')

    args = parser.parse_args()

    suite = TestSuite(args)
    sys.exit(suite.run_all())


if __name__ == '__main__':
    main()
