#!/bin/bash
# This script activates the conda environment, runs the Klareco integration tests
# with coverage, and then generates a coverage report.

# Activate conda environment
source /home/marc/miniconda3/bin/activate klareco-env

# Run the integration test script with coverage using python -m
python -m coverage run --source=klareco scripts/run_integration_test.py "$@" > run_output.txt 2>&1

# Generate the coverage report
python -m coverage report -m >> run_output.txt 2>&1