"""
Flexible Integration Test Runner for the Klareco Pipeline.

This script runs sentences from a test corpus through the pipeline and can
stop at a specified step for targeted testing.
"""
import os
import sys
import json
import argparse
import logging

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from klareco.pipeline import KlarecoPipeline
from klareco.logging_config import setup_logging, ProgressLogger
from tqdm import tqdm  # Keep for console progress bar

def run_integration_test(corpus_path: str, num_sentences: int = None, stop_after: str = None, debug: bool = False):
    """
    Runs the integration test on the specified corpus.

    Args:
        corpus_path: Path to test corpus JSON
        num_sentences: Limit test to N sentences (None = all)
        stop_after: Stop pipeline at specific step
        debug: Enable debug-level logging with context
    """
    # Setup logging with debug mode if requested
    setup_logging(debug=debug)

    logging.info("--- Klareco Integration Test Runner ---")
    logging.info(f"Attempting to load test corpus from: {corpus_path}")
    
    # Load test corpus
    try:
        with open(corpus_path, 'r', encoding='utf-8') as f:
            test_corpus = json.load(f)
        logging.info(f"Successfully loaded test corpus with {len(test_corpus)} sentences.")
    except FileNotFoundError:
        logging.error(f"ERROR: Test corpus not found at {corpus_path}")
        print(f"ERROR: Test corpus not found at {corpus_path}")
        return
    except json.JSONDecodeError:
        logging.error(f"ERROR: Could not decode JSON from {corpus_path}. Is it a valid JSON file?")
        print(f"ERROR: Could not decode JSON from {corpus_path}. Is it a valid JSON file?")
        return

    # Determine number of sentences to test
    if num_sentences is not None:
        sentences_to_test = test_corpus[:num_sentences]
        logging.info(f"Limiting test to {num_sentences} sentences.")
    else:
        sentences_to_test = test_corpus

    print(f"Test corpus loaded. Running {len(sentences_to_test)} sentences.")
    if stop_after:
        print(f"Pipeline will stop after step: '{stop_after}'")
        logging.info(f"Pipeline configured to stop after step: '{stop_after}'")
    print("-" * 30)

    pipeline = KlarecoPipeline()

    # Create dual progress tracking: tqdm for console, ProgressLogger for log file
    progress_log = ProgressLogger(total=len(sentences_to_test), desc="Testing sentences")
    tqdm_bar = tqdm(sentences_to_test, desc="Testing sentences")

    passed = 0
    failed = 0

    for i, sentence in enumerate(tqdm_bar):
        # Truncate long sentences for display
        display_sentence = sentence[:50] + "..." if len(sentence) > 50 else sentence

        # Update progress loggers
        progress_log.update(1, item_desc=f"Sentence {i+1}: {display_sentence}")

        logging.info(f"Running pipeline for sentence {i+1}/{len(sentences_to_test)}")
        if debug:
            logging.debug(f"Full input: {sentence}")

        trace = pipeline.run(sentence, stop_after=stop_after)

        if trace.error:
            failed += 1
            logging.error(f"FAILED sentence {i+1}: {trace.error[:100]}")
            if debug:
                logging.debug(f"Full error: {trace.error}")
                logging.debug(f"Trace: {trace.to_json()[:500]}...")
            print(f"\n--- FAILED on sentence {i+1}: '{display_sentence}' ---")
            print(trace.to_json())
            print("-" * 30)
            # Optionally, stop on first failure
            # break
        else:
            passed += 1
            logging.info(f"PASSED sentence {i+1}")

    progress_log.close()
    tqdm_bar.close()

    # Summary
    logging.info("=" * 80)
    logging.info(f"Integration test COMPLETE: {passed} passed, {failed} failed out of {len(sentences_to_test)}")
    logging.info("=" * 80)
    print(f"\nIntegration test complete: {passed} passed, {failed} failed out of {len(sentences_to_test)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Klareco Integration Test Runner")
    parser.add_argument(
        '--corpus',
        type=str,
        default='data/test_corpus.json',
        help='Path to the test corpus JSON file.'
    )
    parser.add_argument(
        '--num-sentences',
        type=int,
        default=None,
        help='Number of sentences from the corpus to test. (Default: all)'
    )
    parser.add_argument(
        '--stop-after',
        type=str,
        default=None,
        choices=['SafetyMonitor', 'FrontDoor', 'Parser', 'SafetyMonitor_AST', 'IntentClassifier', 'Responder'],
        help='The pipeline step to stop after.'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug-level logging with full context (inputs, AST state, stack traces)'
    )

    args = parser.parse_args()

    run_integration_test(
        corpus_path=args.corpus,
        num_sentences=args.num_sentences,
        stop_after=args.stop_after,
        debug=args.debug
    )
