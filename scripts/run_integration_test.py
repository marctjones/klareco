"""
Flexible Integration Test Runner for the Klareco Pipeline.

This script runs sentences from a test corpus through the pipeline and can
stop at a specified step for targeted testing.
"""
import os
import sys
import json
import argparse

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from klareco.pipeline import KlarecoPipeline
from tqdm import tqdm

def run_integration_test(corpus_path: str, num_sentences: int = None, stop_after: str = None):
    """
    Runs the integration test on the specified corpus.
    """
    print("--- Klareco Integration Test Runner ---")
    
    # Load test corpus
    try:
        with open(corpus_path, 'r', encoding='utf-8') as f:
            test_corpus = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Test corpus not found at {corpus_path}")
        return

    # Determine number of sentences to test
    if num_sentences is not None:
        sentences_to_test = test_corpus[:num_sentences]
    else:
        sentences_to_test = test_corpus

    print(f"Test corpus loaded. Running {len(sentences_to_test)} sentences.")
    if stop_after:
        print(f"Pipeline will stop after step: '{stop_after}'")
    print("-" * 30)

    pipeline = KlarecoPipeline()
    
    for i, sentence in enumerate(tqdm(sentences_to_test, desc="Testing sentences")):
        trace = pipeline.run(sentence, stop_after=stop_after)
        
        if trace.error:
            print(f"\n--- FAILED on sentence {i+1}: '{sentence}' ---")
            print(trace.to_json())
            print("-" * 30)
            # Optionally, stop on first failure
            # break 
    
    print("\nIntegration test run complete.")

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
    
    args = parser.parse_args()
    
    run_integration_test(
        corpus_path=args.corpus,
        num_sentences=args.num_sentences,
        stop_after=args.stop_after
    )
