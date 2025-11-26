#!/usr/bin/env python3
"""
Evaluates the end-to-end pipeline against a formal test set.

This script runs each question from the test set through the pipeline,
captures the generated answer, and computes NLP metrics (BLEU and ROUGE)
by comparing it to the ideal answer.
"""

import argparse
import logging
from pathlib import Path
import sys
import json
import torch

# This script requires NLTK and rouge-score, prompt the user to install
try:
    from nltk.translate.bleu_score import sentence_bleu
    from rouge_score import rouge_scorer
except ImportError:
    print("This script requires 'nltk' and 'rouge-score'.")
    print("Please install them using: pip install nltk rouge-score")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import necessary components from other scripts
from scripts.run_pipeline import run_pipeline
from klareco.parser import parse
from klareco.ast_to_graph import ASTToGraphConverter
from klareco.rag.retriever import create_retriever
from klareco.models.generator import Graph2SeqGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def evaluate(model, retriever, converter, test_data, device):
    """
    Runs the evaluation loop.
    """
    scores = {'bleu': [], 'rouge1': [], 'rougeL': []}
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

    for item in test_data:
        question = item['question']
        ideal_answer = item['ideal_answer']
        
        logging.info(f"Evaluating question: {question}")

        # Since run_pipeline prints the output, we need to capture it.
        # A better implementation would have run_pipeline return the value.
        # For now, we know the model is not trained, so we'll use a placeholder.
        generated_answer = "(Model output is not implemented yet as the model is not trained)"
        
        # --- When the model is trained, the following lines would be used ---
        # from io import StringIO
        # old_stdout = sys.stdout
        # sys.stdout = captured_output = StringIO()
        # run_pipeline(question, retriever, model, converter, device)
        # sys.stdout = old_stdout
        # generated_answer = captured_output.getvalue().strip()
        # --------------------------------------------------------------------

        logging.info(f"  -> Ideal: {ideal_answer}")
        logging.info(f"  -> Generated: {generated_answer}")

        # Calculate BLEU score
        # NLTK expects tokenized input
        reference = [ideal_answer.split()]
        candidate = generated_answer.split()
        bleu = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))
        scores['bleu'].append(bleu)

        # Calculate ROUGE scores
        rouge = scorer.score(ideal_answer, generated_answer)
        scores['rouge1'].append(rouge['rouge1'].fmeasure)
        scores['rougeL'].append(rouge['rougeL'].fmeasure)
        
        logging.info(f"  -> Scores: BLEU={bleu:.3f}, ROUGE-1={rouge['rouge1'].fmeasure:.3f}, ROUGE-L={rouge['rougeL'].fmeasure:.3f}")

    # Calculate average scores
    avg_bleu = sum(scores['bleu']) / len(scores['bleu'])
    avg_rouge1 = sum(scores['rouge1']) / len(scores['rouge1'])
    avg_rougeL = sum(scores['rougeL']) / len(scores['rougeL'])

    print("\n" + "="*50)
    print("EVALUATION COMPLETE")
    print("="*50)
    print(f"Average BLEU: {avg_bleu:.4f}")
    print(f"Average ROUGE-1 F-measure: {avg_rouge1:.4f}")
    print(f"Average ROUGE-L F-measure: {avg_rougeL:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate the Klareco pipeline.")
    parser.add_argument(
        "--test-set",
        type=Path,
        default=Path("data/test_set/evaluation.jsonl"),
        help="Path to the evaluation JSONL file."
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load components (same as run_pipeline) ---
    logging.info("Loading components for evaluation...")
    converter = ASTToGraphConverter()
    retriever = create_retriever()
    vocab_size = 1000  # Placeholder
    model = Graph2SeqGenerator(
        vocab_size=vocab_size,
        embed_dim=128,
        hidden_dim=256,
        gnn_out_dim=256
    ).to(device)
    model.eval()

    # --- Load test data ---
    test_data = []
    with open(args.test_set, 'r', encoding='utf-8') as f:
        for line in f:
            test_data.append(json.loads(line))
    
    logging.info(f"Loaded {len(test_data)} evaluation examples.")

    # --- Run evaluation ---
    evaluate(model, retriever, converter, test_data, device)


if __name__ == "__main__":
    main()
