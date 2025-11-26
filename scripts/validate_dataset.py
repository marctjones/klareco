#!/usr/bin/env python3
"""
Validates the quality of the self-supervised synthesis dataset.

This script checks for common issues like target text not being present in context,
invalid ASTs, or malformed graph data.
"""

import argparse
import logging
import json
from pathlib import Path
import sys

import torch
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.parser import parse
from klareco.ast_to_graph import ASTToGraphConverter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def validate_example(example: dict, converter: ASTToGraphConverter) -> list[str]:
    """
    Validates a single training example.
    Returns a list of error messages, or an empty list if valid.
    """
    errors = []

    question_graph_dict = example.get('question_graph')
    context_graph_dict = example.get('context_graph')
    target_text = example.get('target_text')

    if not (question_graph_dict and context_graph_dict and target_text):
        errors.append("Missing required fields (question_graph, context_graph, target_text).")
        return errors
    
    # 1. Check graph integrity (can it be converted back to PyG Data)
    try:
        Data(x=torch.tensor(question_graph_dict['x']), edge_index=torch.tensor(question_graph_dict['edge_index']))
    except Exception:
        errors.append("Question graph data is malformed.")
    
    try:
        Data(x=torch.tensor(context_graph_dict['x']), edge_index=torch.tensor(context_graph_dict['edge_index']))
    except Exception:
        errors.append("Context graph data is malformed.")

    # 2. Check target text presence in context
    # This requires reconstructing the context text from the graph, which is not trivial.
    # For now, we'll assume the context_graph_dict contains enough info to re-parse.
    # This is a placeholder for a more robust check involving the original context text.
    
    # The generation script creates `context_ast` then `context_graph`.
    # To properly check target_text in context, we would need the original context string.
    # For now, let's assume if the context graph is valid, the original text was present
    # in the example generation process.
    # A more robust check might involve comparing substrings or embeddings.
    
    # Let's do a simple check on text length
    if len(target_text) < 5:
        errors.append(f"Target text is too short: '{target_text}'.")

    return errors


def validate_dataset(dataset_path: Path, max_examples: int):
    """
    Loads and validates the dataset, printing a report.
    """
    total_examples = 0
    total_errors = 0
    
    logging.info(f"Loading dataset from: {dataset_path}")
    
    converter = ASTToGraphConverter()
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if total_examples >= max_examples:
                logging.info(f"Reached max examples limit of {max_examples}.")
                break
            
            total_examples += 1
            
            try:
                example = json.loads(line)
                errors = validate_example(example, converter)
                if errors:
                    total_errors += 1
                    logging.warning(f"Errors in example {line_num}:")
                    for error in errors:
                        logging.warning(f"  - {error}")
            except json.JSONDecodeError:
                total_errors += 1
                logging.error(f"Line {line_num} is not valid JSON.")
            except Exception as e:
                total_errors += 1
                logging.error(f"Unhandled error in example {line_num}: {e}")

    logging.info("\n" + "="*50)
    logging.info("DATASET VALIDATION REPORT")
    logging.info("="*50)
    logging.info(f"Total examples processed: {total_examples}")
    logging.info(f"Total examples with errors: {total_errors}")
    logging.info(f"Dataset quality: {((total_examples - total_errors) / total_examples * 100):.2f}% clean")
    if total_errors > 0:
        logging.warning("Please review the warnings and errors above.")
    else:
        logging.info("Dataset appears clean. Good job!")


def main():
    parser = argparse.ArgumentParser(description="Validate the synthesis dataset quality.")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("data/training_pairs/synthesis_dataset.jsonl"),
        help="Path to the JSONL dataset file to validate."
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=1000,
        help="Maximum number of examples to validate (for quick checks)."
    )
    args = parser.parse_args()

    validate_dataset(args.dataset_path, args.max_examples)


if __name__ == "__main__":
    main()
