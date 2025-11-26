#!/usr/bin/env python3
"""
Generate a self-supervised dataset for training the Graph2Seq generator.

This script reads from the Esperanto corpus, and for each paragraph,
it creates multiple training examples. Each example consists of:
- A "context" (a set of sentences)
- A "target" (a single sentence from that context)
- A "question" (a synthetically generated question about the target)

The output is a JSONL file where each line is a dictionary containing the
AST-graphs for the question, context, and the raw text of the target.
"""

import argparse
import logging
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.parser import parse
from klareco.ast_to_graph import ASTToGraphConverter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_training_example(paragraph: str, converter: ASTToGraphConverter):
    """
    From a paragraph of text, create one or more training examples.

    This is a placeholder for the core logic.
    """
    # TODO: Implement sentence splitting.
    sentences = [s.strip() for s in paragraph.split('.') if s.strip()]
    if not sentences:
        return None

    # For now, let's just use a simple example logic.
    # The real logic will be more sophisticated.
    if len(sentences) < 2:
        return None

    target_sentence = sentences[-1]
    context_sentences = sentences[:-1]
    context_text = ". ".join(context_sentences) + "."

    # TODO: Implement the rule-based T2Q logic.
    # For now, use a placeholder question.
    placeholder_question = "Kio estas la resumo?" # "What is the summary?"

    try:
        # Parse all components into ASTs
        question_ast = parse(placeholder_question)
        context_ast = parse(context_text)
        
        # Convert ASTs to graphs
        question_graph = converter.ast_to_graph(question_ast).to_dict()
        context_graph = converter.ast_to_graph(context_ast).to_dict()

        return {
            "question_graph": question_graph,
            "context_graph": context_graph,
            "target_text": target_sentence
        }
    except Exception as e:
        logging.error(f"Failed to process paragraph: {e}")
        return None


def process_corpus(corpus_dir: Path, output_file: Path, max_examples: int):
    """
    Process all text files in the corpus directory and generate the dataset.
    """
    converter = ASTToGraphConverter()
    example_count = 0

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for text_file in corpus_dir.glob("*.txt"):
            if example_count >= max_examples:
                logging.info(f"Reached max examples limit of {max_examples}.")
                break
            
            logging.info(f"Processing file: {text_file.name}")
            with open(text_file, 'r', encoding='utf-8') as f_in:
                # We'll treat paragraphs as text separated by double newlines
                paragraphs = f_in.read().split('\n\n')
                for para in paragraphs:
                    if example_count >= max_examples:
                        break
                    
                    para = para.strip().replace('\n', ' ')
                    if not para:
                        continue

                    example = create_training_example(para, converter)
                    if example:
                        f_out.write(json.dumps(example) + '\n')
                        example_count += 1
                        if example_count % 100 == 0:
                            logging.info(f"Generated {example_count} examples...")

    logging.info(f"Done. Total examples generated: {example_count}")


def main():
    parser = argparse.ArgumentParser(description="Generate a self-supervised dataset for the Graph2Seq model.")
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=Path("data/clean_corpus"),
        help="Directory containing the cleaned Esperanto corpus files."
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("data/training_pairs/synthesis_dataset.jsonl"),
        help="Path to save the generated JSONL dataset."
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=10000,
        help="Maximum number of training examples to generate."
    )
    args = parser.parse_args()

    # Ensure output directory exists
    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    logging.info("Starting dataset generation...")
    logging.info(f"Corpus directory: {args.corpus_dir}")
    logging.info(f"Output file: {args.output_file}")
    logging.info(f"Max examples: {args.max_examples}")

    process_corpus(args.corpus_dir, args.output_file, args.max_examples)


if __name__ == "__main__":
    main()
