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

def generate_question_from_ast(ast_node):
    """
    Analyzes an AST and generates a rule-based question.
    Returns the question string.
    """
    if "OBJ" in str(ast_node):
        return "Kion faris la subjekto?"
    if "SUBJ" in str(ast_node):
        return "Kiu faris tion?"
    return "Kio okazis?"

def create_training_examples(paragraph: str, converter: ASTToGraphConverter):
    """
    From a paragraph of text, create one or more training examples.
    Yields each valid example.
    """
    sentences = [s.strip() for s in paragraph.split('.') if len(s.strip()) > 3]
    if len(sentences) < 2:
        return

    for i, target_sentence in enumerate(sentences):
        context_sentences = sentences[:i] + sentences[i+1:]
        if not context_sentences:
            continue
        
        context_text = ". ".join(context_sentences) + "."

        try:
            target_ast = parse(target_sentence)
            question_text = generate_question_from_ast(target_ast)
            question_ast = parse(question_text)
            context_ast = parse(context_text)
            
            question_graph = converter.ast_to_graph(question_ast).to_dict()
            context_graph = converter.ast_to_graph(context_ast).to_dict()

            yield {
                "question_graph": question_graph,
                "context_graph": context_graph,
                "target_text": target_sentence
            }
        except Exception as e:
            logging.debug(f"Could not process sentence pair in '{paragraph}': {e}")
            continue


def process_corpus(corpus_dir: Path, output_file: Path, max_examples: int):
    """
    Process all text files in the corpus directory and generate the dataset.
    """
    converter = ASTToGraphConverter()
    
    # Check for existing examples to make the script resumable
    existing_examples = 0
    if output_file.exists():
        with open(output_file, 'r', encoding='utf-8') as f:
            existing_examples = sum(1 for line in f)
        logging.info(f"Resuming. Found {existing_examples} existing examples in output file.")

    example_count = existing_examples

    if example_count >= max_examples:
        logging.info("Max examples limit already met or exceeded by existing file. Exiting.")
        return

    with open(output_file, 'a', encoding='utf-8') as f_out:
        # Use sorted() to ensure deterministic order for reproducibility
        for text_file in sorted(corpus_dir.glob("*.txt")):
            if example_count >= max_examples:
                logging.info(f"Reached max examples limit of {max_examples}.")
                break
            
            logging.info(f"Processing file: {text_file.name}")
            with open(text_file, 'r', encoding='utf-8') as f_in:
                content = f_in.read()
                paragraphs = content.split('\n\n')

                for para in paragraphs:
                    if example_count >= max_examples:
                        break
                    
                    para = para.strip().replace('\n', ' ')
                    if not para or len(para) < 20:
                        continue

                    for example in create_training_examples(para, converter):
                        if example_count >= max_examples:
                            break
                        f_out.write(json.dumps(example) + '\n')
                        example_count += 1
                        if example_count % 100 == 0:
                            logging.info(f"Generated {example_count} total examples...")
                        
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

    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    logging.info("Starting dataset generation...")
    logging.info(f"Corpus directory: {args.corpus_dir}")
    logging.info(f"Output file: {args.output_file}")
    logging.info(f"Max examples: {args.max_examples}")

    process_corpus(args.corpus_dir, args.output_file, args.max_examples)


if __name__ == "__main__":
    main()