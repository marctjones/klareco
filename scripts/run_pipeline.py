#!/usr/bin/env python3
"""
The main end-to-end pipeline for Klareco.

This script takes a question, retrieves context, and uses the Graph2Seq
generator to synthesize an answer.
"""

import argparse
import logging
from pathlib import Path
import sys
import torch
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.parser import parse
from klareco.ast_to_graph import ASTToGraphConverter
from klareco.rag.retriever import create_retriever
from klareco.models.generator import Graph2SeqGenerator
from klareco.deparser import deparse_from_tokens

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def combine_graphs(question_graph: Data, context_graphs: list[Data]) -> Data:
    """
    Combines a question graph and a list of context graphs into a single graph.
    
    This is a complex operation. A simple placeholder is implemented here.
    A robust implementation would need to handle node and edge re-indexing
    and add segment IDs to distinguish nodes from different graphs.
    """
    # For now, we will placeholder this by just returning the question graph.
    # This is incorrect but allows the pipeline to run end-to-end.
    logging.warning("Graph combination is a placeholder. Only using question graph.")
    return question_graph


def run_pipeline(question: str, retriever, model, converter, device):
    """
    Executes the full RAG + generation pipeline.
    """
    logging.info("1. Parsing question...")
    question_ast = parse(question)
    question_graph = converter.ast_to_graph(question_ast)

    logging.info("2. Retrieving context...")
    # The retriever should be modified to work with AST-graphs.
    # For now, we assume it can take an AST.
    context_docs = retriever.retrieve_hybrid(question_ast, k=5)
    
    context_graphs = []
    for doc in context_docs:
        try:
            ctx_ast = parse(doc.get('text', ''))
            ctx_graph = converter.ast_to_graph(ctx_ast)
            context_graphs.append(ctx_graph)
        except Exception:
            continue
    
    logging.info(f"Retrieved {len(context_graphs)} valid context documents.")

    logging.info("3. Combining graphs...")
    reasoning_graph = combine_graphs(question_graph, context_graphs)
    reasoning_graph = reasoning_graph.to(device)

    logging.info("4. Generating answer...")
    # The model is not trained, so the output will be random.
    with torch.no_grad():
        output_tensor = model(reasoning_graph)

    # Placeholder for token-to-text deparsing
    # output_tokens = [vocab.id_to_token[idx] for idx in output_tensor[0].argmax(dim=-1).tolist()]
    # answer = deparse_from_tokens(output_tokens)
    
    answer = "(Model output is not implemented yet as the model is not trained)"

    logging.info("5. Synthesizing Final Answer...")
    # The deparsing step will be called here once the model is trained.
    # For now, we return a placeholder.
    answer = "[Placeholder: Model is not trained. No answer generated.]"

    return answer


def main():
    parser = argparse.ArgumentParser(description="Run the full Klareco pipeline.")
    parser.add_argument("question", type=str, help="The question to ask in Esperanto.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # --- Load components ---
    logging.info("Loading components...")
    converter = ASTToGraphConverter()
    retriever = create_retriever()
    
    # TODO: Load a vocabulary and a trained model checkpoint.
    # Using placeholder values for now.
    vocab_size = 1000 
    model = Graph2SeqGenerator(
        vocab_size=vocab_size,
        embed_dim=128,
        hidden_dim=256,
        gnn_out_dim=256
    ).to(device)
    model.eval()
    logging.info("Components loaded.")
    
    # --- Run pipeline ---
    final_answer = run_pipeline(args.question, retriever, model, converter, device)
    
    print("\n--- Klareco Respondo ---")
    print(final_answer)
    print("------------------------")


if __name__ == "__main__":
    main()
