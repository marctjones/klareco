#!/usr/bin/env python3
"""
Test QA Decoder generation quality.

Loads the trained QA Decoder and tests it on sample questions with retrieved context.
"""

import argparse
import json
import sys
import torch
from pathlib import Path
from typing import List, Dict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.parser import parse
from klareco.deparser import deparse
from klareco.models.qa_decoder import QADecoder
from klareco.models.tree_lstm import TreeLSTMEncoder
from klareco.ast_to_graph import ASTToGraphConverter
from klareco.logging_config import setup_logging
import logging

logger = logging.getLogger(__name__)


class QADecoderTester:
    """Test QA Decoder generation."""

    def __init__(self, model_path: Path, vocab_path: Path, gnn_path: Path, device: str = 'cpu'):
        """
        Initialize tester.

        Args:
            model_path: Path to trained QA Decoder
            vocab_path: Path to vocabulary JSON
            gnn_path: Path to GNN encoder checkpoint
            device: Device to use
        """
        self.device = device
        self.converter = ASTToGraphConverter()

        # Load vocabulary
        logger.info(f"Loading vocabulary from {vocab_path}")
        with open(vocab_path, 'r') as f:
            vocab_data = json.load(f)
        self.vocab = vocab_data['token2id']
        self.id_to_token = vocab_data['id2token']
        self.vocab_size = len(self.vocab)
        logger.info(f"Vocabulary size: {self.vocab_size}")

        # Load GNN encoder
        logger.info(f"Loading GNN encoder from {gnn_path}")
        self.gnn_encoder = TreeLSTMEncoder(
            vocab_size=10000,  # From GNN training
            embed_dim=128,
            hidden_dim=256,
            output_dim=512
        )
        checkpoint = torch.load(gnn_path, map_location=device)
        self.gnn_encoder.load_state_dict(checkpoint['model_state_dict'])
        self.gnn_encoder.eval()
        self.gnn_encoder.to(device)
        logger.info("GNN encoder loaded")

        # Load QA Decoder
        logger.info(f"Loading QA Decoder from {model_path}")
        self.qa_decoder = QADecoder(
            vocab_size=self.vocab_size,
            d_model=512,
            n_heads=8,
            n_layers=8,
            d_ff=2048
        )
        checkpoint = torch.load(model_path, map_location=device)
        self.qa_decoder.load_state_dict(checkpoint['model_state_dict'])
        self.qa_decoder.eval()
        self.qa_decoder.to(device)
        logger.info("QA Decoder loaded")

    def ast_to_tokens(self, ast: Dict) -> List[str]:
        """
        Convert AST to token sequence.

        Args:
            ast: Parsed AST

        Returns:
            List of tokens
        """
        tokens = []

        def traverse(node):
            if isinstance(node, dict):
                # Add root
                if 'radiko' in node and node['radiko']:
                    tokens.append(node['radiko'])

                # Add POS
                if 'vortspeco' in node and node['vortspeco']:
                    tokens.append(node['vortspeco'])

                # Add grammatical markers
                if node.get('nombro') == 'pluralo':
                    tokens.append('PLURAL')
                if node.get('kazo') == 'akuzativo':
                    tokens.append('ACC')

                # Traverse children
                for key, value in node.items():
                    if key not in ['radiko', 'vortspeco', 'nombro', 'kazo', 'tipo', 'plena_vorto']:
                        traverse(value)
            elif isinstance(node, list):
                for item in node:
                    traverse(item)

        traverse(ast)
        return tokens

    def tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Convert tokens to IDs."""
        return [self.vocab.get(token, self.vocab.get('<UNK>', 0)) for token in tokens]

    def ids_to_tokens(self, ids: List[int]) -> List[str]:
        """Convert IDs to tokens."""
        return [self.id_to_token.get(str(id), '<UNK>') for id in ids]

    def encode_with_gnn(self, ast: Dict) -> torch.Tensor:
        """
        Encode AST with GNN.

        Args:
            ast: Parsed AST

        Returns:
            Embedding tensor (output_dim,)
        """
        graph = self.converter.ast_to_graph(ast)
        graph.x = graph.x.to(self.device)
        graph.edge_index = graph.edge_index.to(self.device)

        with torch.no_grad():
            embedding = self.gnn_encoder(graph)

        return embedding

    def generate_answer(
        self,
        question_text: str,
        context_texts: List[str],
        max_length: int = 50
    ) -> Dict:
        """
        Generate answer for question with context.

        Args:
            question_text: Question text
            context_texts: List of context sentences
            max_length: Maximum answer length

        Returns:
            Dict with generated tokens and answer text
        """
        logger.info(f"Question: {question_text}")
        logger.info(f"Context: {context_texts}")

        # Parse question
        try:
            question_ast = parse(question_text)
        except Exception as e:
            logger.error(f"Failed to parse question: {e}")
            return {'error': str(e)}

        # Parse context
        context_asts = []
        for ctx_text in context_texts:
            try:
                ctx_ast = parse(ctx_text)
                context_asts.append(ctx_ast)
            except Exception:
                continue

        if not context_asts:
            logger.error("No valid context ASTs")
            return {'error': 'No valid context'}

        # Encode with GNN
        question_emb = self.encode_with_gnn(question_ast)
        context_embs = [self.encode_with_gnn(ast) for ast in context_asts]

        # Stack embeddings
        question_emb = question_emb.unsqueeze(0)  # (1, d_model)
        context_embs = torch.stack(context_embs).unsqueeze(0)  # (1, num_ctx, d_model)

        # Generate
        logger.info("Generating answer...")
        with torch.no_grad():
            generated_ids = self.qa_decoder.generate(
                question_emb,
                context_embs,
                max_length=max_length
            )

        # Convert to tokens
        generated_tokens = self.ids_to_tokens(generated_ids[0].cpu().tolist())

        # Filter special tokens
        filtered_tokens = [t for t in generated_tokens if t not in ['<PAD>', '<UNK>', '<SOS>', '<EOS>']]

        logger.info(f"Generated tokens: {filtered_tokens}")

        return {
            'question': question_text,
            'context': context_texts,
            'generated_tokens': filtered_tokens,
            'num_tokens': len(filtered_tokens)
        }


def main():
    """Test QA Decoder."""
    parser = argparse.ArgumentParser(description='Test QA Decoder generation')
    parser.add_argument('--model', type=str, default='models/qa_decoder/best_model.pt',
                        help='Path to trained QA Decoder')
    parser.add_argument('--vocab', type=str, default='models/qa_decoder/vocabulary.json',
                        help='Path to vocabulary')
    parser.add_argument('--gnn', type=str, default='models/tree_lstm/checkpoint_epoch_20.pt',
                        help='Path to GNN checkpoint')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    args = parser.parse_args()

    # Setup logging
    setup_logging(level=logging.DEBUG if args.debug else logging.INFO)

    logger.info("=" * 70)
    logger.info("QA DECODER GENERATION TEST")
    logger.info("=" * 70)

    # Initialize tester
    tester = QADecoderTester(
        model_path=Path(args.model),
        vocab_path=Path(args.vocab),
        gnn_path=Path(args.gnn),
        device=args.device
    )

    # Test cases
    test_cases = [
        {
            'question': 'Kiu estas Frodo?',
            'context': [
                'Frodo estas hobito.',
                'Li loĝas en Hobbiton.',
                'Frodo havas ringon.'
            ]
        },
        {
            'question': 'Kie Frodo loĝas?',
            'context': [
                'Frodo loĝas en Hobbiton.',
                'Hobbiton estas vilaĝo.',
                'La vilaĝo troviĝas en Ŝiro.'
            ]
        },
        {
            'question': 'Kio estas Esperanto?',
            'context': [
                'Esperanto estas planlingvo.',
                'Zamenhof kreis Esperanton.',
                'La lingvo havas regulan gramatikon.'
            ]
        }
    ]

    logger.info("")
    logger.info("Running test cases...")
    logger.info("")

    results = []
    for i, test_case in enumerate(test_cases, 1):
        logger.info(f"Test case {i}/{len(test_cases)}")
        logger.info("-" * 70)

        result = tester.generate_answer(
            question_text=test_case['question'],
            context_texts=test_case['context'],
            max_length=30
        )

        results.append(result)

        logger.info(f"Result: {result.get('generated_tokens', 'ERROR')}")
        logger.info("")

    # Summary
    logger.info("=" * 70)
    logger.info("TEST COMPLETE")
    logger.info("=" * 70)

    successful = sum(1 for r in results if 'error' not in r)
    logger.info(f"Successful: {successful}/{len(results)}")

    for i, result in enumerate(results, 1):
        logger.info(f"\nTest {i}:")
        logger.info(f"  Question: {result.get('question', 'N/A')}")
        if 'error' in result:
            logger.info(f"  Error: {result['error']}")
        else:
            logger.info(f"  Generated: {' '.join(result['generated_tokens'][:10])}...")
            logger.info(f"  Tokens: {result['num_tokens']}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
