#!/usr/bin/env python3
"""
End-to-end QA pipeline test.

Tests the full flow: Question → Parse → RAG → GNN → QA Decoder → Answer
"""

import argparse
import json
import sys
import torch
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.parser import parse
from klareco.deparser import deparse
from klareco.models.qa_decoder import QADecoder
from klareco.models.tree_lstm import TreeLSTMEncoder
from klareco.ast_to_graph import ASTToGraphConverter
from klareco.rag.retriever import KlarecoRetriever
from klareco.logging_config import setup_logging
import logging

logger = logging.getLogger(__name__)


class EndToEndQASystem:
    """Complete QA system with RAG + GNN + Decoder."""

    def __init__(
        self,
        qa_model_path: Path,
        vocab_path: Path,
        gnn_path: Path,
        index_path: Path,
        device: str = 'cpu'
    ):
        """Initialize end-to-end QA system."""
        self.device = device
        self.converter = ASTToGraphConverter()

        # Load vocabulary
        logger.info(f"Loading vocabulary from {vocab_path}")
        with open(vocab_path, 'r') as f:
            vocab_data = json.load(f)
        self.vocab = vocab_data['token2id']
        self.id_to_token = vocab_data['id2token']
        self.vocab_size = len(self.vocab)

        # Load GNN encoder
        logger.info(f"Loading GNN encoder from {gnn_path}")
        self.gnn_encoder = TreeLSTMEncoder(
            vocab_size=10000,
            embed_dim=128,
            hidden_dim=256,
            output_dim=512
        )
        checkpoint = torch.load(gnn_path, map_location=device)
        self.gnn_encoder.load_state_dict(checkpoint['model_state_dict'])
        self.gnn_encoder.eval()
        self.gnn_encoder.to(device)

        # Load QA Decoder
        logger.info(f"Loading QA Decoder from {qa_model_path}")
        self.qa_decoder = QADecoder(
            vocab_size=self.vocab_size,
            d_model=512,
            n_heads=8,
            n_layers=8,
            d_ff=2048
        )
        checkpoint = torch.load(qa_model_path, map_location=device)
        self.qa_decoder.load_state_dict(checkpoint['model_state_dict'])
        self.qa_decoder.eval()
        self.qa_decoder.to(device)

        # Load RAG retriever
        logger.info(f"Loading RAG index from {index_path}")
        self.retriever = KlarecoRetriever(
            index_dir=str(index_path),
            model_path=str(gnn_path),
            mode='tree_lstm',
            device=device
        )
        logger.info("End-to-end QA system initialized")

    def answer_question(
        self,
        question_text: str,
        top_k: int = 3,
        max_length: int = 30
    ) -> Dict:
        """
        Answer a question end-to-end.

        Args:
            question_text: Question text
            top_k: Number of context documents to retrieve
            max_length: Max answer length

        Returns:
            Dict with answer and intermediate results
        """
        logger.info(f"Question: {question_text}")

        # Step 1: Parse question
        try:
            question_ast = parse(question_text)
            logger.info("✓ Parsed question to AST")
        except Exception as e:
            logger.error(f"Failed to parse question: {e}")
            return {'error': f'Parse failed: {e}'}

        # Step 2: Retrieve context with RAG
        try:
            results = self.retriever.retrieve(question_text, k=top_k)
            context_texts = [r['text'] for r in results]
            logger.info(f"✓ Retrieved {len(context_texts)} context sentences")
        except Exception as e:
            logger.error(f"RAG retrieval failed: {e}")
            return {'error': f'RAG failed: {e}'}

        if not context_texts:
            return {'error': 'No context retrieved'}

        # Step 3: Parse context
        context_asts = []
        for ctx_text in context_texts:
            try:
                ctx_ast = parse(ctx_text)
                context_asts.append(ctx_ast)
            except Exception:
                continue

        if not context_asts:
            return {'error': 'No valid context ASTs'}

        logger.info(f"✓ Parsed {len(context_asts)} context ASTs")

        # Step 4: Encode with GNN
        try:
            question_emb = self._encode_ast(question_ast)
            context_embs = [self._encode_ast(ast) for ast in context_asts]

            question_emb = question_emb.unsqueeze(0)
            context_embs = torch.stack(context_embs).unsqueeze(0)
            logger.info("✓ Encoded question and context with GNN")
        except Exception as e:
            logger.error(f"GNN encoding failed: {e}")
            return {'error': f'GNN failed: {e}'}

        # Step 5: Generate answer with QA Decoder
        try:
            with torch.no_grad():
                generated_ids = self.qa_decoder.generate(
                    question_emb,
                    context_embs,
                    max_length=max_length
                )

            generated_tokens = [self.id_to_token.get(str(id), '<UNK>')
                              for id in generated_ids[0].cpu().tolist()]

            # Filter special tokens
            filtered_tokens = [t for t in generated_tokens
                             if t not in ['<PAD>', '<END>']][:10]  # Limit to first 10 real tokens

            logger.info(f"✓ Generated answer tokens: {filtered_tokens}")
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return {'error': f'Generation failed: {e}'}

        return {
            'question': question_text,
            'context': context_texts,
            'generated_tokens': filtered_tokens,
            'status': 'success'
        }

    def _encode_ast(self, ast: Dict) -> torch.Tensor:
        """Encode AST with GNN."""
        graph = self.converter.ast_to_graph(ast)
        graph.x = graph.x.to(self.device)
        graph.edge_index = graph.edge_index.to(self.device)

        with torch.no_grad():
            embedding = self.gnn_encoder(graph)

        return embedding


def main():
    """Run end-to-end QA tests."""
    parser = argparse.ArgumentParser(description='Test end-to-end QA pipeline')
    parser.add_argument('--qa-model', type=str, default='models/qa_decoder/best_model.pt')
    parser.add_argument('--vocab', type=str, default='models/qa_decoder/vocabulary.json')
    parser.add_argument('--gnn', type=str, default='models/tree_lstm/checkpoint_epoch_20.pt')
    parser.add_argument('--index', type=str, default='data/corpus_index_v2')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    setup_logging(level=logging.INFO)

    logger.info("=" * 70)
    logger.info("END-TO-END QA PIPELINE TEST")
    logger.info("=" * 70)

    # Initialize system
    qa_system = EndToEndQASystem(
        qa_model_path=Path(args.qa_model),
        vocab_path=Path(args.vocab),
        gnn_path=Path(args.gnn),
        index_path=Path(args.index),
        device=args.device
    )

    # Test questions
    test_questions = [
        "Kiu estas Frodo?",
        "Kio estas hobito?",
        "Kie estas Hobbiton?"
    ]

    logger.info("")
    logger.info("Running end-to-end tests...")
    logger.info("")

    results = []
    for i, question in enumerate(test_questions, 1):
        logger.info(f"Test {i}/{len(test_questions)}")
        logger.info("-" * 70)

        result = qa_system.answer_question(question, top_k=3)
        results.append(result)

        if 'error' in result:
            logger.error(f"ERROR: {result['error']}")
        else:
            logger.info(f"Generated: {' '.join(result.get('generated_tokens', []))}")
            logger.info(f"Context sentences: {len(result.get('context', []))}")

        logger.info("")

    # Summary
    logger.info("=" * 70)
    logger.info("END-TO-END TEST COMPLETE")
    logger.info("=" * 70)

    successful = sum(1 for r in results if 'error' not in r)
    logger.info(f"Successful: {successful}/{len(results)}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
