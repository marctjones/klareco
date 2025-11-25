#!/usr/bin/env python3
"""
Query with Local QA Decoder - Fully automatic answer generation

Uses your trained QA Decoder model (no Claude Code needed, fully local).

Usage:
    python scripts/query_with_local_model.py "Kiu estas Frodo?"
    python scripts/query_with_local_model.py "Kiu estas Frodo?" --translate
    python scripts/query_with_local_model.py "Kiu estas Frodo?" -k 10
"""

import sys
import argparse
import logging
from pathlib import Path
import json

# Suppress logging early
if '--debug' not in sys.argv:
    logging.basicConfig(level=logging.CRITICAL)
    logging.disable(logging.WARNING)

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from klareco.parser import parse
from klareco.rag.retriever import create_retriever
from klareco.models.qa_decoder import QADecoder
from klareco.models.tree_lstm import TreeLSTMEncoder
from klareco.ast_to_graph import ASTToGraphConverter
from klareco.translator import TranslationService


class LocalQASystem:
    """Local QA system using trained QA Decoder."""

    def __init__(
        self,
        qa_model_path: Path,
        vocab_path: Path,
        gnn_path: Path,
        device: str = 'cpu'
    ):
        """
        Initialize local QA system.

        Args:
            qa_model_path: Path to QA decoder checkpoint
            vocab_path: Path to vocabulary JSON
            gnn_path: Path to GNN encoder checkpoint
            device: Device ('cpu' or 'cuda')
        """
        self.device = device
        self.converter = ASTToGraphConverter()

        print(f"⚙️  Loading vocabulary from {vocab_path.name}...")
        with open(vocab_path, 'r') as f:
            vocab_data = json.load(f)
        self.vocab = vocab_data['token2id']
        self.id_to_token = {int(k): v for k, v in vocab_data['id2token'].items()}
        self.vocab_size = len(self.vocab)
        print(f"✓ Vocabulary loaded ({self.vocab_size:,} tokens)")

        print(f"⚙️  Loading GNN encoder from {gnn_path.name}...")
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
        print(f"✓ GNN encoder loaded (epoch {checkpoint.get('epoch', '?')})")

        print(f"⚙️  Loading QA Decoder from {qa_model_path.name}...")
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

        epoch = checkpoint.get('epoch', '?')
        val_loss = checkpoint.get('val_loss', 0)
        print(f"✓ QA Decoder loaded (epoch {epoch}, val_loss: {val_loss:.4f})")

    def answer_question(
        self,
        question_ast,
        context_asts: list,
        max_length: int = 30
    ) -> str:
        """
        Generate answer using QA decoder.

        Args:
            question_ast: Parsed question AST
            context_asts: List of context ASTs
            max_length: Maximum answer length (tokens)

        Returns:
            Generated answer tokens as string
        """
        # Encode question with GNN
        question_emb = self._encode_ast(question_ast).unsqueeze(0)

        # Encode context with GNN
        if context_asts:
            context_embs = torch.stack([
                self._encode_ast(ast) for ast in context_asts[:5]
            ]).unsqueeze(0)
        else:
            # No context - use zero embeddings
            context_embs = torch.zeros(1, 1, 512, device=self.device)

        # Generate answer with QA decoder
        with torch.no_grad():
            generated_ids = self.qa_decoder.generate(
                question_emb,
                context_embs,
                max_length=max_length
            )

        # Decode tokens
        tokens = []
        for id in generated_ids[0].cpu().tolist():
            token = self.id_to_token.get(id, '<UNK>')
            if token in ['<PAD>', '<END>']:
                break
            if token not in ['<START>', '<UNK>']:
                tokens.append(token)

        return ' '.join(tokens[:10]) if tokens else None

    def _encode_ast(self, ast) -> torch.Tensor:
        """Encode AST with GNN."""
        graph = self.converter.ast_to_graph(ast)
        graph.x = graph.x.to(self.device)
        graph.edge_index = graph.edge_index.to(self.device)

        with torch.no_grad():
            embedding = self.gnn_encoder(graph)

        return embedding


def generate_answer_with_local_model(
    query: str,
    retriever,
    qa_system: LocalQASystem,
    k: int = 5
):
    """
    Generate answer using local QA decoder.

    Args:
        query: Query text (Esperanto)
        retriever: RAG retriever
        qa_system: Local QA system
        k: Number of context documents

    Returns:
        Dict with answer and sources
    """
    # Parse query
    try:
        query_ast = parse(query)
    except Exception as e:
        return {
            'error': f'Failed to parse query: {e}',
            'answer': None,
            'sources': []
        }

    # Retrieve context
    try:
        results = retriever.retrieve_hybrid(
            query_ast,
            k=k,
            return_scores=True
        )
    except Exception as e:
        return {
            'error': f'Failed to retrieve: {e}',
            'answer': None,
            'sources': []
        }

    if not results:
        return {
            'error': 'No results found',
            'answer': 'Mi ne trovis rilatan informon en la korpuso.',
            'sources': []
        }

    # Parse context ASTs
    context_asts = []
    for result in results[:k]:
        try:
            ctx_text = result.get('text', '')
            ctx_ast = parse(ctx_text)
            context_asts.append(ctx_ast)
        except:
            continue

    # Generate answer with local model
    try:
        print("🤖 Generating answer with local QA Decoder...")
        answer_tokens = qa_system.answer_question(
            query_ast,
            context_asts,
            max_length=30
        )

        if answer_tokens:
            print(f"✓ Generated {len(answer_tokens.split())} tokens")
            answer = f"Laŭ la modela respondo: {answer_tokens}"
            tokens_generated = len(answer_tokens.split())
        else:
            logging.info("QA Decoder returned no tokens. Falling back to top retrieved document.")
            print("🤔 The model is uncertain. Falling back to the most relevant document.")
            top_document = results[0].get('text', 'Neniu teksto trovita.')
            answer = f"Jen la plej trafa dokumento, kiun mi trovis:\n\n---\n{top_document}\n---"
            tokens_generated = 0

    except Exception as e:
        return {
            'error': f'Generation failed: {e}',
            'answer': None,
            'sources': results
        }

    return {
        'answer': answer,
        'sources': results,
        'context_used': len(context_asts),
        'tokens_generated': tokens_generated
    }


def main():
    parser = argparse.ArgumentParser(
        description='Query with local QA decoder (fully automatic)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s "Kiu estas Frodo?"              # Generate answer (local model)
  %(prog)s "Kiu estas Frodo?" --translate  # Translate to English
  %(prog)s "Kiu estas Frodo?" -k 10        # Use 10 context docs
        '''
    )
    parser.add_argument(
        'query',
        nargs='?',
        default='Kiu estas Frodo?',
        help='Query in Esperanto'
    )
    parser.add_argument(
        '-k',
        type=int,
        default=30,
        help='Number of context documents (default: 30, trained with 75)'
    )
    parser.add_argument(
        '--translate',
        action='store_true',
        help='Translate answer to English'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Show debug logging'
    )

    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    translator = TranslationService() if args.translate else None

    print("=" * 70)
    print("KLARECO - LOCAL QA DECODER (Fully Automatic)")
    print("=" * 70)
    print()
    print(f"📝 Query: {args.query}")
    if args.translate:
        print("   [English translation enabled for OUTPUT]")
    else:
        print("   [Pure Esperanto mode - use --translate to show English]")
    print()

    # Find the best available QA Decoder model
    def find_best_model():
        """Find the best QA Decoder model from available options."""
        import torch

        candidates = [
            ('models/qa_decoder_production', 'Production model (k=75)'),
            ('models/qa_decoder_test', 'Test model (k=20)'),
            ('models/qa_decoder', 'Original model (k=3)'),
        ]

        best_model = None
        best_loss = float('inf')
        best_name = None

        for model_dir, description in candidates:
            model_path = Path(model_dir) / 'best_model.pt'
            vocab_path = Path(model_dir) / 'vocabulary.json'

            if model_path.exists() and vocab_path.exists():
                try:
                    # Load checkpoint to check validation loss
                    checkpoint = torch.load(model_path, map_location='cpu')
                    val_loss = checkpoint.get('val_loss', float('inf'))

                    print(f"✓ Found: {description}")
                    print(f"  Location: {model_dir}")
                    print(f"  Val loss: {val_loss:.4f}")

                    if val_loss < best_loss:
                        best_loss = val_loss
                        best_model = (model_path, vocab_path)
                        best_name = description
                except:
                    # If we can't load it, skip
                    pass

        if best_model:
            print()
            print(f"🏆 Using best model: {best_name} (val_loss: {best_loss:.4f})")
            print()
            return best_model

        return None

    print("🔍 Searching for best available QA Decoder model...")
    print()

    model_info = find_best_model()

    if not model_info:
        print("✗ No QA Decoder models found!")
        print()
        print("To train the QA decoder:")
        print("  ./retrain_production.sh")
        return 1

    qa_model_path, vocab_path = model_info
    gnn_path = Path('models/tree_lstm/checkpoint_epoch_20.pt')

    # Initialize systems
    print("⚙️  Initializing local QA system...")
    print()

    try:
        qa_system = LocalQASystem(
            qa_model_path=qa_model_path,
            vocab_path=vocab_path,
            gnn_path=gnn_path,
            device='cpu'
        )
    except Exception as e:
        print(f"✗ Failed to load QA system: {e}")
        return 1

    print()
    print(f"⚙️  Loading RAG retriever (k={args.k})...")
    try:
        retriever = create_retriever()
        print(f"✓ Loaded {retriever.index.ntotal:,} sentences")
    except Exception as e:
        print(f"✗ Failed to load retriever: {e}")
        return 1

    print()
    print("🔍 Retrieving context...")
    print()

    # Generate answer
    result = generate_answer_with_local_model(
        args.query,
        retriever,
        qa_system,
        k=args.k
    )

    if result.get('error'):
        print(f"✗ ERROR: {result['error']}")
        return 1

    # Show context
    print()
    print("=" * 70)
    print(f"📚 RETRIEVED CONTEXT ({result.get('context_used', 0)} documents)")
    print("=" * 70)
    print()

    for i, source in enumerate(result.get('sources', [])[:3], 1):
        score = source.get('score', 0.0)
        text = source.get('text', '')
        source_name = source.get('source_name', 'Unknown')

        print(f"{i}. [{score:.3f}] {source_name}")

        if args.translate and translator:
            try:
                en_text = translator.translate(text[:150], 'eo', 'en')
                print(f"   {en_text}...")
            except:
                print(f"   {text[:150]}...")
        else:
            print(f"   {text[:150]}...")
        print()

    if len(result.get('sources', [])) > 3:
        print(f"   ... and {len(result['sources']) - 3} more")
        print()

    # Show answer
    print("=" * 70)
    print("💬 GENERATED ANSWER (Local QA Decoder)")
    print("=" * 70)
    print()
    print(f"Tokens generated: {result.get('tokens_generated', 0)}")
    print()

    answer = result.get('answer', '')

    if args.translate and translator and answer:
        try:
            en_answer = translator.translate(answer, 'eo', 'en')
            print(en_answer)
        except:
            print(answer)
    else:
        print(answer)

    print()
    print("=" * 70)
    print("✓ Done")
    print("=" * 70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
