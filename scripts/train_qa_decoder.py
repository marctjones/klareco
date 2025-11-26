#!/usr/bin/env python3
"""
Train QA Decoder for question answering over ASTs.

This script trains the 8-layer transformer decoder to generate answer ASTs
from query ASTs and retrieved context ASTs.

Architecture:
- Input: Query AST (encoded by GNN) + Context ASTs (encoded by GNN)
- Output: Answer AST tokens (autoregressive generation)
- Training: Teacher forcing with cross-entropy loss
"""

import argparse
import json
import sys
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional
from collections import Counter
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.models.tree_lstm import TreeLSTMEncoder
from klareco.models.qa_decoder import QADecoder, create_qa_decoder
from klareco.logging_config import setup_logging

logger = logging.getLogger(__name__)


# Special tokens
PAD_TOKEN = "<PAD>"
START_TOKEN = "<START>"
END_TOKEN = "<END>"
UNK_TOKEN = "<UNK>"

SPECIAL_TOKENS = [PAD_TOKEN, START_TOKEN, END_TOKEN, UNK_TOKEN]
PAD_ID = 0
START_ID = 1
END_ID = 2
UNK_ID = 3


class Vocabulary:
    """Vocabulary for AST tokens."""

    def __init__(self, min_freq: int = 2):
        """
        Initialize vocabulary.

        Args:
            min_freq: Minimum frequency for token to be included
        """
        self.min_freq = min_freq
        self.token2id = {}
        self.id2token = {}
        self.token_freq = Counter()

        # Add special tokens
        for i, token in enumerate(SPECIAL_TOKENS):
            self.token2id[token] = i
            self.id2token[i] = token

    def build_from_asts(self, asts: List[Dict]):
        """
        Build vocabulary from ASTs.

        Args:
            asts: List of AST dictionaries
        """
        logger.info("Building vocabulary from ASTs...")

        # Count token frequencies
        for ast in asts:
            tokens = self._ast_to_tokens(ast)
            self.token_freq.update(tokens)

        # Add tokens above minimum frequency
        next_id = len(SPECIAL_TOKENS)
        for token, freq in self.token_freq.most_common():
            if freq >= self.min_freq and token not in self.token2id:
                self.token2id[token] = next_id
                self.id2token[next_id] = token
                next_id += 1

        logger.info(f"Vocabulary size: {len(self.token2id)} tokens")
        logger.info(f"Most common tokens: {self.token_freq.most_common(20)}")

    def _ast_to_tokens(self, ast: Dict) -> List[str]:
        """
        Convert AST to token sequence.

        Extracts root words and grammatical markers.

        Args:
            ast: AST dictionary

        Returns:
            List of tokens
        """
        tokens = []

        if ast.get('tipo') == 'vorto':
            # Word-level AST
            tokens.append(ast.get('radiko', '<UNK>'))
            tokens.append(ast.get('vortspeco', '<UNK>'))

            # Add grammatical markers
            if ast.get('nombro') == 'pluralo':
                tokens.append('PLURAL')
            if ast.get('kazo') == 'akuzativo':
                tokens.append('ACC')

        elif ast.get('tipo') == 'frazo':
            # Sentence-level AST
            if ast.get('subjekto'):
                tokens.extend(self._ast_to_tokens(ast['subjekto']))
            if ast.get('verbo'):
                tokens.extend(self._ast_to_tokens(ast['verbo']))
            if ast.get('objekto'):
                tokens.extend(self._ast_to_tokens(ast['objekto']))
            for other in ast.get('aliaj', []):
                tokens.extend(self._ast_to_tokens(other))

        elif ast.get('tipo') == 'vortgrupo':
            # Word group
            for word in ast.get('vortoj', []):
                tokens.extend(self._ast_to_tokens(word))

        return tokens

    def encode(self, ast: Dict) -> List[int]:
        """
        Encode AST to token IDs.

        Args:
            ast: AST dictionary

        Returns:
            List of token IDs
        """
        tokens = self._ast_to_tokens(ast)
        return [self.token2id.get(token, UNK_ID) for token in tokens]

    def decode(self, ids: List[int]) -> List[str]:
        """
        Decode token IDs to tokens.

        Args:
            ids: List of token IDs

        Returns:
            List of tokens
        """
        return [self.id2token.get(id, UNK_TOKEN) for id in ids]

    def __len__(self):
        return len(self.token2id)


class QADataset(Dataset):
    """Dataset for QA training."""

    def __init__(
        self,
        data: List[Dict],
        vocabulary: Vocabulary,
        max_seq_len: int = 256
    ):
        """
        Initialize dataset.

        Args:
            data: List of QA pairs with ASTs
            vocabulary: Vocabulary instance
            max_seq_len: Maximum sequence length
        """
        self.data = data
        self.vocabulary = vocabulary
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get item at index.

        Returns:
            Dictionary with:
            - question_ast: Question AST
            - answer_tokens: Answer token IDs (with START/END)
            - context_asts: List of context ASTs
        """
        item = self.data[idx]

        # Get ASTs
        question_ast = item['question_ast']
        answer_ast = item['answer_ast']
        context_asts = item.get('context_asts', [])  # Use all provided context ASTs

        # Encode answer to tokens
        answer_tokens = self.vocabulary.encode(answer_ast)

        # Add START and END tokens
        answer_tokens = [START_ID] + answer_tokens[:self.max_seq_len - 2] + [END_ID]

        # Pad to fixed length
        answer_tokens = answer_tokens + [PAD_ID] * (self.max_seq_len - len(answer_tokens))

        return {
            'question_ast': question_ast,
            'answer_tokens': torch.tensor(answer_tokens, dtype=torch.long),
            'context_asts': context_asts
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate batch.

    Args:
        batch: List of items from dataset

    Returns:
        Batched dictionary
    """
    return {
        'question_asts': [item['question_ast'] for item in batch],
        'answer_tokens': torch.stack([item['answer_tokens'] for item in batch]),
        'context_asts': [item['context_asts'] for item in batch]
    }


def encode_asts_with_gnn(
    asts: List[Dict],
    encoder: TreeLSTMEncoder,
    device: torch.device
) -> torch.Tensor:
    """
    Encode ASTs using GNN encoder.

    Args:
        asts: List of AST dictionaries
        encoder: GNN encoder model
        device: Device to use

    Returns:
        Encoded embeddings (batch_size, d_model)
    """
    from klareco.ast_to_graph import ASTToGraphConverter

    converter = ASTToGraphConverter()
    embeddings = []

    for ast in asts:
        try:
            # Convert to graph
            graph = converter.ast_to_graph(ast)
            graph.x = graph.x.to(device)
            graph.edge_index = graph.edge_index.to(device)

            # Encode
            with torch.no_grad():
                emb = encoder(graph)
                embeddings.append(emb)
        except Exception as e:
            logger.warning(f"Failed to encode AST: {e}")
            # Use zero embedding as fallback
            embeddings.append(torch.zeros(encoder.output_dim, device=device))

    return torch.stack(embeddings)


def train_epoch(
    model: QADecoder,
    encoder: TreeLSTMEncoder,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """
    Train for one epoch.

    Args:
        model: QA Decoder model
        encoder: GNN encoder (frozen)
        dataloader: Training dataloader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on

    Returns:
        (average_loss, accuracy)
    """
    model.train()
    encoder.eval()  # GNN is frozen

    total_loss = 0.0
    total_tokens = 0
    correct_tokens = 0

    pbar = tqdm(dataloader, desc="Training")

    for batch in pbar:
        question_asts = batch['question_asts']
        answer_tokens = batch['answer_tokens'].to(device)
        context_asts_batch = batch['context_asts']

        batch_size = answer_tokens.size(0)

        # Encode question ASTs with GNN
        question_embeddings = encode_asts_with_gnn(question_asts, encoder, device)

        # Encode context ASTs with GNN
        context_embeddings_list = []
        for context_asts in context_asts_batch:
            if context_asts:
                ctx_embs = encode_asts_with_gnn(context_asts, encoder, device)
                context_embeddings_list.append(ctx_embs)
            else:
                # No context - use zero embeddings
                ctx_embs = torch.zeros(1, encoder.output_dim, device=device)
                context_embeddings_list.append(ctx_embs)

        # Pad context to same length
        max_ctx_len = max(ctx.size(0) for ctx in context_embeddings_list)
        padded_contexts = []
        for ctx in context_embeddings_list:
            if ctx.size(0) < max_ctx_len:
                padding = torch.zeros(
                    max_ctx_len - ctx.size(0),
                    encoder.output_dim,
                    device=device
                )
                ctx = torch.cat([ctx, padding], dim=0)
            padded_contexts.append(ctx)

        context_embeddings = torch.stack(padded_contexts)  # (batch_size, max_ctx_len, d_model)

        # Split answer tokens into input and target
        input_tokens = answer_tokens[:, :-1]   # All but last
        target_tokens = answer_tokens[:, 1:]   # All but first

        # Forward pass
        logits = model(question_embeddings, context_embeddings, input_tokens)

        # Compute loss
        # Reshape for cross-entropy: (batch_size * seq_len, vocab_size)
        logits_flat = logits.view(-1, logits.size(-1))
        target_flat = target_tokens.contiguous().view(-1)

        # Mask out padding
        mask = (target_flat != PAD_ID)
        loss = criterion(logits_flat[mask], target_flat[mask])

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
        optimizer.step()

        # Statistics
        total_loss += loss.item() * mask.sum().item()
        total_tokens += mask.sum().item()

        # Compute accuracy
        predictions = torch.argmax(logits_flat[mask], dim=-1)
        correct_tokens += (predictions == target_flat[mask]).sum().item()

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{correct_tokens/total_tokens:.4f}'
        })

    avg_loss = total_loss / total_tokens
    accuracy = correct_tokens / total_tokens

    return avg_loss, accuracy


def validate_epoch(
    model: QADecoder,
    encoder: TreeLSTMEncoder,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """
    Validate for one epoch.

    Args:
        model: QA Decoder model
        encoder: GNN encoder (frozen)
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device to validate on

    Returns:
        (average_loss, accuracy)
    """
    model.eval()
    encoder.eval()

    total_loss = 0.0
    total_tokens = 0
    correct_tokens = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")

        for batch in pbar:
            question_asts = batch['question_asts']
            answer_tokens = batch['answer_tokens'].to(device)
            context_asts_batch = batch['context_asts']

            # Encode with GNN (same as training)
            question_embeddings = encode_asts_with_gnn(question_asts, encoder, device)

            context_embeddings_list = []
            for context_asts in context_asts_batch:
                if context_asts:
                    ctx_embs = encode_asts_with_gnn(context_asts, encoder, device)
                    context_embeddings_list.append(ctx_embs)
                else:
                    ctx_embs = torch.zeros(1, encoder.output_dim, device=device)
                    context_embeddings_list.append(ctx_embs)

            max_ctx_len = max(ctx.size(0) for ctx in context_embeddings_list)
            padded_contexts = []
            for ctx in context_embeddings_list:
                if ctx.size(0) < max_ctx_len:
                    padding = torch.zeros(
                        max_ctx_len - ctx.size(0),
                        encoder.output_dim,
                        device=device
                    )
                    ctx = torch.cat([ctx, padding], dim=0)
                padded_contexts.append(ctx)

            context_embeddings = torch.stack(padded_contexts)

            # Forward pass
            input_tokens = answer_tokens[:, :-1]
            target_tokens = answer_tokens[:, 1:]

            logits = model(question_embeddings, context_embeddings, input_tokens)

            # Compute loss
            logits_flat = logits.view(-1, logits.size(-1))
            target_flat = target_tokens.contiguous().view(-1)

            mask = (target_flat != PAD_ID)
            loss = criterion(logits_flat[mask], target_flat[mask])

            # Statistics
            total_loss += loss.item() * mask.sum().item()
            total_tokens += mask.sum().item()

            predictions = torch.argmax(logits_flat[mask], dim=-1)
            correct_tokens += (predictions == target_flat[mask]).sum().item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct_tokens/total_tokens:.4f}'
            })

    avg_loss = total_loss / total_tokens
    accuracy = correct_tokens / total_tokens

    return avg_loss, accuracy


def main():
    """Train QA Decoder."""
    parser = argparse.ArgumentParser(description='Train QA Decoder')
    parser.add_argument('--dataset', type=str, default='data/qa_dataset.jsonl',
                        help='QA dataset file')
    parser.add_argument('--gnn-checkpoint', type=str, default=None,
                        help='GNN checkpoint (auto-detects latest if not specified)')
    parser.add_argument('--output', type=str, default='models/qa_decoder',
                        help='Output directory for checkpoints')
    parser.add_argument('--vocab-size', type=int, default=10000,
                        help='Vocabulary size')
    parser.add_argument('--d-model', type=int, default=512,
                        help='Model dimension (must match GNN)')
    parser.add_argument('--n-layers', type=int, default=8,
                        help='Number of decoder layers')
    parser.add_argument('--n-heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--d-ff', type=int, default=2048,
                        help='Feed-forward dimension')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--val-split', type=float, default=0.1,
                        help='Validation split fraction')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (cpu or cuda)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    args = parser.parse_args()

    # Setup logging
    setup_logging(level=logging.DEBUG if args.debug else logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("=" * 70)
    logger.info("QA DECODER TRAINING")
    logger.info("=" * 70)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Model config:")
    logger.info(f"  Vocab size: {args.vocab_size}")
    logger.info(f"  Model dim: {args.d_model}")
    logger.info(f"  Layers: {args.n_layers}")
    logger.info(f"  Heads: {args.n_heads}")
    logger.info(f"  FF dim: {args.d_ff}")
    logger.info(f"Training config:")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Learning rate: {args.lr}")
    logger.info("")

    try:
        device = torch.device(args.device)

        # Load dataset
        logger.info(f"Loading dataset from {args.dataset}...")
        dataset_path = Path(args.dataset)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        data = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))

        logger.info(f"Loaded {len(data)} QA pairs")

        # Split train/val
        random.shuffle(data)
        val_size = int(len(data) * args.val_split)
        train_data = data[val_size:]
        val_data = data[:val_size]

        logger.info(f"Train: {len(train_data)}, Val: {len(val_data)}")

        # Build vocabulary
        logger.info("Building vocabulary...")
        vocabulary = Vocabulary(min_freq=2)
        all_asts = []
        for item in train_data:
            all_asts.append(item['question_ast'])
            all_asts.append(item['answer_ast'])
        vocabulary.build_from_asts(all_asts)

        # Create datasets
        train_dataset = QADataset(train_data, vocabulary)
        val_dataset = QADataset(val_data, vocabulary)

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )

        # Load GNN encoder
        logger.info("Loading GNN encoder...")
        if args.gnn_checkpoint:
            gnn_path = Path(args.gnn_checkpoint)
        else:
            # Auto-detect latest checkpoint
            model_dir = Path("models/tree_lstm")
            checkpoints = list(model_dir.glob("checkpoint_epoch_*.pt"))
            if not checkpoints:
                raise FileNotFoundError("No GNN checkpoints found")
            checkpoints.sort(key=lambda p: int(p.stem.split('_')[-1]))
            gnn_path = checkpoints[-1]

        logger.info(f"Using GNN checkpoint: {gnn_path}")

        # Create GNN encoder
        gnn_encoder = TreeLSTMEncoder(
            vocab_size=10000,  # From GNN training
            embed_dim=128,
            hidden_dim=256,
            output_dim=512
        ).to(device)

        # Load checkpoint
        checkpoint = torch.load(gnn_path, map_location=device)
        gnn_encoder.load_state_dict(checkpoint['model_state_dict'])
        gnn_encoder.eval()  # Freeze GNN

        logger.info(f"Loaded GNN from epoch {checkpoint['epoch']}")

        # Create QA Decoder
        logger.info("Creating QA Decoder...")
        qa_decoder = create_qa_decoder(
            vocab_size=len(vocabulary),
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            d_ff=args.d_ff
        ).to(device)

        # Optimizer and loss
        optimizer = torch.optim.Adam(qa_decoder.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()

        # Output directory
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save vocabulary
        vocab_file = output_dir / "vocabulary.json"
        with open(vocab_file, 'w') as f:
            json.dump({
                'token2id': vocabulary.token2id,
                'id2token': vocabulary.id2token
            }, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved vocabulary to {vocab_file}")

        # Training loop
        logger.info("\nStarting training...")
        best_val_loss = float('inf')

        for epoch in range(1, args.epochs + 1):
            logger.info(f"\nEpoch {epoch}/{args.epochs}")
            logger.info("-" * 70)

            # Train
            train_loss, train_acc = train_epoch(
                qa_decoder, gnn_encoder, train_loader, optimizer, criterion, device
            )
            logger.info(f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}")

            # Validate
            val_loss, val_acc = validate_epoch(
                qa_decoder, gnn_encoder, val_loader, criterion, device
            )
            logger.info(f"Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}")

            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': qa_decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
            }

            checkpoint_file = output_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save(checkpoint, checkpoint_file)
            logger.info(f"Saved checkpoint: {checkpoint_file}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_file = output_dir / 'best_model.pt'
                torch.save(checkpoint, best_file)
                logger.info(f"Saved best model: {best_file}")

        # Summary
        logger.info("")
        logger.info("=" * 70)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Best validation loss: {best_val_loss:.4f}")
        logger.info(f"Model saved to: {output_dir}")
        logger.info("")

        return 0

    except Exception as e:
        logger.error(f"ERROR: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
