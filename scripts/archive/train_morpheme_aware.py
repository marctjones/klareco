#!/usr/bin/env python3
"""
Train Morpheme-Aware Compositional Embeddings.

Approach: Learn root and affix embeddings that can compose to predict word meanings.
This is based on Discussion #39's morpheme-aware training strategy.

Training objective:
- Masked Language Modeling: Predict masked morphemes from context
- Composition Quality: Ensure root + affix embeddings compose well
- Parameter Efficiency: 5K roots + affixes = ~709K params (vs 3.9M baseline)

Architecture:
- Root embeddings: 5K x 128d = 640K params (semantic)
- Prefix embeddings: 18 x 8d = 144 params (semantic)
- Suffix embeddings: 46 x 8d = 368 params (semantic)
- Ending embeddings: 17 x 8d = 136 params (grammatical, deterministic)
- Composer: 128d -> 128d projection = 16.5K params
- OOV handling: 10 entity types + 500 hash buckets = 65K params
- **Total: ~709K params**
"""

import argparse
import json
import logging
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.embeddings.compositional import CompositionalEmbedding, ENDINGS
from klareco import parser as eo_parser_module


def setup_logging(output_dir: Path):
    """Setup logging to both console and file."""
    log_file = output_dir / 'training.log'

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # File handler
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    return logging.getLogger(__name__)


def save_checkpoint_atomic(checkpoint_path: Path, checkpoint: dict):
    """
    Save checkpoint atomically to prevent corruption.

    Uses temp file + rename pattern for atomic writes on POSIX systems.
    """
    temp_path = checkpoint_path.with_suffix('.pt.tmp')
    try:
        torch.save(checkpoint, temp_path)
        temp_path.rename(checkpoint_path)  # Atomic on POSIX
    except Exception as e:
        logger.error(f"Failed to save checkpoint to {checkpoint_path}: {e}")
        if temp_path.exists():
            temp_path.unlink()
        raise


def rotate_best_checkpoint(output_dir: Path):
    """
    Rotate best checkpoint: best_model.pt -> best_model.prev.pt

    Keeps previous best model as backup.
    """
    best_path = output_dir / 'best_model.pt'
    prev_path = output_dir / 'best_model.prev.pt'

    if best_path.exists():
        # Remove old previous backup if exists
        if prev_path.exists():
            prev_path.unlink()
        # Move current best to previous
        best_path.rename(prev_path)
        logger.info("Rotated previous best model to best_model.prev.pt")


# Placeholder - will be properly initialized in main()
logger = logging.getLogger(__name__)


def build_root_vocabulary(
    corpus_path: Path,
    vocab_size: int = 5000,
    sample_size: Optional[int] = None,
) -> Dict[str, int]:
    """
    Build root vocabulary from corpus.

    Args:
        corpus_path: Path to M1 enhanced corpus
        vocab_size: Number of most frequent roots to include
        sample_size: Optional limit on sentences to scan

    Returns:
        Dictionary mapping root strings to indices
    """
    logger.info(f"Building {vocab_size}-root vocabulary from {corpus_path}")

    root_counter = Counter()
    scanned = 0

    with open(corpus_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, desc="Scanning corpus")):
            if sample_size and i >= sample_size:
                break

            entry = json.loads(line)
            scanned += 1

            # Extract roots from AST
            if 'ast' in entry and entry['ast']:
                ast = entry['ast']
                if ast.get('tipo') == 'frazo':
                    for role in ['subjekto', 'verbo', 'objekto', 'aliaj']:
                        if role in ast and ast[role]:
                            component = ast[role]
                            _extract_roots_from_component(component, root_counter)

    logger.info(f"Scanned {scanned} sentences")
    logger.info(f"Found {len(root_counter)} unique roots")

    # Build vocabulary with special tokens
    vocab = {
        '<PAD>': 0,
        '<UNK>': 1,
        '<MASK>': 2,
    }

    # Add most frequent roots
    for root, count in root_counter.most_common(vocab_size):
        if root not in vocab:
            vocab[root] = len(vocab)

    logger.info(f"Built vocabulary with {len(vocab)} roots (including special tokens)")

    # Calculate coverage
    top_roots_count = sum(count for root, count in root_counter.most_common(vocab_size))
    total_count = sum(root_counter.values())
    coverage = top_roots_count / total_count if total_count > 0 else 0
    logger.info(f"Coverage: {coverage:.1%} ({top_roots_count:,} / {total_count:,} instances)")

    return vocab


def _extract_roots_from_component(component, counter: Counter):
    """Recursively extract roots from AST component."""
    if isinstance(component, dict):
        if component.get('tipo') == 'vorto' and 'radiko' in component:
            counter[component['radiko']] += 1
        elif component.get('tipo') == 'vortgrupo':
            if 'kerno' in component and isinstance(component['kerno'], dict):
                if 'radiko' in component['kerno']:
                    counter[component['kerno']['radiko']] += 1
            if 'priskriboj' in component:
                for item in component['priskriboj']:
                    _extract_roots_from_component(item, counter)
    elif isinstance(component, list):
        for item in component:
            _extract_roots_from_component(item, counter)


class MorphemeMaskingDataset(Dataset):
    """
    Dataset for training morpheme embeddings via masked language modeling.

    Masks random morphemes (roots, prefixes, suffixes) and trains the model
    to reconstruct them from context.
    """

    def __init__(
        self,
        corpus_path: Path,
        embedding_layer: CompositionalEmbedding,
        max_samples: Optional[int] = None,
        mask_prob: float = 0.15,
    ):
        """
        Initialize dataset.

        Args:
            corpus_path: Path to M1 enhanced corpus
            embedding_layer: Compositional embedding layer for vocabulary
            max_samples: Optional limit on samples to load
            mask_prob: Probability of masking each word
        """
        self.embedding_layer = embedding_layer
        self.mask_prob = mask_prob
        self.samples = []  # List of (words, masked_indices)

        logger.info(f"Loading corpus from {corpus_path}")

        with open(corpus_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(tqdm(f, desc="Loading corpus")):
                if max_samples and i >= max_samples:
                    break

                entry = json.loads(line)

                # Extract words from AST
                if 'ast' in entry and entry['ast']:
                    words = self._extract_words_from_ast(entry['ast'])
                    if words:  # Only add if we extracted words
                        self.samples.append(words)

        logger.info(f"Loaded {len(self.samples)} sentences")

    def _extract_words_from_ast(self, ast: dict) -> List[dict]:
        """
        Extract word information from AST.

        Returns list of dicts with: root, prefix, suffixes, ending
        """
        words = []

        if ast.get('tipo') != 'frazo':
            return words

        for role in ['subjekto', 'verbo', 'objekto', 'aliaj']:
            if role in ast and ast[role]:
                component = ast[role]
                self._extract_words_from_component(component, words)

        return words

    def _extract_words_from_component(self, component, words: List):
        """Recursively extract words from component."""
        if isinstance(component, dict):
            if component.get('tipo') == 'vorto':
                word_info = {
                    'root': component.get('radiko', '<UNK>'),
                    'prefix': component.get('prefikso'),
                    'suffixes': component.get('sufiksoj', []),
                    'ending': component.get('vortspeco_ending'),
                }
                words.append(word_info)
            elif component.get('tipo') == 'vortgrupo':
                if 'kerno' in component:
                    self._extract_words_from_component(component['kerno'], words)
                if 'priskriboj' in component:
                    for item in component['priskriboj']:
                        self._extract_words_from_component(item, words)
        elif isinstance(component, list):
            for item in component:
                self._extract_words_from_component(item, words)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get a sample with random masking.

        Returns:
            (words, mask_indices): Original words and which indices to mask
        """
        words = self.samples[idx]

        # Randomly select indices to mask
        mask_indices = [
            i for i in range(len(words))
            if random.random() < self.mask_prob
        ]

        # Ensure at least one word is masked
        if not mask_indices and words:
            mask_indices = [random.randint(0, len(words) - 1)]

        return words, mask_indices


def collate_morpheme_batch(batch: List[Tuple[List[dict], List[int]]]):
    """
    Collate batch of morpheme samples.

    Returns:
        Dictionary with tensors for training
    """
    # For now, just return the raw batch - we'll process in training loop
    return batch


class MorphemeReconstructionModel(nn.Module):
    """
    Model for learning morpheme embeddings via reconstruction.

    Architecture:
    - CompositionalEmbedding layer
    - Context encoder (simple averaging for now)
    - Reconstruction heads for each morpheme type
    """

    def __init__(
        self,
        compositional_embedding: CompositionalEmbedding,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.embedding = compositional_embedding
        self.hidden_dim = hidden_dim

        # Reconstruction heads
        self.root_classifier = nn.Linear(hidden_dim, len(compositional_embedding.root_vocab))
        self.prefix_classifier = nn.Linear(hidden_dim, len(compositional_embedding.prefix_vocab))
        self.suffix_classifier = nn.Linear(hidden_dim, len(compositional_embedding.suffix_vocab))

    def forward(self, words: List[dict], mask_idx: int):
        """
        Predict masked word from context.

        Args:
            words: List of word dicts (root, prefix, suffixes, ending)
            mask_idx: Index of word to mask

        Returns:
            Logits for (root, prefix, suffix) reconstruction
        """
        # Encode all words
        embeddings = []
        for i, word in enumerate(words):
            if i == mask_idx:
                # Use mask token
                emb = self.embedding.encode_word(
                    root='<MASK>',
                    prefix=None,
                    suffixes=None,
                    ending=None,
                )
            else:
                emb = self.embedding.encode_word(
                    root=word['root'],
                    prefix=word['prefix'],
                    suffixes=word['suffixes'],
                    ending=word['ending'],
                )
            embeddings.append(emb)

        # Simple context: average of all word embeddings
        context = torch.stack(embeddings).mean(dim=0)

        # Predict masked morphemes
        root_logits = self.root_classifier(context)
        prefix_logits = self.prefix_classifier(context)
        suffix_logits = self.suffix_classifier(context)

        return root_logits, prefix_logits, suffix_logits


def train_epoch(
    model: MorphemeReconstructionModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch in pbar:
        for words, mask_indices in batch:
            if not words or not mask_indices:
                continue

            # Train on each masked position
            for mask_idx in mask_indices:
                optimizer.zero_grad()

                # Forward pass
                root_logits, prefix_logits, suffix_logits = model(words, mask_idx)

                # Get targets
                masked_word = words[mask_idx]
                root_target = torch.tensor([
                    model.embedding.get_root_idx(masked_word['root'])
                ], device=device)

                # Compute loss (just root for now)
                loss = F.cross_entropy(root_logits.unsqueeze(0), root_target)

                # Backward
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        if num_batches > 0:
            pbar.set_postfix({'loss': total_loss / num_batches})

    return total_loss / max(num_batches, 1)


def main():
    parser = argparse.ArgumentParser(description="Train Morpheme-Aware Embeddings")
    parser.add_argument('--corpus', type=Path, default=Path('data/corpus_enhanced_m1.jsonl'))
    parser.add_argument('--vocab-dir', type=Path, default=Path('data/vocabularies'))
    parser.add_argument('--output-dir', type=Path, default=Path('models/morpheme_aware'))
    parser.add_argument('--vocab-size', type=int, default=5000, help='Number of roots in vocabulary')
    parser.add_argument('--embed-dim', type=int, default=128)
    parser.add_argument('--composition-method', type=str, default='sum', choices=['sum', 'concat', 'gated', 'attention'])
    parser.add_argument('--max-samples', type=int, help='Limit training samples (for testing)')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--fresh', action='store_true', help='Start fresh (ignore existing checkpoints)')

    args = parser.parse_args()

    # Setup
    args.output_dir.mkdir(parents=True, exist_ok=True)
    global logger
    logger = setup_logging(args.output_dir)

    logger.info("=" * 80)
    logger.info("Morpheme-Aware Embedding Training")
    logger.info("=" * 80)
    logger.info(f"Corpus: {args.corpus}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Vocabulary size: {args.vocab_size} roots")
    logger.info(f"Embedding dimension: {args.embed_dim}")
    logger.info(f"Composition method: {args.composition_method}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # Step 1: Build or load root vocabulary
    root_vocab_path = args.output_dir / 'root_vocabulary.json'

    if root_vocab_path.exists() and not args.fresh:
        logger.info(f"Loading existing root vocabulary from {root_vocab_path}")
        with open(root_vocab_path, 'r', encoding='utf-8') as f:
            root_vocab = json.load(f)
    else:
        logger.info("Building root vocabulary from corpus...")
        root_vocab = build_root_vocabulary(
            args.corpus,
            vocab_size=args.vocab_size,
            sample_size=args.max_samples,
        )

        # Save vocabulary
        with open(root_vocab_path, 'w', encoding='utf-8') as f:
            json.dump(root_vocab, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved root vocabulary to {root_vocab_path}")

    # Step 2: Load affix vocabularies
    logger.info(f"Loading affix vocabularies from {args.vocab_dir}")

    with open(args.vocab_dir / 'affix_vocabulary.json', 'r', encoding='utf-8') as f:
        affix_data = json.load(f)

    prefix_vocab = affix_data['prefixes']
    suffix_vocab = affix_data['suffixes']

    logger.info(f"Loaded {len(prefix_vocab)} prefixes, {len(suffix_vocab)} suffixes")

    # Step 3: Create compositional embedding layer
    logger.info("Initializing compositional embedding layer...")

    embedding_layer = CompositionalEmbedding(
        root_vocab=root_vocab,
        prefix_vocab=prefix_vocab,
        suffix_vocab=suffix_vocab,
        embed_dim=args.embed_dim,
        composition_method=args.composition_method,
        dropout=0.1,
    )

    # Step 4: Create model
    model = MorphemeReconstructionModel(embedding_layer, hidden_dim=args.embed_dim)
    model = model.to(device)

    # Calculate parameter count
    total_params = sum(p.numel() for p in model.parameters())
    embedding_params = sum(p.numel() for p in embedding_layer.parameters())

    logger.info(f"Model parameters:")
    logger.info(f"  Total: {total_params:,}")
    logger.info(f"  Embeddings: {embedding_params:,}")
    logger.info(f"  Other: {total_params - embedding_params:,}")

    # Step 5: Create dataset and dataloader
    logger.info("Loading training data...")

    dataset = MorphemeMaskingDataset(
        corpus_path=args.corpus,
        embedding_layer=embedding_layer,
        max_samples=args.max_samples,
        mask_prob=0.15,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_morpheme_batch,
    )

    # Step 6: Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Step 7: Resume from checkpoint if exists
    start_epoch = 1
    best_loss = float('inf')
    epochs_without_improvement = 0
    patience = 3  # Early stopping patience

    checkpoint_path = args.output_dir / 'latest_checkpoint.pt'

    if checkpoint_path.exists() and not args.fresh:
        try:
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_loss = checkpoint.get('best_loss', float('inf'))
            epochs_without_improvement = checkpoint.get('epochs_without_improvement', 0)
            logger.info(f"Resumed from epoch {checkpoint['epoch']}, best loss: {best_loss:.4f}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            logger.info("Starting from scratch")
            start_epoch = 1
            best_loss = float('inf')
    elif args.fresh and checkpoint_path.exists():
        logger.info("--fresh flag set, ignoring existing checkpoint")

    # Step 8: Training loop with best model tracking
    logger.info(f"Starting training from epoch {start_epoch} to {args.epochs}...")

    for epoch in range(start_epoch, args.epochs + 1):
        avg_loss = train_epoch(model, dataloader, optimizer, device, epoch)
        logger.info(f"Epoch {epoch}/{args.epochs} - Loss: {avg_loss:.4f}")

        # Save latest checkpoint (for resume)
        try:
            save_checkpoint_atomic(
                checkpoint_path,
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_loss,
                    'best_loss': best_loss,
                    'epochs_without_improvement': epochs_without_improvement,
                    'root_vocab': root_vocab,
                    'prefix_vocab': prefix_vocab,
                    'suffix_vocab': suffix_vocab,
                }
            )
            logger.info(f"Saved latest checkpoint")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            logger.info("Continuing training despite checkpoint save failure")

        # Save best model if improved
        if avg_loss < best_loss:
            logger.info(f"New best model! Loss: {avg_loss:.4f} (prev: {best_loss:.4f})")
            best_loss = avg_loss
            epochs_without_improvement = 0

            # Rotate old best model
            rotate_best_checkpoint(args.output_dir)

            # Save new best model
            try:
                save_checkpoint_atomic(
                    args.output_dir / 'best_model.pt',
                    {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'train_loss': avg_loss,
                        'root_vocab': root_vocab,
                        'prefix_vocab': prefix_vocab,
                        'suffix_vocab': suffix_vocab,
                        'config': {
                            'vocab_size': args.vocab_size,
                            'embed_dim': args.embed_dim,
                            'composition_method': args.composition_method,
                        },
                    }
                )
                logger.info(f"Saved best model to best_model.pt")
            except Exception as e:
                logger.error(f"Failed to save best model: {e}")
        else:
            epochs_without_improvement += 1
            logger.info(f"No improvement for {epochs_without_improvement} epoch(s)")

        # Early stopping
        if epochs_without_improvement >= patience:
            logger.info(f"Early stopping after {epoch} epochs (no improvement for {patience} epochs)")
            break

    logger.info("=" * 80)
    logger.info(f"Training complete!")
    logger.info(f"Best loss: {best_loss:.4f}")
    logger.info(f"Best model: {args.output_dir / 'best_model.pt'}")
    logger.info(f"Latest checkpoint: {checkpoint_path}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
