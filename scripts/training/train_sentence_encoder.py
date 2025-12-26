#!/usr/bin/env python3
"""
Train AST-aware sentence encoder using compositional embeddings.

Phase 4 of Fundamento-Centered Training (Issue #71)

This phase:
1. Loads trained root and affix embeddings
2. Composes word embeddings from roots + affixes
3. Learns AST-role attention weights for pooling words into sentences
4. Does NOT learn grammar - uses deterministic AST structure

Key insight: The AST is produced by the deterministic parser.
We only learn how to weight different AST roles when pooling.

Output: models/sentence_encoder/best_model.pt
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def setup_file_logging(log_path: Path):
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)


# AST role weights - these are the ONLY learned parameters for pooling
# Higher weight = more contribution to sentence meaning
AST_ROLES = ['subjekto', 'verbo', 'objekto', 'predikativo', 'aliaj', 'unknown']

# Function words to exclude from sentence embeddings.
# These are grammatical/structural words handled by the AST, not learned embeddings.
# Including them causes sentence embeddings to collapse (all sentences become similar
# because they all contain the same high-frequency function words).
# See Wiki: "Function Word Exclusion Principle"
FUNCTION_WORDS = {
    # Conjunctions
    'kaj', 'aŭ', 'sed', 'nek', 'do', 'tamen', 'ĉar', 'ke', 'se',
    # Prepositions
    'al', 'de', 'en', 'el', 'kun', 'per', 'por', 'pri', 'sen', 'sur', 'sub', 'ĉe', 'tra', 'ĉirkaŭ',
    # Pronouns/correlatives
    'mi', 'vi', 'li', 'ŝi', 'ĝi', 'ni', 'ili', 'si', 'oni',
    'kiu', 'kio', 'kia', 'kie', 'kiel', 'kiam', 'kiom', 'kial',
    'tiu', 'tio', 'tia', 'tie', 'tiel', 'tiam', 'tiom', 'tial',
    'ĉiu', 'ĉio', 'ĉia', 'ĉie', 'ĉiel', 'ĉiam', 'ĉiom', 'ĉial',
    'neniu', 'nenio', 'nenia', 'nenie', 'neniel', 'neniam', 'neniom', 'nenial',
    'iu', 'io', 'ia', 'ie', 'iel', 'iam', 'iom', 'ial',
    # Common verbs/copula
    'est', 'far', 'hav', 'pov', 'dev', 'vol', 'deb',
    # Articles/particles
    'la', 'ne', 'tre', 'nur', 'ankaŭ', 'eĉ', 'ja', 'jen', 'jes', 'plej', 'pli', 'tro',
    # Note: Numbers (unu, du, tri, etc.) are NOT included here.
    # Unlike true function words, numbers carry semantic content (quantity).
    # See Issue #83 for discussion.
}


class ASTRoleAttention(nn.Module):
    """
    Learn attention weights for AST roles.

    This is a TINY model - just 6 learnable scalars (one per role).
    The idea: verbs and subjects typically carry more semantic weight
    than modifiers in 'aliaj'.
    """

    def __init__(self, num_roles: int = 6):
        super().__init__()
        # Initialize with linguistic priors:
        # verbo=1.5, subjekto=1.2, objekto=1.0, predikativo=0.8, aliaj=0.5, unknown=0.3
        initial_weights = torch.tensor([1.2, 1.5, 1.0, 0.8, 0.5, 0.3])
        self.role_weights = nn.Parameter(initial_weights)
        self.role_to_idx = {role: i for i, role in enumerate(AST_ROLES)}

    def get_weight(self, role: str) -> float:
        """Get attention weight for a role."""
        idx = self.role_to_idx.get(role, self.role_to_idx['unknown'])
        return F.softplus(self.role_weights[idx])  # Ensure positive


class SentenceEncoder(nn.Module):
    """
    Compose sentence embeddings from word embeddings using AST structure.

    Architecture:
    1. Look up root embedding
    2. Apply affix transformations (if affix embeddings available)
    3. Pool words using AST-role attention weights

    Total learned parameters: ~6 (role attention weights)
    Everything else uses pre-trained embeddings.
    """

    def __init__(self, root_embeddings: torch.Tensor, root_to_idx: Dict[str, int],
                 affix_embeddings: Optional[torch.Tensor] = None,
                 affix_to_idx: Optional[Dict[str, int]] = None):
        super().__init__()

        self.embedding_dim = root_embeddings.shape[1]

        # Freeze pre-trained embeddings
        self.root_embeddings = nn.Embedding.from_pretrained(root_embeddings, freeze=True)
        self.root_to_idx = root_to_idx

        # Optional affix embeddings
        if affix_embeddings is not None:
            self.affix_embeddings = nn.Embedding.from_pretrained(affix_embeddings, freeze=True)
            self.affix_to_idx = affix_to_idx
        else:
            self.affix_embeddings = None
            self.affix_to_idx = None

        # The only learned component: role attention
        self.role_attention = ASTRoleAttention()

        # Unknown root embedding (average of all roots)
        self.register_buffer('unk_embedding', root_embeddings.mean(dim=0))

    def embed_word(self, root: str, prefixes: List[str] = None,
                   suffixes: List[str] = None) -> torch.Tensor:
        """
        Compose word embedding from root + affixes.

        For now, we just use root embedding.
        Affix composition can be added later.
        """
        if root in self.root_to_idx:
            idx = self.root_to_idx[root]
            return self.root_embeddings.weight[idx]
        else:
            return self.unk_embedding

    def embed_sentence(self, words: List[Dict], negita: bool = False) -> torch.Tensor:
        """
        Compose sentence embedding from word embeddings + AST roles.

        words: List of dicts with 'root', 'role', optionally 'prefixes', 'suffixes'
        negita: Whether the sentence is negated (from AST 'negita' flag, Issue #78)

        Returns: sentence embedding (embedding_dim,)
        """
        if not words:
            return self.unk_embedding

        word_embs = []
        weights = []

        for word in words:
            root = word.get('root', word.get('radiko', ''))
            role = word.get('role', 'unknown')

            # Get word embedding
            emb = self.embed_word(root)
            word_embs.append(emb)

            # Get role attention weight
            weight = self.role_attention.get_weight(role)
            weights.append(weight)

        # Stack embeddings
        word_embs = torch.stack(word_embs)  # (num_words, embedding_dim)
        weights = torch.stack(weights)  # (num_words,)

        # Normalize weights
        weights = weights / weights.sum()

        # Weighted average
        sentence_emb = (word_embs * weights.unsqueeze(1)).sum(dim=0)

        # Apply negation transformation (Issue #78)
        # Negated sentences get a deterministic transformation to distinguish them
        # We flip the sign of half the dimensions - this ensures negated and
        # non-negated sentences are distinguishable while preserving magnitude
        if negita:
            # Flip sign of first half of embedding dimensions
            half = self.embedding_dim // 2
            sentence_emb = sentence_emb.clone()
            sentence_emb[:half] = -sentence_emb[:half]

        return sentence_emb

    def forward(self, batch_words: List[List[Dict]], batch_negita: List[bool] = None) -> torch.Tensor:
        """
        Embed a batch of sentences.

        batch_words: List of sentences, each sentence is a list of word dicts
        batch_negita: List of negation flags per sentence (from AST 'negita', Issue #78)

        Returns: (batch_size, embedding_dim)
        """
        if batch_negita is None:
            batch_negita = [False] * len(batch_words)
        embeddings = [self.embed_sentence(words, negita)
                      for words, negita in zip(batch_words, batch_negita)]
        return torch.stack(embeddings)


def extract_words_from_ast(ast: dict) -> tuple:
    """
    Extract content words with their AST roles from a parsed AST.

    Function words are excluded - they're grammatical, not semantic.
    See Wiki: "Function Word Exclusion Principle"

    Returns tuple: (words, negita)
        words: list of dicts [{'root': ..., 'role': ...}, ...]
        negita: bool indicating if sentence is negated (Issue #78)
    """
    words = []

    def extract_from_node(node, role='unknown'):
        if isinstance(node, dict):
            # Check if this is a word node
            if 'radiko' in node:
                root = node['radiko']
                # Filter out function words - they don't carry semantic meaning
                if root not in FUNCTION_WORDS:
                    words.append({
                        'root': root,
                        'role': role,
                        'vortspeco': node.get('vortspeco', 'unknown')
                    })

            # Recurse with role information
            for key, value in node.items():
                if key in AST_ROLES:
                    extract_from_node(value, role=key)
                elif key == 'kerno':
                    extract_from_node(value, role=role)
                elif key == 'priskriboj':
                    extract_from_node(value, role='aliaj')
                elif isinstance(value, (dict, list)):
                    extract_from_node(value, role=role)

        elif isinstance(node, list):
            for item in node:
                extract_from_node(item, role=role)

    extract_from_node(ast)

    # Get sentence-level negation flag (Issue #78)
    negita = ast.get('negita', False)

    return words, negita


class SentencePairDataset(Dataset):
    """Dataset of sentence pairs for contrastive learning."""

    def __init__(self, pairs: List[Tuple]):
        """
        pairs: [(words1, negita1, words2, negita2, similarity_target), ...]
        """
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def collate_fn(batch):
    """Custom collate for variable-length word lists with negation flags."""
    words1_batch = [item[0] for item in batch]
    negita1_batch = [item[1] for item in batch]
    words2_batch = [item[2] for item in batch]
    negita2_batch = [item[3] for item in batch]
    targets = torch.tensor([item[4] for item in batch], dtype=torch.float)
    return words1_batch, negita1_batch, words2_batch, negita2_batch, targets


def build_training_pairs(corpus_path: Path, max_sentences: int = 10000) -> List[Tuple]:
    """
    Build training pairs from corpus.

    Positive pairs: Sentences from same document/context
    Negative pairs: Random sentence pairs

    Each pair is: (words1, negita1, words2, negita2, similarity_target)
    """
    logger.info(f"Building training pairs from {corpus_path}")

    sentences = []
    with open(corpus_path) as f:
        for i, line in enumerate(f):
            if i >= max_sentences:
                break
            try:
                entry = json.loads(line)
                ast = entry.get('ast', {})
                words, negita = extract_words_from_ast(ast)
                if len(words) >= 2:  # Need at least 2 words
                    sentences.append({
                        'words': words,
                        'negita': negita,
                        'source': entry.get('source', 'unknown'),
                        'idx': i
                    })
            except (json.JSONDecodeError, KeyError):
                continue

    logger.info(f"Loaded {len(sentences)} sentences with AST")
    negated_count = sum(1 for s in sentences if s['negita'])
    logger.info(f"  Negated sentences: {negated_count} ({100*negated_count/len(sentences):.1f}%)")

    # Build pairs
    pairs = []

    # Group by source for positive pairs
    by_source = {}
    for sent in sentences:
        source = sent['source']
        if source not in by_source:
            by_source[source] = []
        by_source[source].append(sent)

    # Create positive pairs (same source = similar context)
    for source, sents in by_source.items():
        if len(sents) < 2:
            continue
        for i in range(min(len(sents) - 1, 100)):  # Limit pairs per source
            j = random.randint(i + 1, len(sents) - 1)
            pairs.append((
                sents[i]['words'], sents[i]['negita'],
                sents[j]['words'], sents[j]['negita'],
                1.0
            ))

    positive_count = len(pairs)
    logger.info(f"Created {positive_count} positive pairs")

    # Create negative pairs (different sources)
    sources = list(by_source.keys())
    for _ in range(positive_count):
        if len(sources) < 2:
            # If only one source, use random sentences
            s1, s2 = random.sample(sentences, 2)
        else:
            src1, src2 = random.sample(sources, 2)
            s1 = random.choice(by_source[src1])
            s2 = random.choice(by_source[src2])
        pairs.append((
            s1['words'], s1['negita'],
            s2['words'], s2['negita'],
            0.0
        ))

    logger.info(f"Total pairs: {len(pairs)}")
    return pairs


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    for words1_batch, negita1_batch, words2_batch, negita2_batch, targets in dataloader:
        targets = targets.to(device)

        # Embed sentences with negation flags (Issue #78)
        emb1 = model(words1_batch, negita1_batch)
        emb2 = model(words2_batch, negita2_batch)

        # Cosine similarity
        emb1_norm = F.normalize(emb1, dim=1)
        emb2_norm = F.normalize(emb2, dim=1)
        preds = (emb1_norm * emb2_norm).sum(dim=1)

        # MSE loss
        loss = F.mse_loss(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for words1_batch, negita1_batch, words2_batch, negita2_batch, targets in dataloader:
            targets = targets.to(device)

            # Embed sentences with negation flags (Issue #78)
            emb1 = model(words1_batch, negita1_batch)
            emb2 = model(words2_batch, negita2_batch)

            emb1_norm = F.normalize(emb1, dim=1)
            emb2_norm = F.normalize(emb2, dim=1)
            preds = (emb1_norm * emb2_norm).sum(dim=1)

            loss = F.mse_loss(preds, targets)
            total_loss += loss.item()

            # Accuracy
            pred_class = (preds > 0.5).float()
            target_class = (targets > 0.5).float()
            correct += (pred_class == target_class).sum().item()
            total += len(targets)

    return total_loss / len(dataloader), correct / total if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description='Train AST-aware sentence encoder')
    parser.add_argument('--root-model', type=Path,
                        default=Path('models/root_embeddings/best_model.pt'))
    parser.add_argument('--affix-model', type=Path,
                        default=Path('models/affix_embeddings/best_model.pt'))
    parser.add_argument('--corpus', type=Path,
                        default=Path('data/corpus_with_sources_v2.jsonl'))
    parser.add_argument('--output-dir', type=Path,
                        default=Path('models/sentence_encoder'))
    parser.add_argument('--log-dir', type=Path, default=Path('logs/training'))
    parser.add_argument('--max-sentences', type=int, default=10000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--dry-run', action='store_true')

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_dir / f'train_sentence_encoder_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    setup_file_logging(log_path)

    logger.info("=" * 60)
    logger.info("Phase 4: AST-Aware Sentence Encoding")
    logger.info("=" * 60)

    # Load root embeddings
    if not args.root_model.exists():
        logger.error(f"Root model not found: {args.root_model}")
        logger.error("Run Phase 2 (root embedding training) first")
        sys.exit(1)

    root_checkpoint = torch.load(args.root_model, map_location='cpu')
    root_embeddings = root_checkpoint['model_state_dict']['embeddings.weight']
    root_to_idx = root_checkpoint['root_to_idx']
    logger.info(f"Loaded root embeddings: {len(root_to_idx)} roots, {root_embeddings.shape[1]}d")

    # Load affix embeddings (optional)
    affix_embeddings = None
    affix_to_idx = None
    if args.affix_model.exists():
        affix_checkpoint = torch.load(args.affix_model, map_location='cpu')
        # Combine prefix and suffix embeddings
        prefix_embs = affix_checkpoint['model_state_dict']['prefix_embeddings.weight']
        suffix_embs = affix_checkpoint['model_state_dict']['suffix_embeddings.weight']
        logger.info(f"Loaded affix embeddings: {prefix_embs.shape[0]} prefixes, {suffix_embs.shape[0]} suffixes")

    # Check corpus
    if not args.corpus.exists():
        logger.error(f"Corpus not found: {args.corpus}")
        sys.exit(1)

    # Build training pairs
    pairs = build_training_pairs(args.corpus, args.max_sentences)

    if args.dry_run:
        logger.info(f"\nDry run - would train on {len(pairs)} pairs")
        logger.info(f"Model would have ~6 learned parameters (role attention weights)")
        return

    if len(pairs) < 100:
        logger.warning("Too few pairs for training")
        return

    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SentenceEncoder(root_embeddings, root_to_idx).to(device)

    # Only optimize the role attention weights
    optimizer = torch.optim.Adam(model.role_attention.parameters(), lr=args.learning_rate)

    logger.info(f"\nModel parameters:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Total: {total_params:,}")
    logger.info(f"  Trainable: {trainable_params}")  # Should be ~6!

    # Data
    random.shuffle(pairs)
    split = int(len(pairs) * 0.9)
    train_pairs, val_pairs = pairs[:split], pairs[split:]

    train_loader = DataLoader(
        SentencePairDataset(train_pairs),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        SentencePairDataset(val_pairs),
        batch_size=args.batch_size,
        collate_fn=collate_fn
    )

    logger.info(f"\nStarting training...")
    logger.info(f"Train pairs: {len(train_pairs)}, Val pairs: {len(val_pairs)}")

    best_accuracy = 0.0
    patience_counter = 0

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, accuracy = evaluate(model, val_loader, device)

        # Log role weights
        with torch.no_grad():
            weights = {role: model.role_attention.get_weight(role).item()
                      for role in AST_ROLES}

        logger.info(f"Epoch {epoch+1}/{args.epochs}: train_loss={train_loss:.4f}, "
                    f"val_loss={val_loss:.4f}, accuracy={accuracy:.4f}")
        logger.info(f"  Role weights: {weights}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            patience_counter = 0

            # Save model
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'role_weights': weights,
                'accuracy': accuracy,
                'root_to_idx': root_to_idx,
                'embedding_dim': root_embeddings.shape[1],
            }
            torch.save(checkpoint, args.output_dir / 'best_model.pt')
            logger.info(f"Saved new best model (accuracy: {accuracy:.4f})")
        else:
            patience_counter += 1

        if epoch >= 10 and patience_counter >= args.patience:
            logger.info(f"Early stopping after {args.patience} epochs without improvement")
            break

    logger.info(f"\nTraining complete! Best accuracy: {best_accuracy:.4f}")
    logger.info(f"\nFinal role weights:")
    for role, weight in weights.items():
        logger.info(f"  {role}: {weight:.3f}")


if __name__ == '__main__':
    main()
