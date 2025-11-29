"""
Compositional Embeddings for Esperanto Morphemes.

Implements embeddings that compose word representations from:
- Root (radiko) embedding
- Prefix (prefikso) embedding (if present)
- Suffix (sufiksoj) embeddings (if present)
- Grammatical ending embedding

This approach:
1. Reduces vocabulary size dramatically (75%+ parameter reduction)
2. Enables handling of novel words through morpheme composition
3. Captures Esperanto's regular morphological structure

Word embedding = f(root, prefix, suffixes, ending)
"""

import json
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .unknown_tracker import UnknownRootTracker

logger = logging.getLogger(__name__)


# Esperanto grammatical endings
ENDINGS = {
    'o': 0,   # noun singular nominative
    'on': 1,  # noun singular accusative
    'oj': 2,  # noun plural nominative
    'ojn': 3, # noun plural accusative
    'a': 4,   # adjective singular nominative
    'an': 5,  # adjective singular accusative
    'aj': 6,  # adjective plural nominative
    'ajn': 7, # adjective plural accusative
    'e': 8,   # adverb
    'en': 9,  # adverb accusative (directional)
    'i': 10,  # verb infinitive
    'as': 11, # verb present
    'is': 12, # verb past
    'os': 13, # verb future
    'us': 14, # verb conditional
    'u': 15,  # verb imperative
    '<NONE>': 16,  # no ending (particles, etc.)
}


class CompositionalEmbedding(nn.Module):
    """
    Compositional embedding layer for Esperanto words.

    Composes word embeddings from morpheme components using learnable
    embeddings for roots, prefixes, suffixes, and grammatical endings.
    """

    def __init__(
        self,
        root_vocab: Dict[str, int],
        prefix_vocab: Dict[str, int],
        suffix_vocab: Dict[str, int],
        embed_dim: int = 128,
        composition_method: str = 'sum',
        dropout: float = 0.1,
    ):
        """
        Initialize compositional embedding layer.

        Args:
            root_vocab: Mapping from root strings to indices
            prefix_vocab: Mapping from prefix strings to indices
            suffix_vocab: Mapping from suffix strings to indices
            embed_dim: Dimension of embeddings
            composition_method: How to combine morphemes ('sum', 'concat', 'gated')
            dropout: Dropout probability
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.composition_method = composition_method

        # Store vocabularies
        self.root_vocab = root_vocab
        self.prefix_vocab = prefix_vocab
        self.suffix_vocab = suffix_vocab
        self.ending_vocab = ENDINGS

        # Embedding layers
        self.root_embed = nn.Embedding(len(root_vocab), embed_dim)
        self.prefix_embed = nn.Embedding(len(prefix_vocab), embed_dim)
        self.suffix_embed = nn.Embedding(len(suffix_vocab), embed_dim)
        self.ending_embed = nn.Embedding(len(ENDINGS), embed_dim)

        # Composition-specific layers
        if composition_method == 'gated':
            # Gated composition: learn how to weight each component
            self.gate_linear = nn.Linear(embed_dim * 4, 4)
            self.output_proj = nn.Linear(embed_dim, embed_dim)
        elif composition_method == 'concat':
            # Concatenation followed by projection
            self.output_proj = nn.Linear(embed_dim * 4, embed_dim)
        elif composition_method == 'attention':
            # Multi-head attention over morpheme embeddings
            self.attention = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
            self.output_proj = nn.Linear(embed_dim, embed_dim)
        else:
            # Sum composition (simplest, often works well)
            self.output_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

        # Unknown root tracker (optional, for vocabulary expansion)
        self._unknown_tracker: Optional['UnknownRootTracker'] = None
        self._track_unknowns = False

        # Initialize embeddings
        self._init_embeddings()

    def _init_embeddings(self):
        """Initialize embedding weights."""
        nn.init.normal_(self.root_embed.weight, mean=0, std=0.1)
        nn.init.normal_(self.prefix_embed.weight, mean=0, std=0.1)
        nn.init.normal_(self.suffix_embed.weight, mean=0, std=0.1)
        nn.init.normal_(self.ending_embed.weight, mean=0, std=0.1)

        # Set <PAD>, <UNK>, <NONE> tokens to zero
        with torch.no_grad():
            if '<PAD>' in self.root_vocab:
                self.root_embed.weight[self.root_vocab['<PAD>']].zero_()
            if '<NONE>' in self.prefix_vocab:
                self.prefix_embed.weight[self.prefix_vocab['<NONE>']].zero_()
            if '<NONE>' in self.suffix_vocab:
                self.suffix_embed.weight[self.suffix_vocab['<NONE>']].zero_()

    def get_root_idx(
        self,
        root: str,
        sentence: Optional[str] = None,
        word: Optional[str] = None,
    ) -> int:
        """Get index for root, using <UNK> for unknown."""
        if root in self.root_vocab:
            return self.root_vocab[root]

        # Unknown root - track if enabled
        if self._track_unknowns and self._unknown_tracker is not None:
            self._unknown_tracker.log(root, sentence=sentence, word=word)

        return self.root_vocab.get('<UNK>', 0)

    def enable_unknown_tracking(
        self,
        tracker: Optional['UnknownRootTracker'] = None,
        storage_path: Optional[Path] = None,
    ) -> None:
        """
        Enable tracking of unknown roots for vocabulary expansion.

        Args:
            tracker: Existing tracker to use
            storage_path: Path for new tracker (if tracker not provided)
        """
        if tracker is not None:
            self._unknown_tracker = tracker
        else:
            from .unknown_tracker import UnknownRootTracker
            default_path = storage_path or Path("data/unknown_roots.json")
            self._unknown_tracker = UnknownRootTracker(default_path)

        self._track_unknowns = True
        logger.info("Unknown root tracking enabled")

    def disable_unknown_tracking(self) -> None:
        """Disable unknown root tracking."""
        self._track_unknowns = False
        if self._unknown_tracker:
            self._unknown_tracker.save()
        logger.info("Unknown root tracking disabled")

    def get_unknown_tracker(self) -> Optional['UnknownRootTracker']:
        """Get the unknown root tracker if enabled."""
        return self._unknown_tracker

    def get_prefix_idx(self, prefix: Optional[str]) -> int:
        """Get index for prefix, using <NONE> if absent."""
        if not prefix:
            return self.prefix_vocab.get('<NONE>', 0)
        return self.prefix_vocab.get(prefix, self.prefix_vocab.get('<NONE>', 0))

    def get_suffix_idx(self, suffix: str) -> int:
        """Get index for suffix, using <NONE> for unknown."""
        return self.suffix_vocab.get(suffix, self.suffix_vocab.get('<NONE>', 0))

    def get_ending_idx(self, ending: Optional[str]) -> int:
        """Get index for grammatical ending."""
        if not ending:
            return ENDINGS.get('<NONE>', 0)
        return ENDINGS.get(ending, ENDINGS.get('<NONE>', 0))

    def encode_word(
        self,
        root: str,
        prefix: Optional[str] = None,
        suffixes: Optional[List[str]] = None,
        ending: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Encode a single word from its morpheme components.

        Args:
            root: Word root (radiko)
            prefix: Optional prefix (prefikso)
            suffixes: Optional list of suffixes (sufiksoj)
            ending: Grammatical ending

        Returns:
            Word embedding tensor (embed_dim,)
        """
        device = self.root_embed.weight.device

        # Get embeddings for each component
        root_idx = torch.tensor([self.get_root_idx(root)], device=device)
        root_emb = self.root_embed(root_idx).squeeze(0)

        prefix_idx = torch.tensor([self.get_prefix_idx(prefix)], device=device)
        prefix_emb = self.prefix_embed(prefix_idx).squeeze(0)

        # Average suffix embeddings if multiple
        if suffixes:
            suffix_idxs = torch.tensor(
                [self.get_suffix_idx(s) for s in suffixes],
                device=device
            )
            suffix_emb = self.suffix_embed(suffix_idxs).mean(dim=0)
        else:
            suffix_idx = torch.tensor([self.suffix_vocab.get('<NONE>', 0)], device=device)
            suffix_emb = self.suffix_embed(suffix_idx).squeeze(0)

        ending_idx = torch.tensor([self.get_ending_idx(ending)], device=device)
        ending_emb = self.ending_embed(ending_idx).squeeze(0)

        # Compose embeddings
        return self._compose(root_emb, prefix_emb, suffix_emb, ending_emb)

    def _compose(
        self,
        root_emb: torch.Tensor,
        prefix_emb: torch.Tensor,
        suffix_emb: torch.Tensor,
        ending_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compose morpheme embeddings into word embedding.

        Args:
            root_emb: Root embedding (embed_dim,)
            prefix_emb: Prefix embedding (embed_dim,)
            suffix_emb: Suffix embedding (embed_dim,)
            ending_emb: Ending embedding (embed_dim,)

        Returns:
            Composed word embedding (embed_dim,)
        """
        if self.composition_method == 'sum':
            # Simple sum with projection
            composed = root_emb + prefix_emb + suffix_emb + ending_emb
            composed = self.output_proj(composed)

        elif self.composition_method == 'concat':
            # Concatenate and project
            composed = torch.cat([root_emb, prefix_emb, suffix_emb, ending_emb], dim=-1)
            composed = self.output_proj(composed)

        elif self.composition_method == 'gated':
            # Learned gating mechanism
            all_embs = torch.stack([root_emb, prefix_emb, suffix_emb, ending_emb], dim=0)
            gate_input = torch.cat([root_emb, prefix_emb, suffix_emb, ending_emb], dim=-1)
            gates = F.softmax(self.gate_linear(gate_input), dim=-1)
            composed = (all_embs * gates.unsqueeze(-1)).sum(dim=0)
            composed = self.output_proj(composed)

        elif self.composition_method == 'attention':
            # Self-attention over morpheme embeddings
            all_embs = torch.stack([root_emb, prefix_emb, suffix_emb, ending_emb], dim=0)
            all_embs = all_embs.unsqueeze(0)  # (1, 4, embed_dim)
            attended, _ = self.attention(all_embs, all_embs, all_embs)
            composed = attended.squeeze(0).mean(dim=0)  # Average attended embeddings
            composed = self.output_proj(composed)

        # Apply layer norm and dropout
        composed = self.layer_norm(composed)
        composed = self.dropout(composed)

        return composed

    def forward_batch(
        self,
        root_indices: torch.Tensor,
        prefix_indices: torch.Tensor,
        suffix_indices: torch.Tensor,
        suffix_mask: torch.Tensor,
        ending_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Batch forward pass for efficiency.

        Args:
            root_indices: (batch_size,) root vocabulary indices
            prefix_indices: (batch_size,) prefix vocabulary indices
            suffix_indices: (batch_size, max_suffixes) suffix indices
            suffix_mask: (batch_size, max_suffixes) mask for valid suffixes
            ending_indices: (batch_size,) ending indices

        Returns:
            Word embeddings (batch_size, embed_dim)
        """
        batch_size = root_indices.size(0)

        # Get embeddings
        root_emb = self.root_embed(root_indices)  # (batch, embed_dim)
        prefix_emb = self.prefix_embed(prefix_indices)  # (batch, embed_dim)
        ending_emb = self.ending_embed(ending_indices)  # (batch, embed_dim)

        # Handle variable-length suffixes
        suffix_emb = self.suffix_embed(suffix_indices)  # (batch, max_suf, embed_dim)
        # Masked average
        suffix_mask_expanded = suffix_mask.unsqueeze(-1).float()  # (batch, max_suf, 1)
        suffix_sum = (suffix_emb * suffix_mask_expanded).sum(dim=1)  # (batch, embed_dim)
        suffix_count = suffix_mask.sum(dim=1, keepdim=True).clamp(min=1).float()  # (batch, 1)
        suffix_emb = suffix_sum / suffix_count  # (batch, embed_dim)

        # Compose (vectorized)
        if self.composition_method == 'sum':
            composed = root_emb + prefix_emb + suffix_emb + ending_emb
            composed = self.output_proj(composed)

        elif self.composition_method == 'concat':
            composed = torch.cat([root_emb, prefix_emb, suffix_emb, ending_emb], dim=-1)
            composed = self.output_proj(composed)

        elif self.composition_method == 'gated':
            # Stack for gating: (batch, 4, embed_dim)
            all_embs = torch.stack([root_emb, prefix_emb, suffix_emb, ending_emb], dim=1)
            gate_input = torch.cat([root_emb, prefix_emb, suffix_emb, ending_emb], dim=-1)
            gates = F.softmax(self.gate_linear(gate_input), dim=-1)  # (batch, 4)
            composed = (all_embs * gates.unsqueeze(-1)).sum(dim=1)  # (batch, embed_dim)
            composed = self.output_proj(composed)

        elif self.composition_method == 'attention':
            # (batch, 4, embed_dim)
            all_embs = torch.stack([root_emb, prefix_emb, suffix_emb, ending_emb], dim=1)
            attended, _ = self.attention(all_embs, all_embs, all_embs)
            composed = attended.mean(dim=1)  # (batch, embed_dim)
            composed = self.output_proj(composed)

        composed = self.layer_norm(composed)
        composed = self.dropout(composed)

        return composed

    @classmethod
    def from_vocabulary_files(
        cls,
        vocab_dir: Path,
        embed_dim: int = 128,
        composition_method: str = 'sum',
        dropout: float = 0.1,
    ) -> 'CompositionalEmbedding':
        """
        Create CompositionalEmbedding from vocabulary files.

        Args:
            vocab_dir: Directory containing vocabulary JSON files
            embed_dim: Embedding dimension
            composition_method: Composition method
            dropout: Dropout rate

        Returns:
            Initialized CompositionalEmbedding
        """
        vocab_dir = Path(vocab_dir)

        # Load root vocabulary
        with open(vocab_dir / 'root_vocabulary.json', 'r', encoding='utf-8') as f:
            root_vocab = json.load(f)

        # Load affix vocabularies
        with open(vocab_dir / 'affix_vocabulary.json', 'r', encoding='utf-8') as f:
            affix_data = json.load(f)

        prefix_vocab = affix_data['prefixes']
        suffix_vocab = affix_data['suffixes']

        return cls(
            root_vocab=root_vocab,
            prefix_vocab=prefix_vocab,
            suffix_vocab=suffix_vocab,
            embed_dim=embed_dim,
            composition_method=composition_method,
            dropout=dropout,
        )

    def get_vocab_stats(self) -> Dict:
        """Get vocabulary statistics."""
        return {
            'num_roots': len(self.root_vocab),
            'num_prefixes': len(self.prefix_vocab),
            'num_suffixes': len(self.suffix_vocab),
            'num_endings': len(self.ending_vocab),
            'embed_dim': self.embed_dim,
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'composition_method': self.composition_method,
        }

    def expand_vocabulary(
        self,
        new_roots: List[str],
        initialization: str = 'average',
        similar_roots: Optional[Dict[str, List[str]]] = None,
    ) -> int:
        """
        Expand vocabulary with new roots.

        Args:
            new_roots: List of new root strings to add
            initialization: How to initialize new embeddings:
                - 'average': Average of all existing embeddings
                - 'similar': Average of similar roots (requires similar_roots)
                - 'random': Random initialization
                - 'zero': Zero initialization
            similar_roots: Optional mapping from new root to similar existing roots

        Returns:
            Number of roots actually added (excludes duplicates)
        """
        # Filter out roots already in vocabulary
        roots_to_add = [r for r in new_roots if r not in self.root_vocab]

        if not roots_to_add:
            logger.info("No new roots to add (all already in vocabulary)")
            return 0

        old_vocab_size = len(self.root_vocab)
        old_weights = self.root_embed.weight.data.clone()

        # Add new roots to vocabulary
        for root in roots_to_add:
            self.root_vocab[root] = len(self.root_vocab)

        # Create new embedding layer with expanded size
        new_vocab_size = len(self.root_vocab)
        new_embed = nn.Embedding(new_vocab_size, self.embed_dim)

        # Copy old weights
        new_embed.weight.data[:old_vocab_size] = old_weights

        # Initialize new embeddings
        with torch.no_grad():
            for i, root in enumerate(roots_to_add):
                new_idx = old_vocab_size + i

                if initialization == 'average':
                    # Average of all non-special embeddings
                    # Skip <PAD>, <UNK>, etc. (assume first few indices)
                    start_idx = min(5, old_vocab_size)
                    new_embed.weight.data[new_idx] = old_weights[start_idx:].mean(dim=0)

                elif initialization == 'similar' and similar_roots:
                    # Average of similar roots
                    if root in similar_roots:
                        similar_idxs = [
                            self.root_vocab[r] for r in similar_roots[root]
                            if r in self.root_vocab and self.root_vocab[r] < old_vocab_size
                        ]
                        if similar_idxs:
                            new_embed.weight.data[new_idx] = old_weights[similar_idxs].mean(dim=0)
                        else:
                            new_embed.weight.data[new_idx] = old_weights[5:].mean(dim=0)
                    else:
                        new_embed.weight.data[new_idx] = old_weights[5:].mean(dim=0)

                elif initialization == 'random':
                    nn.init.normal_(new_embed.weight.data[new_idx], mean=0, std=0.1)

                elif initialization == 'zero':
                    new_embed.weight.data[new_idx].zero_()

                else:
                    # Default to average
                    new_embed.weight.data[new_idx] = old_weights[5:].mean(dim=0)

        # Replace embedding layer
        self.root_embed = new_embed.to(old_weights.device)

        # Mark roots as added in tracker
        if self._unknown_tracker:
            self._unknown_tracker.mark_added(roots_to_add)

        logger.info(f"Expanded vocabulary: {old_vocab_size} -> {new_vocab_size} (+{len(roots_to_add)} roots)")
        return len(roots_to_add)

    def save_vocabularies(self, vocab_dir: Path) -> None:
        """
        Save current vocabularies to files.

        Args:
            vocab_dir: Directory to save vocabulary files
        """
        vocab_dir = Path(vocab_dir)
        vocab_dir.mkdir(parents=True, exist_ok=True)

        # Save root vocabulary
        with open(vocab_dir / 'root_vocabulary.json', 'w', encoding='utf-8') as f:
            json.dump(self.root_vocab, f, ensure_ascii=False, indent=2)

        # Save affix vocabulary
        affix_data = {
            'prefixes': self.prefix_vocab,
            'suffixes': self.suffix_vocab,
        }
        with open(vocab_dir / 'affix_vocabulary.json', 'w', encoding='utf-8') as f:
            json.dump(affix_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved vocabularies to {vocab_dir}")

    def get_new_root_embeddings_for_finetuning(
        self,
        new_root_start_idx: int,
    ) -> List[nn.Parameter]:
        """
        Get parameters for fine-tuning only new roots.

        Usage:
            # After expanding vocabulary
            new_params = model.get_new_root_embeddings_for_finetuning(old_vocab_size)
            optimizer = torch.optim.Adam(new_params, lr=1e-3)

        Args:
            new_root_start_idx: Index where new roots start

        Returns:
            List of parameters (the embedding weights slice)
        """
        # Create a wrapper that only optimizes the new embeddings
        # This is a bit hacky but works for fine-tuning
        class NewRootEmbeddingWrapper(nn.Module):
            def __init__(self, embed_layer, start_idx):
                super().__init__()
                self.embed_layer = embed_layer
                self.start_idx = start_idx
                # Register the slice as a parameter
                self.new_embeddings = nn.Parameter(
                    embed_layer.weight.data[start_idx:].clone()
                )

            def sync_back(self):
                """Copy fine-tuned embeddings back to main embedding layer."""
                with torch.no_grad():
                    self.embed_layer.weight.data[self.start_idx:] = self.new_embeddings.data

        wrapper = NewRootEmbeddingWrapper(self.root_embed, new_root_start_idx)
        return [wrapper.new_embeddings], wrapper


def main():
    """Test compositional embedding."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    print("Testing Compositional Embeddings")
    print("=" * 60)

    # Check if vocabulary files exist
    vocab_dir = Path(__file__).parent.parent.parent / 'data' / 'vocabularies'

    if not vocab_dir.exists():
        print(f"Vocabulary directory not found: {vocab_dir}")
        print("Creating test vocabularies...")

        # Create test vocabularies
        root_vocab = {'hund': 0, 'kat': 1, 'bird': 2, '<PAD>': 3, '<UNK>': 4}
        prefix_vocab = {'mal': 0, 're': 1, '<NONE>': 2}
        suffix_vocab = {'et': 0, 'eg': 1, 'ul': 2, '<NONE>': 3}

        emb = CompositionalEmbedding(
            root_vocab=root_vocab,
            prefix_vocab=prefix_vocab,
            suffix_vocab=suffix_vocab,
            embed_dim=64,
            composition_method='sum',
        )
    else:
        print(f"Loading vocabularies from: {vocab_dir}")
        emb = CompositionalEmbedding.from_vocabulary_files(vocab_dir, embed_dim=64)

    print("\nVocabulary Statistics:")
    stats = emb.get_vocab_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Test single word encoding
    print("\nTesting word encoding:")

    # Test: hundeto (little dog)
    embedding = emb.encode_word(
        root='hund',
        prefix=None,
        suffixes=['et'],
        ending='o'
    )
    print(f"  hundeto: shape={embedding.shape}, norm={embedding.norm().item():.4f}")

    # Test: malhundo (opposite-dog, nonsense but valid morphology)
    embedding = emb.encode_word(
        root='hund',
        prefix='mal',
        suffixes=None,
        ending='o'
    )
    print(f"  malhundo: shape={embedding.shape}, norm={embedding.norm().item():.4f}")

    # Test: katidoj (kittens)
    if 'kat' in emb.root_vocab or '<UNK>' in emb.root_vocab:
        embedding = emb.encode_word(
            root='kat',
            prefix=None,
            suffixes=['id'],
            ending='oj'
        )
        print(f"  katidoj: shape={embedding.shape}, norm={embedding.norm().item():.4f}")

    # Test unknown root (should use <UNK>)
    embedding = emb.encode_word(
        root='zzzzz',
        prefix=None,
        suffixes=None,
        ending='o'
    )
    print(f"  zzzzz'o (unknown): shape={embedding.shape}, norm={embedding.norm().item():.4f}")

    print("\n✅ Compositional embedding test successful!")


if __name__ == '__main__':
    main()
