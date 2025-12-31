#!/usr/bin/env python3
"""
Index Esperanto corpus with compositional embeddings for RAG retrieval.

Uses the trained Stage 1 models:
- Root embeddings: models/root_embeddings/best_model.pt
- Affix transforms: models/affix_transforms_v2/best_model.pt (low-rank transformation matrices)

The key difference from static embeddings:
- Affixes are TRANSFORMATIONS, not additive vectors
- mal- flips polarity by transforming the embedding
- -ej adds "place" semantics through transformation
- Composition: prefixes → root → suffixes (each transform applied in order)

Features:
- Compositional word embeddings: root → prefix_transform → suffix_transforms
- Sentence embeddings via mean pooling of content words
- Automatic checkpointing and resume
- FAISS index building for efficient similarity search

Usage:
    python scripts/index_corpus_compositional.py
    python scripts/index_corpus_compositional.py --fresh
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.parser import parse

# Function words (grammatical, not semantic) - excluded from embeddings
FUNCTION_WORDS = {
    # Articles
    'la',
    # Pronouns
    'mi', 'vi', 'li', 'ŝi', 'ĝi', 'ni', 'ili', 'oni', 'si',
    # Possessive correlatives
    'mia', 'via', 'lia', 'ŝia', 'ĝia', 'nia', 'ilia', 'sia',
    # Prepositions
    'al', 'de', 'en', 'el', 'kun', 'per', 'por', 'pri', 'sur', 'sub',
    'tra', 'trans', 'ĉe', 'ĉi', 'ĉirkaŭ', 'ekster', 'inter', 'kontraŭ',
    'antaŭ', 'post', 'super', 'apud', 'preter', 'malgraŭ', 'krom', 'laŭ',
    'anstataŭ', 'ĝis', 'sen', 'pro', 'spite',
    # Conjunctions
    'kaj', 'aŭ', 'sed', 'nek', 'ke', 'ĉar', 'se', 'dum', 'kvankam',
    'tamen', 'do', 'tial', 'ĉu',
    # Correlatives (function word subset)
    'kio', 'kiu', 'kia', 'kie', 'kiel', 'kiam', 'kiom', 'kial', 'kies',
    'tio', 'tiu', 'tia', 'tie', 'tiel', 'tiam', 'tiom', 'tial', 'ties',
    'io', 'iu', 'ia', 'ie', 'iel', 'iam', 'iom', 'ial', 'ies',
    'ĉio', 'ĉiu', 'ĉia', 'ĉie', 'ĉiel', 'ĉiam', 'ĉiom', 'ĉial', 'ĉies',
    'nenio', 'neniu', 'nenia', 'nenie', 'neniel', 'neniam', 'neniom', 'nenial', 'nenies',
    # Common adverbs that are grammatical
    'ne', 'ankaŭ', 'nur', 'eĉ', 'ja', 'jen', 'tre', 'pli', 'plej', 'tro',
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class LowRankTransform(nn.Module):
    """Low-rank transformation for affixes: x + up(down(x))"""

    def __init__(self, dim: int, rank: int = 4):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.down(x))


class CompositionalIndexer:
    """Index corpus using compositional root+affix transform embeddings."""

    def __init__(
        self,
        root_model_path: Path,
        affix_model_path: Path,
        output_dir: Path,
        batch_size: int = 100,
        root_only: bool = False,
    ):
        self.root_model_path = root_model_path
        self.affix_model_path = affix_model_path
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.root_only = root_only

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Paths
        self.checkpoint_path = self.output_dir / "indexing_checkpoint.json"
        self.embeddings_path = self.output_dir / "embeddings.npy"
        self.metadata_path = self.output_dir / "metadata.jsonl"
        self.failed_path = self.output_dir / "failed_sentences.jsonl"
        self.index_path = self.output_dir / "faiss_index.bin"
        self.log_path = self.output_dir / "indexing.log"

        # Load models
        self.device = torch.device('cpu')
        self.root_emb, self.root_to_idx, self.embedding_dim = self._load_root_model()

        # Load affix transforms unless root_only mode
        if self.root_only:
            logger.info("ROOT-ONLY MODE: Skipping affix transforms")
            self.prefix_transforms = {}
            self.suffix_transforms = {}
        else:
            self.prefix_transforms, self.suffix_transforms = self._load_affix_transforms()

        # Compute average root embedding for morpheme-only fallback
        self.avg_root_embedding = self.root_emb.mean(dim=0)

        # Stats
        self.stats = {
            "processed": 0,
            "successful": 0,
            "failed": 0,
            "fallback_proper_noun": 0,
            "fallback_char_hash": 0,
            "fallback_morpheme_only": 0,
            "total_sentences": 0,
        }

    def _load_root_model(self) -> Tuple[torch.Tensor, Dict[str, int], int]:
        """Load trained root embeddings."""
        logger.info(f"Loading root embeddings from {self.root_model_path}")
        checkpoint = torch.load(self.root_model_path, map_location='cpu', weights_only=False)
        embeddings = checkpoint['model_state_dict']['embeddings.weight']
        root_to_idx = checkpoint['root_to_idx']
        dim = embeddings.shape[1]
        logger.info(f"  Loaded {len(root_to_idx)} roots, {dim}d")
        return embeddings, root_to_idx, dim

    def _load_affix_transforms(self) -> Tuple[Dict[str, LowRankTransform], Dict[str, LowRankTransform]]:
        """Load trained affix transformation matrices."""
        logger.info(f"Loading affix transforms from {self.affix_model_path}")
        checkpoint = torch.load(self.affix_model_path, map_location='cpu', weights_only=False)

        rank = checkpoint['rank']
        prefixes = checkpoint['prefixes']
        suffixes = checkpoint['suffixes']
        state_dict = checkpoint['model_state_dict']

        # Reconstruct prefix transforms
        prefix_transforms = {}
        for p in prefixes:
            transform = LowRankTransform(self.embedding_dim, rank)
            transform.down.weight.data = state_dict[f'prefix_transforms.{p}.down.weight']
            transform.up.weight.data = state_dict[f'prefix_transforms.{p}.up.weight']
            transform.eval()
            prefix_transforms[p] = transform

        # Reconstruct suffix transforms
        suffix_transforms = {}
        for s in suffixes:
            transform = LowRankTransform(self.embedding_dim, rank)
            transform.down.weight.data = state_dict[f'suffix_transforms.{s}.down.weight']
            transform.up.weight.data = state_dict[f'suffix_transforms.{s}.up.weight']
            transform.eval()
            suffix_transforms[s] = transform

        logger.info(f"  Loaded {len(prefixes)} prefix transforms, {len(suffixes)} suffix transforms (rank={rank})")
        return prefix_transforms, suffix_transforms

    def _is_proper_noun(self, root: str) -> bool:
        """
        Detect if a root is likely a proper noun.

        Proper nouns in Esperanto are typically:
        - Capitalized (e.g., "Zamenhof", "Parizo")
        - Not in our root vocabulary (since proper nouns aren't roots)
        """
        if not root:
            return False
        # Check if first letter is uppercase
        return root[0].isupper()

    def _char_hash_embedding(self, root: str) -> torch.Tensor:
        """
        Create embedding for unknown root using character trigram hashing.

        This provides a deterministic embedding based on character patterns,
        allowing similar-looking unknown words to have similar embeddings.
        """
        root_lower = root.lower()

        # Pad with boundary markers
        padded = f"^{root_lower}$"

        # Extract character trigrams and hash them
        trigram_hashes = []
        for i in range(len(padded) - 2):
            trigram = padded[i:i+3]
            # Simple hash to bucket index
            h = hash(trigram) % self.embedding_dim
            trigram_hashes.append(h)

        # Create embedding by accumulating at hash positions
        emb = torch.zeros(self.embedding_dim)
        for h in trigram_hashes:
            emb[h] += 1.0

        # Normalize to unit length
        norm = torch.norm(emb)
        if norm > 0:
            emb = emb / norm

        # Blend with average root embedding (70% hash, 30% average)
        # This ensures unknown roots are in a similar space as known roots
        emb = 0.7 * emb + 0.3 * self.avg_root_embedding

        return emb

    def embed_word(self, root: str, prefixes: List[str], suffixes: List[str],
                   fallback_type: Optional[str] = None) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """
        Create compositional embedding for a word using transformation approach.

        Order of operations: prefixes → root → suffixes
        Each affix transforms the embedding rather than adding to it.

        Returns:
            Tuple of (embedding, fallback_type_used) where fallback_type is:
            - None: used known root embedding
            - "proper_noun": used proper noun fallback
            - "char_hash": used character n-gram hashing
            - "morpheme_only": used average root + affix transforms
        """
        used_fallback = None

        # Skip function words
        if root.lower() in FUNCTION_WORDS:
            return None, None

        # Get root embedding
        root_lower = root.lower()
        if root_lower not in self.root_to_idx:
            if root not in self.root_to_idx:
                # Unknown root - use fallback strategies
                has_known_affixes = any(p in self.prefix_transforms for p in prefixes if p)
                has_known_affixes = has_known_affixes or any(s in self.suffix_transforms for s in suffixes if s)

                if self._is_proper_noun(root):
                    # Strategy 1: Proper noun - use char hash embedding
                    # Proper nouns should be somewhat unique based on their spelling
                    emb = self._char_hash_embedding(root)
                    used_fallback = "proper_noun"
                elif has_known_affixes:
                    # Strategy 3: Morpheme-only - use average root + apply transforms
                    # The affixes give us semantic information even without the root
                    emb = self.avg_root_embedding.clone()
                    used_fallback = "morpheme_only"
                else:
                    # Strategy 2: Character n-gram hashing
                    # Unknown root with no known affixes - hash the characters
                    emb = self._char_hash_embedding(root)
                    used_fallback = "char_hash"
            else:
                root_lower = root
                root_idx = self.root_to_idx[root_lower]
                emb = self.root_emb[root_idx].clone()
        else:
            root_idx = self.root_to_idx[root_lower]
            emb = self.root_emb[root_idx].clone()

        # Apply prefix transforms (in order)
        for p in prefixes:
            if p and p in self.prefix_transforms:
                with torch.no_grad():
                    emb = self.prefix_transforms[p](emb.unsqueeze(0)).squeeze(0)

        # Apply suffix transforms (in order)
        for s in suffixes:
            if s and s in self.suffix_transforms:
                with torch.no_grad():
                    emb = self.suffix_transforms[s](emb.unsqueeze(0)).squeeze(0)

        return emb.numpy(), used_fallback

    def embed_sentence(self, text: str) -> Tuple[Optional[np.ndarray], Dict]:
        """
        Create sentence embedding by mean-pooling content word embeddings.

        Returns (embedding, metadata) where metadata includes parse info.
        """
        metadata = {
            "text": text,
            "words_total": 0,
            "words_embedded": 0,
            "roots_found": [],
            "fallbacks_used": {"proper_noun": 0, "char_hash": 0, "morpheme_only": 0},
        }

        try:
            ast = parse(text)
        except Exception as e:
            metadata["parse_error"] = str(e)
            return None, metadata

        # Extract words from AST
        word_embeddings = []

        def extract_words(node):
            if isinstance(node, dict):
                if node.get('tipo') == 'vorto':
                    metadata["words_total"] += 1
                    root = node.get('radiko', '')

                    # Get prefixes (handle both 'prefiksoj' list and 'prefikso' string)
                    prefixes = node.get('prefiksoj', [])
                    if not prefixes:
                        p = node.get('prefikso')
                        if p:
                            prefixes = [p]

                    # Get suffixes
                    suffixes = node.get('sufiksoj', [])

                    emb, fallback_type = self.embed_word(root, prefixes, suffixes)
                    if emb is not None:
                        word_embeddings.append(emb)
                        metadata["words_embedded"] += 1
                        metadata["roots_found"].append(root)
                        # Track fallback usage
                        if fallback_type:
                            metadata["fallbacks_used"][fallback_type] += 1

                for v in node.values():
                    extract_words(v)
            elif isinstance(node, list):
                for item in node:
                    extract_words(item)

        extract_words(ast)

        if not word_embeddings:
            return None, metadata

        # Mean pooling
        sentence_emb = np.mean(word_embeddings, axis=0)
        # Normalize
        norm = np.linalg.norm(sentence_emb)
        if norm > 0:
            sentence_emb = sentence_emb / norm

        return sentence_emb, metadata

    def load_checkpoint(self) -> int:
        """Load checkpoint and return number of processed sentences."""
        if self.checkpoint_path.exists():
            with open(self.checkpoint_path) as f:
                self.stats = json.load(f)
            return self.stats["processed"]
        return 0

    def save_checkpoint(self):
        """Save current progress."""
        temp_path = self.checkpoint_path.with_suffix('.tmp')
        with open(temp_path, 'w') as f:
            json.dump(self.stats, f)
        temp_path.rename(self.checkpoint_path)

    def index_corpus(self, corpus_path: Path, resume: bool = True):
        """Index entire corpus."""
        # Setup file logging
        file_handler = logging.FileHandler(self.log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)

        logger.info("=" * 60)
        logger.info("Compositional Corpus Indexing (Transform-based)")
        logger.info("=" * 60)
        logger.info(f"Corpus: {corpus_path}")
        logger.info(f"Output: {self.output_dir}")
        logger.info(f"Root embeddings: {self.embedding_dim}d, {len(self.root_to_idx)} roots")
        logger.info(f"Prefix transforms: {len(self.prefix_transforms)}")
        logger.info(f"Suffix transforms: {len(self.suffix_transforms)}")

        # Load corpus
        with open(corpus_path) as f:
            if corpus_path.suffix == '.jsonl':
                sentences = []
                for line in f:
                    data = json.loads(line)
                    sentences.append(data.get('text', ''))
            else:
                sentences = [line.strip() for line in f if line.strip()]

        self.stats["total_sentences"] = len(sentences)
        logger.info(f"Total sentences: {len(sentences)}")

        # Resume from checkpoint
        start_idx = 0
        if resume and self.checkpoint_path.exists():
            start_idx = self.load_checkpoint()
            logger.info(f"Resuming from sentence {start_idx}")

        # Load existing embeddings if resuming
        if start_idx > 0 and self.embeddings_path.exists():
            all_embeddings = list(np.load(self.embeddings_path))
            logger.info(f"Loaded {len(all_embeddings)} existing embeddings")
        else:
            all_embeddings = []
            start_idx = 0
            self.stats = {
                "processed": 0,
                "successful": 0,
                "failed": 0,
                "fallback_proper_noun": 0,
                "fallback_char_hash": 0,
                "fallback_morpheme_only": 0,
                "total_sentences": len(sentences)
            }

        # Open metadata and failed files
        metadata_mode = 'a' if start_idx > 0 else 'w'
        metadata_file = open(self.metadata_path, metadata_mode)
        failed_file = open(self.failed_path, metadata_mode)

        try:
            start_time = time.time()
            last_scroll_pct = 0  # Track last percentage we scrolled at
            last_scroll_time = start_time  # Track last time we scrolled

            for i in range(start_idx, len(sentences), self.batch_size):
                batch = sentences[i:i + self.batch_size]
                batch_embeddings = []

                for text in batch:
                    emb, meta = self.embed_sentence(text)

                    if emb is not None:
                        batch_embeddings.append(emb)
                        meta["index"] = len(all_embeddings) + len(batch_embeddings) - 1
                        metadata_file.write(json.dumps(meta, ensure_ascii=False) + '\n')
                        self.stats["successful"] += 1
                        # Track fallback usage from metadata
                        fallbacks = meta.get("fallbacks_used", {})
                        self.stats["fallback_proper_noun"] += fallbacks.get("proper_noun", 0)
                        self.stats["fallback_char_hash"] += fallbacks.get("char_hash", 0)
                        self.stats["fallback_morpheme_only"] += fallbacks.get("morpheme_only", 0)
                    else:
                        failed_file.write(json.dumps(meta, ensure_ascii=False) + '\n')
                        self.stats["failed"] += 1

                    self.stats["processed"] += 1

                all_embeddings.extend(batch_embeddings)

                # Progress update - in place on terminal, scroll at milestones
                if (i + len(batch)) % 1000 == 0 or i + len(batch) == len(sentences):
                    elapsed = time.time() - start_time
                    rate = (self.stats["processed"] - start_idx) / elapsed if elapsed > 0 else 0
                    remaining = (len(sentences) - self.stats["processed"]) / rate if rate > 0 else 0
                    pct = 100 * self.stats["processed"] / len(sentences)
                    current_time = time.time()

                    progress_msg = (f"Progress: {self.stats['processed']:,}/{len(sentences):,} ({pct:.1f}%) "
                                   f"| {self.stats['successful']:,} OK, {self.stats['failed']:,} failed "
                                   f"| {rate:.0f}/s | ETA: {remaining/60:.1f}m")

                    # Scroll (print with newline) at 10% milestones OR every 10 minutes
                    current_pct_milestone = int(pct // 10) * 10  # 0, 10, 20, ... 100
                    time_since_scroll = current_time - last_scroll_time

                    if current_pct_milestone > last_scroll_pct or time_since_scroll >= 600:
                        # Scroll: print with newline for history
                        print(f"\r{progress_msg:<100}")
                        last_scroll_pct = current_pct_milestone
                        last_scroll_time = current_time
                        logger.info(progress_msg)  # Also log to file
                    else:
                        # In-place update
                        print(f"\r{progress_msg:<100}", end='', flush=True)

                # Save checkpoint every 5000 sentences
                if self.stats["processed"] % 5000 == 0:
                    self.save_checkpoint()
                    # Save embeddings incrementally
                    np.save(self.embeddings_path, np.array(all_embeddings, dtype=np.float32))

        finally:
            metadata_file.close()
            failed_file.close()

        # End progress line
        print()  # Newline after in-place progress

        # Save final embeddings
        embeddings_array = np.array(all_embeddings, dtype=np.float32)
        np.save(self.embeddings_path, embeddings_array)
        logger.info(f"Saved embeddings: {embeddings_array.shape}")

        # Build FAISS index
        self._build_faiss_index(embeddings_array)

        # Save final checkpoint
        self.save_checkpoint()

        logger.info("")
        logger.info("=" * 60)
        logger.info("Indexing Complete")
        logger.info("=" * 60)
        logger.info(f"Total: {self.stats['total_sentences']}")
        logger.info(f"Successful: {self.stats['successful']} ({self.stats['successful']/self.stats['total_sentences']*100:.1f}%)")
        logger.info(f"Failed: {self.stats['failed']}")
        # Report fallback usage
        total_fallbacks = (self.stats.get('fallback_proper_noun', 0) +
                          self.stats.get('fallback_char_hash', 0) +
                          self.stats.get('fallback_morpheme_only', 0))
        if total_fallbacks > 0:
            logger.info(f"Fallback words embedded: {total_fallbacks}")
            logger.info(f"  - Proper noun: {self.stats.get('fallback_proper_noun', 0)}")
            logger.info(f"  - Char hash: {self.stats.get('fallback_char_hash', 0)}")
            logger.info(f"  - Morpheme-only: {self.stats.get('fallback_morpheme_only', 0)}")

    def _build_faiss_index(self, embeddings: np.ndarray):
        """Build FAISS index for fast similarity search."""
        try:
            import faiss
        except ImportError:
            logger.warning("FAISS not installed, skipping index build")
            logger.warning("Install with: pip install faiss-cpu")
            return

        logger.info(f"Building FAISS index for {len(embeddings)} embeddings...")

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)

        # Create index
        index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product = cosine for normalized vectors
        index.add(embeddings)

        # Save index
        faiss.write_index(index, str(self.index_path))
        logger.info(f"Saved FAISS index: {self.index_path}")


def main():
    parser = argparse.ArgumentParser(description='Index corpus with compositional embeddings')
    parser.add_argument('--corpus', type=Path,
                        default=Path('data/corpus/unified_corpus.jsonl'),
                        help='Path to corpus file (.txt or .jsonl)')
    parser.add_argument('--root-model', type=Path,
                        default=Path('models/root_embeddings/best_model.pt'))
    parser.add_argument('--affix-model', type=Path,
                        default=Path('models/affix_transforms_v2/best_model.pt'))
    parser.add_argument('--output-dir', type=Path,
                        default=Path('data/corpus_index_compositional'))
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--resume', action='store_true', default=True,
                        help='Resume from checkpoint (default)')
    parser.add_argument('--fresh', action='store_true',
                        help='Start fresh, ignore checkpoint')
    parser.add_argument('--root-only', action='store_true',
                        help='Use only root embeddings, skip affix transforms (for testing)')

    args = parser.parse_args()

    # Validate inputs
    if not args.corpus.exists():
        logger.error(f"Corpus not found: {args.corpus}")
        sys.exit(1)
    if not args.root_model.exists():
        logger.error(f"Root model not found: {args.root_model}")
        sys.exit(1)
    if not args.root_only and not args.affix_model.exists():
        logger.error(f"Affix model not found: {args.affix_model}")
        sys.exit(1)

    indexer = CompositionalIndexer(
        root_model_path=args.root_model,
        affix_model_path=args.affix_model,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        root_only=args.root_only,
    )

    indexer.index_corpus(args.corpus, resume=not args.fresh)


if __name__ == '__main__':
    main()
