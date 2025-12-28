#!/usr/bin/env python3
"""
Index Esperanto corpus with compositional embeddings (root + affix) for RAG retrieval.

Uses the trained Stage 1 models:
- Root embeddings: models/root_embeddings/best_model.pt
- Affix embeddings: models/affix_embeddings/best_model.pt

Features:
- Compositional word embeddings: root + prefix_transform + suffix_transforms
- Sentence embeddings via mean pooling of content words
- Automatic checkpointing and resume
- FAISS index building for efficient similarity search

Usage:
    python scripts/index_corpus_compositional.py
    python scripts/index_corpus_compositional.py --resume
    python scripts/index_corpus_compositional.py --fresh
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.parser import parse

# Function words (grammatical, not semantic) - excluded from embeddings
# These are handled by the deterministic AST layer, not learned embeddings
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


class CompositionalIndexer:
    """Index corpus using compositional root+affix embeddings."""

    def __init__(
        self,
        root_model_path: Path,
        affix_model_path: Path,
        output_dir: Path,
        batch_size: int = 100,
    ):
        self.root_model_path = root_model_path
        self.affix_model_path = affix_model_path
        self.output_dir = output_dir
        self.batch_size = batch_size

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Paths
        self.checkpoint_path = self.output_dir / "indexing_checkpoint.json"
        self.embeddings_path = self.output_dir / "embeddings.npy"
        self.metadata_path = self.output_dir / "metadata.jsonl"
        self.failed_path = self.output_dir / "failed_sentences.jsonl"
        self.index_path = self.output_dir / "faiss_index.bin"
        self.log_path = self.output_dir / "indexing.log"

        # Load models
        self.root_emb, self.root_to_idx, self.root_dim = self._load_root_model()
        self.prefix_emb, self.suffix_emb, self.prefix_vocab, self.suffix_vocab, self.affix_dim = self._load_affix_model()

        # Output dimension is root dimension (we project affixes to match)
        self.embedding_dim = self.root_dim

        # Stats
        self.stats = {
            "processed": 0,
            "successful": 0,
            "failed": 0,
            "total_sentences": 0,
        }

    def _load_root_model(self) -> Tuple[np.ndarray, Dict[str, int], int]:
        """Load trained root embeddings."""
        logger.info(f"Loading root embeddings from {self.root_model_path}")
        checkpoint = torch.load(self.root_model_path, map_location='cpu', weights_only=False)
        embeddings = checkpoint['model_state_dict']['embeddings.weight'].numpy()
        root_to_idx = checkpoint['root_to_idx']
        dim = embeddings.shape[1]
        logger.info(f"  Loaded {len(root_to_idx)} roots, {dim}d")
        return embeddings, root_to_idx, dim

    def _load_affix_model(self) -> Tuple[np.ndarray, np.ndarray, Dict, Dict, int]:
        """Load trained affix embeddings."""
        logger.info(f"Loading affix embeddings from {self.affix_model_path}")
        checkpoint = torch.load(self.affix_model_path, map_location='cpu', weights_only=False)
        prefix_emb = checkpoint['model_state_dict']['prefix_embeddings.weight'].numpy()
        suffix_emb = checkpoint['model_state_dict']['suffix_embeddings.weight'].numpy()
        prefix_vocab = checkpoint['prefix_vocab']
        suffix_vocab = checkpoint['suffix_vocab']
        dim = prefix_emb.shape[1]
        logger.info(f"  Loaded {len(prefix_vocab)} prefixes, {len(suffix_vocab)} suffixes, {dim}d")
        return prefix_emb, suffix_emb, prefix_vocab, suffix_vocab, dim

    def embed_word(self, root: str, prefixes: List[str], suffixes: List[str]) -> Optional[np.ndarray]:
        """
        Create compositional embedding for a word.

        Approach: root embedding + scaled affix embeddings (projected to root dimension)
        """
        # Skip function words
        if root.lower() in FUNCTION_WORDS:
            return None

        # Get root embedding
        if root not in self.root_to_idx:
            # Try lowercase
            root_lower = root.lower()
            if root_lower not in self.root_to_idx:
                return None
            root = root_lower

        root_idx = self.root_to_idx[root]
        emb = self.root_emb[root_idx].copy()

        # Add prefix contributions (scaled and zero-padded to root dimension)
        for p in prefixes:
            if p and p in self.prefix_vocab and p != '<NONE>':
                prefix_idx = self.prefix_vocab[p]
                prefix_vec = self.prefix_emb[prefix_idx]
                # Zero-pad or truncate to match root dimension
                if self.affix_dim < self.root_dim:
                    padded = np.zeros(self.root_dim)
                    padded[:self.affix_dim] = prefix_vec * 0.3  # Scale down affix contribution
                    emb = emb + padded
                else:
                    emb = emb + prefix_vec[:self.root_dim] * 0.3

        # Add suffix contributions
        for s in suffixes:
            if s and s in self.suffix_vocab and s != '<NONE>':
                suffix_idx = self.suffix_vocab[s]
                suffix_vec = self.suffix_emb[suffix_idx]
                if self.affix_dim < self.root_dim:
                    padded = np.zeros(self.root_dim)
                    padded[:self.affix_dim] = suffix_vec * 0.2  # Scale down
                    emb = emb + padded
                else:
                    emb = emb + suffix_vec[:self.root_dim] * 0.2

        return emb

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

                    # Get prefixes
                    prefixes = node.get('prefiksoj', [])
                    if not prefixes:
                        p = node.get('prefikso')
                        if p:
                            prefixes = [p]

                    # Get suffixes
                    suffixes = node.get('sufiksoj', [])

                    emb = self.embed_word(root, prefixes, suffixes)
                    if emb is not None:
                        word_embeddings.append(emb)
                        metadata["words_embedded"] += 1
                        metadata["roots_found"].append(root)

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
        logger.info("Compositional Corpus Indexing")
        logger.info("=" * 60)
        logger.info(f"Corpus: {corpus_path}")
        logger.info(f"Output: {self.output_dir}")
        logger.info(f"Root embeddings: {self.root_dim}d, {len(self.root_to_idx)} roots")
        logger.info(f"Affix embeddings: {self.affix_dim}d")

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
            self.stats = {"processed": 0, "successful": 0, "failed": 0, "total_sentences": len(sentences)}

        # Open metadata and failed files
        metadata_mode = 'a' if start_idx > 0 else 'w'
        metadata_file = open(self.metadata_path, metadata_mode)
        failed_file = open(self.failed_path, metadata_mode)

        try:
            start_time = time.time()

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
                    else:
                        failed_file.write(json.dumps(meta, ensure_ascii=False) + '\n')
                        self.stats["failed"] += 1

                    self.stats["processed"] += 1

                all_embeddings.extend(batch_embeddings)

                # Progress update
                if (i + len(batch)) % 1000 == 0 or i + len(batch) == len(sentences):
                    elapsed = time.time() - start_time
                    rate = self.stats["processed"] / elapsed if elapsed > 0 else 0
                    remaining = (len(sentences) - self.stats["processed"]) / rate if rate > 0 else 0
                    logger.info(f"Progress: {self.stats['processed']}/{len(sentences)} "
                               f"({self.stats['successful']} OK, {self.stats['failed']} failed) "
                               f"| {rate:.1f} sent/s | ETA: {remaining/60:.1f}m")

                # Save checkpoint every 5000 sentences
                if self.stats["processed"] % 5000 == 0:
                    self.save_checkpoint()
                    # Save embeddings incrementally
                    np.save(self.embeddings_path, np.array(all_embeddings))

        finally:
            metadata_file.close()
            failed_file.close()

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
                        default=Path('data/training/combined_training.jsonl'),
                        help='Path to corpus file (.txt or .jsonl)')
    parser.add_argument('--root-model', type=Path,
                        default=Path('models/root_embeddings/best_model.pt'))
    parser.add_argument('--affix-model', type=Path,
                        default=Path('models/affix_embeddings/best_model.pt'))
    parser.add_argument('--output-dir', type=Path,
                        default=Path('data/corpus_index_compositional'))
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--resume', action='store_true', default=True,
                        help='Resume from checkpoint (default)')
    parser.add_argument('--fresh', action='store_true',
                        help='Start fresh, ignore checkpoint')

    args = parser.parse_args()

    # Validate inputs
    if not args.corpus.exists():
        logger.error(f"Corpus not found: {args.corpus}")
        sys.exit(1)
    if not args.root_model.exists():
        logger.error(f"Root model not found: {args.root_model}")
        sys.exit(1)
    if not args.affix_model.exists():
        logger.error(f"Affix model not found: {args.affix_model}")
        sys.exit(1)

    indexer = CompositionalIndexer(
        root_model_path=args.root_model,
        affix_model_path=args.affix_model,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )

    indexer.index_corpus(args.corpus, resume=not args.fresh)


if __name__ == '__main__':
    main()
