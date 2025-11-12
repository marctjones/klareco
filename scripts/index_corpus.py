#!/usr/bin/env python3
"""
Index Esperanto corpus with Tree-LSTM embeddings for RAG retrieval.

Features:
- Automatic checkpointing and resume from last completed batch
- Progress tracking with ETA
- Detailed logging
- Batch processing to manage memory
- FAISS index building for efficient similarity search
- Graceful interrupt handling

Usage:
    python scripts/index_corpus.py --corpus data/gutenberg_sentences.txt --output data/corpus_index
    python scripts/index_corpus.py --resume  # Resume from last checkpoint
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.parser import parse
from klareco.models.tree_lstm import TreeLSTMEncoder
from klareco.ast_to_graph import ASTToGraphConverter


class CorpusIndexer:
    """Index corpus with Tree-LSTM embeddings and build FAISS index."""

    def __init__(
        self,
        model_path: str,
        output_dir: str,
        batch_size: int = 32,
        embedding_dim: int = 256,
    ):
        """Initialize corpus indexer.

        Args:
            model_path: Path to trained Tree-LSTM model checkpoint
            output_dir: Directory to save index and metadata
            batch_size: Number of sentences to process per batch
            embedding_dim: Dimension of Tree-LSTM embeddings
        """
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Paths for checkpointing
        self.checkpoint_path = self.output_dir / "indexing_checkpoint.json"
        self.embeddings_path = self.output_dir / "embeddings.npy"
        self.metadata_path = self.output_dir / "metadata.jsonl"
        self.failed_path = self.output_dir / "failed_sentences.jsonl"
        self.index_path = self.output_dir / "faiss_index.bin"

        # Initialize model and converter
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.converter = ASTToGraphConverter()

        # Statistics
        self.stats = {
            "total_sentences": 0,
            "processed": 0,
            "successful": 0,
            "failed": 0,
            "start_time": None,
        }

        logging.info(f"Initializing CorpusIndexer")
        logging.info(f"  Output directory: {self.output_dir}")
        logging.info(f"  Batch size: {self.batch_size}")
        logging.info(f"  Device: {self.device}")

    def load_model(self):
        """Load trained Tree-LSTM model."""
        logging.info(f"Loading model from {self.model_path}")

        # Load checkpoint to get hyperparameters
        checkpoint = torch.load(self.model_path, map_location=self.device)

        # Get hyperparameters from checkpoint or infer from saved model
        if "vocab_size" in checkpoint:
            vocab_size = checkpoint["vocab_size"]
            embed_dim = checkpoint["embed_dim"]
            hidden_dim = checkpoint["hidden_dim"]
            output_dim = checkpoint["output_dim"]
        else:
            # Infer from model state dict
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            vocab_size = state_dict["embed.weight"].shape[0]
            embed_dim = state_dict["embed.weight"].shape[1]
            hidden_dim = state_dict["tree_lstm.cell.W_i.weight"].shape[0]
            output_dim = state_dict["tree_lstm.output_proj.weight"].shape[0]

        logging.info(f"  Vocabulary size: {vocab_size}")
        logging.info(f"  Embedding dim: {embed_dim}")
        logging.info(f"  Hidden dim: {hidden_dim}")
        logging.info(f"  Output dim: {output_dim}")

        # Create model
        self.model = TreeLSTMEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        )

        # Load weights
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()

        logging.info("Model loaded successfully")

    def load_checkpoint(self) -> Dict:
        """Load checkpoint if exists."""
        if self.checkpoint_path.exists():
            with open(self.checkpoint_path, 'r') as f:
                checkpoint = json.load(f)
            logging.info(f"Loaded checkpoint: {checkpoint['processed']}/{checkpoint['total_sentences']} sentences processed")
            return checkpoint
        return None

    def save_checkpoint(self):
        """Save checkpoint."""
        checkpoint = {
            "total_sentences": self.stats["total_sentences"],
            "processed": self.stats["processed"],
            "successful": self.stats["successful"],
            "failed": self.stats["failed"],
            "timestamp": time.time(),
        }

        with open(self.checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)

        logging.debug(f"Checkpoint saved: {self.stats['processed']}/{self.stats['total_sentences']}")

    def encode_sentence(self, sentence: str) -> Optional[np.ndarray]:
        """Encode single sentence to embedding.

        Args:
            sentence: Esperanto sentence

        Returns:
            Embedding vector or None if parsing failed
        """
        try:
            # Parse sentence
            ast = parse(sentence)

            # Convert AST to graph
            graph_data = self.converter.ast_to_graph(ast)
            if graph_data is None:
                return None

            # Move to device
            graph_data = graph_data.to(self.device)

            # Encode with Tree-LSTM
            with torch.no_grad():
                embedding = self.model(graph_data)
                embedding = embedding.cpu().numpy()

            return embedding

        except Exception as e:
            logging.debug(f"Failed to encode sentence: {e}")
            return None

    def load_corpus(self, corpus_path: str) -> List[str]:
        """Load corpus sentences from file.

        Args:
            corpus_path: Path to corpus file (one sentence per line)

        Returns:
            List of sentences
        """
        logging.info(f"Loading corpus from {corpus_path}")

        sentences = []
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and len(line) > 5:  # Filter very short lines
                    sentences.append(line)

        logging.info(f"  Loaded {len(sentences)} sentences")
        return sentences

    def index_corpus(
        self,
        corpus_path: str,
        resume: bool = True,
    ):
        """Index entire corpus with checkpointing.

        Args:
            corpus_path: Path to corpus file
            resume: Whether to resume from checkpoint
        """
        # Load model
        if self.model is None:
            self.load_model()

        # Load corpus
        sentences = self.load_corpus(corpus_path)
        self.stats["total_sentences"] = len(sentences)
        self.stats["start_time"] = time.time()

        # Check for existing checkpoint
        start_idx = 0
        if resume:
            checkpoint = self.load_checkpoint()
            if checkpoint:
                start_idx = checkpoint["processed"]
                self.stats["processed"] = checkpoint["processed"]
                self.stats["successful"] = checkpoint["successful"]
                self.stats["failed"] = checkpoint["failed"]

                if start_idx >= len(sentences):
                    logging.info("Corpus already fully indexed!")
                    return

                logging.info(f"Resuming from sentence {start_idx}")

        # Open files for appending
        metadata_file = open(self.metadata_path, 'a', encoding='utf-8')
        failed_file = open(self.failed_path, 'a', encoding='utf-8')

        # Determine if we need to create or append to embeddings
        if start_idx == 0 and self.embeddings_path.exists():
            # Starting fresh, remove old file
            self.embeddings_path.unlink()

        # Process sentences in batches
        try:
            with tqdm(total=len(sentences), initial=start_idx, desc="Indexing corpus") as pbar:
                for i in range(start_idx, len(sentences), self.batch_size):
                    batch_sentences = sentences[i:i + self.batch_size]
                    batch_embeddings = []

                    for j, sentence in enumerate(batch_sentences):
                        global_idx = i + j

                        # Encode sentence
                        embedding = self.encode_sentence(sentence)

                        if embedding is not None:
                            # Save embedding
                            batch_embeddings.append(embedding)

                            # Save metadata
                            metadata = {
                                "idx": global_idx,
                                "sentence": sentence,
                                "embedding_idx": self.stats["successful"],
                            }
                            metadata_file.write(json.dumps(metadata, ensure_ascii=False) + '\n')
                            metadata_file.flush()

                            self.stats["successful"] += 1
                        else:
                            # Save failed sentence
                            failed = {
                                "idx": global_idx,
                                "sentence": sentence,
                            }
                            failed_file.write(json.dumps(failed, ensure_ascii=False) + '\n')
                            failed_file.flush()

                            self.stats["failed"] += 1

                        self.stats["processed"] += 1
                        pbar.update(1)

                        # Update progress bar description
                        if (global_idx + 1) % 100 == 0:
                            success_rate = self.stats["successful"] / self.stats["processed"] * 100
                            pbar.set_description(f"Indexing corpus (success rate: {success_rate:.1f}%)")

                    # Save batch embeddings to disk
                    if batch_embeddings:
                        batch_array = np.array(batch_embeddings, dtype=np.float32)

                        if self.embeddings_path.exists():
                            # Append to existing
                            existing = np.load(str(self.embeddings_path))
                            combined = np.vstack([existing, batch_array])
                            np.save(str(self.embeddings_path), combined)
                        else:
                            # Create new
                            np.save(str(self.embeddings_path), batch_array)

                    # Save checkpoint every batch
                    self.save_checkpoint()

        finally:
            metadata_file.close()
            failed_file.close()

        # Print final statistics
        elapsed = time.time() - self.stats["start_time"]
        success_rate = self.stats["successful"] / self.stats["processed"] * 100

        logging.info("")
        logging.info("Indexing complete!")
        logging.info(f"  Total sentences: {self.stats['total_sentences']}")
        logging.info(f"  Successfully encoded: {self.stats['successful']} ({success_rate:.1f}%)")
        logging.info(f"  Failed: {self.stats['failed']}")
        logging.info(f"  Time elapsed: {elapsed:.1f}s ({self.stats['processed']/elapsed:.1f} sentences/s)")
        logging.info(f"  Embeddings saved to: {self.embeddings_path}")
        logging.info(f"  Metadata saved to: {self.metadata_path}")

        # Build FAISS index
        self.build_faiss_index()

    def build_faiss_index(self):
        """Build FAISS index from embeddings."""
        logging.info("Building FAISS index...")

        try:
            import faiss
        except ImportError:
            logging.warning("FAISS not installed. Skipping index building.")
            logging.warning("Install with: pip install faiss-cpu")
            return

        # Load embeddings
        embeddings = np.load(str(self.embeddings_path))
        logging.info(f"  Loaded {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)

        # Build index (using IndexFlatIP for inner product / cosine similarity)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        # Save index
        faiss.write_index(index, str(self.index_path))

        logging.info(f"  FAISS index saved to: {self.index_path}")
        logging.info(f"  Index size: {index.ntotal} vectors")


def main():
    parser = argparse.ArgumentParser(
        description="Index Esperanto corpus with Tree-LSTM embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--corpus",
        type=str,
        default="data/gutenberg_sentences.txt",
        help="Path to corpus file (one sentence per line)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="models/tree_lstm/checkpoint_epoch_12.pt",
        help="Path to trained Tree-LSTM model",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/corpus_index",
        help="Output directory for index and metadata",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for processing",
    )

    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=256,
        help="Dimension of Tree-LSTM embeddings",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from last checkpoint (default: True)",
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh, ignore checkpoints",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / "indexing.log"),
            logging.StreamHandler(),
        ]
    )

    logging.info("Starting corpus indexing")
    logging.info(f"  Corpus: {args.corpus}")
    logging.info(f"  Model: {args.model}")
    logging.info(f"  Output: {args.output}")
    logging.info(f"  Batch size: {args.batch_size}")
    logging.info(f"  Resume: {args.resume and not args.no_resume}")

    # Create indexer
    indexer = CorpusIndexer(
        model_path=args.model,
        output_dir=args.output,
        batch_size=args.batch_size,
        embedding_dim=args.embedding_dim,
    )

    # Index corpus
    indexer.index_corpus(
        corpus_path=args.corpus,
        resume=args.resume and not args.no_resume,
    )

    logging.info("Done!")


if __name__ == "__main__":
    main()
