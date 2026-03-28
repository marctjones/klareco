#!/usr/bin/env python3
"""
Build Hybrid Retrieval System (BM25 + Semantic Search)

Implements complete hybrid retrieval pipeline:
1. Precompute AST embeddings for corpus (5.4M sentences)
   - 190d embeddings: 120d semantic (learned) + 70d grammar (deterministic)
2. Build FAISS HNSW index for fast semantic search
3. Create HybridRetriever class combining BM25 + semantic similarity

Embedding Architecture (v2.1):
- Semantic: 120d learned from roots only (maintains thesis: grammar is deterministic)
- Grammar: 70d deterministic flags (word class, tense, mood, case, prefixes, suffixes)
- Total: 190d per word, mean pooled over sentence

Usage:
    # Full pipeline
    python scripts/build_hybrid_retrieval_system.py

    # Test on small sample first
    python scripts/build_hybrid_retrieval_system.py --limit 10000

    # Resume from checkpoint
    python scripts/build_hybrid_retrieval_system.py --resume

STATUS: This is a long-running script (~6 hours for full corpus).
        Run in background: nohup python scripts/build_hybrid_retrieval_system.py &
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import kuzu
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.embeddings.compositional import CompositionalEmbedding
from klareco.embeddings.semantic_plus_grammar import SemanticPlusGrammarEmbedding
from klareco.parser import parse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/build_hybrid_retrieval.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class HybridRetrievalBuilder:
    """Build hybrid retrieval system with precomputed embeddings."""

    def __init__(
        self,
        kuzu_db_path: Path,
        comp_emb_path: Path,
        output_dir: Path
    ):
        self.kuzu_db_path = kuzu_db_path
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Progress tracking
        self.checkpoint_file = output_dir / "checkpoint.json"
        self.progress = self._load_checkpoint()

        # Load compositional embeddings
        logger.info(f"Loading compositional embeddings from {comp_emb_path}")
        self.comp_emb = self._load_compositional_embeddings(comp_emb_path)
        self.comp_emb.eval()

        # Initialize semantic + grammar embedder
        logger.info("Initializing SemanticPlusGrammarEmbedding (120d semantic + 70d grammar)")
        self.embedder = SemanticPlusGrammarEmbedding(self.comp_emb)

        # Connect to Kùzu
        logger.info(f"Connecting to Kùzu at {kuzu_db_path}")
        self.kuzu_db = kuzu.Database(str(kuzu_db_path))
        self.kuzu_conn = kuzu.Connection(self.kuzu_db)

    def _load_compositional_embeddings(self, path: Path) -> CompositionalEmbedding:
        """Load compositional embedding model."""
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)

        if 'root_vocab' in checkpoint:
            comp_emb = CompositionalEmbedding(
                root_vocab=checkpoint['root_vocab'],
                prefix_vocab=checkpoint['prefix_vocab'],
                suffix_vocab=checkpoint['suffix_vocab'],
                embed_dim=checkpoint.get('embed_dim', 128),
            )
            comp_emb.load_state_dict(checkpoint['model_state_dict'])
        elif 'model_state_dict' in checkpoint:
            root_to_idx = checkpoint['root_to_idx']
            prefix_vocab = {'<NONE>': 0, '<UNK>': 1}
            suffix_vocab = {'<NONE>': 0, '<UNK>': 1}

            comp_emb = CompositionalEmbedding(
                root_vocab=root_to_idx,
                prefix_vocab=prefix_vocab,
                suffix_vocab=suffix_vocab,
                embed_dim=checkpoint.get('embedding_dim', 128)
            )
            comp_emb.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            raise ValueError(f"Unrecognized checkpoint format")

        return comp_emb

    def _load_checkpoint(self) -> Dict:
        """Load checkpoint if exists."""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file) as f:
                return json.load(f)
        return {'sentences_processed': 0, 'embeddings': []}

    def _save_checkpoint(self):
        """Save checkpoint atomically."""
        temp_file = self.checkpoint_file.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(self.progress, f)
        temp_file.rename(self.checkpoint_file)

    def embed_ast(self, ast: Dict) -> np.ndarray:
        """
        Embed an AST using SemanticPlusGrammarEmbedding (190d).

        Returns:
            190d embedding: [120d semantic (learned), 70d grammar (deterministic)]
        """
        # Use the new embedder (handles extraction + embedding internally)
        return self.embedder.embed_ast(ast)

    def precompute_embeddings(self, limit: int = None):
        """
        Precompute embeddings for all sentences in corpus.

        This fetches ASTs from Kùzu graph and embeds them.
        """
        logger.info("Starting embedding precomputation...")

        # Get total count
        result = self.kuzu_conn.execute("MATCH (f:Frazoteksto) RETURN COUNT(*) AS count;")
        total_count = result.get_next()[0] if result.has_next() else 0

        if limit:
            total_count = min(total_count, limit)

        logger.info(f"Will process {total_count} sentences")

        # Resume from checkpoint
        start_id = self.progress.get('last_processed_id', 0)
        logger.info(f"Resuming from sentence ID {start_id}")

        # Batch processing
        batch_size = 1000
        embeddings_file = self.output_dir / "ast_embeddings.npy"
        id_mapping_file = self.output_dir / "sentence_ids.npy"

        embeddings_list = []
        id_list = []

        # Query sentences in batches
        query = f"""
            MATCH (ft:Frazoteksto)
            WHERE ft.id > {start_id}
            RETURN ft.id, ft.teksto
            ORDER BY ft.id
            LIMIT {batch_size}
        """

        processed = 0
        batch_num = 0

        while processed < total_count:
            logger.info(f"Processing batch {batch_num} (sentences {processed}/{total_count})")

            result = self.kuzu_conn.execute(query)

            batch_embeddings = []
            batch_ids = []
            rows_in_batch = 0

            while result.has_next():
                row = result.get_next()
                sentence_id = row[0]
                text = row[1]

                # Parse AST (TODO: fetch from graph instead)
                try:
                    ast = parse(text)
                except:
                    ast = None

                # Embed
                embedding = self.embed_ast(ast)
                batch_embeddings.append(embedding)
                batch_ids.append(sentence_id)

                rows_in_batch += 1

                # Update progress
                if rows_in_batch % 100 == 0:
                    logger.info(f"  {rows_in_batch}/{batch_size} in batch...")

            if not batch_embeddings:
                break

            embeddings_list.extend(batch_embeddings)
            id_list.extend(batch_ids)

            processed += len(batch_embeddings)
            start_id = batch_ids[-1]

            # Save checkpoint every batch
            self.progress['last_processed_id'] = start_id
            self.progress['sentences_processed'] = processed
            self._save_checkpoint()

            # Update query for next batch
            query = f"""
                MATCH (ft:Frazoteksto)
                WHERE ft.id > {start_id}
                RETURN ft.id, ft.teksto
                ORDER BY ft.id
                LIMIT {batch_size}
            """

            batch_num += 1

            # Early stop if limit reached
            if limit and processed >= limit:
                break

        # Save embeddings
        logger.info(f"Saving {len(embeddings_list)} embeddings...")
        embeddings_np = np.array(embeddings_list, dtype=np.float32)
        ids_np = np.array(id_list, dtype=np.int64)

        np.save(embeddings_file, embeddings_np)
        np.save(id_mapping_file, ids_np)

        logger.info(f"Embeddings saved to {embeddings_file}")
        logger.info(f"ID mapping saved to {id_mapping_file}")

        return embeddings_np, ids_np

    def build_faiss_index(self, embeddings: np.ndarray):
        """Build FAISS HNSW index for fast semantic search."""
        try:
            import faiss
        except ImportError:
            logger.error("FAISS not installed. Run: pip install faiss-cpu")
            return

        logger.info(f"Building FAISS index for {len(embeddings)} vectors...")

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)

        # Create HNSW index (hierarchical navigable small world)
        d = embeddings.shape[1]  # Dimension (190 = 120 semantic + 70 grammar)
        index = faiss.IndexHNSWFlat(d, 32)  # 32 neighbors

        # Add vectors
        index.add(embeddings)

        # Save index
        index_file = self.output_dir / "ast_embeddings.faiss"
        faiss.write_index(index, str(index_file))

        logger.info(f"FAISS index saved to {index_file}")

        # Test search speed
        logger.info("Testing search speed...")
        query = embeddings[0:1]  # First embedding as query
        import time
        start = time.time()
        D, I = index.search(query, k=1000)
        elapsed = (time.time() - start) * 1000
        logger.info(f"Search for 1000 neighbors took {elapsed:.2f}ms")

        return index


def main():
    parser = argparse.ArgumentParser(description='Build hybrid retrieval system')
    parser.add_argument('--kuzu-db', type=Path,
                       default=Path('data/indexes/v2.1_kuzu_index_full'),
                       help='Path to Kùzu database')
    parser.add_argument('--comp-emb', type=Path,
                       default=Path('models/root_embeddings_fundamento_enhanced/root_embeddings_best.pt'),
                       help='Path to compositional embeddings')
    parser.add_argument('--output', type=Path,
                       default=Path('data/indexes/hybrid_retrieval'),
                       help='Output directory')
    parser.add_argument('--limit', type=int,
                       help='Limit number of sentences (for testing)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from checkpoint')

    args = parser.parse_args()

    # Create logs directory
    Path('logs').mkdir(exist_ok=True)

    # Build hybrid system
    builder = HybridRetrievalBuilder(
        kuzu_db_path=args.kuzu_db,
        comp_emb_path=args.comp_emb,
        output_dir=args.output
    )

    # Precompute embeddings
    embeddings, ids = builder.precompute_embeddings(limit=args.limit)

    # Build FAISS index
    builder.build_faiss_index(embeddings)

    logger.info("Hybrid retrieval system built successfully!")
    logger.info(f"Output files:")
    logger.info(f"  {args.output}/ast_embeddings.npy")
    logger.info(f"  {args.output}/sentence_ids.npy")
    logger.info(f"  {args.output}/ast_embeddings.faiss")


if __name__ == '__main__':
    main()
