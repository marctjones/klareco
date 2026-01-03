"""
Solution 4: SQLite backend with vector storage.

Uses SQLite for metadata and feature filtering,
stores embeddings as BLOBs, computes similarities in Python.
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from klareco.parser import parse
from klareco.rag.slot_indexer import SlotBasedIndexer

logger = logging.getLogger(__name__)


class SQLiteSlotRetriever:
    """Slot-based retriever using SQLite backend."""

    def __init__(
        self,
        index_path: Path,
        indexer: SlotBasedIndexer,
        slot_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize SQLite-based retriever.

        Args:
            index_path: Path to slot index directory
            indexer: SlotBasedIndexer instance for query embedding
            slot_weights: Weights for each slot (default: equal)
        """
        self.index_path = Path(index_path)
        self.indexer = indexer
        self.db_path = self.index_path / "slot_index.db"

        # Default slot weights
        self.slot_weights = slot_weights or {
            'SUBJ': 0.3,
            'VERB': 0.4,
            'OBJ': 0.3,
        }

        # Load or create database
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database."""
        logger.info(f"Initializing SQLite database at {self.db_path}")

        # Check if database exists
        if not self.db_path.exists():
            logger.info("  Database not found, creating...")
            self._create_db()
        else:
            logger.info("  Database exists, connecting...")

        # Connect
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row

        # Count documents
        cursor = self.conn.execute("SELECT COUNT(*) FROM documents")
        count = cursor.fetchone()[0]
        logger.info(f"  Loaded {count:,} documents")

    def _create_db(self):
        """Create SQLite database from slot index."""
        # Create connection
        conn = sqlite3.connect(str(self.db_path))

        # Create schema
        conn.execute("""
            CREATE TABLE documents (
                id INTEGER PRIMARY KEY,
                text TEXT NOT NULL,
                subj_embedding BLOB,
                verb_embedding BLOB,
                obj_embedding BLOB,
                full_embedding BLOB NOT NULL,
                negita INTEGER,
                tempo TEXT,
                fraztipo TEXT,
                modo TEXT,
                source_type TEXT,
                source_title TEXT
            )
        """)

        # Create indexes for feature filtering
        conn.execute("CREATE INDEX idx_negita ON documents(negita)")
        conn.execute("CREATE INDEX idx_tempo ON documents(tempo)")
        conn.execute("CREATE INDEX idx_fraztipo ON documents(fraztipo)")

        # Load data from JSONL
        index_file = self.index_path / "slot_index.jsonl"

        with open(index_file) as f:
            documents = []
            total_count = 0
            for line in f:
                doc = json.loads(line)

                # Convert embeddings to blobs
                subj_blob = self._embedding_to_blob(doc['slots'].get('SUBJ'))
                verb_blob = self._embedding_to_blob(doc['slots'].get('VERB'))
                obj_blob = self._embedding_to_blob(doc['slots'].get('OBJ'))
                full_blob = self._embedding_to_blob(doc['full_embedding'])

                documents.append((
                    doc['text'],
                    subj_blob,
                    verb_blob,
                    obj_blob,
                    full_blob,
                    1 if doc['features'].get('negita') else 0,
                    doc['features'].get('tempo', 'prezenco'),
                    doc['features'].get('fraztipo', 'deklaro'),
                    doc['features'].get('modo', 'indikativo'),
                    doc.get('source', {}).get('type'),
                    doc.get('source', {}).get('title'),
                ))

                if len(documents) >= 1000:
                    conn.executemany("""
                        INSERT INTO documents (
                            text, subj_embedding, verb_embedding, obj_embedding,
                            full_embedding, negita, tempo, fraztipo, modo,
                            source_type, source_title
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, documents)
                    total_count += len(documents)
                    documents = []

            # Insert remaining
            if documents:
                conn.executemany("""
                    INSERT INTO documents (
                        text, subj_embedding, verb_embedding, obj_embedding,
                        full_embedding, negita, tempo, fraztipo, modo,
                        source_type, source_title
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, documents)
                total_count += len(documents)

        conn.commit()
        conn.close()

        logger.info(f"  Created SQLite database with {total_count:,} documents")

    def _embedding_to_blob(self, embedding: Optional[List[float]]) -> Optional[bytes]:
        """Convert embedding list to binary blob."""
        if embedding is None:
            return None
        arr = np.array(embedding, dtype=np.float32)
        return arr.tobytes()

    def _blob_to_embedding(self, blob: Optional[bytes]) -> Optional[np.ndarray]:
        """Convert binary blob to numpy array."""
        if blob is None:
            return None
        return np.frombuffer(blob, dtype=np.float32)

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return np.dot(a, b) / (norm_a * norm_b)

    def slot_similarity(
        self,
        query_slots: Dict[str, Optional[np.ndarray]],
        doc_slots: Dict[str, Optional[np.ndarray]],
    ) -> float:
        """Compute weighted slot similarity."""
        score = 0.0
        total_weight = 0.0

        for slot, weight in self.slot_weights.items():
            query_emb = query_slots.get(slot)
            doc_emb = doc_slots.get(slot)

            if query_emb is not None and doc_emb is not None:
                sim = self.cosine_similarity(query_emb, doc_emb)
                score += weight * sim
                total_weight += weight
            elif query_emb is None and doc_emb is not None:
                # Partial credit when query slot is missing but doc has it
                score += weight * 0.5
                total_weight += weight

        # Normalize by total weight used (not number of slots)
        if total_weight > 0:
            return score / total_weight
        else:
            return 0.0

    def feature_similarity(
        self,
        query_features: Dict,
        doc_features: Dict,
    ) -> float:
        """Compute feature matching bonus."""
        bonus = 1.0

        if query_features.get('negita') == doc_features.get('negita'):
            bonus *= 1.1
        if query_features.get('tempo') == doc_features.get('tempo'):
            bonus *= 1.05
        if query_features.get('fraztipo') == doc_features.get('fraztipo'):
            bonus *= 1.05

        return bonus

    def search(
        self,
        query: str,
        top_k: int = 10,
        prefilter_limit: int = 5000,
        slot_weight: float = 0.6,
        full_weight: float = 0.4,
        use_feature_prefilter: bool = True,
    ) -> List[Tuple[float, Dict]]:
        """
        Two-stage retrieval: SQL feature filtering + slot scoring.

        Args:
            query: Query text in Esperanto
            top_k: Number of results to return
            prefilter_limit: Max candidates from SQL prefilter
            slot_weight: Weight for slot similarity in final score
            full_weight: Weight for full embedding in final score
            use_feature_prefilter: Use SQL WHERE for feature filtering

        Returns:
            List of (score, document) tuples sorted by score descending
        """
        # Parse query
        try:
            query_ast = parse(query)
        except Exception as e:
            logger.error(f"Failed to parse query: {query} - {e}")
            return []

        # Extract query slots and features
        query_slots = self.indexer.extract_slots(query_ast)
        query_features = self.indexer.extract_features(query_ast)

        # Compute query full embedding
        query_word_embs = [emb for emb in query_slots.values() if emb is not None]
        if not query_word_embs:
            logger.warning(f"No content words in query: {query}")
            return []

        query_full_emb = np.mean(query_word_embs, axis=0)
        query_full_emb = query_full_emb / np.linalg.norm(query_full_emb)

        # Stage 1: SQL prefilter
        logger.debug(f"Stage 1: SQL feature prefilter")

        # Build WHERE clause for feature matching (optional)
        where_clauses = []
        params = []

        if use_feature_prefilter:
            # Exact matches preferred, but also allow non-matches
            # This is a soft filter - we'll use it for scoring
            pass

        # Query database
        sql = f"""
            SELECT
                id, text,
                subj_embedding, verb_embedding, obj_embedding, full_embedding,
                negita, tempo, fraztipo, modo,
                source_type, source_title
            FROM documents
            {"WHERE " + " AND ".join(where_clauses) if where_clauses else ""}
            LIMIT {prefilter_limit}
        """

        cursor = self.conn.execute(sql, params)
        rows = cursor.fetchall()

        logger.debug(f"  {len(rows)} candidates from SQL")

        # Stage 2: Slot-based scoring
        logger.debug(f"Stage 2: Slot-based scoring")

        final_results = []
        for row in rows:
            # Extract embeddings
            doc_slots = {
                'SUBJ': self._blob_to_embedding(row['subj_embedding']),
                'VERB': self._blob_to_embedding(row['verb_embedding']),
                'OBJ': self._blob_to_embedding(row['obj_embedding']),
            }
            full_emb = self._blob_to_embedding(row['full_embedding'])

            # Extract features
            doc_features = {
                'negita': bool(row['negita']),
                'tempo': row['tempo'],
                'fraztipo': row['fraztipo'],
                'modo': row['modo'],
            }

            # Compute slot similarity
            slot_sim = self.slot_similarity(query_slots, doc_slots)

            # Apply feature bonus
            feature_bonus = self.feature_similarity(query_features, doc_features)
            slot_score = slot_sim * feature_bonus

            # Compute full embedding similarity
            full_sim = self.cosine_similarity(query_full_emb, full_emb)

            # Combine scores
            final_score = slot_weight * slot_score + full_weight * full_sim

            # Build document dict
            doc = {
                'text': row['text'],
                'features': doc_features,
                'source': {
                    'type': row['source_type'],
                    'title': row['source_title'],
                },
                'slots_np': doc_slots,
            }

            final_results.append((final_score, doc))

        # Sort by final score
        final_results.sort(key=lambda x: x[0], reverse=True)

        if final_results:
            logger.debug(f"  Top final score: {final_results[0][0]:.3f}")

        return final_results[:top_k]

    def explain_match(
        self,
        query: str,
        doc: Dict,
    ) -> Dict:
        """Explain why a document matched the query."""
        # Parse query
        query_ast = parse(query)
        query_slots = self.indexer.extract_slots(query_ast)

        # Compute slot similarities
        explanation = {
            'query': query,
            'document': doc['text'],
            'slot_matches': {},
        }

        for slot in ['SUBJ', 'VERB', 'OBJ']:
            query_emb = query_slots.get(slot)
            doc_emb = doc['slots_np'].get(slot)

            if query_emb is not None and doc_emb is not None:
                sim = self.cosine_similarity(query_emb, doc_emb)
                explanation['slot_matches'][slot] = {
                    'similarity': float(sim),
                    'status': 'matched',
                }
            elif query_emb is None:
                explanation['slot_matches'][slot] = {
                    'status': 'query_missing',
                }
            elif doc_emb is None:
                explanation['slot_matches'][slot] = {
                    'status': 'doc_missing',
                }

        return explanation

    def close(self):
        """Close database connection."""
        if hasattr(self, 'conn'):
            self.conn.close()

    def __del__(self):
        """Cleanup on deletion."""
        self.close()
