"""
Solution 4: SQLite backend with vector storage.

Uses SQLite for metadata and feature filtering,
stores embeddings as BLOBs, computes similarities in Python.

Optimizations:
- P1: Pre-computed norms stored in database (15-20% speedup) - #214
- P2: SQL feature prefilter using indexed columns (5-10% speedup) - #216
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
        # P1: Added norm columns for pre-computed norms (#214)
        conn.execute("""
            CREATE TABLE documents (
                id INTEGER PRIMARY KEY,
                text TEXT NOT NULL,
                subj_embedding BLOB,
                verb_embedding BLOB,
                obj_embedding BLOB,
                full_embedding BLOB NOT NULL,
                subj_norm REAL,
                verb_norm REAL,
                obj_norm REAL,
                full_norm REAL NOT NULL,
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

                # Convert embeddings to blobs and compute norms (P1: #214)
                subj_emb = doc['slots'].get('SUBJ')
                verb_emb = doc['slots'].get('VERB')
                obj_emb = doc['slots'].get('OBJ')
                full_emb = doc['full_embedding']

                subj_blob = self._embedding_to_blob(subj_emb)
                verb_blob = self._embedding_to_blob(verb_emb)
                obj_blob = self._embedding_to_blob(obj_emb)
                full_blob = self._embedding_to_blob(full_emb)

                # P1: Pre-compute norms
                subj_norm = float(np.linalg.norm(np.array(subj_emb, dtype=np.float32))) if subj_emb else None
                verb_norm = float(np.linalg.norm(np.array(verb_emb, dtype=np.float32))) if verb_emb else None
                obj_norm = float(np.linalg.norm(np.array(obj_emb, dtype=np.float32))) if obj_emb else None
                full_norm = float(np.linalg.norm(np.array(full_emb, dtype=np.float32)))

                documents.append((
                    doc['text'],
                    subj_blob,
                    verb_blob,
                    obj_blob,
                    full_blob,
                    subj_norm,
                    verb_norm,
                    obj_norm,
                    full_norm,
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
                            full_embedding, subj_norm, verb_norm, obj_norm, full_norm,
                            negita, tempo, fraztipo, modo,
                            source_type, source_title
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, documents)
                    total_count += len(documents)
                    documents = []

            # Insert remaining
            if documents:
                conn.executemany("""
                    INSERT INTO documents (
                        text, subj_embedding, verb_embedding, obj_embedding,
                        full_embedding, subj_norm, verb_norm, obj_norm, full_norm,
                        negita, tempo, fraztipo, modo,
                        source_type, source_title
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray, norm_a: Optional[float] = None, norm_b: Optional[float] = None) -> float:
        """Compute cosine similarity between two vectors.

        P1: Optionally use pre-computed norms (#214) to avoid recomputing.
        """
        if norm_a is None:
            norm_a = np.linalg.norm(a)
        if norm_b is None:
            norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return np.dot(a, b) / (norm_a * norm_b)

    def slot_similarity(
        self,
        query_slots: Dict[str, Optional[np.ndarray]],
        doc_slots: Dict[str, Optional[np.ndarray]],
        doc_norms: Optional[Dict[str, Optional[float]]] = None,
        is_question: bool = False,
    ) -> float:
        """
        Compute weighted slot similarity.

        P1: Optionally use pre-computed norms (#214) to avoid recomputing.

        Args:
            query_slots: Query slot embeddings
            doc_slots: Document slot embeddings
            doc_norms: Pre-computed norms for document slots
            is_question: Whether query is a question (affects partial bonus)
        """
        score = 0.0
        total_weight = 0.0

        # Bug #2 fix: Higher partial bonus for questions
        partial_bonus = 0.8 if is_question else 0.5

        for slot, weight in self.slot_weights.items():
            query_emb = query_slots.get(slot)
            doc_emb = doc_slots.get(slot)

            if query_emb is not None and doc_emb is not None:
                # P1: Use pre-computed norm if available
                doc_norm = doc_norms.get(slot) if doc_norms else None
                sim = self.cosine_similarity(query_emb, doc_emb, norm_b=doc_norm)
                score += weight * sim
                total_weight += weight
            elif query_emb is None and doc_emb is not None:
                # Partial credit when query slot is missing but doc has it
                score += weight * partial_bonus
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

        # Detect if query is a question (Bug #2 fix)
        is_question = query.strip().endswith('?') or query_ast.get('fraztipo') == 'demando'

        # Compute query full embedding
        query_word_embs = [emb for emb in query_slots.values() if emb is not None]
        if not query_word_embs:
            logger.warning(f"No content words in query: {query}")
            return []

        query_full_emb = np.mean(query_word_embs, axis=0)
        query_full_emb = query_full_emb / np.linalg.norm(query_full_emb)

        # Stage 1: SQL prefilter
        logger.debug(f"Stage 1: SQL feature prefilter")

        # P2: Build WHERE clause for feature matching (#216)
        where_clauses = []
        params = []

        if use_feature_prefilter:
            # Filter by negation (indexed column)
            if 'negita' in query_features:
                where_clauses.append("negita = ?")
                params.append(1 if query_features['negita'] else 0)

            # Filter by tense (indexed column)
            if 'tempo' in query_features and query_features['tempo']:
                where_clauses.append("tempo = ?")
                params.append(query_features['tempo'])

            # Filter by sentence type (indexed column)
            if 'fraztipo' in query_features and query_features['fraztipo']:
                where_clauses.append("fraztipo = ?")
                params.append(query_features['fraztipo'])

        # Query database
        # P1: Load pre-computed norms (#214)
        sql = f"""
            SELECT
                id, text,
                subj_embedding, verb_embedding, obj_embedding, full_embedding,
                subj_norm, verb_norm, obj_norm, full_norm,
                negita, tempo, fraztipo, modo,
                source_type, source_title
            FROM documents
            {"WHERE " + " AND ".join(where_clauses) if where_clauses else ""}
            LIMIT {prefilter_limit}
        """

        cursor = self.conn.execute(sql, params)
        rows = cursor.fetchall()

        if where_clauses:
            logger.debug(f"  {len(rows)} candidates from SQL (filtered by {len(where_clauses)} features)")
        else:
            logger.debug(f"  {len(rows)} candidates from SQL (no feature filter)")

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

            # P1: Extract pre-computed norms (#214)
            doc_norms = {
                'SUBJ': row['subj_norm'],
                'VERB': row['verb_norm'],
                'OBJ': row['obj_norm'],
            }
            full_norm = row['full_norm']

            # Extract features
            doc_features = {
                'negita': bool(row['negita']),
                'tempo': row['tempo'],
                'fraztipo': row['fraztipo'],
                'modo': row['modo'],
            }

            # Compute slot similarity (P1: using pre-computed norms)
            slot_sim = self.slot_similarity(query_slots, doc_slots, doc_norms, is_question=is_question)

            # Apply feature bonus
            feature_bonus = self.feature_similarity(query_features, doc_features)
            slot_score = slot_sim * feature_bonus

            # Compute full embedding similarity (P1: using pre-computed norm)
            full_sim = self.cosine_similarity(query_full_emb, full_emb, norm_b=full_norm)

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
