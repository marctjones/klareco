"""
Corpus Manager - Database and management system for Esperanto texts.

Features:
- Track all texts in SQLite database
- Add/remove texts from index
- Validate Esperanto content
- Automatic cleaning pipeline
- Incremental indexing (no need to rebuild everything)
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .lang_id import identify_language
from .parser import parse


class CorpusDatabase:
    """SQLite database for tracking corpus texts and their index status."""

    def __init__(self, db_path: Path):
        """
        Initialize corpus database.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.conn = None
        self._init_database()

    def _init_database(self):
        """Create database tables if they don't exist."""
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row

        # Texts table - tracks all texts in the corpus
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS texts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL,
                source_type TEXT,  -- 'literature', 'wikipedia', 'dictionary', etc.
                language_code TEXT DEFAULT 'eo',
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                indexed_at TIMESTAMP,
                is_indexed BOOLEAN DEFAULT 0,
                sentence_count INTEGER DEFAULT 0,
                file_size INTEGER,
                md5_hash TEXT,
                validation_status TEXT,  -- 'valid', 'invalid', 'pending'
                validation_score REAL,  -- 0.0-1.0, parse success rate
                metadata TEXT  -- JSON blob for custom metadata
            )
        """)

        # Indexed sentences table - tracks which sentences belong to which text
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS indexed_sentences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text_id INTEGER NOT NULL,
                sentence TEXT NOT NULL,
                line_num INTEGER,
                embedding_idx INTEGER,  -- Index in FAISS
                indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (text_id) REFERENCES texts(id) ON DELETE CASCADE
            )
        """)

        # Index for fast lookups
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_text_id
            ON indexed_sentences(text_id)
        """)

        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_embedding_idx
            ON indexed_sentences(embedding_idx)
        """)

        self.conn.commit()

    def add_text(
        self,
        filename: str,
        title: str,
        source_type: str = 'literature',
        metadata: Optional[Dict] = None
    ) -> int:
        """
        Add a new text to the database.

        Args:
            filename: Unique filename identifier
            title: Human-readable title
            source_type: Type of source (literature, wikipedia, etc.)
            metadata: Optional metadata dict

        Returns:
            Text ID
        """
        cursor = self.conn.execute(
            """
            INSERT INTO texts (filename, title, source_type, metadata, validation_status)
            VALUES (?, ?, ?, ?, 'pending')
            """,
            (filename, title, source_type, json.dumps(metadata or {}))
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_text(self, text_id: int) -> Optional[Dict]:
        """Get text by ID."""
        cursor = self.conn.execute(
            "SELECT * FROM texts WHERE id = ?", (text_id,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_text_by_filename(self, filename: str) -> Optional[Dict]:
        """Get text by filename."""
        cursor = self.conn.execute(
            "SELECT * FROM texts WHERE filename = ?", (filename,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def list_texts(self, indexed_only: bool = False) -> List[Dict]:
        """
        List all texts.

        Args:
            indexed_only: Only return indexed texts

        Returns:
            List of text records
        """
        query = "SELECT * FROM texts"
        if indexed_only:
            query += " WHERE is_indexed = 1"
        query += " ORDER BY added_at DESC"

        cursor = self.conn.execute(query)
        return [dict(row) for row in cursor.fetchall()]

    def update_validation(
        self,
        text_id: int,
        status: str,
        score: float
    ):
        """Update validation status for a text."""
        self.conn.execute(
            """
            UPDATE texts
            SET validation_status = ?, validation_score = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (status, score, text_id)
        )
        self.conn.commit()

    def mark_indexed(
        self,
        text_id: int,
        sentence_count: int,
        sentence_data: List[Tuple[str, int, int]]
    ):
        """
        Mark a text as indexed and store sentence mappings.

        Args:
            text_id: Text ID
            sentence_count: Number of sentences indexed
            sentence_data: List of (sentence, line_num, embedding_idx) tuples
        """
        # Update text record
        self.conn.execute(
            """
            UPDATE texts
            SET is_indexed = 1,
                indexed_at = CURRENT_TIMESTAMP,
                sentence_count = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (sentence_count, text_id)
        )

        # Store sentence mappings
        for sentence, line_num, embedding_idx in sentence_data:
            self.conn.execute(
                """
                INSERT INTO indexed_sentences (text_id, sentence, line_num, embedding_idx)
                VALUES (?, ?, ?, ?)
                """,
                (text_id, sentence, line_num, embedding_idx)
            )

        self.conn.commit()

    def mark_unindexed(self, text_id: int):
        """Mark a text as not indexed and remove sentence mappings."""
        self.conn.execute(
            """
            UPDATE texts
            SET is_indexed = 0,
                indexed_at = NULL,
                sentence_count = 0,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (text_id,)
        )

        # Remove sentence mappings
        self.conn.execute(
            "DELETE FROM indexed_sentences WHERE text_id = ?",
            (text_id,)
        )

        self.conn.commit()

    def get_indexed_sentences(self, text_id: int) -> List[Dict]:
        """Get all indexed sentences for a text."""
        cursor = self.conn.execute(
            """
            SELECT * FROM indexed_sentences
            WHERE text_id = ?
            ORDER BY line_num
            """,
            (text_id,)
        )
        return [dict(row) for row in cursor.fetchall()]

    def remove_text(self, text_id: int):
        """Remove a text and all its sentence mappings."""
        self.conn.execute("DELETE FROM texts WHERE id = ?", (text_id,))
        self.conn.commit()

    def get_stats(self) -> Dict:
        """Get corpus statistics."""
        cursor = self.conn.execute("""
            SELECT
                COUNT(*) as total_texts,
                SUM(CASE WHEN is_indexed = 1 THEN 1 ELSE 0 END) as indexed_texts,
                SUM(sentence_count) as total_sentences,
                SUM(file_size) as total_size
            FROM texts
        """)
        row = cursor.fetchone()
        return dict(row)

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class TextValidator:
    """Validate that a text file contains valid Esperanto."""

    @staticmethod
    def validate_file(file_path: Path) -> Tuple[bool, float, str]:
        """
        Validate an Esperanto text file.

        Args:
            file_path: Path to text file

        Returns:
            Tuple of (is_valid, score, message)
            - is_valid: Whether file passes validation
            - score: Validation score (0.0-1.0)
            - message: Description of validation result
        """
        if not file_path.exists():
            return False, 0.0, "File does not exist"

        # Check file size (too small = probably metadata only)
        file_size = file_path.stat().st_size
        if file_size < 100:  # Less than 100 bytes
            return False, 0.0, f"File too small ({file_size} bytes)"

        # Read sample of lines
        with file_path.open('r', encoding='utf-8', errors='ignore') as f:
            lines = [line.strip() for line in f if line.strip()]

        if len(lines) < 10:
            return False, 0.1, f"Too few lines ({len(lines)})"

        # Check language on sample (first 100 lines)
        sample_text = '\n'.join(lines[:100])
        detected_lang = identify_language(sample_text)

        if detected_lang != 'eo':
            return False, 0.2, f"Language detected as '{detected_lang}', not Esperanto"

        # Try parsing sample sentences
        parse_successes = 0
        parse_attempts = 0

        for line in lines[:100]:
            if len(line) < 10:  # Skip very short lines
                continue

            parse_attempts += 1
            try:
                ast = parse(line)
                if ast and 'tipo' in ast:
                    parse_successes += 1
            except:
                pass

        if parse_attempts == 0:
            return False, 0.3, "No parseable sentences found"

        success_rate = parse_successes / parse_attempts

        # Validation thresholds
        if success_rate >= 0.7:
            return True, success_rate, f"Valid Esperanto ({success_rate:.1%} parse rate)"
        elif success_rate >= 0.4:
            return True, success_rate, f"Acceptable Esperanto ({success_rate:.1%} parse rate, may need cleaning)"
        else:
            return False, success_rate, f"Poor parse rate ({success_rate:.1%}), likely not clean Esperanto"


class CorpusManager:
    """High-level manager for corpus operations."""

    def __init__(self, data_dir: Path):
        """
        Initialize corpus manager.

        Args:
            data_dir: Directory containing data/ folder
        """
        self.data_dir = data_dir
        self.db = CorpusDatabase(data_dir / "corpus.db")
        self.validator = TextValidator()

    def add_text_from_file(
        self,
        file_path: Path,
        title: Optional[str] = None,
        source_type: str = 'literature',
        auto_clean: bool = True,
        auto_index: bool = False
    ) -> Tuple[bool, str, Optional[int]]:
        """
        Add a text file to the corpus.

        Args:
            file_path: Path to text file
            title: Title (defaults to filename)
            source_type: Type of source
            auto_clean: Automatically clean if needed
            auto_index: Automatically index after adding

        Returns:
            Tuple of (success, message, text_id)
        """
        if not file_path.exists():
            return False, f"File not found: {file_path}", None

        filename = file_path.name
        title = title or filename

        # Check if already exists
        existing = self.db.get_text_by_filename(filename)
        if existing:
            return False, f"Text '{filename}' already exists (ID: {existing['id']})", existing['id']

        # Validate
        is_valid, score, msg = self.validator.validate_file(file_path)

        # Add to database
        text_id = self.db.add_text(filename, title, source_type)

        # Update validation
        status = 'valid' if is_valid else 'invalid'
        self.db.update_validation(text_id, status, score)

        if not is_valid:
            return False, f"Validation failed: {msg}", text_id

        success_msg = f"Added '{title}' (ID: {text_id}). {msg}"

        if auto_index:
            # TODO: Implement auto-indexing
            success_msg += " [Auto-indexing not yet implemented]"

        return True, success_msg, text_id

    def remove_text(self, text_id: int) -> Tuple[bool, str]:
        """
        Remove a text from the corpus.

        Args:
            text_id: Text ID to remove

        Returns:
            Tuple of (success, message)
        """
        text = self.db.get_text(text_id)
        if not text:
            return False, f"Text ID {text_id} not found"

        # If indexed, need to rebuild index
        if text['is_indexed']:
            return False, f"Text '{text['title']}' is indexed. Rebuild index after removal."

        self.db.remove_text(text_id)
        return True, f"Removed '{text['title']}' (ID: {text_id})"

    def list_texts(self, indexed_only: bool = False) -> List[Dict]:
        """List all texts in corpus."""
        return self.db.list_texts(indexed_only)

    def get_stats(self) -> Dict:
        """Get corpus statistics."""
        return self.db.get_stats()

    def close(self):
        """Close manager and database."""
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
