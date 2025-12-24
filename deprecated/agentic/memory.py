"""
Memory System - Short-Term and Long-Term Memory

The memory system stores interactions and facts as structured ASTs
rather than raw text, enabling symbolic querying and semantic retrieval.

Short-Term Memory (STM):
- Recent interactions stored as ASTs
- Fixed size (FIFO eviction)
- Fast access for contextual queries

Long-Term Memory (LTM):
- Consolidated facts from STM
- Persistent storage (SQLite)
- Indexed for efficient retrieval

This is Phase 6 of the Klareco development roadmap.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import sqlite3
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """Type of memory entry"""
    USER_QUERY = "user_query"
    SYSTEM_RESPONSE = "system_response"
    FACT = "fact"
    EVENT = "event"


@dataclass
class MemoryEntry:
    """
    A single memory entry.

    Memories are stored as ASTs (structured representations)
    rather than raw text for better querying and reasoning.
    """
    entry_id: str
    memory_type: MemoryType
    timestamp: datetime
    ast: Dict[str, Any]
    text: str  # Original text for reference
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'entry_id': self.entry_id,
            'memory_type': self.memory_type.value,
            'timestamp': self.timestamp.isoformat(),
            'ast': self.ast,
            'text': self.text,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryEntry':
        """Create from dictionary"""
        return cls(
            entry_id=data['entry_id'],
            memory_type=MemoryType(data['memory_type']),
            timestamp=datetime.fromisoformat(data['timestamp']),
            ast=data['ast'],
            text=data['text'],
            metadata=data.get('metadata', {})
        )


class ShortTermMemory:
    """
    Short-Term Memory (STM) - Recent interactions.

    Stores recent queries and responses as ASTs with FIFO eviction.
    Designed for contextual queries within a session.
    """

    def __init__(self, max_size: int = 50):
        """
        Initialize STM.

        Args:
            max_size: Maximum number of entries (FIFO eviction)
        """
        self.max_size = max_size
        self.entries: List[MemoryEntry] = []
        self.entry_counter = 0

        logger.info(f"ShortTermMemory initialized (max_size={max_size})")

    def add(self, memory_type: MemoryType, ast: Dict[str, Any], text: str,
            metadata: Optional[Dict[str, Any]] = None) -> MemoryEntry:
        """
        Add entry to STM.

        Args:
            memory_type: Type of memory entry
            ast: Structured AST representation
            text: Original text
            metadata: Optional metadata

        Returns:
            Created memory entry
        """
        # Generate entry ID
        self.entry_counter += 1
        entry_id = f"STM-{self.entry_counter:06d}"

        # Create entry
        entry = MemoryEntry(
            entry_id=entry_id,
            memory_type=memory_type,
            timestamp=datetime.now(),
            ast=ast,
            text=text,
            metadata=metadata or {}
        )

        # Add to entries
        self.entries.append(entry)

        # Evict oldest if over capacity
        if len(self.entries) > self.max_size:
            evicted = self.entries.pop(0)
            logger.debug(f"Evicted {evicted.entry_id} from STM")

        logger.debug(f"Added {entry_id} to STM")
        return entry

    def get_recent(self, n: int = 10) -> List[MemoryEntry]:
        """Get n most recent entries"""
        return self.entries[-n:]

    def get_all(self) -> List[MemoryEntry]:
        """Get all entries"""
        return self.entries.copy()

    def search(self, query_fn) -> List[MemoryEntry]:
        """
        Search STM with custom function.

        Args:
            query_fn: Function that takes MemoryEntry and returns bool

        Returns:
            Matching entries
        """
        return [entry for entry in self.entries if query_fn(entry)]

    def clear(self):
        """Clear all entries"""
        self.entries.clear()
        logger.info("STM cleared")

    def __len__(self) -> int:
        return len(self.entries)

    def __repr__(self) -> str:
        return f"STM({len(self.entries)}/{self.max_size} entries)"


class LongTermMemory:
    """
    Long-Term Memory (LTM) - Persistent storage.

    Stores consolidated facts and important information in SQLite.
    Provides indexed access for efficient retrieval.
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize LTM.

        Args:
            db_path: Path to SQLite database (uses in-memory if None)
        """
        self.db_path = db_path or ":memory:"
        self.conn = sqlite3.connect(self.db_path)
        self._init_database()

        logger.info(f"LongTermMemory initialized (db={self.db_path})")

    def _init_database(self):
        """Initialize database schema"""
        cursor = self.conn.cursor()

        # Create memories table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                entry_id TEXT PRIMARY KEY,
                memory_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                ast TEXT NOT NULL,
                text TEXT NOT NULL,
                metadata TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create index on timestamp for efficient temporal queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp
            ON memories(timestamp DESC)
        """)

        # Create index on memory_type
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_memory_type
            ON memories(memory_type)
        """)

        self.conn.commit()
        logger.debug("LTM database schema initialized")

    def add(self, memory_type: MemoryType, ast: Dict[str, Any], text: str,
            metadata: Optional[Dict[str, Any]] = None, entry_id: Optional[str] = None) -> str:
        """
        Add entry to LTM.

        Args:
            memory_type: Type of memory
            ast: Structured AST
            text: Original text
            metadata: Optional metadata
            entry_id: Optional entry ID (generated if None)

        Returns:
            Entry ID
        """
        cursor = self.conn.cursor()

        # Generate ID if not provided
        if not entry_id:
            cursor.execute("SELECT COUNT(*) FROM memories")
            count = cursor.fetchone()[0]
            entry_id = f"LTM-{count+1:06d}"

        # Serialize AST and metadata
        ast_json = json.dumps(ast)
        metadata_json = json.dumps(metadata or {})
        timestamp = datetime.now().isoformat()

        # Insert
        cursor.execute("""
            INSERT INTO memories (entry_id, memory_type, timestamp, ast, text, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (entry_id, memory_type.value, timestamp, ast_json, text, metadata_json))

        self.conn.commit()
        logger.debug(f"Added {entry_id} to LTM")

        return entry_id

    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        """Get entry by ID"""
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT entry_id, memory_type, timestamp, ast, text, metadata
            FROM memories
            WHERE entry_id = ?
        """, (entry_id,))

        row = cursor.fetchone()
        if not row:
            return None

        return self._row_to_entry(row)

    def get_recent(self, n: int = 10) -> List[MemoryEntry]:
        """Get n most recent entries"""
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT entry_id, memory_type, timestamp, ast, text, metadata
            FROM memories
            ORDER BY timestamp DESC
            LIMIT ?
        """, (n,))

        return [self._row_to_entry(row) for row in cursor.fetchall()]

    def search(self, memory_type: Optional[MemoryType] = None,
               since: Optional[datetime] = None,
               limit: int = 100) -> List[MemoryEntry]:
        """
        Search LTM with filters.

        Args:
            memory_type: Filter by type
            since: Only entries after this timestamp
            limit: Maximum results

        Returns:
            Matching entries
        """
        cursor = self.conn.cursor()

        query = "SELECT entry_id, memory_type, timestamp, ast, text, metadata FROM memories WHERE 1=1"
        params = []

        if memory_type:
            query += " AND memory_type = ?"
            params.append(memory_type.value)

        if since:
            query += " AND timestamp >= ?"
            params.append(since.isoformat())

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)

        return [self._row_to_entry(row) for row in cursor.fetchall()]

    def count(self) -> int:
        """Get total number of entries"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM memories")
        return cursor.fetchone()[0]

    def clear(self):
        """Clear all entries"""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM memories")
        self.conn.commit()
        logger.info("LTM cleared")

    def _row_to_entry(self, row) -> MemoryEntry:
        """Convert database row to MemoryEntry"""
        entry_id, memory_type, timestamp, ast_json, text, metadata_json = row

        return MemoryEntry(
            entry_id=entry_id,
            memory_type=MemoryType(memory_type),
            timestamp=datetime.fromisoformat(timestamp),
            ast=json.loads(ast_json),
            text=text,
            metadata=json.loads(metadata_json) if metadata_json else {}
        )

    def close(self):
        """Close database connection"""
        self.conn.close()
        logger.info("LTM database closed")

    def __repr__(self) -> str:
        return f"LTM({self.count()} entries, db={self.db_path})"


class MemorySystem:
    """
    Complete memory system with STM and LTM.

    Coordinates between short-term and long-term memory,
    handles consolidation, and provides unified query interface.
    """

    def __init__(self, stm_size: int = 50, ltm_db_path: Optional[str] = None):
        """
        Initialize memory system.

        Args:
            stm_size: Maximum STM entries
            ltm_db_path: Path to LTM database
        """
        self.stm = ShortTermMemory(max_size=stm_size)
        self.ltm = LongTermMemory(db_path=ltm_db_path)

        logger.info(f"MemorySystem initialized: {self.stm}, {self.ltm}")

    def remember(self, memory_type: MemoryType, ast: Dict[str, Any], text: str,
                 metadata: Optional[Dict[str, Any]] = None, persistent: bool = False):
        """
        Store a memory.

        Args:
            memory_type: Type of memory
            ast: Structured AST
            text: Original text
            metadata: Optional metadata
            persistent: If True, also store in LTM
        """
        # Always add to STM
        entry = self.stm.add(memory_type, ast, text, metadata)

        # Optionally add to LTM
        if persistent:
            self.ltm.add(memory_type, ast, text, metadata, entry_id=entry.entry_id)

        logger.debug(f"Remembered: {entry.entry_id} (persistent={persistent})")

    def recall_recent(self, n: int = 10, from_ltm: bool = False) -> List[MemoryEntry]:
        """
        Recall recent memories.

        Args:
            n: Number of entries
            from_ltm: If True, query LTM instead of STM

        Returns:
            Recent memory entries
        """
        if from_ltm:
            return self.ltm.get_recent(n)
        else:
            return self.stm.get_recent(n)

    def consolidate(self, entry: MemoryEntry):
        """
        Consolidate STM entry to LTM.

        Used for important memories that should be preserved.
        """
        # Check if already in LTM
        existing = self.ltm.get(entry.entry_id)
        if existing:
            logger.debug(f"{entry.entry_id} already in LTM, skipping")
            return

        self.ltm.add(
            memory_type=entry.memory_type,
            ast=entry.ast,
            text=entry.text,
            metadata=entry.metadata,
            entry_id=entry.entry_id
        )

        logger.info(f"Consolidated {entry.entry_id} to LTM")

    def consolidate_all(self):
        """Consolidate all STM entries to LTM"""
        for entry in self.stm.get_all():
            self.consolidate(entry)

        logger.info(f"Consolidated {len(self.stm)} entries to LTM")

    def __repr__(self) -> str:
        return f"MemorySystem(STM: {self.stm}, LTM: {self.ltm})"


# Factory function
def create_memory_system(stm_size: int = 50, ltm_db_path: Optional[str] = None) -> MemorySystem:
    """
    Create and return a MemorySystem instance.

    Args:
        stm_size: Maximum STM size
        ltm_db_path: Path to LTM database

    Returns:
        Initialized MemorySystem
    """
    return MemorySystem(stm_size, ltm_db_path)


if __name__ == "__main__":
    # Test memory system
    print("Testing Memory System")
    print("=" * 80)

    # Create memory system
    memory = create_memory_system(stm_size=5)

    print(f"\n{memory}\n")

    # Add some memories
    print("Adding memories to STM...")
    memory.remember(
        MemoryType.USER_QUERY,
        {'tipo': 'frazo', 'teksto': 'Kio estas Esperanto?'},
        "Kio estas Esperanto?"
    )

    memory.remember(
        MemoryType.SYSTEM_RESPONSE,
        {'tipo': 'respondo', 'teksto': 'Esperanto estas internacia lingvo.'},
        "Esperanto estas internacia lingvo."
    )

    memory.remember(
        MemoryType.FACT,
        {'tipo': 'fakto', 'teksto': 'Esperanto estis kreita de Zamenhof.'},
        "Esperanto estis kreita de Zamenhof.",
        persistent=True  # This goes to LTM too
    )

    print(f"STM: {memory.stm}")
    print(f"LTM: {memory.ltm}\n")

    # Recall recent
    print("Recent STM entries:")
    for entry in memory.recall_recent(n=3):
        print(f"  {entry.entry_id}: {entry.text[:50]}")

    print("\nRecent LTM entries:")
    for entry in memory.recall_recent(n=3, from_ltm=True):
        print(f"  {entry.entry_id}: {entry.text[:50]}")

    # Consolidate
    print("\nConsolidating all STM to LTM...")
    memory.consolidate_all()
    print(f"LTM: {memory.ltm}")

    print("\n✅ Memory system test complete!")
