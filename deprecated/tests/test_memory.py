"""
Unit tests for Memory System
"""

import pytest
from datetime import datetime, timedelta
from klareco.memory import (
    MemorySystem, ShortTermMemory, LongTermMemory,
    MemoryEntry, MemoryType,
    create_memory_system
)


class TestMemoryEntry:
    """Test MemoryEntry dataclass"""

    def test_create_entry(self):
        """Test creating a memory entry"""
        entry = MemoryEntry(
            entry_id="TEST-001",
            memory_type=MemoryType.USER_QUERY,
            timestamp=datetime.now(),
            ast={'tipo': 'frazo'},
            text="Test query"
        )

        assert entry.entry_id == "TEST-001"
        assert entry.memory_type == MemoryType.USER_QUERY
        assert entry.text == "Test query"
        assert entry.metadata == {}

    def test_to_dict(self):
        """Test serialization"""
        now = datetime.now()
        entry = MemoryEntry(
            entry_id="TEST-001",
            memory_type=MemoryType.FACT,
            timestamp=now,
            ast={'tipo': 'fakto'},
            text="Test fact",
            metadata={'source': 'test'}
        )

        data = entry.to_dict()

        assert data['entry_id'] == "TEST-001"
        assert data['memory_type'] == 'fact'
        assert data['text'] == "Test fact"
        assert data['metadata'] == {'source': 'test'}

    def test_from_dict(self):
        """Test deserialization"""
        now = datetime.now()
        data = {
            'entry_id': "TEST-001",
            'memory_type': 'user_query',
            'timestamp': now.isoformat(),
            'ast': {'tipo': 'frazo'},
            'text': "Test",
            'metadata': {}
        }

        entry = MemoryEntry.from_dict(data)

        assert entry.entry_id == "TEST-001"
        assert entry.memory_type == MemoryType.USER_QUERY
        assert entry.text == "Test"


class TestShortTermMemory:
    """Test Short-Term Memory"""

    def test_create_stm(self):
        """Test creating STM"""
        stm = ShortTermMemory(max_size=10)

        assert stm.max_size == 10
        assert len(stm) == 0

    def test_add_entry(self):
        """Test adding entry to STM"""
        stm = ShortTermMemory(max_size=5)

        entry = stm.add(
            MemoryType.USER_QUERY,
            {'tipo': 'frazo'},
            "Test query"
        )

        assert entry is not None
        assert entry.entry_id == "STM-000001"
        assert len(stm) == 1

    def test_fifo_eviction(self):
        """Test FIFO eviction when max size reached"""
        stm = ShortTermMemory(max_size=3)

        # Add 4 entries
        for i in range(4):
            stm.add(MemoryType.USER_QUERY, {}, f"Query {i}")

        # Should only have 3 (oldest evicted)
        assert len(stm) == 3

        # First entry should be gone
        all_entries = stm.get_all()
        assert all_entries[0].text == "Query 1"  # Query 0 was evicted

    def test_get_recent(self):
        """Test getting recent entries"""
        stm = ShortTermMemory(max_size=10)

        for i in range(5):
            stm.add(MemoryType.USER_QUERY, {}, f"Query {i}")

        recent = stm.get_recent(n=3)

        assert len(recent) == 3
        assert recent[0].text == "Query 2"
        assert recent[-1].text == "Query 4"

    def test_search(self):
        """Test searching STM"""
        stm = ShortTermMemory(max_size=10)

        stm.add(MemoryType.USER_QUERY, {}, "Query about Esperanto")
        stm.add(MemoryType.USER_QUERY, {}, "Query about Python")
        stm.add(MemoryType.FACT, {}, "Fact about Esperanto")

        # Search for entries with "Esperanto"
        results = stm.search(lambda e: "Esperanto" in e.text)

        assert len(results) == 2

    def test_clear(self):
        """Test clearing STM"""
        stm = ShortTermMemory(max_size=10)

        for i in range(5):
            stm.add(MemoryType.USER_QUERY, {}, f"Query {i}")

        assert len(stm) == 5

        stm.clear()
        assert len(stm) == 0


class TestLongTermMemory:
    """Test Long-Term Memory"""

    def test_create_ltm(self):
        """Test creating LTM"""
        ltm = LongTermMemory()  # In-memory database

        assert ltm.count() == 0

    def test_add_entry(self):
        """Test adding entry to LTM"""
        ltm = LongTermMemory()

        entry_id = ltm.add(
            MemoryType.FACT,
            {'tipo': 'fakto'},
            "Test fact"
        )

        assert entry_id == "LTM-000001"
        assert ltm.count() == 1

    def test_get_entry(self):
        """Test retrieving entry by ID"""
        ltm = LongTermMemory()

        entry_id = ltm.add(
            MemoryType.FACT,
            {'tipo': 'fakto'},
            "Test fact"
        )

        entry = ltm.get(entry_id)

        assert entry is not None
        assert entry.entry_id == entry_id
        assert entry.text == "Test fact"

    def test_get_recent(self):
        """Test getting recent entries"""
        ltm = LongTermMemory()

        for i in range(5):
            ltm.add(MemoryType.FACT, {}, f"Fact {i}")

        recent = ltm.get_recent(n=3)

        assert len(recent) == 3
        # Most recent first
        assert recent[0].text == "Fact 4"

    def test_search_by_type(self):
        """Test searching by memory type"""
        ltm = LongTermMemory()

        ltm.add(MemoryType.USER_QUERY, {}, "Query 1")
        ltm.add(MemoryType.FACT, {}, "Fact 1")
        ltm.add(MemoryType.FACT, {}, "Fact 2")

        facts = ltm.search(memory_type=MemoryType.FACT)

        assert len(facts) == 2
        assert all(f.memory_type == MemoryType.FACT for f in facts)

    def test_search_by_time(self):
        """Test searching by timestamp"""
        ltm = LongTermMemory()

        # Add entry
        ltm.add(MemoryType.FACT, {}, "Old fact")

        # Search for entries since now (should be empty)
        since = datetime.now() + timedelta(seconds=1)
        results = ltm.search(since=since)

        assert len(results) == 0

        # Search for entries since past (should find it)
        since = datetime.now() - timedelta(days=1)
        results = ltm.search(since=since)

        assert len(results) == 1

    def test_clear(self):
        """Test clearing LTM"""
        ltm = LongTermMemory()

        for i in range(5):
            ltm.add(MemoryType.FACT, {}, f"Fact {i}")

        assert ltm.count() == 5

        ltm.clear()
        assert ltm.count() == 0


class TestMemorySystem:
    """Test complete Memory System"""

    def test_create_memory_system(self):
        """Test creating memory system"""
        memory = create_memory_system(stm_size=10)

        assert memory.stm.max_size == 10
        assert memory.ltm.count() == 0

    def test_remember_to_stm(self):
        """Test remembering to STM only"""
        memory = create_memory_system()

        memory.remember(
            MemoryType.USER_QUERY,
            {'tipo': 'frazo'},
            "Test query",
            persistent=False
        )

        assert len(memory.stm) == 1
        assert memory.ltm.count() == 0

    def test_remember_persistent(self):
        """Test remembering to both STM and LTM"""
        memory = create_memory_system()

        memory.remember(
            MemoryType.FACT,
            {'tipo': 'fakto'},
            "Important fact",
            persistent=True
        )

        assert len(memory.stm) == 1
        assert memory.ltm.count() == 1

    def test_recall_recent_from_stm(self):
        """Test recalling recent from STM"""
        memory = create_memory_system()

        for i in range(5):
            memory.remember(MemoryType.USER_QUERY, {}, f"Query {i}")

        recent = memory.recall_recent(n=3, from_ltm=False)

        assert len(recent) == 3

    def test_recall_recent_from_ltm(self):
        """Test recalling recent from LTM"""
        memory = create_memory_system()

        for i in range(5):
            memory.remember(MemoryType.FACT, {}, f"Fact {i}", persistent=True)

        recent = memory.recall_recent(n=3, from_ltm=True)

        assert len(recent) == 3

    def test_consolidate_single(self):
        """Test consolidating single entry"""
        memory = create_memory_system()

        memory.remember(MemoryType.USER_QUERY, {}, "Query")
        entry = memory.stm.get_all()[0]

        assert memory.ltm.count() == 0

        memory.consolidate(entry)

        assert memory.ltm.count() == 1

    def test_consolidate_all(self):
        """Test consolidating all STM to LTM"""
        memory = create_memory_system()

        for i in range(5):
            memory.remember(MemoryType.USER_QUERY, {}, f"Query {i}")

        assert memory.ltm.count() == 0

        memory.consolidate_all()

        assert memory.ltm.count() == 5

    def test_consolidate_skip_existing(self):
        """Test that consolidation skips existing entries"""
        memory = create_memory_system()

        memory.remember(MemoryType.FACT, {}, "Fact", persistent=True)

        assert memory.ltm.count() == 1

        # Try to consolidate again
        entry = memory.stm.get_all()[0]
        memory.consolidate(entry)

        # Should still be 1 (not duplicated)
        assert memory.ltm.count() == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
