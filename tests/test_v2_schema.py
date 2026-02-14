#!/usr/bin/env python3
"""
Tests for v2.0 AST-Native Kuzu Schema

Validates that the schema can be created without errors.
"""

import tempfile
from pathlib import Path

import pytest

try:
    import kuzu
    KUZU_AVAILABLE = True
except ImportError:
    KUZU_AVAILABLE = False

from klareco.schema.kuzu_ast_schema import (
    SCHEMA_VERSION,
    get_create_statements,
    FULL_SCHEMA,
)


def test_schema_version():
    """Schema version is defined."""
    assert SCHEMA_VERSION.startswith("2.0")


def test_get_create_statements():
    """Can extract CREATE statements from schema."""
    statements = get_create_statements()

    assert len(statements) > 0
    assert all(stmt.strip().endswith(';') for stmt in statements)
    assert all('CREATE' in stmt.upper() for stmt in statements)


def test_schema_blocks_defined():
    """All schema blocks are defined."""
    assert len(FULL_SCHEMA) == 8  # 4 NODE schemas + 4 REL schemas


@pytest.mark.skipif(not KUZU_AVAILABLE, reason="Kuzu not installed")
def test_schema_creates_without_errors():
    """Schema can be created in Kuzu without errors."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        db = kuzu.Database(str(db_path))
        conn = kuzu.Connection(db)

        statements = get_create_statements()

        # Execute all statements
        for i, stmt in enumerate(statements):
            try:
                conn.execute(stmt)
            except Exception as e:
                pytest.fail(f"Statement {i+1} failed: {stmt}\nError: {e}")

        # Verify tables exist
        # Check a few key tables
        result = conn.execute("MATCH (n:Sentence) RETURN count(n)")
        assert result is not None  # Table exists

        result = conn.execute("MATCH (n:Vorto) RETURN count(n)")
        assert result is not None

        result = conn.execute("MATCH (n:Root) RETURN count(n)")
        assert result is not None


@pytest.mark.skipif(not KUZU_AVAILABLE, reason="Kuzu not installed")
def test_can_create_sample_ast():
    """Can create a minimal AST graph."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        db = kuzu.Database(str(db_path))
        conn = kuzu.Connection(db)

        # Create schema
        for stmt in get_create_statements():
            conn.execute(stmt)

        # Create minimal document hierarchy
        conn.execute("""
            CREATE (c:SourceCollection {
                id: 1,
                name: 'test',
                source_type: 'test',
                language: 'eo',
                metadata: '{}'
            })
        """)

        conn.execute("""
            CREATE (d:Document {
                id: 1,
                collection_id: 1,
                title: 'Test Document',
                external_id: 'test_1',
                doc_type: 'article',
                author: 'Test',
                year: 2026,
                quality: 'GOLD',
                metadata: '{}'
            })
        """)

        conn.execute("""
            CREATE (s:Sentence {
                id: 1,
                paragraph_id: 1,
                text: 'Mi estas hundo.',
                sentence_order: 1,
                global_order: 1
            })
        """)

        # Create minimal AST
        conn.execute("""
            CREATE (ast:AST {
                id: 1,
                sentence_id: 1,
                version: 1,
                created_at: timestamp('2026-02-13T19:00:00'),
                created_by: 'parser_v1.0',
                is_current: true,
                fraztipo: 'deklaro',
                demandotipo: NULL,
                negita: false,
                total_words: 3,
                esperanto_words: 3,
                non_esperanto_words: 0,
                success_rate: 1.0,
                parse_categories: '{}'
            })
        """)

        # Verify we can query it
        result = conn.execute("MATCH (ast:AST) RETURN count(ast)")
        count = result.get_next()[0]
        assert count == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
