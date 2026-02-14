#!/usr/bin/env python3
"""
Tests for v2.0 corpus indexer.

Validates that corpus can be indexed into AST-native graph structure.
"""

import tempfile
from pathlib import Path
import json

import pytest

try:
    import kuzu
    KUZU_AVAILABLE = True
except ImportError:
    KUZU_AVAILABLE = False

# Skip all tests if kuzu not available
pytestmark = pytest.mark.skipif(not KUZU_AVAILABLE, reason="Kuzu not installed")

from scripts.index_corpus_v2 import CorpusIndexer


@pytest.fixture
def sample_corpus_entry():
    """Sample corpus entry for testing."""
    return {
        "text": "La hundo vidas la katon.",
        "source": {
            "name": "test_source",
            "source_type": "test",
            "source_name": "Test Source",
            "author": "Test Author",
            "year": 2026,
            "quality": "GOLD",
            "sentence_type": "text"
        },
        "ast": {
            "tipo": "frazo",
            "fraztipo": "deklaro",
            "demandotipo": None,
            "negita": False,
            "subjekto": {
                "tipo": "vortgrupo",
                "kerno": {
                    "tipo": "vorto",
                    "plena_vorto": "hundo",
                    "radiko": "hund",
                    "vortspeco": "substantivo",
                    "nombro": "singularo",
                    "kazo": "nominativo",
                    "prefiksoj": [],
                    "sufiksoj": [],
                    "parse_status": "success"
                },
                "priskriboj": [
                    {
                        "tipo": "vorto",
                        "plena_vorto": "La",
                        "radiko": "la",
                        "vortspeco": "artikolo",
                        "prefiksoj": [],
                        "sufiksoj": [],
                        "parse_status": "success"
                    }
                ]
            },
            "verbo": {
                "tipo": "vorto",
                "plena_vorto": "vidas",
                "radiko": "vid",
                "vortspeco": "verbo",
                "tempo": "prezenco",
                "modo": "indikativo",
                "prefiksoj": [],
                "sufiksoj": [],
                "parse_status": "success"
            },
            "objekto": {
                "tipo": "vortgrupo",
                "kerno": {
                    "tipo": "vorto",
                    "plena_vorto": "katon",
                    "radiko": "kat",
                    "vortspeco": "substantivo",
                    "nombro": "singularo",
                    "kazo": "akuzativo",
                    "prefiksoj": [],
                    "sufiksoj": [],
                    "parse_status": "success"
                },
                "priskriboj": [
                    {
                        "tipo": "vorto",
                        "plena_vorto": "la",
                        "radiko": "la",
                        "vortspeco": "artikolo",
                        "prefiksoj": [],
                        "sufiksoj": [],
                        "parse_status": "success"
                    }
                ]
            },
            "parse_statistics": {
                "total_words": 5,
                "esperanto_words": 5,
                "non_esperanto_words": 0,
                "success_rate": 1.0,
                "parse_categories": {}
            }
        }
    }


@pytest.fixture
def temp_corpus_file(sample_corpus_entry, tmp_path):
    """Create temporary corpus JSONL file."""
    corpus_file = tmp_path / "test_corpus.jsonl"

    # Write 3 entries
    with open(corpus_file, 'w') as f:
        for _ in range(3):
            f.write(json.dumps(sample_corpus_entry) + '\n')

    return corpus_file


@pytest.fixture
def temp_vocab_file(tmp_path):
    """Create temporary vocabulary file."""
    vocab_file = tmp_path / "vocab.json"
    vocab = ["hund", "kat", "vid", "la"]
    with open(vocab_file, 'w') as f:
        json.dump(vocab, f)
    return vocab_file


@pytest.fixture
def indexer(tmp_path):
    """Create indexer with temporary database."""
    db_path = tmp_path / "test.db"
    indexer = CorpusIndexer(db_path)
    indexer.connect()
    indexer.create_schema()
    return indexer


def test_indexer_initialization(tmp_path):
    """Indexer can be initialized."""
    db_path = tmp_path / "test.db"
    indexer = CorpusIndexer(db_path)
    assert indexer.db_path == db_path
    assert indexer.next_collection_id == 1
    assert indexer.next_document_id == 1


def test_schema_creation(indexer):
    """Schema can be created in database."""
    # Verify key tables exist
    result = indexer.conn.execute("MATCH (n:Sentence) RETURN count(n)")
    assert result is not None

    result = indexer.conn.execute("MATCH (n:AST) RETURN count(n)")
    assert result is not None

    result = indexer.conn.execute("MATCH (n:Vorto) RETURN count(n)")
    assert result is not None


def test_create_collection(indexer, sample_corpus_entry):
    """SourceCollection can be created."""
    source = sample_corpus_entry['source']
    collection_id = indexer.get_or_create_collection(source)

    assert collection_id == 1

    # Verify node exists
    result = indexer.conn.execute("MATCH (c:SourceCollection) RETURN count(c)")
    count = result.get_next()[0]
    assert count == 1

    # Getting again returns same ID
    collection_id2 = indexer.get_or_create_collection(source)
    assert collection_id2 == collection_id


def test_create_document(indexer, sample_corpus_entry):
    """Document can be created and linked to collection."""
    source = sample_corpus_entry['source']
    collection_id = indexer.get_or_create_collection(source)
    doc_id = indexer.create_document(source, collection_id)

    assert doc_id == 1

    # Verify node exists
    result = indexer.conn.execute("MATCH (d:Document) RETURN count(d)")
    count = result.get_next()[0]
    assert count == 1

    # Verify link exists
    result = indexer.conn.execute("""
        MATCH (d:Document)-[:IN_COLLECTION]->(c:SourceCollection)
        RETURN count(*)
    """)
    count = result.get_next()[0]
    assert count == 1


def test_create_sentence(indexer):
    """Sentence can be created."""
    sentence_id = indexer.create_sentence("Test sentence.", 1)

    assert sentence_id == 1

    # Verify node exists
    result = indexer.conn.execute("MATCH (s:Sentence) RETURN s.text")
    text = result.get_next()[0]
    assert text == "Test sentence."


def test_create_ast(indexer, sample_corpus_entry):
    """AST can be created with metadata."""
    sentence_id = indexer.create_sentence("Test.", 1)
    ast_dict = sample_corpus_entry['ast']
    ast_id = indexer.create_ast(ast_dict, sentence_id)

    assert ast_id == 1

    # Verify node exists
    result = indexer.conn.execute("MATCH (ast:AST) RETURN ast.fraztipo, ast.total_words")
    fraztipo, total_words = result.get_next()
    assert fraztipo == "deklaro"
    assert total_words == 5

    # Verify link exists
    result = indexer.conn.execute("""
        MATCH (s:Sentence)-[:SENTENCE_HAS_AST]->(ast:AST)
        RETURN count(*)
    """)
    count = result.get_next()[0]
    assert count == 1


def test_create_vorto(indexer, sample_corpus_entry):
    """Vorto can be created with all fields."""
    vorto_dict = sample_corpus_entry['ast']['verbo']
    vorto_id = indexer.create_vorto(vorto_dict, 1)

    assert vorto_id == 1

    # Verify node exists with correct fields
    result = indexer.conn.execute("""
        MATCH (v:Vorto)
        RETURN v.plena_vorto, v.radiko, v.vortspeco, v.tempo
    """)
    plena_vorto, radiko, vortspeco, tempo = result.get_next()
    assert plena_vorto == "vidas"
    assert radiko == "vid"
    assert vortspeco == "verbo"
    assert tempo == "prezenco"


def test_create_vortgrupo(indexer, sample_corpus_entry):
    """Vortgrupo can be created with kerno and priskriboj."""
    vg_dict = sample_corpus_entry['ast']['subjekto']
    vg_id = indexer.create_vortgrupo(vg_dict, 1)

    assert vg_id == 1

    # Verify node exists
    result = indexer.conn.execute("MATCH (vg:Vortgrupo) RETURN count(vg)")
    count = result.get_next()[0]
    assert count == 1

    # Verify kerno link
    result = indexer.conn.execute("""
        MATCH (vg:Vortgrupo)-[:HAS_KERNO]->(v:Vorto)
        RETURN v.radiko
    """)
    radiko = result.get_next()[0]
    assert radiko == "hund"

    # Verify priskribo link
    result = indexer.conn.execute("""
        MATCH (vg:Vortgrupo)-[:HAS_PRISKRIBO]->(v:Vorto)
        RETURN v.radiko
    """)
    radiko = result.get_next()[0]
    assert radiko == "la"


def test_create_frazo(indexer, sample_corpus_entry):
    """Frazo can be created with full structure."""
    ast_dict = sample_corpus_entry['ast']
    frazo_id = indexer.create_frazo(ast_dict, 1)

    assert frazo_id == 1

    # Verify node exists
    result = indexer.conn.execute("MATCH (f:Frazo) RETURN count(f)")
    count = result.get_next()[0]
    assert count == 1

    # Verify subjekto link (Vortgrupo)
    result = indexer.conn.execute("""
        MATCH (f:Frazo)-[:HAS_SUBJEKTO_VORTGRUPO]->(vg:Vortgrupo)
        RETURN count(*)
    """)
    count = result.get_next()[0]
    assert count == 1

    # Verify verbo link
    result = indexer.conn.execute("""
        MATCH (f:Frazo)-[:HAS_VERBO]->(v:Vorto)
        RETURN v.radiko
    """)
    radiko = result.get_next()[0]
    assert radiko == "vid"

    # Verify objekto link (Vortgrupo)
    result = indexer.conn.execute("""
        MATCH (f:Frazo)-[:HAS_OBJEKTO_VORTGRUPO]->(vg:Vortgrupo)
        RETURN count(*)
    """)
    count = result.get_next()[0]
    assert count == 1


def test_index_entry(indexer, sample_corpus_entry):
    """Full corpus entry can be indexed."""
    indexer.index_entry(sample_corpus_entry)

    # Verify complete structure was created
    result = indexer.conn.execute("MATCH (c:SourceCollection) RETURN count(c)")
    assert result.get_next()[0] == 1

    result = indexer.conn.execute("MATCH (d:Document) RETURN count(d)")
    assert result.get_next()[0] == 1

    result = indexer.conn.execute("MATCH (s:Sentence) RETURN count(s)")
    assert result.get_next()[0] == 1

    result = indexer.conn.execute("MATCH (ast:AST) RETURN count(ast)")
    assert result.get_next()[0] == 1

    result = indexer.conn.execute("MATCH (f:Frazo) RETURN count(f)")
    assert result.get_next()[0] == 1

    # Should have 2 Vortgrupo (subjekto + objekto)
    result = indexer.conn.execute("MATCH (vg:Vortgrupo) RETURN count(vg)")
    assert result.get_next()[0] == 2

    # Should have 5 Vorto nodes (2 kerno + 2 priskribo + 1 verbo)
    result = indexer.conn.execute("MATCH (v:Vorto) RETURN count(v)")
    assert result.get_next()[0] == 5


def test_index_corpus(indexer, temp_corpus_file):
    """Multiple corpus entries can be indexed."""
    indexer.index_corpus(temp_corpus_file, max_entries=3)

    # Should have 3 sentences
    result = indexer.conn.execute("MATCH (s:Sentence) RETURN count(s)")
    assert result.get_next()[0] == 3

    # Should have 3 ASTs
    result = indexer.conn.execute("MATCH (ast:AST) RETURN count(ast)")
    assert result.get_next()[0] == 3

    # But only 1 collection (all from same source)
    result = indexer.conn.execute("MATCH (c:SourceCollection) RETURN count(c)")
    assert result.get_next()[0] == 1


def test_build_root_index(indexer, sample_corpus_entry, temp_vocab_file):
    """Root index can be built from vocabulary and corpus."""
    # Index an entry first
    indexer.index_entry(sample_corpus_entry)

    # Build root index
    indexer.build_root_index(temp_vocab_file)

    # Should have roots from vocab
    result = indexer.conn.execute("MATCH (r:Root) RETURN count(r)")
    count = result.get_next()[0]
    assert count >= 4  # At least hund, kat, vid, la

    # Verify frequency stats
    result = indexer.conn.execute("""
        MATCH (r:Root {root: 'hund'})
        RETURN r.doc_freq, r.total_freq
    """)
    doc_freq, total_freq = result.get_next()
    assert doc_freq == 1  # Appears in 1 document
    assert total_freq == 1  # Appears 1 time total

    # Verify Vorto->Root links
    result = indexer.conn.execute("""
        MATCH (v:Vorto)-[:HAS_ROOT]->(r:Root {root: 'hund'})
        RETURN count(*)
    """)
    count = result.get_next()[0]
    assert count == 1  # One word "hundo" links to root "hund"


def test_get_stats(indexer, sample_corpus_entry):
    """Statistics can be retrieved."""
    indexer.index_entry(sample_corpus_entry)
    indexer.build_root_index()

    stats = indexer.get_stats()

    assert 'SourceCollection_count' in stats
    assert 'Document_count' in stats
    assert 'Sentence_count' in stats
    assert 'AST_count' in stats
    assert 'Frazo_count' in stats
    assert 'Vortgrupo_count' in stats
    assert 'Vorto_count' in stats
    assert 'Root_count' in stats

    assert stats['Sentence_count'] == 1
    assert stats['AST_count'] == 1
    assert stats['Frazo_count'] == 1


def test_full_indexing_pipeline(temp_corpus_file, temp_vocab_file, tmp_path):
    """Full indexing pipeline works end-to-end."""
    db_path = tmp_path / "full_test.db"

    # Create indexer
    indexer = CorpusIndexer(db_path)
    indexer.connect()
    indexer.create_schema()

    # Index corpus
    indexer.index_corpus(temp_corpus_file, max_entries=3)

    # Build root index
    indexer.build_root_index(temp_vocab_file)

    # Get stats
    stats = indexer.get_stats()

    # Verify complete pipeline
    assert stats['SourceCollection_count'] == 1
    assert stats['Document_count'] == 3
    assert stats['Sentence_count'] == 3
    assert stats['AST_count'] == 3
    assert stats['Frazo_count'] == 3
    assert stats['Vorto_count'] == 15  # 5 words per sentence * 3
    assert stats['Root_count'] >= 4


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
