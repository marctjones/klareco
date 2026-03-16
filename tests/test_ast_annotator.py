"""
Tests for AST Annotator Protocol

Validates that:
1. ASTAnnotator base class interface works correctly
2. Models follow the protocol (never modify deterministic features)
3. Annotations are properly added and preserved
4. Example implementation (RootEmbeddingsAnnotator) works correctly
"""

import pytest
import torch
import json
from pathlib import Path

from klareco.ast_annotator import ASTAnnotator, DeterministicAnnotator
from klareco.embeddings.root_annotator import RootEmbeddingsAnnotator


# ============================================================================
# Mock Implementations for Testing
# ============================================================================

class MockAnnotator(ASTAnnotator):
    """Simple mock annotator for testing base class."""

    def __init__(self):
        super().__init__(model_name="MockAnnotator")

    def annotate(self, ast, context=None):
        ast = self._ensure_annotations_dict(ast)
        ast = self._add_annotation(ast, 'mock_score', 0.85)
        return ast


class BadAnnotator(ASTAnnotator):
    """Annotator that violates the protocol (for testing validation)."""

    def __init__(self):
        super().__init__(model_name="BadAnnotator")

    def annotate(self, ast, context=None):
        # VIOLATION: Modifying deterministic feature
        if 'verbo' in ast and ast['verbo'] is not None:
            ast['verbo']['kazo'] = 'MODIFIED'  # Should never do this!
        return ast


class MockDeterministicAnnotator(DeterministicAnnotator):
    """Mock deterministic annotator for testing fallback mechanism."""

    def __init__(self, deterministic_success=True, fallback_model=None):
        super().__init__(model_name="MockDeterministicAnnotator", fallback_model=fallback_model)
        self.deterministic_success = deterministic_success

    def _annotate_deterministic(self, ast, context=None):
        if self.deterministic_success:
            ast = self._ensure_annotations_dict(ast)
            ast = self._add_annotation(ast, 'deterministic_annotation', 'RULE_BASED')
            return ast, True
        else:
            return ast, False


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def simple_ast():
    """Simple AST with one word."""
    return {
        'tipo': 'frazo',
        'verbo': {
            'tipo': 'vorto',
            'radiko': 'kur',
            'vortspeco': 'verbo',
            'kazo': 'nominativo',
            'nombro': 'singularo',
            'tempo': 'estanto'
        }
    }


@pytest.fixture
def complex_ast():
    """Complex AST with subject, verb, object."""
    return {
        'tipo': 'frazo',
        'subjekto': {
            'tipo': 'vortgrupo',
            'kerno': {
                'tipo': 'vorto',
                'radiko': 'hund',
                'vortspeco': 'substantivo',
                'kazo': 'nominativo',
                'nombro': 'singularo'
            },
            'priskriboj': [
                {
                    'tipo': 'vorto',
                    'radiko': 'bel',
                    'vortspeco': 'adjektivo',
                    'kazo': 'nominativo',
                    'nombro': 'singularo'
                }
            ]
        },
        'verbo': {
            'tipo': 'vorto',
            'radiko': 'vid',
            'vortspeco': 'verbo',
            'tempo': 'estanto'
        },
        'objekto': {
            'tipo': 'vorto',
            'radiko': 'kat',
            'vortspeco': 'substantivo',
            'kazo': 'akuzativo',
            'nombro': 'singularo'
        }
    }


# ============================================================================
# Tests for Base ASTAnnotator Class
# ============================================================================

def test_ast_annotator_base_class():
    """Test that ASTAnnotator base class works correctly."""
    annotator = MockAnnotator()

    assert annotator.model_name == "MockAnnotator"
    assert repr(annotator) == "<MockAnnotator(model_name='MockAnnotator')>"


def test_ensure_annotations_dict(simple_ast):
    """Test that _ensure_annotations_dict creates annotations dict if missing."""
    annotator = MockAnnotator()

    # Initially no annotations
    assert 'annotations' not in simple_ast

    # After ensuring, annotations dict exists
    annotator._ensure_annotations_dict(simple_ast)
    assert 'annotations' in simple_ast
    assert isinstance(simple_ast['annotations'], dict)
    assert len(simple_ast['annotations']) == 0


def test_add_annotation(simple_ast):
    """Test that _add_annotation adds annotations correctly."""
    annotator = MockAnnotator()

    # Add first annotation
    annotator._add_annotation(simple_ast, 'score', 0.9)
    assert simple_ast['annotations']['score'] == 0.9

    # Add second annotation (should preserve first)
    annotator._add_annotation(simple_ast, 'label', 'positive')
    assert simple_ast['annotations']['score'] == 0.9
    assert simple_ast['annotations']['label'] == 'positive'


def test_get_annotation(simple_ast):
    """Test that _get_annotation retrieves annotations correctly."""
    annotator = MockAnnotator()

    # No annotations yet
    assert annotator._get_annotation(simple_ast, 'score') is None
    assert annotator._get_annotation(simple_ast, 'score', default=0.5) == 0.5

    # Add annotation
    annotator._add_annotation(simple_ast, 'score', 0.9)
    assert annotator._get_annotation(simple_ast, 'score') == 0.9


def test_read_deterministic_feature(simple_ast):
    """Test that _read_deterministic_feature reads features correctly."""
    annotator = MockAnnotator()

    verbo = simple_ast['verbo']

    # Read deterministic features
    assert annotator._read_deterministic_feature(verbo, 'radiko') == 'kur'
    assert annotator._read_deterministic_feature(verbo, 'kazo') == 'nominativo'
    assert annotator._read_deterministic_feature(verbo, 'nombro') == 'singularo'
    assert annotator._read_deterministic_feature(verbo, 'tempo') == 'estanto'

    # Missing feature raises KeyError
    with pytest.raises(KeyError):
        annotator._read_deterministic_feature(verbo, 'missing_feature')


def test_annotate_preserves_deterministic_features(simple_ast):
    """Test that annotate() preserves deterministic features."""
    annotator = MockAnnotator()

    # Save original values
    original_kazo = simple_ast['verbo']['kazo']
    original_nombro = simple_ast['verbo']['nombro']
    original_radiko = simple_ast['verbo']['radiko']

    # Annotate
    annotated_ast = annotator.annotate(simple_ast)

    # Deterministic features unchanged
    assert annotated_ast['verbo']['kazo'] == original_kazo
    assert annotated_ast['verbo']['nombro'] == original_nombro
    assert annotated_ast['verbo']['radiko'] == original_radiko

    # Annotation added
    assert 'annotations' in annotated_ast
    assert annotated_ast['annotations']['mock_score'] == 0.85


def test_bad_annotator_violates_protocol(simple_ast):
    """Test that we can detect when annotators violate the protocol."""
    bad_annotator = BadAnnotator()

    # Save original value
    original_kazo = simple_ast['verbo']['kazo']

    # Annotate (violates protocol by modifying kazo)
    annotated_ast = bad_annotator.annotate(simple_ast)

    # Violation detected: kazo was modified
    assert annotated_ast['verbo']['kazo'] != original_kazo
    assert annotated_ast['verbo']['kazo'] == 'MODIFIED'


def test_annotate_batch(complex_ast, simple_ast):
    """Test that annotate_batch processes multiple ASTs."""
    annotator = MockAnnotator()

    asts = [simple_ast, complex_ast]
    annotated_asts = annotator.annotate_batch(asts)

    # Both ASTs annotated
    assert len(annotated_asts) == 2
    assert annotated_asts[0]['annotations']['mock_score'] == 0.85
    assert annotated_asts[1]['annotations']['mock_score'] == 0.85


# ============================================================================
# Tests for DeterministicAnnotator
# ============================================================================

def test_deterministic_annotator_success(simple_ast):
    """Test that deterministic annotator uses rules when successful."""
    annotator = MockDeterministicAnnotator(deterministic_success=True)

    annotated_ast = annotator.annotate(simple_ast)

    # Deterministic annotation added
    assert 'annotations' in annotated_ast
    assert annotated_ast['annotations']['deterministic_annotation'] == 'RULE_BASED'


def test_deterministic_annotator_fallback(simple_ast):
    """Test that deterministic annotator uses fallback model when rules fail."""
    fallback = MockAnnotator()
    annotator = MockDeterministicAnnotator(deterministic_success=False, fallback_model=fallback)

    annotated_ast = annotator.annotate(simple_ast)

    # Fallback annotation added
    assert 'annotations' in annotated_ast
    assert annotated_ast['annotations']['mock_score'] == 0.85


def test_deterministic_annotator_no_fallback_raises(simple_ast):
    """Test that deterministic annotator raises error when no fallback and rules fail."""
    annotator = MockDeterministicAnnotator(deterministic_success=False, fallback_model=None)

    with pytest.raises(RuntimeError, match="No fallback model configured"):
        annotator.annotate(simple_ast)


# ============================================================================
# Tests for RootEmbeddingsAnnotator
# ============================================================================

@pytest.fixture
def mock_root_embeddings_model(tmp_path):
    """Create mock root embeddings model and vocabulary."""
    # Create vocabulary
    vocab = {
        'hund': 0,
        'kat': 1,
        'vid': 2,
        'kur': 3,
        'bel': 4
    }
    vocab_path = tmp_path / 'root_vocab.json'
    with open(vocab_path, 'w') as f:
        json.dump(vocab, f)

    # Create embedding table
    embedding_table = torch.randn(5, 64)  # 5 roots, 64d
    checkpoint = {
        'model_state_dict': {
            'embeddings.weight': embedding_table
        }
    }
    model_path = tmp_path / 'best_model.pt'
    torch.save(checkpoint, model_path)

    return str(model_path), str(vocab_path)


def test_root_embeddings_annotator_initialization(mock_root_embeddings_model):
    """Test that RootEmbeddingsAnnotator initializes correctly."""
    model_path, vocab_path = mock_root_embeddings_model

    annotator = RootEmbeddingsAnnotator(
        model_path=model_path,
        vocab_path=vocab_path
    )

    assert annotator.model_name == "RootEmbeddings"
    assert annotator.embedding_table is not None
    assert annotator.root_to_idx is not None
    assert len(annotator.root_to_idx) == 5
    assert annotator.embed_dim == 64


def test_root_embeddings_annotates_simple_ast(mock_root_embeddings_model, simple_ast):
    """Test that RootEmbeddingsAnnotator annotates a simple AST."""
    model_path, vocab_path = mock_root_embeddings_model

    annotator = RootEmbeddingsAnnotator(
        model_path=model_path,
        vocab_path=vocab_path
    )

    annotated_ast = annotator.annotate(simple_ast)

    # Root embedding added to verb
    assert 'annotations' in annotated_ast['verbo']
    assert 'root_embedding' in annotated_ast['verbo']['annotations']
    assert len(annotated_ast['verbo']['annotations']['root_embedding']) == 64

    # Deterministic features preserved
    assert annotated_ast['verbo']['radiko'] == 'kur'
    assert annotated_ast['verbo']['kazo'] == 'nominativo'


def test_root_embeddings_annotates_complex_ast(mock_root_embeddings_model, complex_ast):
    """Test that RootEmbeddingsAnnotator annotates a complex AST."""
    model_path, vocab_path = mock_root_embeddings_model

    annotator = RootEmbeddingsAnnotator(
        model_path=model_path,
        vocab_path=vocab_path
    )

    annotated_ast = annotator.annotate(complex_ast)

    # Root embeddings added to all words
    # Subject (kerno)
    assert 'root_embedding' in annotated_ast['subjekto']['kerno']['annotations']
    assert len(annotated_ast['subjekto']['kerno']['annotations']['root_embedding']) == 64

    # Subject (priskriboj)
    assert 'root_embedding' in annotated_ast['subjekto']['priskriboj'][0]['annotations']

    # Verb
    assert 'root_embedding' in annotated_ast['verbo']['annotations']

    # Object
    assert 'root_embedding' in annotated_ast['objekto']['annotations']

    # All deterministic features preserved
    assert annotated_ast['subjekto']['kerno']['radiko'] == 'hund'
    assert annotated_ast['verbo']['radiko'] == 'vid'
    assert annotated_ast['objekto']['radiko'] == 'kat'


def test_root_embeddings_handles_oov(mock_root_embeddings_model):
    """Test that RootEmbeddingsAnnotator handles OOV roots gracefully."""
    model_path, vocab_path = mock_root_embeddings_model

    annotator = RootEmbeddingsAnnotator(
        model_path=model_path,
        vocab_path=vocab_path
    )

    oov_ast = {
        'tipo': 'frazo',
        'verbo': {
            'tipo': 'vorto',
            'radiko': 'unknown_root',  # OOV
            'vortspeco': 'verbo'
        }
    }

    annotated_ast = annotator.annotate(oov_ast)

    # OOV root gets zero vector
    assert 'root_embedding' in annotated_ast['verbo']['annotations']
    embedding = annotated_ast['verbo']['annotations']['root_embedding']
    assert all(x == 0.0 for x in embedding)


def test_root_embeddings_get_similar_roots(mock_root_embeddings_model):
    """Test that get_similar_roots finds similar roots."""
    model_path, vocab_path = mock_root_embeddings_model

    annotator = RootEmbeddingsAnnotator(
        model_path=model_path,
        vocab_path=vocab_path
    )

    similar = annotator.get_similar_roots('hund', top_k=3)

    # Returns list of (root, similarity) tuples
    assert len(similar) == 3
    assert all(isinstance(item, tuple) for item in similar)
    assert all(isinstance(item[0], str) for item in similar)
    assert all(isinstance(item[1], float) for item in similar)

    # Self not included
    assert 'hund' not in [root for root, sim in similar]


def test_root_embeddings_get_similar_roots_oov(mock_root_embeddings_model):
    """Test that get_similar_roots handles OOV roots."""
    model_path, vocab_path = mock_root_embeddings_model

    annotator = RootEmbeddingsAnnotator(
        model_path=model_path,
        vocab_path=vocab_path
    )

    similar = annotator.get_similar_roots('unknown_root', top_k=3)

    # OOV returns empty list
    assert similar == []


# ============================================================================
# Integration Tests: Chaining Multiple Annotators
# ============================================================================

def test_chaining_multiple_annotators(mock_root_embeddings_model, complex_ast):
    """Test that multiple annotators can be chained in a pipeline."""
    model_path, vocab_path = mock_root_embeddings_model

    # Create pipeline
    root_annotator = RootEmbeddingsAnnotator(
        model_path=model_path,
        vocab_path=vocab_path
    )
    mock_annotator = MockAnnotator()

    # Chain annotators
    ast = root_annotator.annotate(complex_ast)
    ast = mock_annotator.annotate(ast)

    # Both annotations present
    # Root embeddings (word-level)
    assert 'root_embedding' in ast['verbo']['annotations']

    # Mock score (sentence-level)
    assert 'mock_score' in ast['annotations']
    assert ast['annotations']['mock_score'] == 0.85

    # All deterministic features preserved
    assert ast['verbo']['radiko'] == 'vid'
    assert ast['objekto']['kazo'] == 'akuzativo'


def test_annotation_protocol_validation(simple_ast):
    """
    Test that the protocol correctly validates that models:
    1. Never modify deterministic features
    2. Only add annotations
    3. Preserve existing annotations
    """
    # Step 1: First annotator adds annotation
    annotator1 = MockAnnotator()
    ast = annotator1.annotate(simple_ast)

    original_kazo = ast['verbo']['kazo']
    original_radiko = ast['verbo']['radiko']
    first_annotation = ast['annotations']['mock_score']

    # Step 2: Second annotator should preserve first annotation
    class SecondAnnotator(ASTAnnotator):
        def __init__(self):
            super().__init__(model_name="SecondAnnotator")

        def annotate(self, ast, context=None):
            ast = self._ensure_annotations_dict(ast)
            # Check first annotation still exists
            assert self._get_annotation(ast, 'mock_score') == 0.85
            # Add second annotation
            ast = self._add_annotation(ast, 'second_score', 0.95)
            return ast

    annotator2 = SecondAnnotator()
    ast = annotator2.annotate(ast)

    # Validate:
    # - Deterministic features unchanged
    assert ast['verbo']['kazo'] == original_kazo
    assert ast['verbo']['radiko'] == original_radiko

    # - First annotation preserved
    assert ast['annotations']['mock_score'] == first_annotation

    # - Second annotation added
    assert ast['annotations']['second_score'] == 0.95


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
