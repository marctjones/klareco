import pytest
import torch
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

# Adjust path for local import
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.create_synthesis_dataset import create_training_examples, generate_question_from_ast, process_corpus
from klareco.ast_to_graph import ASTToGraphConverter


# Mock the parse function from klareco.parser
@pytest.fixture
def mock_parse():
    with patch('scripts.create_synthesis_dataset.parse') as mock:
        # Default mock behavior: return a simple mock object for AST
        mock.return_value = MagicMock()
        yield mock

# Mock the ASTToGraphConverter
@pytest.fixture
def mock_converter():
    converter = MagicMock(spec=ASTToGraphConverter)
    # Default mock behavior: return a simple dict for graph
    converter.ast_to_graph.return_value.to_dict.return_value = {"x": [0, 1], "edge_index": [[0,1]]}
    return converter

def test_generate_question_from_ast_kion():
    """Test 'Kion?' rule for objects."""
    mock_ast_node = MagicMock()
    # Simulate an AST string representation that indicates an object
    mock_ast_node.__str__.return_value = "OBJ(akuzativo)"
    question = generate_question_from_ast(mock_ast_node)
    assert question.startswith("Kion")

def test_generate_question_from_ast_kiu():
    """Test 'Kiu?' rule for subjects."""
    mock_ast_node = MagicMock()
    # Simulate an AST string representation that indicates a subject
    mock_ast_node.__str__.return_value = "SUBJ(nominativo)"
    question = generate_question_from_ast(mock_ast_node)
    assert question.startswith("Kiu")

def test_generate_question_from_ast_fallback():
    """Test fallback rule when no specific rule matches."""
    mock_ast_node = MagicMock()
    mock_ast_node.__str__.return_value = "VERB(prezenco)" # Some other AST string
    question = generate_question_from_ast(mock_ast_node)
    assert question == "Kio okazis?"

def test_create_training_examples_single_paragraph(mock_parse, mock_converter):
    """Test that create_training_examples generates correct number of examples."""
    paragraph = "La knabo ludas. La kato dormas. La hundo kuras."
    # Expect 3 examples if each sentence can be a target
    examples = list(create_training_examples(paragraph, mock_converter))
    assert len(examples) == 3

    # Check structure of one example
    first_example = examples[0]
    assert "question_graph" in first_example
    assert "context_graph" in first_example
    assert "target_text" in first_example
    assert first_example["target_text"] == "La knabo ludas"

def test_create_training_examples_short_paragraph(mock_parse, mock_converter):
    """Test that short paragraphs yield no examples."""
    paragraph = "Tro mallonga frazo."
    examples = list(create_training_examples(paragraph, mock_converter))
    assert len(examples) == 0

def test_process_corpus_limit(tmp_path, mock_parse, mock_converter):
    """Test that process_corpus respects max_examples."""
    # Create dummy corpus files with simple sentences
    (tmp_path / "corpus").mkdir()
    (tmp_path / "corpus" / "file1.txt").write_text("Unua frazo. Dua frazo. Tria frazo.\n\nDua paragrafo unua frazo. Dua paragrafo dua frazo.")
    (tmp_path / "corpus" / "file2.txt").write_text("Tria paragrafo unua frazo. Tria paragrafo dua frazo.")

    output_file = tmp_path / "output.jsonl"
    
    # Paragraph 1 has 3 sentences -> 3 examples
    # Paragraph 2 has 2 sentences -> 2 examples
    # Paragraph 3 has 2 sentences -> 2 examples
    # Total possible = 7 examples.
    
    # Limit to 4 examples
    process_corpus(tmp_path / "corpus", output_file, max_examples=4)
    
    with open(output_file, 'r') as f:
        lines = f.readlines()
    assert len(lines) == 4

def test_process_corpus_file_processing(tmp_path, mock_parse, mock_converter):
    """Test that process_corpus processes files and produces valid JSONL."""
    (tmp_path / "corpus").mkdir()
    (tmp_path / "corpus" / "file1.txt").write_text("Unua frazo en la unua dosiero. Dua frazo.")
    output_file = tmp_path / "output.jsonl"
    
    process_corpus(tmp_path / "corpus", output_file, max_examples=10)
    
    with open(output_file, 'r') as f:
        lines = f.readlines()
    
    assert len(lines) > 0
    # Try to parse one line to ensure it's valid JSON
    json.loads(lines[0])
