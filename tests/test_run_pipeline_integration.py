import pytest
import torch
from unittest.mock import MagicMock, patch
from torch_geometric.data import Data

# Adjust path for local import
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_pipeline import run_pipeline
from klareco.parser import parse
from klareco.ast_to_graph import ASTToGraphConverter
from klareco.rag.retriever import create_retriever
from klareco.models.generator import Graph2SeqGenerator

# Fixture for mocking the parser and converter
@pytest.fixture
def mock_pipeline_deps():
    with patch('scripts.run_pipeline.parse') as mock_parse, \
         patch('scripts.run_pipeline.ASTToGraphConverter') as MockASTToGraphConverter:
        
        mock_ast = MagicMock()
        mock_parse.return_value = mock_ast
        
        mock_graph = MagicMock(spec=Data)
        mock_graph.x = torch.tensor([0])
        mock_graph.edge_index = torch.tensor([[0],[0]])
        mock_graph.to.return_value = mock_graph

        mock_converter_instance = MagicMock(spec=ASTToGraphConverter)
        mock_converter_instance.ast_to_graph.return_value = mock_graph
        MockASTToGraphConverter.return_value = mock_converter_instance

        yield mock_parse, mock_converter_instance

# Fixture for mocking the retriever
@pytest.fixture
def mock_retriever():
    with patch('scripts.run_pipeline.create_retriever') as MockCreateRetriever:
        mock_retriever_instance = MagicMock()
        mock_retriever_instance.retrieve_hybrid.return_value = [
            {'text': 'This is context sentence one.', 'score': 0.9},
            {'text': 'This is context sentence two.', 'score': 0.8}
        ]
        MockCreateRetriever.return_value = mock_retriever_instance
        yield mock_retriever_instance

# Fixture for mocking the generator model
@pytest.fixture
def mock_generator_model():
    with patch('scripts.run_pipeline.Graph2SeqGenerator') as MockGenerator:
        mock_model_instance = MagicMock(spec=Graph2SeqGenerator)
        mock_model_instance.return_value = torch.tensor([[[0, 1, 0, 0, 0], [0, 0, 1, 0, 0]]])
        mock_model_instance.eval.return_value = None
        mock_model_instance.to.return_value = mock_model_instance
        MockGenerator.return_value = mock_model_instance
        yield mock_model_instance

def test_run_pipeline_integration_flow(
    mock_pipeline_deps, 
    mock_retriever, 
    mock_generator_model
):
    """
    Test the complete flow of the run_pipeline script using mocked components.
    """
    mock_parse, mock_converter_instance = mock_pipeline_deps
    dummy_question = "Kio estas la koloro de la ĉielo?"
    device = torch.device("cpu")

    result_answer = run_pipeline(
        dummy_question, 
        mock_retriever, 
        mock_generator_model, 
        mock_converter_instance, 
        device
    )

    # Assert that parsing happened for the question
    mock_parse.assert_any_call(dummy_question)

    # Assert that retrieval happened
    mock_retriever.retrieve_hybrid.assert_called_once()

    # Assert that the generator model was called
    mock_generator_model.assert_called_once()
    
    # Check the returned answer
    assert result_answer == "[Placeholder: Model is not trained. No answer generated.]"