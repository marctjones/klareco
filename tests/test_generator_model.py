import pytest
import torch
from torch_geometric.data import Data
from klareco.models.generator import Graph2SeqGenerator

@pytest.fixture
def sample_graph_data():
    """Provides a dummy PyG Data object for testing."""
    # Represents a graph with 5 nodes, each having 128 features (embed_dim)
    # and 4 edges.
    num_nodes = 5
    embed_dim = 128
    
    x = torch.randint(0, 100, (num_nodes,)) # Node features as token IDs
    edge_index = torch.tensor([[0, 1, 2, 3],
                               [1, 2, 3, 4]], dtype=torch.long)
    
    return Data(x=x, edge_index=edge_index)

@pytest.fixture
def generator_model_params():
    """Provides common parameters for initializing the generator model."""
    return {
        "vocab_size": 1000,
        "embed_dim": 128,
        "hidden_dim": 256,
        "gnn_out_dim": 256
    }

def test_generator_model_initialization(generator_model_params):
    """Test that the Graph2SeqGenerator model initializes without errors."""
    model = Graph2SeqGenerator(**generator_model_params)
    assert isinstance(model, Graph2SeqGenerator)
    assert model.vocab_size == generator_model_params["vocab_size"]
    assert model.hidden_dim == generator_model_params["hidden_dim"]
    assert isinstance(model.embedding, torch.nn.Embedding)
    assert isinstance(model.encoder, torch.nn.Module) # GraphEncoder is a Module
    assert isinstance(model.decoder_gru, torch.nn.GRU)
    assert isinstance(model.out, torch.nn.Linear)

def test_generator_model_forward_pass_inference(generator_model_params, sample_graph_data):
    """Test the forward pass for inference mode (no teacher forcing)."""
    model = Graph2SeqGenerator(**generator_model_params)
    model.eval() # Set to evaluation mode
    
    # Clone to prevent in-place modification if any
    graph_data_clone = sample_graph_data.clone()

    with torch.no_grad():
        output = model(graph_data_clone)
    
    # Expected output shape: [batch_size (1), max_len, vocab_size]
    # max_len is defined as 100 in generator.py for inference, but could be shorter
    # if <EOS> token is predicted early. So check vocab_size for last dim.
    assert output.ndim == 3
    assert output.shape[0] == 1 # Batch size
    assert output.shape[2] == generator_model_params["vocab_size"]
    # The actual length (output.shape[1]) depends on <EOS> prediction, so just check it's > 0
    assert output.shape[1] > 0

def test_generator_model_forward_pass_teacher_forcing(generator_model_params, sample_graph_data):
    """Test the forward pass for training mode with teacher forcing."""
    model = Graph2SeqGenerator(**generator_model_params)
    model.train() # Set to training mode

    # Create a dummy target sequence for teacher forcing
    # Batch size 1, sequence length 10
    target_sequence = torch.randint(0, generator_model_params["vocab_size"], (1, 10))

    graph_data_clone = sample_graph_data.clone()

    output = model(graph_data_clone, target_sequence=target_sequence, teacher_forcing_ratio=1.0) # Full teacher forcing
    
    # Expected output shape: [batch_size (1), target_sequence_len, vocab_size]
    assert output.ndim == 3
    assert output.shape[0] == 1 # Batch size
    assert output.shape[1] == target_sequence.shape[1] # Should match target sequence length
    assert output.shape[2] == generator_model_params["vocab_size"]
