#-*- coding: utf-8 -*-
"""
This file defines the Graph2Seq generative model for Klareco.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool

class GraphEncoder(nn.Module):
    """
    Encodes the combined question and context graph into contextualized node embeddings.
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=4, concat=True)
        self.conv2 = GATv2Conv(hidden_channels * 4, out_channels, heads=1, concat=False)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class Attention(nn.Module):
    """
    Simple dot-product attention mechanism.
    """
    def __init__(self):
        super().__init__()

    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: [1, batch_size, hidden_size]
        # encoder_outputs: [seq_len, batch_size, hidden_size]
        # We need to permute encoder_outputs to [batch_size, seq_len, hidden_size]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # And permute decoder_hidden to [batch_size, hidden_size, 1]
        decoder_hidden = decoder_hidden.permute(1, 2, 0)

        # attn_weights: [batch_size, seq_len, 1]
        attn_weights = torch.bmm(encoder_outputs, decoder_hidden).squeeze(2)
        return F.softmax(attn_weights, dim=1)


class Graph2SeqGenerator(nn.Module):
    """
    A Graph-to-Sequence model that generates an answer from a reasoning graph.
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim, gnn_out_dim):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # The GNN encoder for the reasoning graph
        self.encoder = GraphEncoder(in_channels=19, hidden_channels=hidden_dim, out_channels=gnn_out_dim)

        # The GRU decoder
        self.decoder_gru = nn.GRU(gnn_out_dim + embed_dim, hidden_dim)
        
        # The attention mechanism
        self.attention = Attention()

        # Final output layer
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, graph_data, target_sequence=None, teacher_forcing_ratio=0.5):
        """
        Forward pass for the generator.

        Args:
            graph_data: A PyG Batch object representing a batch of reasoning graphs.
            target_sequence (optional): The ground truth output sequence for teacher forcing.
            teacher_forcing_ratio (optional): The probability to use teacher forcing.
        
        Returns:
            A tensor of output vocabulary logits.
        """
        batch_size = target_sequence.shape[0] if target_sequence is not None else graph_data.num_graphs
        
        # Encoder pass
        # The node features 'x' are already computed features.
        node_features = graph_data.x
        encoder_outputs = self.encoder(node_features, graph_data.edge_index)
        
        # Prepare for decoding
        # The encoder outputs become the 'sequence' the decoder attends to.
        # [num_nodes, gnn_out_dim] -> [num_nodes, batch_size, gnn_out_dim]
        # This is tricky with PyG batches. We will handle context inside the decode_step.
        
        # Decoder initialization
        decoder_hidden = torch.zeros(1, batch_size, self.hidden_dim).to(graph_data.x.device)
        
        # <SOS> token (assuming vocab id 0 is SOS)
        decoder_input = torch.full((1, batch_size), 0, dtype=torch.long, device=graph_data.x.device)

        outputs = []
        
        if target_sequence is not None:
            # Training with teacher forcing
            max_len = target_sequence.shape[1]
            for t in range(max_len):
                output, decoder_hidden = self.decode_step(decoder_input, decoder_hidden, encoder_outputs, graph_data.batch)
                outputs.append(output)
                
                # Decide if we are using teacher forcing
                teacher_force = torch.rand(1).item() < teacher_forcing_ratio
                top1 = output.argmax(1)
                decoder_input = target_sequence[:, t] if teacher_force else top1
                decoder_input = decoder_input.unsqueeze(0)
        else:
            # Inference
            max_len = 100 # Max generation length
            for t in range(max_len):
                output, decoder_hidden = self.decode_step(decoder_input, decoder_hidden, encoder_outputs, graph_data.batch)
                outputs.append(output)
                
                top1 = output.argmax(1)
                decoder_input = top1.unsqueeze(0)
                # Stop if we hit <EOS> token (placeholder id 1)
                if top1.item() == 1: # This will only work for batch_size=1 inference
                    break
        
        # Stack outputs: [max_len, batch_size, vocab_size]
        outputs = torch.stack(outputs)
        # Reshape to [batch_size, max_len, vocab_size]
        return outputs.permute(1, 0, 2)


    def decode_step(self, decoder_input, decoder_hidden, encoder_outputs, batch):
        """
        Performs a single decoding step.
        """
        embedded = self.embedding(decoder_input.squeeze(0))
        embedded = embedded.unsqueeze(0) # [1, batch_size, embed_dim]

        # Attention
        # This is a simplified attention. A more robust implementation would be needed.
        # For now, we'll just use the last hidden state to inform the GRU,
        # and a full attention mechanism can be added later.
        # attn_weights = self.attention(decoder_hidden, encoder_outputs)
        
        # Use global mean pooling to get a context vector for each graph in the batch.
        context_vec = global_mean_pool(encoder_outputs, batch)
        context_vec = context_vec.unsqueeze(0) # [1, batch_size, gnn_out_dim]

        rnn_input = torch.cat((embedded, context_vec), dim=2)
        
        output, decoder_hidden = self.decoder_gru(rnn_input, decoder_hidden)
        
        output = self.out(output.squeeze(0))
        return output, decoder_hidden