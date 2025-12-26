"""
Script to initialize and save random morpheme embeddings.

These embeddings will serve as a placeholder for the GNN Encoder's node features.
"""
import os
import json
import torch

def initialize_random_embeddings(vocab_path: str, output_path: str, embedding_dim: int = 128):
    """
    Initializes random embeddings for each morpheme in the vocabulary.
    """
    print(f"Loading vocabulary from {vocab_path}...")
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    all_morphemes = []
    for morpheme_type, counts in vocab.items():
        all_morphemes.extend(counts.keys())
    
    unique_morphemes = sorted(list(set(all_morphemes)))
    vocab_size = len(unique_morphemes)
    print(f"Found {vocab_size} unique morphemes.")

    # Create a mapping from morpheme to index
    morpheme_to_idx = {morpheme: i for i, morpheme in enumerate(unique_morphemes)}

    # Initialize random embeddings
    # Using torch.randn for random initialization
    embeddings = torch.randn(vocab_size, embedding_dim)

    # Save embeddings and morpheme_to_idx mapping
    torch.save({
        'embeddings': embeddings,
        'morpheme_to_idx': morpheme_to_idx,
        'idx_to_morpheme': unique_morphemes
    }, output_path)

    print(f"Random embeddings (size {vocab_size}x{embedding_dim}) and vocabulary mapping saved to {output_path}")

if __name__ == '__main__':
    vocab_path = "data/morpheme_vocab.json"
    embeddings_output_path = "models/morpheme_embeddings.pt"
    embedding_dimension = 128 # A common embedding dimension
    initialize_random_embeddings(vocab_path, embeddings_output_path, embedding_dimension)
