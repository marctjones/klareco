"""
Script to build a morpheme vocabulary from the chunked corpus.

This script parses each text chunk into an AST and extracts all unique
roots, prefixes, suffixes, and endings, along with their frequencies.
"""
import os
import json
from collections import defaultdict
import sys

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from klareco.parser import parse
from tqdm import tqdm

def build_morpheme_vocabulary(chunked_corpus_dir: str, output_path: str):
    """
    Builds a morpheme vocabulary from the chunked corpus.
    """
    morpheme_counts = defaultdict(lambda: defaultdict(int))
    total_chunks = 0

    json_files = [f for f in os.listdir(chunked_corpus_dir) if f.endswith(".json")]

    print(f"Building morpheme vocabulary from {len(json_files)} chunked files...")

    for filename in tqdm(json_files, desc="Processing chunked files", unit=" file"):
        filepath = os.path.join(chunked_corpus_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        total_chunks += len(chunks)

        for chunk_data in chunks:
            text = chunk_data['text']
            try:
                ast = parse(text)
                # Traverse AST to extract morphemes
                _extract_morphemes_from_ast(ast, morpheme_counts)
            except Exception as e:
                print(f"WARNING: Could not parse chunk from {filename}: {text[:50]}... Error: {e}")
                pass # Suppress verbose error logging for now

    print(f"\nFinished processing {total_chunks} chunks.")
    print(f"Saving morpheme vocabulary to {output_path}")

    # Convert defaultdict to regular dict for JSON serialization
    final_vocab = {m_type: dict(counts) for m_type, counts in morpheme_counts.items()}
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_vocab, f, indent=2, ensure_ascii=False)
    print("Vocabulary saved.")

def _extract_morphemes_from_ast(ast_node: dict, morpheme_counts: defaultdict):
    """
    Recursively extracts morphemes from an AST node and updates counts.
    """
    if isinstance(ast_node, dict):
        if ast_node.get('type') == 'word':
            if 'root' in ast_node:
                morpheme_counts['ROOT'][ast_node['root']] += 1
            if 'prefix' in ast_node:
                morpheme_counts['PREFIX'][ast_node['prefix']] += 1
            for suffix in ast_node.get('suffixes', []):
                morpheme_counts['SUFFIX'][suffix] += 1
            for ending in ast_node.get('endings', []):
                morpheme_counts['ENDING'][ending] += 1
        
        for value in ast_node.values():
            _extract_morphemes_from_ast(value, morpheme_counts)
    elif isinstance(ast_node, list):
        for item in ast_node:
            _extract_morphemes_from_ast(item, morpheme_counts)

if __name__ == '__main__':
    chunked_dir = "data/chunked_corpus"
    vocab_output_path = "data/morpheme_vocab.json"
    build_morpheme_vocabulary(chunked_dir, vocab_output_path)
