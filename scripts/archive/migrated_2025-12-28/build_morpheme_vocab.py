"""
Script to build a morpheme vocabulary from the cleaned corpus.

This script iterates through all cleaned text files, splits them into unique
words, and then parses each word to extract its constituent morphemes (roots,
prefixes, suffixes, and endings). It aggregates the counts of these morphemes
and saves them to a JSON file.

This approach avoids the chicken-and-egg problem of needing a full sentence
parser to build the vocabulary that the parser itself needs.
"""
import os
import json
import re
import sys
import logging
from collections import defaultdict
from typing import DefaultDict, Dict, Set

# Add project root to the Python path to allow importing from 'klareco'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from klareco.parser import parse_word
from klareco.logging_config import setup_logging
from tqdm import tqdm

# Setup logging for the script
setup_logging()

def build_morpheme_vocabulary(cleaned_corpus_dir: str, output_path: str) -> None:
    """
    Builds a morpheme vocabulary by parsing individual words from the cleaned corpus.

    Args:
        cleaned_corpus_dir: The path to the directory containing cleaned .txt files.
        output_path: The path where the resulting morpheme_vocab.json will be saved.
    """
    morpheme_counts: DefaultDict[str, DefaultDict[str, int]] = defaultdict(lambda: defaultdict(int))
    
    try:
        source_files = [f for f in os.listdir(cleaned_corpus_dir) if f.endswith(".txt")]
        if not source_files:
            logging.error(f"No .txt files found in the specified directory: {cleaned_corpus_dir}")
            return
    except FileNotFoundError:
        logging.error(f"Cleaned corpus directory not found: {cleaned_corpus_dir}", exc_info=True)
        return

    logging.info(f"Starting morpheme vocabulary build from {len(source_files)} cleaned files...")

    all_unique_words: Set[str] = set()

    # First, gather all unique words from all files to avoid redundant processing
    for filename in tqdm(source_files, desc="Gathering unique words", unit="file"):
        filepath = os.path.join(cleaned_corpus_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read().lower()
                # Regex to find word-like sequences, including those with hyphens
                words = set(re.findall(r'[\w-]+', text))
                all_unique_words.update(words)
        except Exception:
            logging.error(f"Failed to read or process file: {filename}", exc_info=True)
            continue
    
    logging.info(f"Found {len(all_unique_words)} unique words to process.")

    # Now, parse each unique word
    for word in tqdm(all_unique_words, desc="Parsing words", unit="word"):
        try:
            # The 'word' rule in our parser is designed to handle single words.
            ast = parse_word(word)
            _extract_morphemes_from_ast(ast, morpheme_counts)
        except Exception:
            # This is expected for non-Esperanto words, numbers, or unparseable tokens.
            # Logging as DEBUG because this is noisy but useful for deep debugging.
            logging.debug(f"Could not parse token: '{word}'")
            pass

    logging.info(f"Finished processing. Found {len(morpheme_counts.get('ROOT', {}))} unique roots.")
    
    # Convert defaultdict to a regular dict for clean JSON serialization
    final_vocab: Dict[str, Dict[str, int]] = {
        m_type: dict(counts) for m_type, counts in morpheme_counts.items()
    }

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_vocab, f, indent=2, ensure_ascii=False)
        logging.info(f"Morpheme vocabulary successfully saved to {output_path}")
    except Exception:
        logging.error(f"Failed to write vocabulary to {output_path}", exc_info=True)

def _extract_morphemes_from_ast(ast_node: Dict, morpheme_counts: DefaultDict) -> None:
    """
    Recursively traverses a word's AST to find and count its morphemes.
    """
    # Since we are parsing with start='word', the top-level AST is the word structure itself
    if isinstance(ast_node, dict) and ast_node.get('type') == 'word':
        if 'root' in ast_node:
            morpheme_counts['ROOT'][ast_node['root']] += 1
        if 'prefix' in ast_node:
            morpheme_counts['PREFIX'][ast_node['prefix']] += 1
        for suffix in ast_node.get('suffixes', []):
            morpheme_counts['SUFFIX'][suffix] += 1
        for ending in ast_node.get('endings', []):
            morpheme_counts['ENDING'][ending] += 1

if __name__ == '__main__':
    # Define the source and destination paths
    cleaned_dir = "data/cleaned"
    vocab_output_path = "data/morpheme_vocab.json"
    
    # Execute the main function
    build_morpheme_vocabulary(cleaned_dir, vocab_output_path)