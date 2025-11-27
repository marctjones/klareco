#!/usr/bin/env python3
"""
Builds a vocabulary from the text corpus.

The vocabulary is a JSON file containing the word-to-index and index-to-word mappings,
as well as special tokens like <PAD>, <SOS>, <EOS>, and <UNK>.
"""

import argparse
import json
from pathlib import Path
from collections import Counter
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define special tokens
PAD_TOKEN = '<PAD>'
SOS_TOKEN = '<SOS>'
EOS_TOKEN = '<EOS>'
UNK_TOKEN = '<UNK>'
SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

def build_vocabulary(corpus_dir: Path, vocab_path: Path, max_size: int):
    """
    Reads all text files in a directory, tokenizes them, and builds a vocabulary.

    Args:
        corpus_dir: Directory containing the .txt corpus files.
        vocab_path: Path to save the JSON vocabulary file.
        max_size: The maximum number of words to include in the vocabulary
                  (most frequent words will be kept).
    """
    logging.info(f"Building vocabulary from corpus: {corpus_dir}")
    word_counts = Counter()

    for text_file in corpus_dir.glob("*.txt"):
        with open(text_file, 'r', encoding='utf-8') as f:
            for line in f:
                # Simple space-based tokenization
                tokens = line.strip().split(' ')
                word_counts.update(tokens)

    # Create vocabulary based on the most common words
    most_common_words = [word for word, count in word_counts.most_common(max_size - len(SPECIAL_TOKENS))]
    
    # Create word-to-index and index-to-word mappings
    word2index = {}
    index2word = {}

    # Add special tokens first
    for token in SPECIAL_TOKENS:
        if token not in word2index:
            idx = len(word2index)
            word2index[token] = idx
            index2word[idx] = token
            
    # Add words from the corpus
    for word in most_common_words:
        if word not in word2index:
            idx = len(word2index)
            word2index[word] = idx
            index2word[idx] = word
            
    vocab = {
        'word2index': word2index,
        'index2word': index2word,
        'n_words': len(word2index)
    }

    # Ensure the output directory exists
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
        
    logging.info(f"Vocabulary with {len(word2index)} words built and saved to {vocab_path}")
    logging.info(f"  First 10 words: {list(word2index.keys())[:10]}")


def main():
    parser = argparse.ArgumentParser(description="Build a vocabulary from a text corpus.")
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=Path("data/clean_corpus"),
        help="Directory containing the cleaned corpus .txt files."
    )
    parser.add_argument(
        "--vocab-path",
        type=Path,
        default=Path("data/vocab.json"),
        help="Path to save the generated vocabulary JSON file."
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=20000,
        help="Maximum size of the vocabulary."
    )
    args = parser.parse_args()

    build_vocabulary(args.corpus_dir, args.vocab_path, args.max_size)


if __name__ == "__main__":
    main()
