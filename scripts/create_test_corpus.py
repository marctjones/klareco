"""
Script to create a diverse test corpus by sampling sentences from the cleaned data.
"""
import os
import json
import random

def create_test_corpus(cleaned_dir: str, output_path: str, num_samples: int = 50):
    """
    Creates a test corpus by sampling sentences from the cleaned data.
    """
    print(f"Creating test corpus with {num_samples} samples...")
    
    source_files = [f for f in os.listdir(cleaned_dir) if f.endswith(".txt")]
    if not source_files:
        print(f"ERROR: No cleaned text files found in {cleaned_dir}")
        return

    samples_per_source = max(1, num_samples // len(source_files))
    test_corpus = []
    
    for filename in source_files:
        if len(test_corpus) >= num_samples:
            break
            
        filepath = os.path.join(cleaned_dir, filename)
        print(f"  Sampling from {filename}...")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
            
        # Simple sentence splitting by period. This is a heuristic.
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        if not sentences:
            print(f"    WARNING: No sentences found in {filename}")
            continue
            
        num_to_sample = min(samples_per_source, num_samples - len(test_corpus))
        
        # Ensure we don't try to sample more than available
        num_to_sample = min(num_to_sample, len(sentences))
        
        sampled_sentences = random.sample(sentences, num_to_sample)
        test_corpus.extend(sampled_sentences)

    print(f"\nTotal sentences sampled: {len(test_corpus)}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(test_corpus, f, indent=2, ensure_ascii=False)
        
    print(f"Test corpus saved to {output_path}")

if __name__ == '__main__':
    cleaned_data_dir = "data/cleaned"
    corpus_output_path = "data/test_corpus.json"
    create_test_corpus(cleaned_data_dir, corpus_output_path)
