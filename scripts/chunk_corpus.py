"""
Script to chunk the cleaned corpus into manageable pieces for retrieval.
"""
import os
import json
import re

import os
import json
import re

def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
    """
    Splits a text into overlapping chunks of a specified size.
    A simple sliding window approach.
    """
    if not text:
        return []
    
    tokens = text.split() # Simple whitespace tokenization
    chunks = []
    
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunks.append(" ".join(chunk_tokens))
        
        if end >= len(tokens):
            break
        start += chunk_size - overlap
        
    return chunks

def main():
    """
    Main function to process cleaned files into individual chunked JSON files.
    """
    cleaned_dir = "data/cleaned"
    output_dir = "data/chunked_corpus"
    os.makedirs(output_dir, exist_ok=True)
    
    files_to_process = sorted([f for f in os.listdir(cleaned_dir) if f.endswith(".txt")])

    print(f"Starting chunking process. Output directory: {output_dir}")
    
    for filename in files_to_process:
        input_path = os.path.join(cleaned_dir, filename)
        output_path = os.path.join(output_dir, f"{filename}.json")

        if os.path.exists(output_path):
            print(f"Skipping already chunked file: {filename}")
            continue

        print(f"Chunking {filename}...")
        file_chunks = []
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        # Split text by paragraphs first for more logical chunks
        paragraphs = re.split(r'\n\s*\n', text)
        
        for i, para in enumerate(paragraphs):
            if not para.strip():
                continue
            
            # Further split large paragraphs if necessary
            if len(para.split()) > 512:
                sub_chunks = chunk_text(para)
                for j, sub_chunk in enumerate(sub_chunks):
                    chunk_data = {
                        "id": f"{filename}-{i}-{j}",
                        "source": filename,
                        "text": sub_chunk
                    }
                    file_chunks.append(chunk_data)
            else:
                chunk_data = {
                    "id": f"{filename}-{i}",
                    "source": filename,
                    "text": para.strip()
                }
                file_chunks.append(chunk_data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(file_chunks, f, indent=2, ensure_ascii=False)
        print(f"  -> Finished chunking {filename}, created {len(file_chunks)} chunks.")

    print(f"\nChunking process complete.")
if __name__ == '__main__':
    main()
