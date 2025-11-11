"""
Script to verify the integrity of the chunked corpus files.
"""
import os
import json

def verify_chunked_corpus(chunked_dir: str) -> bool:
    """
    Verifies that all chunked JSON files are valid and non-empty.

    Args:
        chunked_dir: The directory containing the chunked JSON files.

    Returns:
        True if all files are valid, False otherwise.
    """
    all_valid = True
    print(f"Starting verification of chunked corpus in: {chunked_dir}")

    json_files = sorted([f for f in os.listdir(chunked_dir) if f.endswith(".json")])
    if not json_files:
        print("No JSON files found in the chunked corpus directory.")
        return False

    for filename in json_files:
        filepath = os.path.join(chunked_dir, filename)
        print(f"  Verifying {filename}...")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                print(f"    ERROR: {filename} does not contain a JSON list.")
                all_valid = False
            elif not data:
                print(f"    ERROR: {filename} contains an empty list of chunks.")
                all_valid = False
            else:
                # Optionally, check structure of first chunk
                if not all(isinstance(chunk, dict) and "id" in chunk and "source" in chunk and "text" in chunk for chunk in data):
                    print(f"    WARNING: {filename} chunks do not match expected structure.")

        except json.JSONDecodeError as e:
            print(f"    ERROR: {filename} is not a valid JSON file: {e}")
            all_valid = False
        except Exception as e:
            print(f"    ERROR: An unexpected error occurred with {filename}: {e}")
            all_valid = False

    if all_valid:
        print("\nAll chunked corpus files verified successfully.")
    else:
        print("\nVerification completed with errors.")
    
    return all_valid

if __name__ == '__main__':
    chunked_corpus_dir = "data/chunked_corpus"
    if verify_chunked_corpus(chunked_corpus_dir):
        print("Corpus is ready for use.")
    else:
        print("Corpus needs attention.")
