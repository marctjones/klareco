"""
Script to clean and prepare the raw text corpus.
"""
import os
import re
import bz2
import xml.etree.ElementTree as ET

import os
import re
import bz2
import xml.etree.ElementTree as ET

def clean_gutenberg_text(text: str) -> str:
    """
    Removes the header and footer from a Project Gutenberg text.
    """
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
    
    start_idx = text.find(start_marker)
    if start_idx == -1:
        # If start marker not found, try a simpler heuristic
        start_idx = text.find("Title:")
    
    end_idx = text.find(end_marker)
    
    if start_idx != -1:
        # Move past the marker line itself
        start_idx = text.find('\n', start_idx) + 1
    else:
        start_idx = 0 # Default to start of file if no header found

    if end_idx != -1:
        text = text[start_idx:end_idx]
    else:
        text = text[start_idx:]
        
    # Basic cleaning
    text = re.sub(r'\r\n', '\n', text) # Normalize line endings
    text = re.sub(r'\[Illustration[^\]]*\]', '', text) # Remove illustration tags
    return text.strip()

import sys
from tqdm import tqdm

def clean_gutenberg_text(text: str) -> str:
    """
    Removes the header and footer from a Project Gutenberg text.
    """
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
    
    start_idx = text.find(start_marker)
    if start_idx == -1:
        start_idx = text.find("Title:")
    
    end_idx = text.find(end_marker)
    
    if start_idx != -1:
        start_idx = text.find('\n', start_idx) + 1
    else:
        start_idx = 0

    if end_idx != -1:
        text = text[start_idx:end_idx]
    else:
        text = text[start_idx:]
        
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\[Illustration[^\]]*\]', '', text)
    return text.strip()

def process_wikipedia_dump(input_path: str, output_path: str):
    """
    Decompresses and parses a Wikipedia XML dump to extract clean article text,
    with robust progress reporting.
    """
    print(f"Processing Wikipedia dump: {input_path}")
    
    # A more robust way to find the text tag, ignoring complex namespaces
    tag_text = "text"
    
    with bz2.open(input_path, 'rt', encoding='utf-8') as bz2f, \
         open(output_path, 'w', encoding='utf-8') as outf:

            context = ET.iterparse(bz2f, events=('end',))
            # Get the root element to clear it periodically
            event, root = next(context)

            # Use tqdm for progress, configured for logging
            pbar = tqdm(
                iterable=context,
                desc="Cleaning Wikipedia",
                unit=" pages",
                mininterval=10.0, # Log every 10 seconds
                file=sys.stderr # Print to stderr to not mix with stdout
            )

            for event, elem in pbar:
                # A more reliable way to get the tag name, ignoring the namespace
                tag = elem.tag.rsplit('}', 1)[-1]

                if tag == tag_text:
                    text = elem.text
                    if text and not text.lstrip().startswith("#REDIRECT"):
                        # Basic cleaning of wikitext markup
                        text = re.sub(r"'''?''", "", text)
                        text = re.sub(r"\[\[(?:[^|\]]+\|)?([^\]]+)\]\]", r"\1", text)
                        text = re.sub(r"\{\{[^}]+\}\}", "", text, flags=re.DOTALL)
                        text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL)
                        outf.write(text)
                        outf.write("\n\n")
                
                # iterparse can build up a lot of memory. Clear elements after processing.
                elem.clear()
                # The root element will accumulate children, so clear it periodically
                # This is a common strategy for handling large XML files with ElementTree
                if tag == "page":
                    root.clear()
    
    # Final check to ensure the file is not empty
    if os.path.getsize(output_path) > 0:
        print(f"\nFinished processing Wikipedia dump. Cleaned text saved to: {output_path}")
    else:
        print(f"\nWARNING: Wikipedia processing resulted in an empty file: {output_path}")



import argparse

def main():
    """
    Main function to run the cleaning process.
    """
    corpus_dir = "data/corpora"
    output_dir = "data/cleaned"
    os.makedirs(output_dir, exist_ok=True)

    files_to_process = os.listdir(corpus_dir)
    
    for filename in files_to_process:
        input_path = os.path.join(corpus_dir, filename)
        
        try:
            if filename.endswith(".txt"):
                output_path = os.path.join(output_dir, f"cleaned_{filename}")
                if os.path.exists(output_path):
                    print(f"Skipping already cleaned file: {filename}")
                    continue
                
                print(f"Cleaning {filename}...")
                with open(input_path, 'r', encoding='utf-8') as f:
                    raw_text = f.read()
                cleaned_text = clean_gutenberg_text(raw_text)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(cleaned_text)
                print(f"Saved cleaned text to {output_path}")

            elif filename.endswith(".xml.bz2"):
                output_path = os.path.join(output_dir, "cleaned_wikipedia.txt")
                if os.path.exists(output_path):
                    print(f"Skipping already cleaned file: {filename}")
                    continue
                
                process_wikipedia_dump(input_path, output_path)
        
        except Exception as e:
            print(f"!!! ERROR processing {filename}: {e}")
            print(f"Skipping file.")
            continue

if __name__ == '__main__':
    main()
