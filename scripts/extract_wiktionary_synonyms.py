#!/usr/bin/env python3
"""
Extract Esperanto Synonyms from Wiktionary Data

Parses the kaikki.org Wiktionary JSONL dump to extract synonym relationships.

Usage:
    python scripts/extract_wiktionary_synonyms.py
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))


def extract_root(word):
    """Extract root from Esperanto word."""
    # Remove common endings
    for ending in ['oj', 'ojn', 'on', 'o', 'aj', 'ajn', 'an', 'a', 'e', 'i', 'is', 'as', 'os', 'us']:
        if word.endswith(ending) and len(word) > len(ending) + 2:
            return word[:-len(ending)]
    return word


def extract_synonyms_from_entry(entry):
    """Extract synonyms from a Wiktionary entry."""
    synonyms = set()
    
    # Check for synonym information in various fields
    if 'senses' in entry:
        for sense in entry['senses']:
            # Check for explicit synonyms field
            if 'synonyms' in sense:
                for syn in sense['synonyms']:
                    if isinstance(syn, dict) and 'word' in syn:
                        synonyms.add(syn['word'])
                    elif isinstance(syn, str):
                        synonyms.add(syn)
            
            # Check for links that might be synonyms
            if 'links' in sense:
                for link in sense['links']:
                    if isinstance(link, list) and len(link) >= 2:
                        # Format: [[word, gloss]]
                        synonyms.add(link[0])
    
    return synonyms


def main():
    # Try to find the Wiktionary file
    wikt_paths = [
        'data/raw/eo/dictionaries/wiktionary/esperanto-wiktionary.jsonl',
        './data/raw/eo/dictionaries/wiktionary/esperanto-wiktionary.jsonl',
    ]
    
    wikt_file = None
    for path in wikt_paths:
        if Path(path).exists():
            wikt_file = Path(path)
            break
    
    if not wikt_file:
        print("ERROR: Wiktionary file not found")
        print("Searched:")
        for path in wikt_paths:
            print(f"  {path}")
        return
    
    print(f"Loading Wiktionary data from: {wikt_file}")
    print(f"File size: {wikt_file.stat().st_size / 1024 / 1024:.1f} MB")
    print()
    
    # Parse JSONL
    synonym_map = defaultdict(set)  # root -> synonyms
    entries_processed = 0
    
    with open(wikt_file, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            
            # Only process Esperanto entries
            if entry.get('lang_code') != 'eo':
                continue
            
            word = entry.get('word', '')
            if not word:
                continue
            
            # Extract synonyms
            synonyms = extract_synonyms_from_entry(entry)
            
            if synonyms:
                root = extract_root(word)
                for syn in synonyms:
                    syn_root = extract_root(syn)
                    if syn_root != root:
                        synonym_map[root].add(syn_root)
            
            entries_processed += 1
            
            if entries_processed % 1000 == 0:
                print(f"  Processed {entries_processed:,} entries...", end='\r')
    
    print(f"  Processed {entries_processed:,} entries total")
    print()
    
    # Convert to JSON-serializable format
    synonym_relations = []
    for root, synonyms in synonym_map.items():
        for syn in synonyms:
            synonym_relations.append({
                'source': root,
                'target': syn,
                'relation': 'WIKTIONARY_SINONIMO',
                'confidence': 1.0
            })
    
    # Save results
    output_file = Path('data/raw/eo/dictionaries/wiktionary_semantic_relations.json')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(synonym_relations, f, ensure_ascii=False, indent=2)
    
    print("="*70)
    print("WIKTIONARY SYNONYM EXTRACTION RESULTS")
    print("="*70)
    print()
    print(f"Entries processed: {entries_processed:,}")
    print(f"Roots with synonyms: {len(synonym_map):,}")
    print(f"Total synonym pairs: {len(synonym_relations):,}")
    print()
    print(f"Output: {output_file}")
    print()
    
    # Show sample
    print("Sample synonyms found:")
    for root, synonyms in list(synonym_map.items())[:10]:
        print(f"  {root}: {', '.join(sorted(synonyms))}")
    
    print()
    print("="*70)
    print("NOTE: This is from partial download (4MB of 129MB)")
    print("For complete coverage, need to download full Wiktionary dump")
    print("="*70)


if __name__ == '__main__':
    main()
