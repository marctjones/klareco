"""Merged vocabulary combining all vocabulary sources.

This file combines roots from:
- Gutenberg English-Esperanto Dictionary (extracted_vocabulary.py)
- Literary corpus analysis
- Manual additions
- Other vocabulary extraction scripts

To populate this vocabulary, run the vocabulary extraction pipeline:
    python scripts/extract_dictionary_roots.py

For now, this is initialized with an empty set.
The parser will fall back to its hardcoded KNOWN_ROOTS vocabulary.
"""

# Combined roots from all sources
MERGED_ROOTS = set()
