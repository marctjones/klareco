"""
Update parser to use enriched vocabulary data.

This script enhances the parser's parse_word() function to:
1. Include semantic category hints in ASTs
2. Mark common vs rare roots
3. Provide frequency information for debugging
"""

import json
from pathlib import Path


def load_enriched_vocabulary(enriched_path: Path) -> dict:
    """Load enriched vocabulary with categories and frequencies."""
    if not enriched_path.exists():
        print(f"Enriched vocabulary not found at {enriched_path}")
        print("Run: python scripts/categorize_vocabulary.py")
        return {}

    with open(enriched_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def enhance_ast_with_metadata(ast: dict, root: str, enriched_vocab: dict) -> dict:
    """
    Enhance AST with metadata from enriched vocabulary.

    Adds:
    - semantic_category: verb/noun/adjective/unknown
    - frequency: usage count
    - frequency_tier: common/moderate/rare/unused
    """
    if not enriched_vocab or root not in enriched_vocab:
        return ast

    root_data = enriched_vocab[root]

    # Add metadata
    ast['metadata'] = {
        'semantic_category': root_data.get('category', 'unknown'),
        'corpus_frequency': root_data.get('corpus_frequency', 0),
        'text_frequency': root_data.get('text_frequency', 0),
        'total_frequency': root_data.get('total_frequency', 0),
    }

    # Add frequency tier
    freq = root_data.get('total_frequency', 0)
    if freq >= 10:
        ast['metadata']['frequency_tier'] = 'common'
    elif freq >= 5:
        ast['metadata']['frequency_tier'] = 'moderate'
    elif freq >= 1:
        ast['metadata']['frequency_tier'] = 'rare'
    else:
        ast['metadata']['frequency_tier'] = 'unused'

    return ast


def demonstrate_enhanced_parsing():
    """Demonstrate parsing with enriched vocabulary."""
    project_root = Path(__file__).parent.parent
    enriched_path = project_root / 'data' / 'enriched_vocabulary.json'

    print("=== Enhanced Parser Demonstration ===\n")

    # Load enriched vocabulary
    print("Loading enriched vocabulary...")
    enriched = load_enriched_vocabulary(enriched_path)
    if not enriched:
        return

    print(f"Loaded {len(enriched)} enriched roots\n")

    # Import parser
    from klareco.parser import parse_word

    # Test words
    test_words = [
        "hundon",  # dog (accusative)
        "manĝas",  # eat (present)
        "bela",    # beautiful (adjective)
        "rapide",  # quickly (adverb)
        "programisto",  # programmer
    ]

    print("Parsing with metadata:\n")

    for word in test_words:
        print(f"Word: {word}")
        ast = parse_word(word)

        # Enhance with metadata
        root = ast.get('radiko')
        if root:
            ast = enhance_ast_with_metadata(ast, root, enriched)

            print(f"  Root: {root}")
            print(f"  Part of speech: {ast.get('vortspeco', 'unknown')}")

            if 'metadata' in ast:
                meta = ast['metadata']
                print(f"  Semantic category: {meta['semantic_category']}")
                print(f"  Frequency tier: {meta['frequency_tier']}")
                print(f"  Total frequency: {meta['total_frequency']}")

        print()

    print("=== Usage Instructions ===\n")
    print("To use enriched vocabulary in your code:")
    print()
    print("```python")
    print("from scripts.update_parser_with_enriched_vocab import \\")
    print("    load_enriched_vocabulary, enhance_ast_with_metadata")
    print("from klareco.parser import parse_word")
    print()
    print("# Load once at startup")
    print("enriched = load_enriched_vocabulary('data/enriched_vocabulary.json')")
    print()
    print("# Use when parsing")
    print("ast = parse_word('hundon')")
    print("ast = enhance_ast_with_metadata(ast, ast['radiko'], enriched)")
    print("```")
    print()


def generate_vocabulary_insights():
    """Generate insights from enriched vocabulary."""
    project_root = Path(__file__).parent.parent
    enriched_path = project_root / 'data' / 'enriched_vocabulary.json'

    enriched = load_enriched_vocabulary(enriched_path)
    if not enriched:
        return

    print("\n=== Vocabulary Insights ===\n")

    # Most common verbs
    verbs = [(r, d) for r, d in enriched.items() if d.get('category') == 'verb']
    verbs_by_freq = sorted(verbs, key=lambda x: x[1]['total_frequency'], reverse=True)

    print("Top 10 most common verbs:")
    for root, data in verbs_by_freq[:10]:
        freq = data['total_frequency']
        print(f"  {root:15} (frequency: {freq})")

    # Most common adjectives
    adjectives = [(r, d) for r, d in enriched.items() if d.get('category') == 'adjective']
    adj_by_freq = sorted(adjectives, key=lambda x: x[1]['total_frequency'], reverse=True)

    print("\nTop 10 most common adjectives:")
    for root, data in adj_by_freq[:10]:
        freq = data['total_frequency']
        print(f"  {root:15} (frequency: {freq})")

    # Most common nouns
    nouns = [(r, d) for r, d in enriched.items() if d.get('category') == 'noun']
    nouns_by_freq = sorted(nouns, key=lambda x: x[1]['total_frequency'], reverse=True)

    print("\nTop 10 most common nouns:")
    for root, data in nouns_by_freq[:10]:
        freq = data['total_frequency']
        print(f"  {root:15} (frequency: {freq})")

    # Roots that appear in corpus
    corpus_roots = [(r, d) for r, d in enriched.items() if d.get('corpus_frequency', 0) > 0]
    print(f"\n{len(corpus_roots)} roots appear in test corpus")

    # Category coverage
    print("\nSemantic category coverage:")
    from collections import Counter
    categories = Counter(d.get('category', 'unknown') for d in enriched.values())
    for cat, count in categories.most_common():
        pct = count / len(enriched) * 100
        print(f"  {cat:15} {count:5} ({pct:5.1f}%)")


def main():
    """Main entry point."""
    demonstrate_enhanced_parsing()
    generate_vocabulary_insights()


if __name__ == '__main__':
    main()
