#!/usr/bin/env python3
"""Simple test of M1 query expansion."""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Set up logging to see what's happening
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

from klareco.rag.ast_aware_retriever import ASTAwareRetriever
from klareco.models.m1_inference import M1Inference
from klareco.parser import parse

# Initialize M1
print("Loading M1...")
m1 = M1Inference(
    model_path=Path('models/m1_compositional/best_model.pt'),
    comp_model_path=Path('models/root_embeddings_tier0/best_model.pt'),
    device='cpu'
)
print("M1 loaded\n")

# Initialize retriever with M1
print("Loading retriever...")
retriever = ASTAwareRetriever(
    index_path=Path('data/indexes/kuzu_index'),
    m1_model=m1
)
print("Retriever loaded\n")

# Test query expansion directly
query = "Kiu fondis Esperanton?"
print(f"Testing query expansion for: {query}\n")

query_ast = parse(query)

# Call expand_query_with_m1 directly to see what it produces
expansions = retriever.expand_query_with_m1(
    query_ast,
    min_plausibility=0.5,
    max_synonyms=10
)

print(f"\nExpansion results:")
print(f"  Total expansions: {len(expansions)}")
for exp in expansions:
    print(f"  - Verb: {exp['verb_root']}, M1 score: {exp['m1_score']:.3f}, Original: {exp['is_original']}")

print("\n" + "="*70)
print("Now testing full search with M1 expansion...")
print("="*70 + "\n")

# Now test full search
results = retriever.search(
    query,
    top_k=5,
    use_m1_expansion=True,
    m1_min_plausibility=0.5
)

print(f"\nSearch results: {len(results)}")
for i, (score, doc, stats) in enumerate(results[:3], 1):
    text = doc.get('text', '')[:100] + '...'
    print(f"{i}. Score: {score:.3f}")
    print(f"   {text}\n")
