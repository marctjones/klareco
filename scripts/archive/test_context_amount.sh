#!/bin/bash
# Test how answer quality changes with different amounts of context

cd /home/marc/klareco

echo "======================================================================="
echo "TESTING: How does context amount affect answer quality?"
echo "======================================================================="
echo ""
echo "Model was trained with 3 context sentences"
echo "Testing: 1, 3, 5, 10 sentences"
echo ""

python3 << 'PYTHON_END'
import sys
from pathlib import Path
sys.path.insert(0, '/home/marc/klareco')

from scripts.test_end_to_end_qa import EndToEndQASystem

question = "Kiu estas Gandalf?"
context_amounts = [1, 3, 5, 10, 20]

print("Initializing QA system...")
qa = EndToEndQASystem(
    qa_model_path=Path('models/qa_decoder/best_model.pt'),
    vocab_path=Path('models/qa_decoder/vocabulary.json'),
    gnn_path=Path('models/tree_lstm/checkpoint_epoch_20.pt'),
    index_path=Path('data/corpus_index'),
    device='cpu'
)
print("✓ Ready!\n")

print(f"Question: {question}\n")
print('='*70)

for k in context_amounts:
    print(f"\n{'='*70}")
    print(f"TESTING WITH {k} CONTEXT SENTENCES")
    print('='*70)
    
    result = qa.answer_question(question, top_k=k, max_length=30)
    
    if 'error' in result:
        print(f"✗ Error: {result['error']}")
        continue
    
    # Show context sources
    print(f"\nContext sources:")
    for i, ctx in enumerate(result.get('context', [])[:5], 1):
        print(f"  [{i}] {ctx[:80]}...")
    if len(result.get('context', [])) > 5:
        print(f"  ... and {len(result['context']) - 5} more")
    
    # Show generated answer
    tokens = result.get('generated_tokens', [])
    print(f"\nGenerated ({len(tokens)} tokens):")
    print(f"  {' '.join(tokens)}")

print(f"\n{'='*70}")
print("ANALYSIS:")
print('='*70)
print("""
Expected behavior:
- k=3: Best quality (matches training distribution)
- k=1: Worse (not enough context)
- k>3: Possibly worse (out of distribution, more noise)

Actual results: See above

Conclusion:
- If k=3 gives best results → use 3 (matches training)
- If k>3 helps → retrain model with more context
- Quality depends on BOTH retrieval relevance AND decoder capacity
""")
PYTHON_END
