#!/bin/bash
# Quick RAG Question Tester
# Run this from any terminal to ask questions and see retrieval + generation

cd /home/marc/klareco

echo "======================================================================="
echo "KLARECO RAG + QA DECODER TEST"
echo "======================================================================="
echo ""
echo "Testing retrieval and answer generation..."
echo ""

python3 << 'PYTHON_END'
import sys
from pathlib import Path
sys.path.insert(0, '/home/marc/klareco')

from scripts.test_end_to_end_qa import EndToEndQASystem

# Questions to test
questions = [
    "Kiu estas Frodo?",           # Who is Frodo?
    "Kiu estas Gandalf?",         # Who is Gandalf?
    "Kio estas hobito?",          # What is a hobbit?
]

print("Initializing QA system...")
qa = EndToEndQASystem(
    qa_model_path=Path('models/qa_decoder/best_model.pt'),
    vocab_path=Path('models/qa_decoder/vocabulary.json'),
    gnn_path=Path('models/tree_lstm/checkpoint_epoch_20.pt'),
    index_path=Path('data/corpus_index'),
    device='cpu'
)
print("✓ Ready!\n")

for i, question in enumerate(questions, 1):
    print(f"\n{'='*70}")
    print(f"QUESTION {i}/{len(questions)}: {question}")
    print('='*70)
    
    result = qa.answer_question(question, top_k=3, max_length=30)
    
    if 'error' in result:
        print(f"✗ Error: {result['error']}")
        continue
    
    # Show retrieved context
    print(f"\nRETRIEVED CONTEXT ({len(result.get('context', []))} sentences):")
    print('-'*70)
    for j, ctx in enumerate(result.get('context', [])[:3], 1):
        print(f"  [{j}] {ctx[:120]}...")
    
    # Show generated answer
    print(f"\nGENERATED ANSWER:")
    print('-'*70)
    tokens = result.get('generated_tokens', [])
    print(f"  {' '.join(tokens)}")
    
print(f"\n{'='*70}")
print("✓ TEST COMPLETE")
print('='*70)
PYTHON_END

echo ""
echo "To test different questions, edit: ask-quick-rag-questions.sh"
echo "Change the 'questions' list in the script."
