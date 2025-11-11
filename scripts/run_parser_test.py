"""
Runs a partial pipeline test, focusing on the FrontDoor and Parser.
"""
import sys
import os
import json

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from klareco.front_door import FrontDoor
from klareco.parser import parse

def test_sentence(sentence: str):
    """
    Tests a single sentence through the initial pipeline stages.
    """
    print(f"--- Testing Sentence: '{sentence}' ---")
    front_door = FrontDoor()
    
    try:
        lang, processed_text = front_door.process(sentence)
        print(f"Language ID: {lang}")
        print(f"Processed Text: '{processed_text}'")
        
        ast = parse(processed_text)
        print("AST:")
        print(json.dumps(ast, indent=2, ensure_ascii=False))
        print("Result: SUCCESS\n")
    except Exception as e:
        print(f"Result: FAILED")
        print(f"Error: {e}\n")

if __name__ == '__main__':
    test_sentence("The dog loves the cat.")
    test_sentence("mi vidas la hundon.")

