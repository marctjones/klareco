"""
The Grammarian (Parser) using the Lark library.

This version performs morphological decomposition, breaking words down into
their constituent morphemes (roots, affixes, endings).
"""
from lark import Lark, Transformer

import json
import os

# --- Dynamic Grammar Construction ---

def find_project_root(start_path):
    """Finds the project root by looking for a .git directory."""
    path = os.path.abspath(start_path)
    while True:
        if os.path.isdir(os.path.join(path, '.git')):
            return path
        parent = os.path.dirname(path)
        if parent == path: # Reached the filesystem root
            return None
        path = parent

try:
    project_root = find_project_root(os.getcwd()) # Start search from current working directory
    if project_root is None:
        raise FileNotFoundError("Could not find project root (.git directory).")
        
    vocab_path = os.path.join(project_root, 'data', 'morpheme_vocab.json')

    with open(vocab_path, 'r', encoding='utf-8') as f:
        morpheme_vocab = json.load(f)
    
    # Get all known roots from the vocabulary
    known_roots = morpheme_vocab.get('ROOT', {}).keys()
    
    # Ensure our simple test roots are also included
    test_roots = {"san", "hund", "kat", "program", "vid", "am", "bon", "mi"}
    all_roots = sorted(list(set(known_roots) | test_roots))
    
    # Create the Lark-compatible string for the ROOT terminal
    root_terminal_str = " | ".join(f'"{root}"' for root in all_roots)

except FileNotFoundError:
    print("WARNING: morpheme_vocab.json not found. Using a minimal root vocabulary.")
    root_terminal_str = '"san" | "hund" | "kat" | "program" | "vid" | "am" | "bon" | "mi"'


esperanto_grammar = f"""
    ?sentence: noun_phrase verb noun_phrase "."?

    noun_phrase: article? word
    verb: word
    word: PREFIX? ROOT SUFFIX* ENDING*

    article: "la"i

    // Morpheme categories defined as Terminals (uppercase)
    PREFIX: "mal" | "re" | "ge"
    SUFFIX: "ul" | "ej" | "in" | "et" | "ad"
    ENDING: "o" | "a" | "e" | "i" | "as" | "is" | "os" | "j" | "n"

    ROOT: {root_terminal_str}

    %import common.WS
    %ignore WS
"""

class MorphemeTransformer(Transformer):
    """
    Transforms the Lark parse tree into a morpheme-based AST.
    """
    def sentence(self, items):
        return {"type": "sentence", "subject": items[0], "verb": items[1], "object": items[2]}

    def noun_phrase(self, items):
        # A noun_phrase can be just a word, or an article + word.
        # The transformed word dictionary will always be the last item in the list.
        return items[-1]

    def verb(self, items):
        # A verb is always just a word.
        return items[0]

    def word(self, items):
        word_structure = {"type": "word", "suffixes": [], "endings": []}
        for item in items:
            if "root" in item:
                word_structure["root"] = item["root"]
            elif "prefix" in item:
                word_structure["prefix"] = item["prefix"]
            elif "suffixes" in item:
                word_structure["suffixes"].extend(item["suffixes"])
            elif "endings" in item:
                word_structure["endings"].extend(item["endings"])
        return word_structure

    def ROOT(self, token):
        return {"root": token.value}
    
    def PREFIX(self, token):
        return {"prefix": token.value}
    
    def SUFFIX(self, token):
        return {"suffixes": [token.value]}

    def ENDING(self, token):
        return {"endings": [token.value]}


# Create the parser instance
esperanto_parser = Lark(esperanto_grammar, start='sentence', parser='lalr', transformer=MorphemeTransformer())

def parse(text: str):
    """
    Parses an Esperanto sentence and returns a structured, morpheme-based AST.
    """
    return esperanto_parser.parse(text)

if __name__ == '__main__':
    # Example Usage
    import json
    
    text = "hundo"
    ast = parse(text)
    print(f"Text: '{text}'")
    print(f"AST: {json.dumps(ast, indent=2, ensure_ascii=False)}")
    def word(self, items):
        word_structure = {"type": "word", "suffixes": [], "endings": []}
        for item in items:
            if item.type == 'ROOT':
                word_structure["root"] = item.value
            elif item.type == 'PREFIX':
                word_structure["prefix"] = item.value
            elif item.type == 'SUFFIX':
                word_structure["suffixes"].append(item.value)
            elif item.type == 'ENDING':
                word_structure["endings"].append(item.value)
        return word_structure

    def ROOT(self, token):
        return {"root": token.value}
    
    def PREFIX(self, token):
        return {"prefix": token.value}
    
    def SUFFIX(self, token):
        return {"suffixes": [token.value]}

    def ENDING(self, token):
        return {"endings": [token.value]}

    def word(self, items):
        # This is the core of the new parser. It assembles the morphemes.
        word_structure = {"type": "word", "suffixes": [], "endings": []}
        for item in items:
            if "suffixes" in item:
                word_structure["suffixes"].extend(item["suffixes"])
            elif "endings" in item:
                word_structure["endings"].extend(item["endings"])
            else:
                word_structure.update(item)
        return word_structure

    def ROOT(self, token):
        return {"root": token.value}
    
    def PREFIX(self, token):
        return {"prefix": token.value}

    def SUFFIX(self, token):
        return {"suffixes": [token.value]}

    def ENDING(self, token):
        return {"endings": [token.value]}
    
    # We don't need to transform the article, it's handled in noun_phrase
    def article(self, items):
        return items[0]


# Create the parser instance
esperanto_parser = Lark(esperanto_grammar, start='sentence', parser='lalr', transformer=MorphemeTransformer())

def parse(text: str):
    """
    Parses an Esperanto sentence and returns a structured, morpheme-based AST.
    """
    return esperanto_parser.parse(text)

if __name__ == '__main__':
    # Example Usage
    import json
    
    text = "la hundo amas la katon."
    ast = parse(text)
    print(f"Text: '{text}'")
    print(f"AST: {json.dumps(ast, indent=2, ensure_ascii=False)}")

    text_2 = "malsanulo vidas bonajn hundojn."
    # We need to add 'malsanulo' parts to the grammar for this to work
    # Let's try a simpler one for now.
    text_2 = "mi vidas bonan hundon."
    # This will fail until we add more roots. Let's stick to the first example.