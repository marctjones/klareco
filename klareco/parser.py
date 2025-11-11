"""A from-scratch, pure Python parser for Esperanto.

This parser is built on the 16 rules of Esperanto and does not use any
external parsing libraries like Lark. It performs morphological and syntactic
analysis to produce a detailed, Esperanto-native Abstract Syntax Tree (AST)."""
import re

# -----------------------------------------------------------------------------
# --- Hardcoded Vocabulary (Lexicon)
# -----------------------------------------------------------------------------
# In a real system, this would be much larger, but for our static parser,
# we define the known morphemes directly here.

KNOWN_PREFIXES = {"mal", "re", "ge"}
KNOWN_SUFFIXES = {"ul", "ej", "in", "et", "ad", "ig"}

# The order of endings matters. Longer ones must be checked first.
KNOWN_ENDINGS = {
    # Tense/Mood
    "as": {"vortspeco": "verbo", "tempo": "prezenco"},
    "is": {"vortspeco": "verbo", "tempo": "pasinteco"},
    "os": {"vortspeco": "verbo", "tempo": "futuro"},
    "us": {"vortspeco": "verbo", "modo": "kondiĉa"},
    "u": {"vortspeco": "verbo", "modo": "vola"},
    "i": {"vortspeco": "verbo", "modo": "infinitivo"},
    # Part of Speech
    "o": {"vortspeco": "substantivo"},
    "a": {"vortspeco": "adjektivo"},
    "e": {"vortspeco": "adverbo"},
    # Case/Number - handled separately
    "j": {},
    "n": {},
}

KNOWN_ROOTS = {"san", "hund", "kat", "program", "vid", "am", "bon", "mi", "grand", "la"}

# -----------------------------------------------------------------------------
# --- Layer 1: Morphological Analyzer (parse_word)
# -----------------------------------------------------------------------------

def parse_word(word: str) -> dict:
    """
    Parses a single Esperanto word and returns its morpheme-based AST.
    This function works by stripping layers from right-to-left.
    """
    original_word = word
    lower_word = word.lower()
    
    # --- Step 1: Initialize AST ---
    ast = {
        "tipo": "vorto",
        "plena_vorto": original_word,
        "radiko": None,
        "vortspeco": "nekonata", # unknown
        "nombro": "singularo",
        "kazo": "nominativo",
        "prefikso": None,
        "sufiksoj": [],
    }

    # --- Handle special, uninflected words first ---
    if lower_word == 'la':
        ast['vortspeco'] = 'artikolo'
        ast['radiko'] = 'la'
        return ast

    # --- Step 2: Decode Grammatical Endings (right-to-left) ---
    remaining_word = lower_word
    # Rule 6: Accusative Case (-n)
    if remaining_word.endswith('n'):
        ast["kazo"] = "akuzativo"
        remaining_word = remaining_word[:-1]

    # Rule 5: Plural (-j)
    if remaining_word.endswith('j'):
        ast["nombro"] = "pluralo"
        remaining_word = remaining_word[:-1]

    # Rule 4, 7, 8, 11, 12: Part of Speech & Tense/Mood
    found_ending = False
    for ending, properties in KNOWN_ENDINGS.items():
        if remaining_word.endswith(ending):
            ast.update(properties)
            remaining_word = remaining_word[:-len(ending)]
            found_ending = True
            break
    
    if not found_ending and lower_word not in KNOWN_ROOTS:
         # If no ending found and it's not a known root (like 'mi'), it's invalid
         raise ValueError(f"Vorto '{original_word}' havas neniun konatan finaĵon.")

    # --- Step 3: Decode Affixes (finds one prefix and multiple suffixes) ---
    stem = remaining_word
    
    # Prefixes (left side)
    for prefix in KNOWN_PREFIXES:
        if stem.startswith(prefix):
            ast["prefikso"] = prefix
            stem = stem[len(prefix):]
            break # Assume only one prefix for now

    # Suffixes (middle) - this is complex, we'll do a simple greedy match for now
    # This is a point for future improvement. A simple loop works for many cases.
    for suffix in KNOWN_SUFFIXES:
        if suffix in stem:
            ast["sufiksoj"].append(suffix)
            stem = stem.replace(suffix, '')

    # --- Step 4: Identify Root ---
    if stem in KNOWN_ROOTS:
        ast["radiko"] = stem
    elif not ast["radiko"]:
        # This could be a word like 'mi' which has no standard ending
        if lower_word in KNOWN_ROOTS:
            ast['radiko'] = lower_word
            ast['vortspeco'] = 'pronomo' # Pronoun
        else:
            raise ValueError(f"Ne povis trovi validan radikon en '{original_word}'. Restaĵo: '{stem}'")

    return ast

# -----------------------------------------------------------------------------
# --- Layer 2: Syntactic Analyzer (parse)
# -----------------------------------------------------------------------------

def parse(text: str):
    """
    Parses an Esperanto sentence and returns a structured, morpheme-based AST.
    """
    # Simple tokenizer: split by space, remove punctuation for now.
    words = text.replace('.', '').replace(',', '').split()

    if not words:
        return None

    # Step 1: Morphological analysis of all words
    word_asts = [parse_word(w) for w in words]

    # Step 2: Syntactic analysis to find sentence structure
    sentence_ast = {
        "tipo": "frazo",
        "subjekto": None,
        "verbo": None,
        "objekto": None,
        "aliaj": [] # Other parts
    }

    # Find the main components (verb, subject noun, object noun)
    for ast in word_asts:
        if ast["vortspeco"] == "verbo" and not sentence_ast["verbo"]:
            sentence_ast["verbo"] = ast
        elif ast["vortspeco"] == "substantivo" and ast["kazo"] == "akuzativo" and not sentence_ast["objekto"]:
            sentence_ast["objekto"] = {"tipo": "vortgrupo", "kerno": ast, "priskriboj": []}
        elif ast["vortspeco"] == "substantivo" and ast["kazo"] == "nominativo" and not sentence_ast["subjekto"]:
            sentence_ast["subjekto"] = {"tipo": "vortgrupo", "kerno": ast, "priskriboj": []}

    # Associate articles and adjectives with their noun groups
    for i, ast in enumerate(word_asts):
        if ast["vortspeco"] == "adjektivo":
            # If it matches the object's case and number, it describes the object
            if sentence_ast["objekto"] and ast["kazo"] == sentence_ast["objekto"]["kerno"]["kazo"] and ast["nombro"] == sentence_ast["objekto"]["kerno"]["nombro"]:
                sentence_ast["objekto"]["priskriboj"].append(ast)
            # If it matches the subject's case and number, it describes the subject
            elif sentence_ast["subjekto"] and ast["kazo"] == sentence_ast["subjekto"]["kerno"]["kazo"] and ast["nombro"] == sentence_ast["subjekto"]["kerno"]["nombro"]:
                sentence_ast["subjekto"]["priskriboj"].append(ast)
            else:
                sentence_ast["aliaj"].append(ast)
        
        elif ast["vortspeco"] == "artikolo":
            # Find the next noun and associate the article with it.
            # This is a simple heuristic that works for "la X" constructions.
            if i + 1 < len(word_asts):
                next_ast = word_asts[i+1]
                if sentence_ast["objekto"] and next_ast == sentence_ast["objekto"]["kerno"]:
                    sentence_ast["objekto"]["artikolo"] = "la"
                elif sentence_ast["subjekto"] and next_ast == sentence_ast["subjekto"]["kerno"]:
                    sentence_ast["subjekto"]["artikolo"] = "la"

    # Clean up unassociated words
    placed_words = []
    if sentence_ast["verbo"]:
        placed_words.append(sentence_ast["verbo"])
    if sentence_ast["subjekto"]:
        placed_words.append(sentence_ast["subjekto"]["kerno"])
        placed_words.extend(sentence_ast["subjekto"]["priskriboj"])
    if sentence_ast["objekto"]:
        placed_words.append(sentence_ast["objekto"]["kerno"])
        placed_words.extend(sentence_ast["objekto"]["priskriboj"])

    for ast in word_asts:
        if ast["vortspeco"] != 'artikolo' and ast not in placed_words:
            sentence_ast["aliaj"].append(ast)


    return sentence_ast

if __name__ == '__main__':
    # Example Usage
    import json

    def pretty_print(data):
        print(json.dumps(data, indent=2, ensure_ascii=False))

    sentence = "malgrandaj hundoj vidas grandan katon"
    print(f"--- Analizante frazon: '{sentence}' ---")
    ast = parse(sentence)
    pretty_print(ast)

    print("\n--- Analizante vorton: 'resanigos' ---")
    word_ast = parse_word("resanigos")
    pretty_print(word_ast)
