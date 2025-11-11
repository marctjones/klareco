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

KNOWN_PREFIXES = {
    "mal",  # opposite
    "re",   # again
    "ge",   # both genders
    "eks",  # former, ex-
    "pra",  # primordial, great- (as in great-grandfather)
}

KNOWN_SUFFIXES = {
    "ul",   # person characterized by
    "ej",   # place for
    "in",   # feminine
    "et",   # diminutive
    "ad",   # continuous action
    "ig",   # make/cause to be
    "iĝ",   # become
    "ism",  # doctrine/system
    "ist",  # professional/adherent
    "ar",   # collection/group
    "aĉ",   # pejorative
    "aĵ",   # concrete thing
    "ebl",  # possible to
    "end",  # must be done
    "ec",   # quality/abstract noun
    "eg",   # augmentative
    "em",   # tendency to
    "er",   # smallest unit
    "estr", # leader/chief
    "id",   # offspring
    "il",   # tool/instrument
    "ind",  # worthy of
    "ing",  # holder/socket
    "uj",   # container/country
    "um",   # indefinite meaning
}

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

# Personal pronouns (personaj pronomoj) - grammatically function exactly like nouns
# Source: Wikipedia Esperanto Grammar, Fundamento de Esperanto (1905)
# "Personal pronouns take the accusative suffix -n as nouns do" - can be subjects/objects
# Rule 5 (Fundamento): mi (I), vi (you), li (he), ŝi (she), ĝi (it),
#                       si (self-reflexive), ni (we), ili (they), oni (one/people)
# Accusative forms: min, vin, lin, ŝin, ĝin, sin, nin, ilin, onin
KNOWN_PRONOUNS = {"mi", "vi", "li", "ŝi", "ĝi", "si", "ni", "ili", "oni"}

# Conjunctions (konjunkcioj) - connect clauses and words
# These are uninflected words (no endings)
KNOWN_CONJUNCTIONS = {
    "kaj",    # and
    "aŭ",     # or
    "sed",    # but
    "nek",    # neither/nor
    "se",     # if
    "ĉar",    # because
    "kvankam", # although
    "ke",     # that (subordinating)
}

# Semantic roots (radikoj) - core vocabulary
# Expanded to cover common Esperanto words
KNOWN_ROOTS = {
    # Original roots
    "san", "hund", "kat", "program", "vid", "am", "bon", "grand", "la",

    # From test corpus (essential for tests to pass)
    "est",     # be/is (most important verb!)
    "manĝ",    # eat
    "dorm",    # sleep
    "kur",     # run
    "tag",     # day
    "amik",    # friend
    "aspekt",  # look/appear
    "labor",   # work
    "vol",     # want
    "help",    # help
    "bel",     # beautiful
    "rapid",   # quick/fast

    # Common verbs
    "far",     # do/make
    "dir",     # say
    "ven",     # come
    "ir",      # go
    "don",     # give
    "pren",    # take
    "hav",     # have
    "pov",     # can/be able
    "dev",     # must
    "sci",     # know
    "komprен",  # understand
    "parol",   # speak
    "skrib",   # write
    "leg",     # read
    "pens",    # think
    "sent",    # feel
    "stud",    # study
    "lern",    # learn
    "instru",  # teach
    "paf",     # shoot

    # Common nouns
    "hom",     # human/person
    "vir",     # man
    "infan",   # child
    "patr",    # father
    "patrın",  # mother
    "frat",    # brother
    "fil",     # son
    "dom",     # house
    "urb",     # city
    "land",    # land/country
    "mond",    # world
    "temp",    # time
    "jar",     # year
    "monat",   # month
    "semajn",  # week
    "hor",     # hour
    "minut",   # minute
    "lok",     # place
    "voj",     # way/road
    "aŭt",     # car
    "libr",    # book
    "tabl",    # table
    "seĝ",     # chair
    "pord",    # door
    "fenеstr", # window
    "akvо",    # water
    "pаn",     # bread
    "viаnd",   # meat
    "frukt",   # fruit
    "arb",     # tree
    "flor",    # flower
    "sun",     # sun
    "lun",     # moon
    "stel",    # star

    # Common adjectives
    "nov",     # new
    "malnov",  # old
    "jung",    # young
    "mal jung", # old (of people)
    "alt",     # high/tall
    "bas",     # low
    "long",    # long
    "kurt",    # short
    "larg",    # wide
    "gras",    # fat/thick
    "dik",     # thick
    "varm",    # warm
    "malvarm", # cold
    "vеr",     # true
    "fals",    # false
    "bon",     # good (duplicate but keep for clarity)
    "malbon",  # bad
    "bel",     # beautiful (duplicate but keep)
    "malbel",  # ugly
    "feliĉ",   # happy
    "trist",   # sad
    "fru",     # early
    "malfru",  # late
    "facil",   # easy
    "malfacil", # difficult
    "fort",    # strong
    "malfort", # weak
    "riĉ",     # rich
    "malriĉ",  # poor
    "plen",    # full
    "malplen", # empty
    "pеz",     # heavy
    "malpеz",  # light

    # Colors
    "ruĝ",     # red
    "blu",     # blue
    "verd",    # green
    "flav",    # yellow
    "nigr",    # black
    "blank",   # white
    "griz",    # gray

    # Numbers (as roots, can take endings)
    "unu",     # one
    "du",      # two
    "tri",     # three
    "kvar",    # four
    "kvin",    # five
    "ses",     # six
    "sep",     # seven
    "ok",      # eight
    "naŭ",     # nine
    "dek",     # ten
    "cent",    # hundred
    "mil",     # thousand
}

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

    # Check for conjunctions - uninflected words (no endings)
    # Must check before trying to strip endings
    if lower_word in KNOWN_CONJUNCTIONS:
        ast['vortspeco'] = 'konjunkcio'
        ast['radiko'] = lower_word
        return ast

    # --- Step 2: Decode Grammatical Endings (right-to-left) ---
    remaining_word = lower_word

    # Rule 6: Accusative Case (-n)
    # Pronouns can take accusative: mi→min, vi→vin, etc.
    if remaining_word.endswith('n'):
        ast["kazo"] = "akuzativo"
        remaining_word = remaining_word[:-1]

    # Rule 5: Plural (-j)
    # Note: Personal pronouns don't take -j (they're already marked for number)
    if remaining_word.endswith('j'):
        ast["nombro"] = "pluralo"
        remaining_word = remaining_word[:-1]

    # Check for pronouns EARLY - before POS ending checks
    # Pronouns (pronomoj) are atomic morphemes - no affixes, no POS endings
    # "mi" ends with "i" but that's NOT the infinitive ending, it's just how it's spelled!
    # Must check before KNOWN_ENDINGS to prevent false matches (mi→m+i, vi→v+i, etc.)
    if remaining_word in KNOWN_PRONOUNS or lower_word in KNOWN_PRONOUNS:
        ast['radiko'] = remaining_word if remaining_word in KNOWN_PRONOUNS else lower_word
        ast['vortspeco'] = 'pronomo'  # Pronoun (pronomoj - functions like substantivo)
        return ast

    # Rule 4, 7, 8, 11, 12: Part of Speech & Tense/Mood
    # Only check these for non-pronouns
    found_ending = False
    for ending, properties in KNOWN_ENDINGS.items():
        if remaining_word.endswith(ending):
            ast.update(properties)
            remaining_word = remaining_word[:-len(ending)]
            found_ending = True
            break

    # If no ending found and it's not a known root, it's invalid
    if not found_ending and lower_word not in KNOWN_ROOTS:
         raise ValueError(f"Vorto '{original_word}' havas neniun konatan finaĵon.")

    # --- Step 3: Decode Affixes (finds one prefix and multiple suffixes) ---
    # Only for non-pronouns - pronouns are atomic
    stem = remaining_word

    # Prefixes (left side)
    for prefix in KNOWN_PREFIXES:
        if stem.startswith(prefix):
            ast["prefikso"] = prefix
            stem = stem[len(prefix):]
            break # Assume only one prefix for now

    # Suffixes (middle) - improved matching logic
    # Only match suffixes if they leave behind a valid root
    # Sort suffixes by length (longest first) to match greedily
    sorted_suffixes = sorted(KNOWN_SUFFIXES, key=len, reverse=True)

    for suffix in sorted_suffixes:
        # Check if suffix is present and would leave a valid root
        if suffix in stem:
            # Try removing the suffix
            potential_root = stem.replace(suffix, '', 1)  # Remove only first occurrence

            # Only accept this suffix if what remains is either:
            # 1. A known root, OR
            # 2. Could still contain other suffixes and a root
            if potential_root in KNOWN_ROOTS or len(potential_root) >= 2:
                # Check if removing this suffix leaves us with a known root eventually
                # For now, only remove if we can verify the root exists
                temp_stem = potential_root
                # Try removing remaining suffixes
                for other_suffix in sorted_suffixes:
                    if other_suffix in temp_stem:
                        temp_stem = temp_stem.replace(other_suffix, '', 1)

                # If we end up with a known root, accept the suffix
                if temp_stem in KNOWN_ROOTS:
                    ast["sufiksoj"].append(suffix)
                    stem = potential_root
                    break  # Only remove one suffix at a time for now

    # --- Step 5: Identify Root ---
    if stem in KNOWN_ROOTS:
        ast["radiko"] = stem
    elif not ast["radiko"]:
        raise ValueError(f"Ne povis trovi validan radikon en '{original_word}'. Restaĵo: '{stem}'")

    return ast

# -----------------------------------------------------------------------------
# --- Layer 2: Syntactic Analyzer (parse)
# -----------------------------------------------------------------------------

def parse(text: str):
    """
    Parses an Esperanto sentence and returns a structured, morpheme-based AST.
    """
    # Simple tokenizer: split by space, remove all punctuation
    # Remove common punctuation marks: . , ! ? : ; " ' ( ) [ ] { }
    import string
    for punct in string.punctuation:
        text = text.replace(punct, ' ')
    words = text.split()

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
    # Rule 6: Case determines grammatical function (nominative=subject, accusative=object)
    # Pronouns (pronomoj) function exactly like nouns (substantivoj) grammatically
    for ast in word_asts:
        if ast["vortspeco"] == "verbo" and not sentence_ast["verbo"]:
            sentence_ast["verbo"] = ast
        # Object: any noun or pronoun in accusative case (-n)
        elif ast["vortspeco"] in ["substantivo", "pronomo"] and ast["kazo"] == "akuzativo" and not sentence_ast["objekto"]:
            sentence_ast["objekto"] = {"tipo": "vortgrupo", "kerno": ast, "priskriboj": []}
        # Subject: any noun or pronoun in nominative case (no -n)
        elif ast["vortspeco"] in ["substantivo", "pronomo"] and ast["kazo"] == "nominativo" and not sentence_ast["subjekto"]:
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
            # Find the noun that this article modifies (may have adjectives in between)
            # Example: "la grandan katon" - article "la" applies to "katon" despite "grandan"
            # Look ahead through adjectives to find the noun
            for j in range(i + 1, len(word_asts)):
                next_ast = word_asts[j]
                # Skip adjectives, look for the noun
                if next_ast["vortspeco"] in ["substantivo", "pronomo"]:
                    # Check if this noun is the object or subject
                    if sentence_ast["objekto"] and next_ast == sentence_ast["objekto"]["kerno"]:
                        sentence_ast["objekto"]["artikolo"] = "la"
                        break
                    elif sentence_ast["subjekto"] and next_ast == sentence_ast["subjekto"]["kerno"]:
                        sentence_ast["subjekto"]["artikolo"] = "la"
                        break
                elif next_ast["vortspeco"] != "adjektivo":
                    # If we hit something that's not an adjective or noun, stop
                    break

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
