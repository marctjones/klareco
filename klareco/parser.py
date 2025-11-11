"""A from-scratch, pure Python parser for Esperanto.

This parser is built on the 16 rules of Esperanto and does not use any
external parsing libraries like Lark. It performs morphological and syntactic
analysis to produce a detailed, Esperanto-native Abstract Syntax Tree (AST)."""
import re
try:
    from data.merged_vocabulary import MERGED_ROOTS as DICTIONARY_ROOTS
except ImportError:
    # Fallback to Gutenberg dictionary if merged vocabulary not available
    from data.extracted_vocabulary import DICTIONARY_ROOTS

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
    # Participial suffixes (active and passive)
    "ant",  # present active participle (seeing)
    "int",  # past active participle (having seen)
    "ont",  # future active participle (about to see)
    "at",   # present passive participle (being seen)
    "it",   # past passive participle (having been seen)
    "ot",   # future passive participle (about to be seen)
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
    "tamen",  # however/nevertheless
    "do",     # therefore/so
}

# Prepositions (prepozicioj) - show relationships
# These are uninflected words (no endings)
KNOWN_PREPOSITIONS = {
    "al",      # to, toward
    "ĉe",      # at, by
    "de",      # of, from
    "da",      # of (quantity)
    "dum",     # during, while
    "el",      # out of, from
    "en",      # in, into
    "ekster",  # outside
    "ĝis",     # until, up to
    "inter",   # between, among
    "kontraŭ", # against
    "krom",    # besides, except
    "kun",     # with
    "laŭ",     # according to, along
    "per",     # by means of, with
    "po",      # at (distributive)
    "post",    # after, behind
    "preter",  # past, by
    "pri",     # about, concerning
    "pro",     # because of
    "sen",     # without
    "sub",     # under, below
    "super",   # above, over
    "sur",     # on, upon
    "tra",     # through
    "trans",   # across
    "antaŭ",   # before, in front of
    "apud",    # beside, next to
    "ĉirkaŭ",  # around
}

# Correlatives (korelativoj) - the famous Esperanto correlative table
# These are uninflected words formed from 5 beginnings × 9 endings
KNOWN_CORRELATIVES = {
    # Ki- (interrogative/relative)
    "kia",     # what kind of
    "kial",    # why
    "kiam",    # when
    "kie",     # where
    "kien",    # where to (direction)
    "kies",    # whose
    "kio",     # what
    "kiom",    # how much/many
    "kiu",     # who, which

    # Ti- (demonstrative)
    "tia",     # that kind of
    "tial",    # therefore
    "tiam",    # then (at that time)
    "tie",     # there
    "tien",    # there (direction)
    "ties",    # that one's
    "tio",     # that
    "tiom",    # that much/many
    "tiu",     # that (one)

    # Ĉi- (universal)
    "ĉia",     # every kind of
    "ĉial",    # for every reason
    "ĉiam",    # always
    "ĉie",     # everywhere
    "ĉien",    # in every direction
    "ĉies",    # everyone's
    "ĉio",     # everything
    "ĉiom",    # all (the amount)
    "ĉiu",     # everyone, each

    # Neni- (negative)
    "nenia",   # no kind of
    "nenial",  # for no reason
    "neniam",  # never
    "nenie",   # nowhere
    "nenien",  # in no direction
    "nenies",  # no one's
    "nenio",   # nothing
    "neniom",  # none (amount)
    "neniu",   # no one, nobody

    # I- (indefinite)
    "ia",      # some kind of
    "ial",     # for some reason
    "iam",     # sometime
    "ie",      # somewhere
    "ien",     # somewhere (direction)
    "ies",     # someone's
    "io",      # something
    "iom",     # some (amount)
    "iu",      # someone, somebody
}

# Common particles and adverbs
KNOWN_PARTICLES = {
    "ajn",     # any (modifier: kiu ajn = whoever)
    "ankaŭ",   # also, too
    "ankoraŭ",  # still, yet
    "apenaŭ",  # hardly, scarcely
    "baldaŭ",  # soon
    "ĉi",      # this/here (modifier)
    "ĉu",      # whether, question particle
    "des",     # the (in correlatives: ju...des = the...the)
    "eĉ",      # even
    "hieraŭ",  # yesterday
    "hodiaŭ",  # today
    "ja",      # indeed, you know
    "jam",     # already
    "jes",     # yes
    "ju",      # the (in correlatives: ju...des = the...the)
    "kvazaŭ",  # as if, as though
    "morgaŭ",  # tomorrow
    "ne",      # no, not
    "nek",     # neither, nor
    "nu",      # well (interjection)
    "nun",     # now
    "nur",     # only
    "pli",     # more (comparative)
    "plej",    # most (superlative)
    "plu",     # more, further
    "preskaŭ", # almost
    "tamen",   # however, nevertheless (also conjunction)
    "tre",     # very
    "tro",     # too (excessive)
    "tuj",     # immediately
}

# Number words (numeraloj) - can function as adjectives or substantives
KNOWN_NUMBERS = {
    "nul",     # zero
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
    "milion",  # million
    "miliard", # billion
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

# Merge with 8,232 roots extracted from Gutenberg English-Esperanto Dictionary
# This massively expands our vocabulary coverage
# Also merge number words so they can be used as roots
KNOWN_ROOTS = KNOWN_ROOTS | DICTIONARY_ROOTS | KNOWN_NUMBERS

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

    # --- Handle foreign words and numbers ---
    # Skip numeric literals (years, etc.) - treat as foreign words
    if word.isdigit():
        ast['vortspeco'] = 'numero'
        ast['radiko'] = word
        return ast

    # Skip capitalized non-Esperanto names (but allow Esperanto proper nouns)
    # If word is capitalized and doesn't end with Esperanto morphology, it's likely foreign
    if word[0].isupper() and len(word) > 1:
        # Check if it has Esperanto endings (could be proper noun: Mario, Johano, etc.)
        if not any(word.endswith(ending) for ending in ['o', 'a', 'e', 'on', 'an', 'en', 'oj', 'aj']):
            # Likely foreign name
            ast['vortspeco'] = 'nomo'  # foreign name
            ast['radiko'] = word
            return ast

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

    # Check for prepositions - uninflected words
    if lower_word in KNOWN_PREPOSITIONS:
        ast['vortspeco'] = 'prepozicio'
        ast['radiko'] = lower_word
        return ast

    # Check for correlatives - uninflected words
    if lower_word in KNOWN_CORRELATIVES:
        ast['vortspeco'] = 'korelativo'
        ast['radiko'] = lower_word
        return ast

    # Check for particles - uninflected adverbs and modifiers
    if lower_word in KNOWN_PARTICLES:
        ast['vortspeco'] = 'partiklo'
        ast['radiko'] = lower_word
        return ast

    # --- Step 2: Decode Grammatical Endings (right-to-left) ---
    remaining_word = lower_word

    # Check for numbers - can be inflected like adjectives (dua, duaj, duan, etc.)
    # Numbers can take -a (adjective), -e (adverb), -o (noun), -j (plural), -n (accusative)
    # So we need to strip endings and check the root
    temp_word = remaining_word
    if temp_word.endswith('n'):
        temp_word = temp_word[:-1]
    if temp_word.endswith('j'):
        temp_word = temp_word[:-1]
    if temp_word.endswith(('a', 'e', 'o')):
        temp_word = temp_word[:-1]
    if temp_word in KNOWN_NUMBERS:
        # It's a number - treat it as a root for morphological analysis
        # Don't return early - let it go through normal ending analysis
        pass

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

    # Extract suffixes from right to left (innermost last, outermost first in final list)
    while True:
        found_suffix = False
        for suffix in sorted_suffixes:
            # Check if suffix is at the END of the stem
            if stem.endswith(suffix):
                # Try removing the suffix from the end
                potential_root = stem[:-len(suffix)]

                # Only accept this suffix if what remains is either:
                # 1. A known root, OR
                # 2. Long enough to potentially contain more suffixes and a root
                if potential_root in KNOWN_ROOTS or len(potential_root) >= 2:
                    # Accept this suffix
                    ast["sufiksoj"].append(suffix)
                    stem = potential_root
                    found_suffix = True
                    break  # Found one suffix, check for more

        # Stop if no more suffixes found
        if not found_suffix:
            break

    # --- Step 5: Identify Root ---
    if stem in KNOWN_ROOTS:
        ast["radiko"] = stem
    elif not ast["radiko"]:
        raise ValueError(f"Ne povis trovi validan radikon en '{original_word}'. Restaĵo: '{stem}'")

    return ast

# -----------------------------------------------------------------------------
# --- Layer 2: Syntactic Analyzer (parse)
# -----------------------------------------------------------------------------

def preprocess_text(text: str) -> str:
    """
    Preprocess text before parsing to normalize punctuation and whitespace.

    - Converts em-dashes, en-dashes to spaces
    - Normalizes smart quotes to straight quotes
    - Normalizes whitespace
    """
    # Replace various dash types with spaces to separate words
    text = text.replace('—', ' ')  # em-dash
    text = text.replace('–', ' ')  # en-dash
    text = text.replace('―', ' ')  # horizontal bar

    # Normalize smart quotes to straight quotes (will be removed later)
    text = text.replace('"', '"')  # left double quote
    text = text.replace('"', '"')  # right double quote
    text = text.replace(''', "'")  # left single quote
    text = text.replace(''', "'")  # right single quote
    text = text.replace('‚', "'")  # low single quote
    text = text.replace('„', '"')  # low double quote

    # Normalize whitespace (multiple spaces to single space)
    text = ' '.join(text.split())

    return text


def parse(text: str):
    """
    Parses an Esperanto sentence and returns a structured, morpheme-based AST.
    """
    # Preprocess: normalize punctuation
    text = preprocess_text(text)

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
