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
    "ek",   # sudden action, beginning
    "pra",  # primordial, great- (as in great-grandfather)
    "for",  # away, completely
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

# Participle metadata for Issue #84
PARTICIPLE_SUFFIXES = {
    "ant": {"voĉo": "aktiva", "tempo": "prezenco"},
    "int": {"voĉo": "aktiva", "tempo": "pasinteco"},
    "ont": {"voĉo": "aktiva", "tempo": "futuro"},
    "at": {"voĉo": "pasiva", "tempo": "prezenco"},
    "it": {"voĉo": "pasiva", "tempo": "pasinteco"},
    "ot": {"voĉo": "pasiva", "tempo": "futuro"},
}

# Correlative decomposition for Issue #76
CORRELATIVE_PREFIXES = {
    "ki": "demanda",      # interrogative/relative
    "ti": "montra",       # demonstrative
    "i": "nedefinita",    # indefinite
    "ĉi": "universala",   # universal
    "neni": "nea",        # negative
}

CORRELATIVE_SUFFIXES = {
    "o": "aĵo",     # thing
    "u": "persono", # person
    "a": "eco",     # quality
    "e": "loko",    # place
    "am": "tempo",  # time
    "el": "maniero",# manner
    "om": "kvanto", # quantity
    "al": "kaŭzo",  # reason
    "es": "posedo", # possession
}

# The order of endings matters. Longer ones must be checked first.
KNOWN_ENDINGS = {
    # Tense (indicative mood - 3 tenses)
    "as": {"vortspeco": "verbo", "tempo": "prezenco"},
    "is": {"vortspeco": "verbo", "tempo": "pasinteco"},
    "os": {"vortspeco": "verbo", "tempo": "futuro"},
    # Mood (non-indicative - no inherent tense) - Issue #91 fix
    "us": {"vortspeco": "verbo", "modo": "kondicionalo"},  # Conditional mood (not tense!)
    "u": {"vortspeco": "verbo", "modo": "imperativo"},     # Imperative/volitional
    "i": {"vortspeco": "verbo", "modo": "infinitivo"},     # Infinitive (non-finite)
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
    "je",      # (undefined meaning - used when no other preposition fits)
    "kontraŭ", # against
    "krom",    # besides, except
    "kun",     # with
    "laŭ",     # according to, along
    "per",     # by means of, with
    "po",      # at (distributive)
    "por",     # for (purpose, benefit) - Issue #89
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
    "kiel",    # how, as (manner)
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
    "tiel",    # thus, so (manner)
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
    "ĉiel",    # in every manner
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
    "neniel",  # in no manner
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
    "iel",     # somehow (manner)
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
    "ha",      # ha (interjection)
    "hieraŭ",  # yesterday
    "ho",      # oh (interjection)
    "hodiaŭ",  # today
    "ja",      # indeed, you know
    "jam",     # already
    "jen",     # behold, here is/are
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
    "ree",     # again (adverb)
    "tamen",   # however, nevertheless (also conjunction)
    "tre",     # very
    "tro",     # too (excessive)
    "tuj",     # immediately
    "tju",     # phew, whew (interjection)
    "ve",      # woe, alas (interjection)
    "ĵus",     # just (recently)
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
    # Compound numbers
    "dek unu", # eleven
    "dek du",  # twelve
    "dek tri", # thirteen
    "dek kvar", # fourteen
    "dek kvin", # fifteen
    "dek ses", # sixteen
    "dek sep", # seventeen
    "dek ok",  # eighteen
    "dek naŭ", # nineteen
    "dudek",   # twenty
    "tridek",  # thirty
    "kvardek", # forty
    "kvindek", # fifty
    "sesdek",  # sixty
    "sepdek",  # seventy
    "okdek",   # eighty
    "naŭdek",  # ninety
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

    # Additional common roots from Gutenberg corpus
    "reĝ",     # king
    "best",    # beast/animal
    "leon",    # lion
    "kolomb",  # dove/pigeon
    "bird",    # bird
    "roz",     # rose
    "pom",     # apple
    "ter",     # earth/ground
    "ŝton",    # stone
    "ĉiel",    # sky/heaven
    "krajon",  # pencil
    "plum",    # pen/feather
    "dent",    # tooth
    "man",     # hand
    "respond", # respond/answer
    "reg",     # rule/reign
    "obed",    # obey
    "rajt",    # right/entitle
    "apart",   # belong to
    "kuŝ",     # lie down
    "bril",    # shine
    "peto",    # request/petition
    "dang",    # danger
    "kuraĝ",   # courage
    "rajd",    # ride
    "mor",     # die
    "pet", # request (alternative)
    "fenestr", # window (fixed spelling)
    "akv",     # water (fixed spelling)
    "pan",     # bread (fixed spelling)
    "viand",   # meat (fixed spelling)
    "aŭt",     # car (fixed spelling)
    "just",    # just/fair
    "ĝust",    # correct/exact
    "hon",     # shame/be ashamed
    "lev",     # lift/raise
    "ŝancel",  # stagger/totter
    "cel",     # aim/goal
    "ekst",    # ecstasy (noun root, not prefix)
    "enu",     # bore/annoy
    "aŭd",     # hear
    "ramp",    # crawl/creep
    "viv",     # live
    "ricev",   # receive
    "konsil",  # advise/counsel
    "turn",    # turn
    "duon",    # half

    # From literary corpus analysis (analyze_failures.py - verified standard Esperanto)
    "region",  # region (regiono)
    "trankv",  # calm, tranquil (trankvila)
    "alfabet", # alphabet (alfabeto)
    "liĝ",     # law (leĝo)
    "punkt",   # point (punkto)
    "manier",  # manner (maniero)
    "preciz",  # precise (preciza)
    "sven",    # faint, swoon (sveni)
    "disting", # distinguish (distingi)
    "renkont", # encounter, meet (renkonti)
    "distanc", # distance (distanco)
    "demand",  # ask, demand (demandi)
    "bord",    # edge, border (bordo)
    "miz",     # misery (mizero)
    "memor",   # memory (memoro)
    "fakt",    # fact (fakto)
    "mir",     # wonder, marvel (miri)
    "ofer",    # offer, sacrifice (oferi)
    "kord",    # cord, heart (koro)
    "nask",    # birth, be born (naski)
    "redakt",  # edit, redact (redakti)
    "prezid",  # preside (prezidi)
    "akademi", # academy (akademio)
    "vok",     # call (voki)
    "konfirm", # confirm (konfirmi)
    "absolut", # absolute (absoluta)
    "dialog",  # dialogue (dialogo)
    "sistematik", # systematic (sistematika)
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

    # --- Handle elision (Rule 16, Issue #88) ---
    # In poetry, final -o can be elided: la → l', hundo → hund'
    if lower_word.endswith("'") or lower_word.endswith("'"):
        elided_word = lower_word.rstrip("'").rstrip("'")
        ast['elidita'] = True

        # Special case: l' = la (article)
        if elided_word == 'l':
            ast['vortspeco'] = 'artikolo'
            ast['radiko'] = 'la'
            return ast

        # Check for prefix + root FIRST (prefer decomposition)
        # This ensures malamik' → mal+amik rather than malamik as single root
        for prefix in KNOWN_PREFIXES:
            if elided_word.startswith(prefix):
                potential_root = elided_word[len(prefix):]
                if potential_root in KNOWN_ROOTS:
                    ast['vortspeco'] = 'substantivo'
                    ast['radiko'] = potential_root
                    ast['prefikso'] = prefix
                    return ast

        # Fall back to full word as known root (noun with -o elided)
        if elided_word in KNOWN_ROOTS:
            ast['vortspeco'] = 'substantivo'
            ast['radiko'] = elided_word
            return ast

    # --- Handle foreign words and numbers ---
    # Skip numeric literals (years, etc.) - treat as foreign words
    if word.isdigit():
        ast['vortspeco'] = 'numero'
        ast['radiko'] = word
        return ast

    # Check for pronouns FIRST - before foreign name check
    # Pronouns can be capitalized at start of sentence: "Mi vidas..." = "I see..."
    # Check both lowercase and (for accusative) with -n ending
    temp_word_for_pronoun = lower_word
    if temp_word_for_pronoun.endswith('n'):
        temp_word_for_pronoun = temp_word_for_pronoun[:-1]

    if temp_word_for_pronoun in KNOWN_PRONOUNS:
        ast['radiko'] = temp_word_for_pronoun
        ast['vortspeco'] = 'pronomo'
        # Check if it had accusative ending
        if lower_word.endswith('n'):
            ast['kazo'] = 'akuzativo'
        return ast

    # Skip capitalized non-Esperanto names (but allow Esperanto proper nouns)
    # If word is capitalized and doesn't end with Esperanto morphology, it's likely foreign
    # BUT first check if it's a known particle (like Ĉu at sentence start)
    if word[0].isupper() and len(word) > 1:
        # Check for particles BEFORE treating as proper name (Issue #87)
        # Words like Ĉu can appear capitalized at sentence start
        if lower_word in KNOWN_PARTICLES:
            ast['vortspeco'] = 'partiklo'
            ast['radiko'] = lower_word
            return ast
        # Check for correlatives (like Kiu, Kio at sentence start)
        if lower_word in KNOWN_CORRELATIVES:
            correl_check = lower_word
            ast['vortspeco'] = 'korelativo'
            ast['radiko'] = correl_check
            # Decompose correlative into prefix + suffix
            for prefix in sorted(CORRELATIVE_PREFIXES.keys(), key=len, reverse=True):
                if correl_check.startswith(prefix):
                    suffix = correl_check[len(prefix):]
                    if suffix in CORRELATIVE_SUFFIXES:
                        ast['korelativo_prefikso'] = prefix
                        ast['korelativo_sufikso'] = suffix
                        ast['korelativo_signifo'] = CORRELATIVE_PREFIXES[prefix]
                        break
            return ast
        # Check if it has Esperanto endings (could be proper noun: Mario, Johano, etc.)
        # Note: Include -u for imperatives (Venu!, Donu!) and -as/-is/-os/-us for verbs
        if not any(word.endswith(ending) for ending in ['o', 'a', 'e', 'u', 'i', 'on', 'an', 'en', 'un', 'oj', 'aj', 'as', 'is', 'os', 'us']):
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

    # Check for correlatives - uninflected words with compositional semantics (Issue #76)
    # Correlatives can also take -n (accusative) for -o, -u types
    correl_check = lower_word
    correl_accusative = False
    if correl_check.endswith('n'):
        correl_check = correl_check[:-1]
        correl_accusative = True

    if correl_check in KNOWN_CORRELATIVES:
        ast['vortspeco'] = 'korelativo'
        ast['radiko'] = correl_check
        if correl_accusative:
            ast['kazo'] = 'akuzativo'

        # Decompose correlative into prefix + suffix (Issue #76)
        # Try each prefix (longest first to handle "neni-" before "i-")
        for prefix in sorted(CORRELATIVE_PREFIXES.keys(), key=len, reverse=True):
            if correl_check.startswith(prefix):
                suffix = correl_check[len(prefix):]
                if suffix in CORRELATIVE_SUFFIXES:
                    ast['korelativo_prefikso'] = prefix
                    ast['korelativo_sufikso'] = suffix
                    ast['korelativo_signifo'] = CORRELATIVE_PREFIXES[prefix]
                    break
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

    # NOTE: Pronouns are now checked earlier (before foreign name check)
    # to handle capitalized pronouns like "Mi" at sentence start

    # Check for bare number words (before ending stripping)
    # Numbers like "du" would otherwise be parsed as "d" + "u" (volitive ending)
    if remaining_word in KNOWN_NUMBERS:
        ast['radiko'] = remaining_word
        ast['vortspeco'] = 'numero'
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
    # Check for decomposable prefixes FIRST, even if the compound form exists
    # This is linguistically correct for Esperanto (compositional semantics)
    # e.g., "malgrand" should be "mal-" + "grand", not a standalone root
    prefix_stripped = False
    for prefix in KNOWN_PREFIXES:
        if stem.startswith(prefix):
            remaining_after_prefix = stem[len(prefix):]
            # Prefer prefix decomposition if what remains is a known root
            if remaining_after_prefix in KNOWN_ROOTS:
                ast["prefikso"] = prefix
                stem = remaining_after_prefix
                prefix_stripped = True
                break # Assume only one prefix for now
            # Also accept if what remains is long enough to potentially have suffixes
            elif stem not in KNOWN_ROOTS and len(remaining_after_prefix) >= 3:
                ast["prefikso"] = prefix
                stem = remaining_after_prefix
                prefix_stripped = True
                break

    # Suffixes (middle) - improved matching logic
    # Only match suffixes if they leave behind a valid root
    # Sort suffixes by length (longest first) to match greedily
    sorted_suffixes = sorted(KNOWN_SUFFIXES, key=len, reverse=True)

    # FIX for Issue #90: Prefer longer roots over shorter roots with spurious suffixes.
    # Example: "rapide" should parse as "rapid" + e, NOT "rap" + "id" + e
    # Even though both "rap" and "rapid" are known roots.
    #
    # Strategy: If the current stem is already a known root, only strip suffixes
    # if the remaining part is ALSO a known root. This prefers the longer
    # indivisible root ("rapid") over spurious decomposition ("rap" + "id").
    #
    # BUT: For genuine compositional words like "belulo" (bel + ul + o), we DO
    # want to strip the suffix because "bel" is the semantic base.

    # Extract suffixes from right to left (innermost last, outermost first in final list)
    # Be conservative - only remove suffix if it leads to a known root eventually
    max_suffix_depth = 3  # Limit suffix chaining to prevent over-matching
    suffix_count = 0

    while suffix_count < max_suffix_depth:
        found_suffix = False
        for suffix in sorted_suffixes:
            # Check if suffix is at the END of the stem
            if stem.endswith(suffix):
                # Try removing the suffix from the end
                potential_root = stem[:-len(suffix)]

                # Only accept this suffix if what remains is either:
                # 1. A known root (BEST case)
                # 2. A known particle/adverb (for compounds like "tre" + "eg")
                if (potential_root in KNOWN_ROOTS or
                    potential_root in KNOWN_PARTICLES or
                    potential_root in KNOWN_PREPOSITIONS):

                    # Issue #90 fix: Check if this is a spurious decomposition.
                    # If the current stem is a known root AND the potential_root
                    # is shorter, prefer the current stem IF the suffix is not
                    # a standard Esperanto derivational suffix for this context.
                    #
                    # Heuristic: If stem is already a known root and potential_root
                    # is ALSO a known root, prefer the SHORTER one (more derivation)
                    # because Esperanto is compositional. This handles "belulo" → "bel".
                    # BUT if the suffix is "id" (not common), and stem is known, skip.
                    if (stem in KNOWN_ROOTS and
                        suffix not in {'ul', 'in', 'et', 'eg', 'ar', 'ej', 'an',
                                       'ig', 'iĝ', 'ad', 'aĵ', 'ec', 'er', 'ebl',
                                       'em', 'end', 'ind', 'ing', 'ist', 'il', 'op',
                                       'uj', 'um', 'obl', 'on',
                                       # Participle suffixes (Issue #84)
                                       'ant', 'int', 'ont', 'at', 'it', 'ot'}):
                        # This is likely a spurious suffix (like "id" in "rapid")
                        # Skip this decomposition
                        continue

                    ast["sufiksoj"].append(suffix)
                    stem = potential_root
                    found_suffix = True
                    suffix_count += 1
                    break  # Found one suffix, check for more

                # 3. Could have a prefix that leads to a known root
                elif len(potential_root) >= 3:
                    # Check if removing a prefix leaves a known root
                    could_have_prefix = False
                    for prefix in KNOWN_PREFIXES:
                        if potential_root.startswith(prefix):
                            root_without_prefix = potential_root[len(prefix):]
                            if root_without_prefix in KNOWN_ROOTS:
                                could_have_prefix = True
                                break

                    if could_have_prefix:
                        ast["sufiksoj"].append(suffix)
                        stem = potential_root
                        found_suffix = True
                        suffix_count += 1
                        break

        # Stop if no more suffixes found
        if not found_suffix:
            break

    # --- Step 4b: Add participle metadata (Issue #84) ---
    # If any participle suffix was found, add voice and tense info
    for suffix in ast["sufiksoj"]:
        if suffix in PARTICIPLE_SUFFIXES:
            participle_info = PARTICIPLE_SUFFIXES[suffix]
            ast['participo_voĉo'] = participle_info['voĉo']
            ast['participo_tempo'] = participle_info['tempo']
            break  # Only one participle suffix per word

    # --- Step 5: Identify Root (with compound word decomposition) ---
    # Strategy: Try compound decomposition for LONG stems that are also known roots
    # Short stems (<=5 chars) that are known roots → use as-is (e.g., ŝton)
    # Long stems (>5 chars) that are known roots → try compound decomposition first
    compound_found = False

    # Short stems that are known roots: use directly
    if len(stem) <= 5 and stem in KNOWN_ROOTS:
        ast["radiko"] = stem
    elif stem in KNOWN_PARTICLES or stem in KNOWN_PREPOSITIONS:
        # It's a particle/preposition used as a root (e.g., "tre" in "treege")
        ast["radiko"] = stem
    # Long stems or unknown stems: try compound decomposition
    elif len(stem) >= 4:  # Minimum for compound: 2-char root + 2-char root

        # Check if starts with a preposition
        for prep in KNOWN_PREPOSITIONS:
            if stem.startswith(prep) and len(stem) > len(prep):
                remaining = stem[len(prep):]
                if remaining in KNOWN_ROOTS:
                    # It's a compound: preposition + root
                    ast["radiko"] = remaining
                    ast["prefikso"] = prep  # Use prefix field for compound marker
                    compound_found = True
                    break

        # Check if starts with a correlative (tiu, kiu, etc.)
        if not compound_found:
            for corr in ['tiu', 'kiu', 'ĉiu', 'neniu', 'iu', 'tio', 'kio']:
                if stem.startswith(corr) and len(stem) > len(corr):
                    remaining = stem[len(corr):]
                    if remaining in KNOWN_ROOTS:
                        ast["radiko"] = remaining
                        ast["prefikso"] = corr
                        compound_found = True
                        break

        # Check if starts with an adverb (tre, pli, plej, etc.)
        if not compound_found:
            for adv in ['tre', 'pli', 'plej', 'tro', 'tre', 'nun', 'jam']:
                if stem.startswith(adv) and len(stem) > len(adv):
                    remaining = stem[len(adv):]
                    if remaining in KNOWN_ROOTS:
                        ast["radiko"] = remaining
                        ast["prefikso"] = adv
                        compound_found = True
                        break

        # Check if starts with sub/super + en/el/etc (suben, superen)
        if not compound_found:
            for compound_prep in ['suben', 'superen', 'ekster', 'interne']:
                if stem.startswith(compound_prep) and len(stem) > len(compound_prep):
                    remaining = stem[len(compound_prep):]
                    if remaining in KNOWN_ROOTS:
                        ast["radiko"] = remaining
                        ast["prefikso"] = compound_prep
                        compound_found = True
                        break

        # Issue #80: True compound word decomposition (root + root)
        # Esperanto compounds often use linking vowel -o- between roots
        # Examples: akvobird (akv+o+bird), vaporŝip (vapor+ŝip), sunflor (sun+flor)
        if not compound_found:
            # Try all possible split points
            for i in range(2, len(stem) - 1):  # Root must be at least 2 chars
                first_part = stem[:i]
                remaining = stem[i:]

                # Pattern 1: root1 + o + root2 (linking vowel)
                if remaining.startswith('o') and len(remaining) > 2:
                    second_part = remaining[1:]  # Skip the linking 'o'
                    if first_part in KNOWN_ROOTS and second_part in KNOWN_ROOTS:
                        ast["radiko"] = second_part  # Head root is typically the second
                        ast["kunmetitaj_radikoj"] = [first_part, second_part]
                        compound_found = True
                        break

                # Pattern 2: root1 + root2 (no linking vowel)
                if first_part in KNOWN_ROOTS and remaining in KNOWN_ROOTS:
                    ast["radiko"] = remaining
                    ast["kunmetitaj_radikoj"] = [first_part, remaining]
                    compound_found = True
                    break

    # If no compound found, try single root
    if not compound_found and not ast["radiko"]:
        if stem in KNOWN_ROOTS:
            ast["radiko"] = stem
        elif stem in KNOWN_PARTICLES or stem in KNOWN_PREPOSITIONS:
            # It's a particle/preposition used as a root (e.g., "tre" in "treege")
            ast["radiko"] = stem
        else:
            raise ValueError(f"Ne povis trovi validan radikon en '{original_word}'. Restaĵo: '{stem}'")

    return ast

def categorize_unknown_word(word: str, error_msg: str = "") -> dict:
    """
    Categorize an unknown word that failed to parse.

    Returns an AST node marking the word as non-Esperanto with best-guess categorization.
    Categories:
    - proper_name_known: Known proper noun from dictionary (parse_status=success!)
    - proper_name: Capitalized word (person, place)
    - foreign_word: Lowercase but not Esperanto
    - number_literal: Numeric
    - unknown: Cannot categorize
    """
    ast = {
        "tipo": "vorto",
        "plena_vorto": word,
        "radiko": word,
        "vortspeco": "nekonata",
        "parse_status": "failed",
        "parse_error": error_msg,
        "category": "unknown",
        "nombro": "singularo",
        "kazo": "nominativo",
        "prefikso": None,
        "sufiksoj": [],
    }

    # Categorization heuristics

    # 1. Number literal (digits)
    if word.isdigit():
        ast["category"] = "number_literal"
        ast["vortspeco"] = "numero"
        return ast

    # 2. Proper name (starts with capital letter)
    if word[0].isupper() and len(word) > 1:
        # Check proper noun dictionary first
        from klareco.proper_nouns import get_proper_noun_dictionary
        pn_dict = get_proper_noun_dictionary()

        if pn_dict.is_proper_noun(word):
            # Known proper noun - mark as SUCCESS, not failed!
            ast["parse_status"] = "success"
            ast["category"] = "proper_name_known"
            ast["vortspeco"] = "propra_nomo"
            ast["parse_error"] = ""

            # Add metadata from dictionary
            metadata = pn_dict.get_metadata(word)
            if metadata:
                ast["proper_noun_category"] = metadata.get("category", "other")
                ast["proper_noun_frequency"] = metadata.get("frequency", 0)

            # Extract case/number from Esperanto endings
            if word.endswith(('o', 'on', 'oj', 'ojn', 'a', 'an', 'aj', 'ajn')):
                if word.endswith('n'):
                    ast["kazo"] = "akuzativo"
                if word.endswith(('j', 'jn')):
                    ast["nombro"] = "pluralo"

            return ast

        # Unknown proper name (not in dictionary)
        ast["category"] = "proper_name"
        ast["vortspeco"] = "propra_nomo"

        # Try to detect if it has Esperanto-like endings (might be Esperantized name)
        if word.endswith(('o', 'on', 'oj', 'ojn')):
            ast["category"] = "proper_name_esperantized"
            # Extract the case/number from ending
            if word.endswith('n'):
                ast["kazo"] = "akuzativo"
                word = word[:-1]
            if word.endswith('j'):
                ast["nombro"] = "pluralo"

        return ast

    # 3. Single letter (often grammar examples)
    if len(word) == 1:
        ast["category"] = "single_letter"
        ast["vortspeco"] = "ekzemplo"
        return ast

    # 4. Foreign word (lowercase, no Esperanto structure)
    # Has no recognizable Esperanto endings or morphology
    ast["category"] = "foreign_word"
    ast["vortspeco"] = "fremda_vorto"

    return ast


# -----------------------------------------------------------------------------
# --- Layer 2: Syntactic Analyzer (parse)
# -----------------------------------------------------------------------------

def preprocess_text(text: str) -> str:
    """
    Preprocess text before parsing to normalize punctuation and whitespace.

    - Converts CX-system (cx, gx, etc.) to Unicode (ĉ, ĝ, etc.)
    - Converts em-dashes, en-dashes to spaces
    - Normalizes smart quotes to straight quotes
    - Normalizes whitespace
    """
    # Normalize CX-system (ASCII representation) to Unicode
    # This is used in older Esperanto texts from Project Gutenberg
    cx_replacements = {
        'Cx': 'Ĉ', 'cx': 'ĉ', 'CX': 'Ĉ',
        'Gx': 'Ĝ', 'gx': 'ĝ', 'GX': 'Ĝ',
        'Hx': 'Ĥ', 'hx': 'ĥ', 'HX': 'Ĥ',
        'Jx': 'Ĵ', 'jx': 'ĵ', 'JX': 'Ĵ',
        'Sx': 'Ŝ', 'sx': 'ŝ', 'SX': 'Ŝ',
        'Ux': 'Ŭ', 'ux': 'ŭ', 'UX': 'Ŭ',
    }
    for old, new in cx_replacements.items():
        text = text.replace(old, new)

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

    # Simple tokenizer: split by space, remove all punctuation EXCEPT apostrophes for elision
    # Remove common punctuation marks: . , ! ? : ; " ( ) [ ] { }
    # Keep apostrophes attached to preceding letter for elision: l', hund'
    import string
    import re
    # First, preserve elision apostrophes by converting "letter'" to a safe form
    # Match: word character followed by apostrophe (straight or curly)
    text = re.sub(r"(\w)([''])", r"\1ELISION_MARKER", text)
    # Remove all punctuation
    for punct in string.punctuation:
        text = text.replace(punct, ' ')
    # Restore elision apostrophes
    text = text.replace("ELISION_MARKER", "'")
    words = text.split()

    if not words:
        raise ValueError("Ne povis analizi malplenan ĉenon.")

    # Step 1: Morphological analysis of all words
    # Gracefully handle unknown words by categorizing them
    word_asts = []
    for w in words:
        try:
            ast = parse_word(w)
            ast["parse_status"] = "success"  # Mark as successfully parsed Esperanto
            word_asts.append(ast)
        except ValueError as e:
            # Word failed to parse - categorize it as non-Esperanto
            unknown_ast = categorize_unknown_word(w, str(e))
            word_asts.append(unknown_ast)
        except Exception as e:
            # Unexpected error - still create a node
            unknown_ast = categorize_unknown_word(w, f"Unexpected error: {str(e)}")
            word_asts.append(unknown_ast)

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
    for i, ast in enumerate(word_asts):
        if ast["vortspeco"] == "verbo" and not sentence_ast["verbo"]:
            sentence_ast["verbo"] = ast
            # Check for negation: 'ne' immediately preceding the verb (Issue #78)
            # In Esperanto, 'ne' typically directly precedes the word it negates
            if i > 0 and word_asts[i-1].get("radiko") == "ne":
                ast["negita"] = True
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

    # --- Issue #87: Sentence type detection ---
    # Determine fraztipo (sentence type): demando, ordono, deklaro
    fraztipo = 'deklaro'  # default: statement
    demandotipo = None

    # Check for question indicators
    is_question = False

    # 1. Check if sentence ends with '?'
    if text.strip().endswith('?'):
        is_question = True

    # 2. Check for ĉu (yes/no question marker)
    for ast in word_asts:
        if ast.get("radiko") == "ĉu":
            is_question = True
            demandotipo = 'ĉu'
            break

    # 3. Check for ki- correlatives (question words: kio, kiu, kie, kiam, kiel, etc.)
    if not demandotipo:
        for ast in word_asts:
            if ast.get("vortspeco") == "korelativo":
                prefix = ast.get("korelativo_prefikso", "")
                if prefix == "ki":
                    is_question = True
                    demandotipo = 'ki'
                    break

    if is_question:
        fraztipo = 'demando'

    # Check for command (imperative mood)
    if not is_question and sentence_ast["verbo"]:
        if sentence_ast["verbo"].get("modo") == "imperativo":
            fraztipo = 'ordono'

    sentence_ast["fraztipo"] = fraztipo
    if demandotipo:
        sentence_ast["demandotipo"] = demandotipo

    # Add parse statistics (word-level success metrics)
    total_words = len(word_asts)
    successful_words = sum(1 for ast in word_asts if ast.get("parse_status") == "success")
    failed_words = total_words - successful_words

    # Categorize the failed words
    categories = {}
    for ast in word_asts:
        if ast.get("parse_status") == "failed":
            category = ast.get("category", "unknown")
            categories[category] = categories.get(category, 0) + 1

    sentence_ast["parse_statistics"] = {
        "total_words": total_words,
        "esperanto_words": successful_words,
        "non_esperanto_words": failed_words,
        "success_rate": successful_words / total_words if total_words > 0 else 0.0,
        "categories": categories
    }

    # Add sentence-level negation flag (Issue #78)
    # Check for explicit 'ne' or negative correlatives (neni- words)
    # Note: correlatives may be parsed as 'korelativo' or 'nomo' depending on context
    has_ne = any(ast.get("radiko", "").lower() == "ne" for ast in word_asts)
    has_negative_correlative = any(
        ast.get("radiko", "").lower().startswith("neni")
        for ast in word_asts
    )
    sentence_ast["negita"] = has_ne or has_negative_correlative

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
