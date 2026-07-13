"""A from-scratch, pure Python parser for Esperanto.

This parser is built on the 16 rules of Esperanto and does not use any
external parsing libraries like Lark. It performs morphological and syntactic
analysis to produce a detailed, Esperanto-native Abstract Syntax Tree (AST)."""
import re
import json
from functools import lru_cache
from pathlib import Path

# Dictionary roots — the language's own vocabulary. This is the artifact the
# parser's NEGATIVE DETECTION rests on: "capitalised AND the root is not a known
# Esperanto morpheme -> proper noun". With an EMPTY lexicon that rule fires on
# every ordinary word it has never heard of.
#
# ⚠️ THIS FILE WENT MISSING IN THE JUNE 2026 MIGRATION AND THE PARSER CARRIED ON.
# `merged_roots.json` was absent, `DICTIONARY_ROOTS` silently became `set()`, and
# the parser ran on 2,481 Fundamento roots — which do not contain `centr`,
# `list`, `muze` or `krom`. Consequence, measured on the live store: 41.8% of all
# 5.39M sentences were given a `propra_nomo` SUBJECT, 700,925 of them an ordinary
# Esperanto word (`Ĝia` x29,222, `Krome` x6,369, `Ambaŭ` x4,783). See #821.
#
# CLAUDE.md: "A silently-degrading dependency is a bug." This was one, in the
# parser's own first twenty lines. It now FAILS LOUDLY.
_VOCAB_DIR = Path(__file__).parent.parent / "data" / "vocabularies"

def _load_root_lexicon() -> set:
    # merged_roots.json is the legacy artifact (a bare list). root_vocab.json is
    # the rebuilt one (scripts/index/build_root_lexicon.py) and carries provenance.
    legacy = _VOCAB_DIR / "merged_roots.json"
    if legacy.exists():
        return set(json.loads(legacy.read_text(encoding="utf-8")))
    rebuilt = _VOCAB_DIR / "root_vocab.json"
    if rebuilt.exists():
        return set(json.loads(rebuilt.read_text(encoding="utf-8"))["roots"])
    raise FileNotFoundError(
        "The parser has NO ROOT LEXICON, and without one its proper-noun rule "
        "misfires on every ordinary word (see #821 — this cost 700,925 corrupted "
        "subjects).\n"
        f"  looked for: {legacy}\n"
        f"              {rebuilt}\n"
        "  build it:   python scripts/index/build_root_lexicon.py\n"
        "Refusing to parse with an empty lexicon rather than degrade silently."
    )

DICTIONARY_ROOTS = _load_root_lexicon()

# Fundamento roots (authoritative, tier-1 vocabulary). Used to disambiguate
# prefix/suffix conflicts. Also a REQUIRED artifact — it too used to fail silently
# (`except Exception: pass`).
_fundamento_path = _VOCAB_DIR / "fundamento_roots.json"
if not _fundamento_path.exists():
    raise FileNotFoundError(
        f"Fundamento roots missing: {_fundamento_path}\n"
        "This is the normative floor of the language. Refusing to degrade silently."
    )
# JSON structure: {"hund": {...}, "libr": {...}, ...}
_FUNDAMENTO_ROOTS = set(json.loads(_fundamento_path.read_text(encoding="utf-8")).keys())

# All function words (grammatical morphemes, not content words)
# Used for negative detection: these should NEVER be marked as proper nouns
# even if capitalized at sentence start
_ALL_FUNCTION_WORDS = set()

def _build_function_word_set():
    """Build comprehensive set of all Esperanto function words."""
    # This will be populated after the constants are defined below
    # We use a function to avoid forward reference issues
    function_words = set()

    # Pronouns
    function_words.update(KNOWN_PRONOUNS)

    # Conjunctions
    function_words.update(KNOWN_CONJUNCTIONS)

    # Prepositions
    function_words.update(KNOWN_PREPOSITIONS)

    # Particles
    function_words.update(KNOWN_PARTICLES)

    # Correlatives
    function_words.update(KNOWN_CORRELATIVES)

    # Article
    function_words.add("la")

    return function_words

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
    "dis",  # dispersal, separation
    "mis",  # wrongly, mis-
    "bo",   # in-law (bopatro = father-in-law)
    "fi",   # shameful, morally bad
    "vic",  # vice-, deputy
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
    "an",   # member of group/place (urbano, kristano)
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
    "ĉj",   # male affectionate diminutive (paĉjo from patro)
    "nj",   # female affectionate diminutive (panjo from patrino)
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

# =============================================================================
# Affectionate suffix root recovery (-ĉj, -nj)
# =============================================================================
# These suffixes truncate the root after the first vowel:
#   patro → pa + ĉj + o → paĉjo (daddy)
#   patrino → pa + nj + o → panjo (mommy)
#   frato → fra + ĉj + o → fraĉjo (bro)
#   Johano → Jo + ĉj + o → Joĉjo (Johnny)
#
# Since the truncated form loses information, we need a lookup table
# to recover the full root for proper semantic embedding.
# =============================================================================
AFFECTIONATE_ROOT_LOOKUP = {
    # -ĉj (male affectionate)
    "pa": "patr",       # paĉjo → patro (daddy)
    "fra": "frat",      # fraĉjo → frato (bro)
    "fi": "fil",        # fiĉjo → filo (sonny)
    "ne": "nev",        # neĉjo → nevo (nephew dear)
    "o": "onkl",        # oĉjo → onklo (uncle dear)
    "a": "av",          # aĉjo → avo (grandpa)
    "ku": "kuz",        # kuĉjo → kuzo (cousin dear)
    # Common names (male)
    "Jo": "Johan",      # Joĉjo → Johano (Johnny)
    "Pe": "Petr",       # Peĉjo → Petro (Pete)
    "Mi": "Miĥael",     # Miĉjo → Miĥaelo (Mike)
    "To": "Tomas",      # Toĉjo → Tomaso (Tommy)
    "Da": "David",      # Daĉjo → Davido (Davey)
    "Ma": "Mark",       # Maĉjo → Marko (Marky)
    "Ja": "Jakob",      # Jaĉjo → Jakobo (Jake)

    # -nj (female affectionate) - same truncated forms often
    # Note: Some overlap with -ĉj forms, context determines meaning
    # "pa" is used for both paĉjo (daddy) and panjo (mommy) but roots differ!
    # We handle this by checking the actual suffix used.
}

# Separate lookup for -nj since truncated "pa" maps to different roots
# NOTE: We recover just the BASE root, not root+in, because -nj already implies feminine
# The -in suffix is implicit in the -nj affectionate form
AFFECTIONATE_ROOT_LOOKUP_NJ = {
    "pa": "patr",       # panjo → patrino (mommy) - base root is patr
    "fra": "frat",      # franjo → fratino (sis) - base root is frat
    "fi": "fil",        # finjo → filino (daughter dear) - base root is fil
    "ne": "nev",        # nenjo → nevino (niece dear) - base root is nev
    "o": "onkl",        # onjo → onklino (auntie) - base root is onkl
    "a": "av",          # anjo → avino (grandma) - base root is av
    "ku": "kuz",        # kunjo → kuzino (cousin dear, female) - base root is kuz
    # Common names (female) - these don't have -in since they're proper nouns
    "Ma": "Mari",       # Manjo → Mario (Mary dear)
    "An": "An",         # Annjo → Anno (Annie)
    "Ka": "Katerin",    # Kanjo → Katerino (Katie)
    "So": "Sofi",       # Sonjo → Sofio (Sophie)
    "El": "Elizabet",   # Elnjo → Elizabeto (Lizzy)
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
    "anstataŭ", # instead of
    "malgraŭ",  # despite
    "ekde",     # since (compound preposition)
    "depost",   # since (compound preposition, formal)
    "spite",    # despite
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

# Generate compound numerals algorithmically: ducent (200), tricent (300), ..., dumil (2000), ...
_DIGIT_WORDS_FOR_COMPOUND = ["du", "tri", "kvar", "kvin", "ses", "sep", "ok", "naŭ"]
for _d in _DIGIT_WORDS_FOR_COMPOUND:
    KNOWN_NUMBERS.add(_d + "cent")   # ducent, tricent, kvarcent, kvincent, sescent, sepcent, okcent, naŭcent
    KNOWN_NUMBERS.add(_d + "mil")    # dumil, trimil, kvarmil, ...
del _DIGIT_WORDS_FOR_COMPOUND, _d

# Simple (non-compound) numeral roots — these CAN take grammatical endings (a/e/o/i)
# to form adjectives/adverbs/nouns and must NOT be caught by the inflected-compound check.
_BASIC_NUMERAL_ROOTS = frozenset({
    "unu", "du", "tri", "kvar", "kvin", "ses", "sep", "ok", "naŭ",
    "dek", "cent", "mil", "nul",
})

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
# --- Protected Roots: Fundamento roots that look like they contain affixes
# --- These must NEVER be decomposed - they are atomic roots.
# --- Loaded from data/vocabularies/protected_roots.json for maintainability
# -----------------------------------------------------------------------------

# Load protected roots from JSON file (single source of truth)
# Word classes that mean "we did NOT successfully analyse this as an Esperanto
# word". `nekonata` = we could not identify it; `fremda_vorto` = we identified it
# as foreign. Counting either as a parse success is how sukcesoprocento came to
# report 1.0 on "Xyzzy plugh frobnicate." and on English. See #818.
NON_ESPERANTO_VORTSPECOJ = frozenset({'nekonata', 'fremda_vorto'})


PROTECTED_PREFIX_ROOTS = set()
PROTECTED_SUFFIX_ROOTS = set()

_protected_roots_path = Path(__file__).parent.parent / "data" / "vocabularies" / "protected_roots.json"
if _protected_roots_path.exists():
    with open(_protected_roots_path, 'r', encoding='utf-8') as f:
        _protected_data = json.load(f)
    # v2 schema (scripts/index/build_surface_lexical_facts.py): a flat `roots`
    # list derived from DERIVATIONAL PRODUCTIVITY over RAW SURFACE TEXT. A stem
    # that takes many distinct derivational tails has LEXICALIZED and must not be
    # split — `esperant` (esperant-ist-o, esperant-uj-o, …), `milit` (NOT mil+it),
    # `regul` (NOT reg+ul). This is a USAGE fact, so no grammar rule recovers it.
    PROTECTED_SUFFIX_ROOTS.update(_protected_data.get('roots', []))
    # v1 schema (hand-grouped by the affix they must not be split on).
    for _prefix, _roots in _protected_data.get('prefix_protected', {}).items():
        PROTECTED_PREFIX_ROOTS.update(_roots)
    for _suffix, _roots in _protected_data.get('suffix_protected', {}).items():
        PROTECTED_SUFFIX_ROOTS.update(_roots)

# ReVo-derived protections (scripts/index/build_root_lexicon.py v2).
#
# **ReVo says X is a root => X is ATOMIC => never split X.**
#
# This is what neutralises the laundering loop (#806) WITHOUT throwing away the
# corpus tier. A laundered root like `org` is harmless sitting in the lexicon —
# it is harmful only when it lets `organo` become org+an. So we do not remove it;
# we make the dictionary's reading WIN. `organ`, `banan`, `milit`, `regul` are
# ReVo headwords and therefore protected. `amerikan` and `kristan` are NOT in
# ReVo, so `amerikano` correctly stays amerik+an.
_rv = _VOCAB_DIR / 'root_vocab.json'
if _rv.exists():
    PROTECTED_SUFFIX_ROOTS.update(
        json.loads(_rv.read_text(encoding='utf-8')).get('protected', []))

# Combined set for fast lookup
PROTECTED_ROOTS = PROTECTED_PREFIX_ROOTS | PROTECTED_SUFFIX_ROOTS

# ---------------------------------------------------------------------------
# CAPITALIZATION RATIO — namehood as a USAGE statistic.
#
# P(capitalised MID-SENTENCE | word type), counted over raw surface text with
# sentence-initial tokens excluded (every sentence starts with a capital, so that
# position carries no information). Names are capitalised mid-sentence almost
# always; common nouns almost never:
#
#     ruslando 1.000   zamenhof 0.996   petro 0.986   esperanto 0.956
#     libro    0.150   hundo    0.116   urbo  0.025
#
# This is the ONLY signal that reaches the residue. `Petro` decomposes to petr+o
# ("rock"): morphology says ordinary word, syntax says ordinary word. Only USAGE
# says name — and usage is countable.
#
# ⚠️ It is a MEMOIZATION OF USAGE, not world knowledge, and it generalises to ZERO
# unseen tokens. The morphological rules still carry those. So it shrinks the
# residue, it does not abolish it. See docs/PROPER_NOUNS.md and #819.
#
# DEGRADING (not REQUIRED): absent -> empty -> the rule simply never fires, and
# the parser falls back to morphology. Build it with:
#     ./scripts/index/build_surface_lexical_facts.sh
CAPITALIZATION_RATIO: dict[str, float] = {}
_cap_ratio_path = Path(__file__).parent.parent / "data" / "vocabularies" / "capitalization_ratio.json"
if _cap_ratio_path.exists():
    CAPITALIZATION_RATIO = json.loads(
        _cap_ratio_path.read_text(encoding='utf-8')).get('types', {})

# The corpus separates names from common nouns with a wide, empty gap (see the
# anchors above), so these thresholds are not finely tuned — anything in
# [0.5, 0.9] and [0.2, 0.4] gives the same partition on the anchors.
_CAP_RATIO_NAME = 0.85      # at or above: behaves like a NAME
_CAP_RATIO_COMMON = 0.30    # at or below: behaves like a COMMON word


def _usage_says_name(surface: str) -> bool | None:
    """Does the corpus's own usage call this type a name? None = it has no opinion."""
    r = CAPITALIZATION_RATIO.get((surface or '').lower())
    if r is None:
        return None
    if r >= _CAP_RATIO_NAME:
        return True
    if r <= _CAP_RATIO_COMMON:
        return False
    return None

# Common Esperanto abbreviations — matched before any morphological analysis
_KNOWN_ABBREVIATIONS: dict[str, str] = {
    "s-ro":   "Sinjoro",        # Mr.
    "s-ino":  "Sinjorino",      # Mrs./Ms.
    "d-ro":   "doktoro",        # Dr. (male)
    "d-ra":   "doktorino",      # Dr. (female)
    "n-ro":   "numero",         # No. (number)
    "k.t.p.": "kaj tiel plu",   # etc.
    "ktp":    "kaj tiel plu",   # etc. (no-dot form)
    "k.a.":   "kaj aliaj",      # et al.
    "prof.":  "profesoro",      # Prof.
    "kp.":    "komparu",        # cf. / compare
}

# Phonological validity check for Esperanto roots (for neologism acceptance)
_EO_VOWELS = frozenset("aeiou")
_EO_VALID_CHARS = frozenset("abcĉdefgĝhĥijĵklmnoprstŭvzŝ") | _EO_VOWELS


# Rules 2-7: every CONTENT word carries a grammatical ending. This is the test
# that separates a ROOT from a WORD — and the parser did not have it.
#
# `sam` is a root (Fundamento: `sama` = "same"). `Sam` is not a word: it carries
# no ending. Negative detection asked only "is the ROOT known?", answered yes,
# and so refused to call `Sam` a name — leaving it `nekonata` forever. Same for
# `Peter` (`pet` + `er`: a suffix is not an ending).
#
# The closed ending-less class (la, kaj, mi, tiu, ĉar, ankaŭ, …) is the grammar's
# own named exception and is handled separately via _ALL_FUNCTION_WORDS.
_GRAMMATICAL_ENDINGS = (
    'ojn', 'oj', 'on', 'o',        # substantivo
    'ajn', 'aj', 'an', 'a',        # adjektivo
    'en', 'e',                     # adverbo
    'as', 'is', 'os', 'us', 'i', 'u',   # verbo
)


def _has_grammatical_ending(surface: str) -> bool:
    """Does the surface form carry one of the grammatical endings (Rules 2-7)?

    A root without an ending is not a word. This is what makes "capitalised AND
    not a well-formed Esperanto word -> proper noun" decidable WITHOUT any list
    of names.
    """
    s = (surface or '').lower()
    return any(s.endswith(e) for e in _GRAMMATICAL_ENDINGS)


_CONTENT_VORTSPECOJ = ('substantivo', 'adjektivo', 'adverbo', 'verbo', 'nekonata')

# Only the EVIDENTIAL rules may be vetoed by a usage count. The DEDUCTIVE ones
# (no_valid_ending, foreign_e_ending, adjective_unlicensed) are grammar, and a
# frequency statistic does not get to overrule grammar.
_USAGE_VETOABLE = ('mid_sentence_capitalization', 'preceded_by_la',
                   'morphology_no_decomposition')


def _apply_capitalization_ratio(word_asts: list) -> None:
    """Promote a capitalised token to `propra_nomo` when the corpus's OWN USAGE
    says so, even though it decomposed to a perfectly good Esperanto word.

    This is the residue rule (#819). `Petro` = petr-o = "rock"; `Esperanto` =
    esper-ant-o; `Ruslando` = rus-land-o. Morphology and syntax both correctly
    analyse them as ordinary words, because *as words* that is what they are.
    Lexicalization is a fact about USAGE, and usage is the one thing the corpus
    can count.

    Deliberately conservative:
      * only CAPITALISED tokens are considered
      * only tokens the corpus has a STRONG opinion about (>= 0.85) are promoted
      * function words and correlatives are never touched
      * silent on unseen types — morphology still carries those, which is why
        this MEMOIZES usage rather than replacing the rules
    """
    if not CAPITALIZATION_RATIO:
        return
    for ast in word_asts:
        if not isinstance(ast, dict):
            continue
        pv = ast.get('plena_vorto') or ''
        if not pv[:1].isupper() or pv.isupper():
            continue                                  # ALL-CAPS carries no signal
        if ast.get('vortspeco') not in _CONTENT_VORTSPECOJ:
            continue                                  # never touch function words
        if pv.lower() in _ALL_FUNCTION_WORDS:
            continue
        if _usage_says_name(pv) is True:
            ast['vortspeco'] = 'propra_nomo'
            ast['kategorio'] = 'propranomo'
            ast['radiko'] = pv
            ast['prefiksoj'] = []
            ast['sufiksoj'] = []
            ast['propra_nomo_evidence'] = 'capitalization_ratio'


def _veto_by_usage(word_asts: list) -> None:
    """Usage also says NO — and that is where the precision was hiding.

    The weak rules (`mid_sentence_capitalization`, `preceded_by_la`,
    `morphology_no_decomposition`) fire on capitalisation and absence-of-evidence.
    When the corpus has a STRONG opinion that a type behaves like a COMMON word
    (`urbo` 0.025, `hundo` 0.116), that opinion beats them.

    Measured: Prago precision 44.3% -> 49.1%, F1 61.4% -> 65.9%, RECALL UNCHANGED
    at 100%. Cairo unchanged (already clean). Pure false-positive removal.

    It does NOT veto the deductive rules — `no_valid_ending`, `foreign_e_ending`
    and `adjective_unlicensed` are grammar, and a frequency count does not
    overrule grammar.
    """
    if not CAPITALIZATION_RATIO:
        return
    for ast in word_asts:
        if not isinstance(ast, dict):
            continue
        if ast.get('vortspeco') != 'propra_nomo':
            continue
        if ast.get('propra_nomo_evidence') not in _USAGE_VETOABLE:
            continue
        if _usage_says_name(ast.get('plena_vorto')) is False:
            ast['vortspeco'] = 'substantivo'
            ast['kategorio'] = None
            ast['propra_nomo_evidence'] = None


def _is_valid_eo_stem(s: str) -> bool:
    """Return True if s is phonologically valid as an Esperanto root.

    Criteria: at least one vowel, only Esperanto characters, minimum length 2.
    Used to accept neologisms whose roots are not yet in any vocabulary.
    """
    if len(s) < 2:
        return False
    s_lower = s.lower()
    return any(c in _EO_VOWELS for c in s_lower) and all(c in _EO_VALID_CHARS for c in s_lower)


# -----------------------------------------------------------------------------
# --- Layer 1: Morphological Analyzer (Fundamento-first design)
# --- See wiki: Esperanto-Parser-Design.md for architecture
# -----------------------------------------------------------------------------

# Build comprehensive function word set (now that all constants are defined)
_ALL_FUNCTION_WORDS = _build_function_word_set()


# Foreign orthography signals: characters and digraphs that aren't Esperanto.
# Esperanto uses ŝ/ĉ/ĝ/ĵ/ŭ for sounds spelled "sh"/"ch"/"j"/etc. in foreign
# alphabets. Words containing these patterns are very likely foreign / propra_nomo.
_FOREIGN_LETTERS = frozenset('qwxy')
_FOREIGN_DIGRAPHS = ('sh', 'ch', 'th', 'ph')


# Compounding-prefix whitelist for science / international vocabulary not in
# Fundamento as bare roots. These are unambiguous: when a compound starts with
# one, it's an Esperanto/Greek-Latin coinage, not a foreign name. Used by
# _is_genuine_esperanto_compound to accept e.g. "mikronaci" = mikro + naci.
EXTENDED_PREFIXES = frozenset({
    'mikro', 'makro', 'meta', 'pseŭdo', 'arki', 'tele', 'multi',
    'omni', 'nano', 'mez', 'kvazaŭ', 'ekster',
})


def _has_foreign_orthography(word: str) -> bool:
    """True if `word` contains characters or digraphs not in Esperanto orthography."""
    if not word:
        return False
    lower = word.lower()
    if any(c in _FOREIGN_LETTERS for c in lower):
        return True
    if any(d in lower for d in _FOREIGN_DIGRAPHS):
        return True
    return False


# Surface-form ending constants (used by validation and post-processing).
# These are inflectional endings, not radikos. Adjective endings are
# unambiguously plural / accusative for the marked forms (-aj/-an/-ajn);
# bare -a is more ambiguous (real adjectives + names like Maria/Mona/Lisa).
_ADJ_SURFACE_ENDINGS = ('ajn', 'aj', 'an', 'a')
_NOUN_SURFACE_ENDINGS = ('ojn', 'oj', 'on', 'o')


def _surface_looks_adj(pv: str) -> bool:
    return bool(pv) and any(pv.endswith(e) for e in _ADJ_SURFACE_ENDINGS)


def _surface_looks_noun(pv: str) -> bool:
    return bool(pv) and any(pv.endswith(e) for e in _NOUN_SURFACE_ENDINGS)


def _is_genuine_esperanto_compound(stem: str) -> bool:
    """Decompose `stem` (lowercase, no inflectional ending) and verify both
    halves are recognized Esperanto morphemes. Used to discriminate genuine
    compounds (sonfilm = son + film) from foreign names whose tail syllables
    coincidentally match Fundamento roots (Madrid → mad + rid is rejected
    because 'mad' / 'rid' aren't recognized; sonfilm passes because 'son' and
    'film' are both in Fundamento).

    Three acceptance paths:

      Path A (binary-split, conservative):
        - second-half in _FUNDAMENTO_ROOTS, AND
        - first-half is in KNOWN_PREFIXES, EXTENDED_PREFIXES, or _FUNDAMENTO_ROOTS
          (with optional trailing linking-o stripped from first-half).

      Path B (suffix-base, productive):
        - stem ends with a known Esperanto suffix (-ist, -ej, -ec, -ad, ...), AND
        - the base before that suffix is in DICTIONARY_ROOTS.
        Catches productive derivations like "atomist" (atom + ist) where atom
        isn't in the small Fundamento but IS in the broad dictionary.

      Path C (extended-prefix, productive):
        - first-half is in EXTENDED_PREFIXES, AND
        - second-half is in DICTIONARY_ROOTS (and ≥3 chars, no foreign orthography).
        Catches scientific compounds like "mikronaci" / "tele(skop)" where the
        first-half is unambiguously an Esperanto/Greek-Latin combining form.

    DICTIONARY_ROOTS is used in Paths B and C only when there's a strong
    Esperanto-morphology signal (known suffix or extended prefix) — that
    signal compensates for the dictionary's foreign-name pollution. Path A
    stays Fundamento-only for the cases without that signal.
    """
    if not stem or len(stem) < 4:
        return False
    s = stem.lower()

    # Path B: suffix-base. Try longest suffixes first so we don't mistakenly
    # match a shorter suffix when a longer one applies (e.g. 'estr' before 'er').
    for suffix in sorted(KNOWN_SUFFIXES, key=len, reverse=True):
        if not s.endswith(suffix):
            continue
        base = s[: -len(suffix)]
        if len(base) < 3:
            continue
        if base in DICTIONARY_ROOTS and not _has_foreign_orthography(base):
            return True

    # Paths A and C: binary split.
    for i in range(2, len(s) - 1):
        first = s[:i]
        second = s[i:]
        # Path A: second-half is Fundamento.
        if second in _FUNDAMENTO_ROOTS:
            if (first in KNOWN_PREFIXES
                    or first in EXTENDED_PREFIXES
                    or first in _FUNDAMENTO_ROOTS):
                return True
            # Linking-o between two roots: first ends in 'o', stripping it
            # yields a Fundamento root. (membr+o+ŝtat → membroŝtato)
            if (first.endswith('o') and len(first) > 2
                    and first[:-1] in _FUNDAMENTO_ROOTS):
                return True
        # Path C: extended-prefix + DICTIONARY_ROOTS second-half.
        if first in EXTENDED_PREFIXES and len(second) >= 3:
            if second in DICTIONARY_ROOTS and not _has_foreign_orthography(second):
                return True
    return False

def parse_word(word: str) -> dict:
    """
    Parse a single Esperanto word using Fundamento-first design.

    Architecture (from Esperanto-Parser-Design.md):
    1. Function word check (closed list)
    2. Correlative check (45 entries)
    3. Strip grammatical ending (-o, -a, -e, -i, -as, -is, -os, -us, -u, -j, -n)
    4. FUNDAMENTO ROOT CHECK (critical) - if match, STOP
    5. Prefix extraction (with Fundamento guard)
    6. Suffix extraction (with Fundamento guard)
    7. Compound word check (last resort)

    Key insight: Esperanto grammar is 100% deterministic. The ONLY complexity
    is vocabulary-level ambiguity (roots that look like they contain affixes).
    This is solved by checking Fundamento roots FIRST.
    """
    original_word = word
    lower_word = word.lower()

    # ALL-CAPS normalization. In running Esperanto text an all-uppercase
    # token is typographic emphasis / a header (DEMOKRATIO, EDUKADO,
    # RAJTOJ, DE) — NOT a proper-noun signal; real proper nouns are
    # Title-Case (Zamenhof, Parizo). The capitalization-based
    # proper-noun routing keys on word[0].isupper(), which fires on
    # ALL-CAPS and (via mid-sentence reanalysis) flips these common
    # words to propra_nomo — the dominant proper-noun false-positive
    # class on UD-Prago (16/49). If an all-caps token's stem is a known
    # Esperanto root, classify it as the common word it is. Genuine
    # all-caps acronyms / foreign names have no Esperanto root and fall
    # through unchanged, so existing proper-noun logic still applies.
    if len(word) > 1 and word.isupper() and word.isalpha():
        _lw = word.lower()
        # Use the real morphology engine, not a stem guess: recursively
        # parse the lowercased form (terminates — _lw is lowercase so
        # this branch can't re-trigger). If it analyses as a genuine
        # Esperanto content word, that IS the correct reading; emphasis
        # capitalization is irrelevant. Only genuine acronyms / foreign
        # names fail to analyse and keep proper-noun treatment.
        _lc = parse_word(_lw)
        if _lc.get('vortspeco') not in (
                'propra_nomo', 'nekonata', 'fremda_vorto', None):
            return _lc

    # --- Initialize AST ---
    ast = {
        "tipo": "vorto",
        "plena_vorto": original_word,
        "radiko": None,
        "vortspeco": "nekonata",
        "nombro": "singularo",
        "kazo": "nominativo",
        "prefiksoj": [],
        "sufiksoj": [],
    }

    # ==========================================================================
    # STEP -1: Abbreviation check (before hyphen split so S-ro is not torn apart)
    # ==========================================================================
    if lower_word in _KNOWN_ABBREVIATIONS:
        ast["radiko"] = lower_word
        ast["vortspeco"] = "mallongigo"
        ast["ekspansiita"] = _KNOWN_ABBREVIATIONS[lower_word]
        ast["estas_mallongigo"] = True
        return ast

    # ==========================================================================
    # STEP 0: Hyphenated Compound Word Check
    # ==========================================================================
    # Esperanto compound words with hyphens: Esperanto-klubo, hundo-domo
    # The RIGHTMOST component is the HEAD, left components are MODIFIERS
    # Example: "Esperanto-klubon" → HEAD=klub, MODIFIER=esperant
    if '-' in word:
        parts = word.split('-')
        if len(parts) >= 2:
            # Parse each component
            modifier_parts = parts[:-1]  # All except last
            head_part = parts[-1]  # Last part is HEAD

            # Parse the HEAD (determines grammatical properties)
            head_ast = parse_word(head_part)

            # Parse modifiers
            modifier_asts = []
            for mod_part in modifier_parts:
                # Modifiers are typically bare roots (no endings) or with -o
                mod_ast = parse_word(mod_part)
                modifier_asts.append(mod_ast)

            # Build compound word AST
            # The HEAD's grammatical properties apply to the whole compound
            ast = head_ast.copy()
            ast["plena_vorto"] = original_word
            ast["kunmetajhoj"] = modifier_asts  # Store modifier ASTs
            ast["estas_kunmetita"] = True  # Flag as compound word

            # The radiko remains the HEAD's root
            # Modifiers are stored separately for retrieval
            return ast

    # ==========================================================================
    # STEP 1: Function Word Check (closed lists - no morphology)
    # ==========================================================================

    # Handle numeric literals
    if word.isdigit():
        ast["vortspeco"] = "numero"
        ast["radiko"] = word
        return ast

    # Article "la" - the only article
    if lower_word == "la":
        ast["vortspeco"] = "artikolo"
        ast["radiko"] = "la"
        return ast

    # Check pronouns (can take -n accusative)
    pronoun_check = lower_word
    if pronoun_check.endswith("n"):
        pronoun_check = pronoun_check[:-1]
    if pronoun_check in KNOWN_PRONOUNS:
        ast["radiko"] = pronoun_check
        ast["vortspeco"] = "pronomo"
        if lower_word.endswith("n"):
            ast["kazo"] = "akuzativo"
        return ast

    # Conjunctions - uninflected
    if lower_word in KNOWN_CONJUNCTIONS:
        ast["vortspeco"] = "konjunkcio"
        ast["radiko"] = lower_word
        return ast

    # Prepositions - uninflected
    if lower_word in KNOWN_PREPOSITIONS:
        ast["vortspeco"] = "prepozicio"
        ast["radiko"] = lower_word
        return ast

    # Particles - uninflected adverbs and modifiers
    if lower_word in KNOWN_PARTICLES:
        ast["vortspeco"] = "partiklo"
        ast["radiko"] = lower_word
        return ast

    # Number words - check before stripping endings
    # Numbers can be inflected: dua (second), duaj, duan, etc.
    temp_num = lower_word
    if temp_num.endswith("n"):
        temp_num = temp_num[:-1]
    if temp_num.endswith("j"):
        temp_num = temp_num[:-1]
    if temp_num.endswith(("a", "e", "o")):
        temp_num = temp_num[:-1]
    if temp_num in KNOWN_NUMBERS or lower_word in KNOWN_NUMBERS:
        # If bare number word, return immediately
        if lower_word in KNOWN_NUMBERS:
            ast["radiko"] = lower_word
            ast["vortspeco"] = "numero"
            return ast
        # Inflected compound numeral: ducentoj → ducent + oj
        # Exclude basic digit roots (du, tri, ...) — they take grammatical endings
        # to form adjectives/adverbs/nouns (dua, trio, duoj) and must fall through.
        if temp_num in KNOWN_NUMBERS and temp_num not in _BASIC_NUMERAL_ROOTS:
            ast["radiko"] = temp_num
            ast["vortspeco"] = "numero"
            suffix_part = lower_word[len(temp_num):]
            if "n" in suffix_part:
                ast["kazo"] = "akuzativo"
            if "j" in suffix_part:
                ast["nombro"] = "pluralo"
            return ast
        # Otherwise continue with regular parsing for inflected numbers

    # ==========================================================================
    # STEP 2: Correlative Check (45 entries × inflectional forms)
    # ==========================================================================
    # Strip plural -j and accusative -n in either order (-jn is plural+acc)
    # before checking against the closed list of 45 correlative bases.
    # Without this, "tiuj"/"ĉiuj"/"iujn" etc. fall through to morphology
    # and end up misclassified (often as verbs, since the parser treats
    # the leftover stem as a verb root candidate).

    correl_check = lower_word
    correl_accusative = False
    correl_plural = False
    if correl_check.endswith("jn"):
        correl_check = correl_check[:-2]
        correl_accusative = True
        correl_plural = True
    elif correl_check.endswith("n") and len(correl_check) > 1:
        correl_check = correl_check[:-1]
        correl_accusative = True
    elif correl_check.endswith("j") and len(correl_check) > 1:
        correl_check = correl_check[:-1]
        correl_plural = True

    if correl_check in KNOWN_CORRELATIVES:
        ast["vortspeco"] = "korelativo"
        ast["radiko"] = correl_check
        if correl_accusative:
            ast["kazo"] = "akuzativo"
        if correl_plural:
            ast["nombro"] = "pluralo"

        # Decompose correlative into prefix + suffix
        for prefix in sorted(CORRELATIVE_PREFIXES.keys(), key=len, reverse=True):
            if correl_check.startswith(prefix):
                suffix = correl_check[len(prefix):]
                if suffix in CORRELATIVE_SUFFIXES:
                    ast["korelativo_prefikso"] = prefix
                    ast["korelativo_sufikso"] = suffix
                    ast["korelativo_signifo"] = CORRELATIVE_PREFIXES[prefix]
                    break
        return ast

    # ==========================================================================
    # STEP 3: Strip Grammatical Ending (-o, -a, -e, -i, -as, -is, -os, -us, -u)
    # Also strip -j (plural) and -n (accusative)
    # ==========================================================================

    remaining = lower_word

    # Handle elision (Rule 16): l' = la, hund' = hundo
    if remaining.endswith(("'", "'")):
        remaining = remaining.rstrip("'").rstrip("'")
        ast["elidita"] = True

        # Special case: l' is the elided article "la"
        if remaining == "l":
            ast["vortspeco"] = "artikolo"
            ast["radiko"] = "la"
            return ast

        # For elided nouns, the ending is implicitly -o
        ast["vortspeco"] = "substantivo"
        # Continue to check if this is a valid stem

    # Strip accusative (-n) first (rightmost)
    if remaining.endswith("n") and len(remaining) > 2:
        ast["kazo"] = "akuzativo"
        remaining = remaining[:-1]

    # Strip plural (-j)
    if remaining.endswith("j") and len(remaining) > 2:
        ast["nombro"] = "pluralo"
        remaining = remaining[:-1]

    # Strip grammatical ending - try longest first
    ending_info = {}
    for ending in ["as", "is", "os", "us"]:  # 2-char verb endings first
        if remaining.endswith(ending) and len(remaining) > len(ending) + 1:
            ending_info = KNOWN_ENDINGS[ending].copy()
            remaining = remaining[:-len(ending)]
            break
    else:
        for ending in ["u", "i", "o", "a", "e"]:  # 1-char endings
            if remaining.endswith(ending) and len(remaining) > len(ending):
                ending_info = KNOWN_ENDINGS[ending].copy()
                remaining = remaining[:-len(ending)]
                break

    if ending_info:
        ast.update(ending_info)

    stem = remaining

    # ==========================================================================
    # STEP 4: FUNDAMENTO ROOT CHECK (CRITICAL!)
    # If the stem is a Fundamento root AND cannot be prefix + another Fundamento root,
    # it's ATOMIC - do NOT decompose!
    # ==========================================================================

    sorted_prefixes = sorted(KNOWN_PREFIXES, key=len, reverse=True)
    sorted_suffixes = sorted(KNOWN_SUFFIXES, key=len, reverse=True)

    def find_fundamento_root(s: str) -> str | None:
        """Find if s or s-minus-suffixes is a Fundamento/protected root."""
        # Direct match
        if s in _FUNDAMENTO_ROOTS or s in PROTECTED_ROOTS:
            return s
        # Try stripping suffixes
        temp = s
        for _ in range(3):  # Max 3 suffix layers
            found = False
            for suffix in sorted_suffixes:
                if temp.endswith(suffix) and len(temp) > len(suffix) + 1:
                    potential = temp[:-len(suffix)]
                    if potential in _FUNDAMENTO_ROOTS or potential in PROTECTED_ROOTS:
                        return potential
                    if potential in KNOWN_ROOTS:
                        temp = potential
                        found = True
                        break
            if not found:
                break
        return None

    def check_prefix_gives_fundamento(s: str) -> tuple[str, str] | None:
        """Check if s = prefix(es) + Fundamento root.

        Returns (first_prefix, ultimate_fundamento_root) or None.
        Handles double-prefix cases (e.g. malrefar → mal + re + far).
        """
        for prefix in sorted_prefixes:
            if s.startswith(prefix) and len(s) > len(prefix):
                remainder = s[len(prefix):]
                # Check if remainder is directly a Fundamento root
                if remainder in _FUNDAMENTO_ROOTS:
                    return (prefix, remainder)
                # Check if remainder minus suffixes leads to Fundamento
                fund = find_fundamento_root(remainder)
                if fund:
                    return (prefix, fund)
                # Double-prefix: remainder itself starts with a prefix + Fundamento root
                for prefix2 in sorted_prefixes:
                    if remainder.startswith(prefix2) and len(remainder) > len(prefix2):
                        remainder2 = remainder[len(prefix2):]
                        if remainder2 in _FUNDAMENTO_ROOTS:
                            return (prefix, remainder2)
                        fund2 = find_fundamento_root(remainder2)
                        if fund2:
                            return (prefix, fund2)
        return None

    def check_suffix_gives_fundamento(s: str) -> tuple[str, list[str]] | None:
        """Check if stripping suffixes from s leads to a known root.

        Returns (root, [suffix1, suffix2, ...]) or None.
        Suffixes are returned in extraction order (right-to-left).
        """
        temp = s
        extracted = []
        for _ in range(3):  # Max 3 suffix layers
            for suffix in sorted_suffixes:
                if temp.endswith(suffix) and len(temp) > len(suffix) + 1:
                    potential = temp[:-len(suffix)]
                    # Stop when we reach a Fundamento or protected root.
                    # KNOWN_ROOTS is intentionally excluded: it contains corpus-extracted stems
                    # (e.g. "bopatr", "bof") that would cause false early stops.
                    # Protected roots handle the key case: "esperant" stops "esperant+an" decomposition.
                    if potential in _FUNDAMENTO_ROOTS or potential in PROTECTED_ROOTS:
                        extracted.append(suffix)
                        return (potential, extracted)
                    # Continue stripping if potential is not a valid root yet
                    extracted.append(suffix)
                    temp = potential
                    break
            else:
                # No suffix matched at this layer
                break
        return None

    # Protected roots: if stem is in PROTECTED_ROOTS, keep it atomic
    if stem in PROTECTED_ROOTS:
        ast["radiko"] = stem
        return ast

    # Check if stem is Fundamento root
    stem_is_fundamento = stem in _FUNDAMENTO_ROOTS

    # Check if stem could be prefix + Fundamento
    prefix_parse = check_prefix_gives_fundamento(stem)

    # Check if stem could be Fundamento + suffix
    suffix_parse = check_suffix_gives_fundamento(stem)

    # Highly productive prefixes that should be preferred when ambiguous
    PRODUCTIVE_PREFIXES = {"mal", "re", "ne", "ek", "eks", "dis", "mis"}

    # Disambiguation logic - Order of priority:
    # 1. If stem is Fundamento and NO affix parses exist → keep atomic
    # 1b. If stem is Fundamento AND suffix parse exists BUT stem is LONGER → keep atomic
    #     Example: "esperant" (language) vs "esper+ant" (one who hopes)
    #     Prefer the longer Fundamento root "esperant"
    # 2. If ONLY suffix parse exists → skip prefix extraction (do suffix)
    # 3. If ONLY prefix parse exists → do prefix extraction
    # 4. If BOTH exist:
    #    a. If prefix is highly productive (re-, mal-) → prefer prefix (re+leg > rel+eg)
    #    b. Otherwise → prefer suffix (bon+eg > bo+neg)
    # 5. If stem is Fundamento AND prefix parse exists → prefer prefix (re+leg > rel)

    skip_prefix = False
    if stem_is_fundamento and not prefix_parse and not suffix_parse:
        # Stem is Fundamento and no affix parse exists - keep atomic
        ast["radiko"] = stem
        return ast
    elif stem_is_fundamento and suffix_parse and not prefix_parse:
        # CRITICAL: stem is Fundamento AND suffix parse exists
        # Prefer the LONGER Fundamento root
        # Example: "esperant" vs "esper+ant" → prefer "esperant"
        suffix_root = suffix_parse[0]  # e.g., "esper"
        if len(stem) > len(suffix_root):
            # stem is longer → keep atomic (e.g., "esperant" > "esper")
            ast["radiko"] = stem
            return ast
        # Otherwise fall through to suffix extraction
    elif suffix_parse and not prefix_parse:
        # Only suffix parse exists - skip prefix extraction
        skip_prefix = True
    elif suffix_parse and prefix_parse:
        # BOTH exist - check if prefix is highly productive
        # prefix_parse = (prefix, root)
        extracted_prefix = prefix_parse[0]
        if extracted_prefix in PRODUCTIVE_PREFIXES:
            # Highly productive prefix - prefer prefix parse
            # Example: releg = re+leg (not rel+eg)
            skip_prefix = False
        else:
            # Less common prefix - prefer suffix parse
            # Example: boneg = bon+eg (not bo+neg)
            skip_prefix = True
    elif not stem_is_fundamento and prefix_parse:
        # Stem is NOT Fundamento, only prefix parse exists - do prefix extraction
        skip_prefix = False
    elif stem_is_fundamento and prefix_parse:
        # AMBIGUOUS: stem is Fundamento AND prefix parse exists
        # Example: releg = re+leg (not rel)
        # Prefer the prefix parse (more compositional)
        skip_prefix = False

    extracted_prefixes = []

    if not skip_prefix:
        max_prefix_depth = 3
        for _ in range(max_prefix_depth):
            # STOP if stem is now a Fundamento/protected root
            if stem in _FUNDAMENTO_ROOTS or stem in PROTECTED_ROOTS:
                break

            # STOP if stem can be cleanly decomposed as fundamento_root + suffix,
            # UNLESS a prefix+Fundamento decomposition also exists (which takes priority).
            # Without this guard, "bonec" after "mal" → incorrectly "bo"+"nec" (corpus root).
            # Exception: "disig" has dis+ig (both Fundamento) — prefix should win.
            _local_suf = check_suffix_gives_fundamento(stem)
            if _local_suf and (_local_suf[0] in _FUNDAMENTO_ROOTS or _local_suf[0] in PROTECTED_ROOTS):
                if not check_prefix_gives_fundamento(stem):
                    break

            found_prefix = False
            for prefix in sorted_prefixes:
                if stem.startswith(prefix) and len(stem) > len(prefix):
                    remainder = stem[len(prefix):]

                    # Check if remainder leads to a Fundamento root (directly or via suffixes)
                    fund_root = find_fundamento_root(remainder)
                    if fund_root:
                        extracted_prefixes.append(prefix)
                        stem = remainder
                        found_prefix = True
                        break

                    # Double-prefix: remainder = prefix2 + fundamento_root (e.g. malrefar → mal + refar)
                    # We extract the outer prefix and let the next loop iteration handle the inner one.
                    if check_prefix_gives_fundamento(remainder) is not None:
                        extracted_prefixes.append(prefix)
                        stem = remainder
                        found_prefix = True
                        break

                    # Corpus-vocabulary fallback: only accept if no Fundamento-based
                    # decomposition was possible and the remainder is a complete word.
                    if remainder in KNOWN_ROOTS and len(remainder) >= 2:
                        extracted_prefixes.append(prefix)
                        stem = remainder
                        found_prefix = True
                        break

            if not found_prefix:
                break

    ast["prefiksoj"] = extracted_prefixes

    # ==========================================================================
    # STEP 6: Suffix Extraction (with Fundamento guard)
    # Strip suffixes right-to-left until we reach a Fundamento/known root
    # ==========================================================================

    extracted_suffixes = []
    max_suffix_depth = 3

    # If we already determined a suffix parse leads to Fundamento AND we didn't extract prefixes,
    # use it directly. This handles cases like belul+in+o where intermediate "belul" is not a known root.
    # BUT: if we extracted prefixes, the suffix_parse was computed on the ORIGINAL stem, not the
    # POST-prefix stem. So we need to recalculate.
    if suffix_parse and not extracted_prefixes:
        # suffix_parse = (root, [suffix1, suffix2, ...]) in extraction order
        root_from_suffix, suffixes_found = suffix_parse
        stem = root_from_suffix
        extracted_suffixes = suffixes_found
    else:
        # Standard suffix extraction - one layer at a time
        for _ in range(max_suffix_depth):
            # If stem is now a Fundamento/protected root, stop
            if stem in _FUNDAMENTO_ROOTS or stem in PROTECTED_ROOTS:
                break

            found_suffix = False
            for suffix in sorted_suffixes:
                if stem.endswith(suffix):
                    potential = stem[:-len(suffix)]

                    # Special case: affectionate suffixes (-ĉj, -nj) accept truncated stems
                    # that exist in our lookup tables, even if they're very short (e.g., "a")
                    if suffix == "ĉj" and potential in AFFECTIONATE_ROOT_LOOKUP:
                        extracted_suffixes.append(suffix)
                        stem = potential
                        found_suffix = True
                        break
                    if suffix == "nj" and potential in AFFECTIONATE_ROOT_LOOKUP_NJ:
                        extracted_suffixes.append(suffix)
                        stem = potential
                        found_suffix = True
                        break

                    # For non-affectionate suffixes, require minimum stem length
                    if len(stem) <= len(suffix) + 1:
                        continue

                    # Accept if potential is a valid root
                    if (potential in _FUNDAMENTO_ROOTS or
                        potential in PROTECTED_ROOTS or
                        potential in KNOWN_ROOTS or
                        potential in KNOWN_PARTICLES or
                        potential in KNOWN_PREPOSITIONS):

                        # Make sure we're not incorrectly splitting a protected root
                        if stem not in PROTECTED_SUFFIX_ROOTS:
                            extracted_suffixes.append(suffix)
                            stem = potential
                            found_suffix = True
                            break

            if not found_suffix:
                break

    # Reverse suffixes (we extracted from right-to-left, but want left-to-right order)
    ast["sufiksoj"] = extracted_suffixes

    # Add participle metadata if found
    for suffix in extracted_suffixes:
        if suffix in PARTICIPLE_SUFFIXES:
            participle_info = PARTICIPLE_SUFFIXES[suffix]
            ast["participo_voĉo"] = participle_info["voĉo"]
            ast["participo_tempo"] = participle_info["tempo"]
            break

    # ==========================================================================
    # STEP 6b: Affectionate Suffix Root Recovery (-ĉj, -nj)
    # ==========================================================================
    # These suffixes truncate the root, so we need to recover the full root
    # from a lookup table. E.g., paĉjo has stem "pa" but root should be "patr"
    # For -nj (female affectionate), we also add the implicit -in suffix since
    # panjo comes from patrino (patr + in + o), not just patro
    if "ĉj" in extracted_suffixes or "nj" in extracted_suffixes:
        truncated_root = stem
        recovered_root = None

        if "nj" in extracted_suffixes:
            # Use -nj lookup (female affectionate)
            recovered_root = AFFECTIONATE_ROOT_LOOKUP_NJ.get(truncated_root)
            # Add implicit -in suffix since -nj words derive from feminine forms
            # panjo = patr + in + nj + o (from patrino)
            # We insert -in BEFORE -nj in the suffix list
            if recovered_root and "in" not in extracted_suffixes:
                nj_idx = extracted_suffixes.index("nj")
                extracted_suffixes.insert(nj_idx, "in")
                ast["sufiksoj"] = extracted_suffixes
        elif "ĉj" in extracted_suffixes:
            # Use -ĉj lookup (male affectionate)
            recovered_root = AFFECTIONATE_ROOT_LOOKUP.get(truncated_root)

        if recovered_root:
            # Store both truncated and recovered for debugging/training
            ast["radiko_trunkita"] = truncated_root  # The truncated form found
            stem = recovered_root  # Use recovered root for embeddings

    # ==========================================================================
    # STEP 7: Identify Root (with compound word fallback)
    # ==========================================================================

    # FOREIGN-ORTHOGRAPHY FAST-PATH: a capitalized word containing
    # characters or digraphs that aren't Esperanto orthography (Q/W/X/Y or
    # sh/ch/th/ph) is foreign with very high confidence. Don't bother trying
    # morphology — these stems coincidentally matching DICTIONARY_ROOTS
    # would otherwise produce misclassifications like "Christian" → adjektivo.
    if original_word[0].isupper() and _has_foreign_orthography(original_word):
        return categorize_unknown_word(original_word)

    # Capitalization guard: if the original word was capitalized and its
    # stem is not a recognized Esperanto root, it's a proper noun (or foreign
    # word). Route to categorize_unknown_word.
    #
    # EXCEPTIONS that fall through to morphology:
    #   1. Adjectival ending (-a/-aj/-an/-ajn) AND stem in DICTIONARY_ROOTS
    #      — catches "Genetika" (stem "genetik"), "Hungaraj" (stem "hungar").
    #      Adjectival endings are reliable because adjectives MUST agree with
    #      a noun, so genuine modifier-position usage self-evidences.
    #   2. Substantivo ending (-o/-oj/-on/-ojn) AND stem in DICTIONARY_ROOTS
    #      — catches "Manifesto" (stem "manifest"), "Espero" (stem "esper"),
    #      "Hundo" (stem "hund"). A capitalized -o-ending word whose stem is
    #      a recognized Esperanto root (full inventory, not just the ~2K
    #      Fundamento set) HAS a valid common-noun reading. Whether it is
    #      actually proper here is then decided structurally by the
    #      sentence-level reanalysis (sentence-initial → trust morphology;
    #      mid-sentence capitalization → proper). This widens the
    #      morphological-validity gate from _FUNDAMENTO_ROOTS to the full
    #      root inventory, mirroring exception 1 for nouns. Foreign names
    #      that merely look -o-final are already filtered upstream by the
    #      foreign-orthography fast-path; the irreducibly-ambiguous residual
    #      (esperantized names like "Leono"/"Marko" sharing a root) is what
    #      the position logic and a future learned tie-breaker resolve.
    #   3. Substantivo ending AND stem is a *verified* Esperanto compound
    #      (both halves recognized via _is_genuine_esperanto_compound).
    #      Catches "Membroŝtatoj" (membr+ŝtat) where no single root matches.
    #
    # We deliberately do NOT extend to -e (adverb) endings — many foreign
    # names end in -e ("Shakespeare", "Goethe", "Marie") and DICTIONARY_ROOTS
    # contains their stems too. The -e adverb case is handled by the
    # sentence-level reanalysis using alt-subject case-marking.
    if (original_word[0].isupper()
            and stem not in _FUNDAMENTO_ROOTS
            and stem not in PROTECTED_ROOTS):
        lower_word = original_word.lower()
        had_adjectival_ending = (
            lower_word != stem
            and (lower_word.endswith(('aj', 'an', 'ajn'))
                 or lower_word.endswith('a'))
        )
        had_substantivo_ending = (
            lower_word != stem
            and (lower_word.endswith(('oj', 'on', 'ojn'))
                 or lower_word.endswith('o'))
        )
        had_substantivo_compound = (
            had_substantivo_ending
            and _is_genuine_esperanto_compound(stem)
        )
        # Adjectival compounds (e.g., "Multiklasa" = multi + klas + a) — same
        # idea as substantivo compounds but for -a-ending words. Stem must
        # decompose into recognized Esperanto morphemes.
        had_adjectival_compound = (
            had_adjectival_ending
            and _is_genuine_esperanto_compound(stem)
        )
        if not (
            (had_adjectival_ending and stem in DICTIONARY_ROOTS)
            or (had_substantivo_ending and stem in DICTIONARY_ROOTS)
            or had_substantivo_compound
            or had_adjectival_compound
        ):
            return categorize_unknown_word(original_word)
        # Fall through: word is capitalized + valid Esperanto morphology.

    if stem in KNOWN_ROOTS or stem in _FUNDAMENTO_ROOTS or stem in PROTECTED_ROOTS:
        ast["radiko"] = stem
        return ast

    if stem in KNOWN_PARTICLES or stem in KNOWN_PREPOSITIONS:
        ast["radiko"] = stem
        return ast

    # Try compound word decomposition (root + root)
    # Patterns run in priority order across ALL split positions before the next pattern.
    # This prevents an early split with a lower-priority pattern from shadowing the
    # correct split found at a later position by a higher-priority pattern.
    if len(stem) >= 4:
        splits = [(stem[:i], stem[i:]) for i in range(2, len(stem) - 2)]

        # Pattern 1 (highest priority): root1 + o + root2 (linking vowel)
        for first, rest in splits:
            if rest.startswith("o") and len(rest) > 2:
                second = rest[1:]
                if first in KNOWN_ROOTS and second in KNOWN_ROOTS:
                    ast["radiko"] = second
                    ast["kunmetitaj_radikoj"] = [first, second]
                    return ast

        # Pattern 2b: root1 + (fundamento_root2 + suffix)
        # Preferred over Pattern 2 to produce a Fundamento root as the head.
        # Example: librvendejo → libr + (vend + ej) rather than libr + vendej.
        for first, rest in splits:
            if (first in KNOWN_ROOTS or first in _FUNDAMENTO_ROOTS) and len(rest) >= 3:
                for suffix in sorted_suffixes:
                    if rest.endswith(suffix) and len(rest) > len(suffix) + 1:
                        second = rest[:-len(suffix)]
                        if second in _FUNDAMENTO_ROOTS or second in PROTECTED_ROOTS:
                            ast["radiko"] = second
                            ast["kunmetitaj_radikoj"] = [first, second]
                            ast["sufiksoj"] = ast.get("sufiksoj", []) + [suffix]
                            return ast

        # Pattern 2: root1 + root2 (no linking vowel)
        for first, rest in splits:
            if first in KNOWN_ROOTS and rest in KNOWN_ROOTS:
                ast["radiko"] = rest
                ast["kunmetitaj_radikoj"] = [first, rest]
                return ast

        # Pattern 3: preposition + root
        for prep in KNOWN_PREPOSITIONS:
            if stem.startswith(prep) and len(stem) > len(prep):
                remainder = stem[len(prep):]
                if remainder in KNOWN_ROOTS:
                    ast["radiko"] = remainder
                    ast["prefiksoj"].insert(0, prep)
                    return ast

    # Last resort: if no valid Esperanto root found, try phonological acceptance
    # for neologisms (valid Esperanto phonology but not yet in any vocabulary),
    # then fall back to categorize_unknown_word for genuine foreign/unknown words.
    if stem not in KNOWN_ROOTS and stem not in _FUNDAMENTO_ROOTS:
        if _is_valid_eo_stem(stem):
            # Phonologically valid Esperanto root not in vocabulary = neologism.
            # Accept it so modern words (komput, ekran, retum, ...) parse correctly.
            ast["radiko"] = stem
            ast["kategorio"] = "neologismo"
            return ast
        return categorize_unknown_word(original_word, f"Ne povis trovi validan radikon. Restaĵo: '{stem}'")

    ast["radiko"] = stem
    return ast


def categorize_unknown_word(word: str, error_msg: str = "") -> dict:
    """
    Categorize an unknown word that failed to parse.

    Returns an AST node marking the word as non-Esperanto with best-guess categorization.
    Categories (Pure Esperanto):
    - propranomo_konata: Known proper noun from dictionary (analizstato=sukceso!)
    - propranomo: Capitalized word (person, place)
    - fremda_vorto: Lowercase but not Esperanto
    - numero_laŭvorta: Numeric
    - nekonata: Cannot categorize
    """
    ast = {
        "tipo": "vorto",
        "plena_vorto": word,
        "radiko": word,
        "vortspeco": "nekonata",
        "analizstato": "malsukceso",
        "analizeraro": error_msg,
        "kategorio": "nekonata",
        "nombro": "singularo",
        "kazo": "nominativo",
        "prefiksoj": [],
        "sufiksoj": [],
    }

    # Categorization heuristics

    # 1. Number literal (digits)
    if word.isdigit():
        ast["kategorio"] = "numero_laŭvorta"
        ast["vortspeco"] = "numero"
        return ast

    # 2. Proper name (starts with capital letter).
    #
    # No gazetteer. A capitalized word only reaches categorize_unknown_word
    # when the morphology layer found NO valid Esperanto decomposition
    # against the root lexicon (the capitalization guard already let every
    # decomposable word fall through to morphology). Non-decomposable +
    # capitalized IS a proper noun by deterministic negative detection —
    # this needs no world-knowledge lookup. Entity *type*
    # (persono/loko/organizaĵo) is deliberately NOT decided here; that is
    # the sense axis and belongs to the ontology/learned layer downstream,
    # not to a Wikipedia-title gazetteer in the parser.
    if word[0].isupper() and len(word) > 1:
        # Successfully classified — deterministically, as a proper noun.
        ast["analizstato"] = "sukceso"
        ast["analizeraro"] = ""
        ast["vortspeco"] = "propra_nomo"
        ast["kategorio"] = "propranomo"
        ast["propra_nomo_evidence"] = "morphology_no_decomposition"

        # Esperantized proper nouns carry Esperanto case/number endings
        # (Parizon, Berlinoj). Extract them deterministically; bare-foreign
        # names (Washington) have none and stay nominative singular.
        if word.endswith(('o', 'on', 'oj', 'ojn')):
            ast["kategorio"] = "propranomo_esperantigita"
            if word.endswith('n'):
                ast["kazo"] = "akuzativo"
                word = word[:-1]
            if word.endswith('j'):
                ast["nombro"] = "pluralo"

        return ast

    # 3. Single letter (often grammar examples)
    if len(word) == 1:
        ast["kategorio"] = "unusola_litero"
        ast["vortspeco"] = "ekzemplo"
        return ast

    # 4. Foreign word (lowercase, no Esperanto structure)
    # Has no recognizable Esperanto endings or morphology
    ast["kategorio"] = "fremda_vorto"
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


# =============================================================================
# Subordinate Clause Parsing (Issue #691)
# =============================================================================

# Subordinating conjunctions that introduce clauses
SUBORDINATING_CONJUNCTIONS = {
    # Complement clauses (ke-clauses) - become objects
    "ke": "complement",

    # Relative clauses (attach to nouns)
    "kiu": "relative",
    "kio": "relative",
    "kiuj": "relative",
    "kiujn": "relative",
    "kiun": "relative",

    # Temporal clauses (adverbial)
    "kiam": "temporal",
    "dum": "temporal",
    "antaŭ": "temporal",  # antaŭ ol
    "post": "temporal",   # post kiam

    # Conditional clauses (adverbial)
    "se": "conditional",

    # Causal clauses (adverbial)
    "ĉar": "causal",

    # Concessive clauses (adverbial)
    "kvankam": "concessive",
    "malgraŭ": "concessive",  # malgraŭ ke

    # Purpose clauses (adverbial)
    "por": "purpose",  # por ke
}


# =============================================================================
# RELATIVE CLAUSE HANDLING
# =============================================================================


def _is_ki_correlative(ast: dict) -> bool:
    """Return True if *ast* is a ki- correlative (kiu/kiun/kio/kie/kiam…)."""
    return (ast.get("vortspeco") == "korelativo" and
            ast.get("korelativo_prefikso") == "ki")


def _find_relative_clause_end(word_asts: list, start: int) -> int:
    """
    Return the end index (exclusive) of the relative clause whose
    ki-correlative is at *start*.

    The clause ends right before the second finite verb at nesting depth 0.
    Each new subordinate opener (another ki-correlative or ke/se/ĉar/…)
    pushes depth; the next verb at that depth pops it.

    Returns len(word_asts) if the clause runs to the end of the sentence.
    """
    depth = 0       # extra nesting inside this relative clause
    verb_count = 0  # finite verbs seen at depth 0

    non_sub_coords = {"kaj", "sed", "aŭ", "nek", "do", "tial"}

    for j in range(start + 1, len(word_asts)):
        ast = word_asts[j]
        radiko = ast.get("radiko", "").lower()
        vortspeco = ast.get("vortspeco", "")

        if _is_ki_correlative(ast):
            depth += 1
        elif radiko in SUBORDINATING_CONJUNCTIONS and radiko not in non_sub_coords:
            depth += 1
        elif vortspeco == "verbo":
            if depth > 0:
                depth -= 1      # verb closes a nested sub-clause
            else:
                verb_count += 1
                if verb_count >= 2:
                    return j    # second verb = first word of main clause

    return len(word_asts)


def find_relative_clause_spans(word_asts: list) -> list:
    """
    Return list of (start, end) spans (end exclusive) for every relative
    clause in *word_asts*.

    A relative clause is introduced by a ki-correlative that is NOT at
    sentence position 0 (position-0 ki-correlatives are question words).
    Multi-level nesting is handled transparently: scanning jumps past each
    already-found span.

    Fronted-PP question exception: when a ki-correlative is preceded by a
    preposition (`En kiu lando…?`, `Al kiu…?`, `Per kio…?`, `Pri kio…?`),
    it's a fronted-PP interrogative pattern, NOT a relative clause. The
    relative-clause analysis would null out the main-clause subjekto /
    verbo / objekto for these very common natural Esperanto questions.
    """
    spans = []
    i = 0
    while i < len(word_asts):
        if i > 0 and _is_ki_correlative(word_asts[i]):
            # Fronted-PP question check: any preposition appears in the
            # prefix [0:i] without an intervening noun/pronoun antecedent.
            # The antecedent of a relative clause is the noun immediately
            # before the ki-correlative; a preceding preposition without
            # an intervening noun means we're inside (or just past) a
            # sentence-initial PP — a question pattern, not a relative.
            is_fronted_pp_question = False
            for j in range(i - 1, -1, -1):
                vs = word_asts[j].get('vortspeco')
                if vs in ('substantivo', 'propra_nomo', 'pronomo'):
                    break  # found antecedent → real relative clause
                if vs == 'prepozicio':
                    is_fronted_pp_question = True
                    break
            if is_fronted_pp_question:
                i += 1
                continue
            end = _find_relative_clause_end(word_asts, i)
            spans.append((i, end))
            i = end         # jump past the whole clause
        else:
            i += 1
    return spans


def _build_relative_clause_node(correlative_ast: dict,
                                 clause_words: list) -> dict:
    """
    Build a *rilata_subfrazo* node from a ki-correlative and its body words.

    Handles multi-level nesting recursively: nested ki-correlatives inside
    *clause_words* are detected and turned into their own rilata_subfrazo
    nodes attached to the appropriate inner noun group.

    The outer correlative's case tells its syntactic role:
      - nominativo (kiu)  → fills inner subjekto if empty
      - akuzativo (kiun)  → fills inner objekto if empty
    """
    # --- nested relative clauses inside the clause body ---
    nested_spans = find_relative_clause_spans(clause_words)
    nested_set = {i for s, e in nested_spans for i in range(s, e)}

    # Parse the main (non-nested) words
    main_words = [w for i, w in enumerate(clause_words) if i not in nested_set]
    if main_words:
        inner = parse_clause(main_words)
    else:
        inner = {
            "tipo": "frazo",
            "subjekto": None, "verbo": None, "objekto": None, "aliaj": []
        }

    # Fill in the outer correlative's own syntactic role
    kazo = correlative_ast.get("kazo", "nominativo")
    if kazo == "nominativo" and inner["subjekto"] is None:
        inner["subjekto"] = {
            "tipo": "vortgrupo", "kerno": correlative_ast, "priskriboj": []
        }
    elif kazo == "akuzativo" and inner["objekto"] is None:
        inner["objekto"] = {
            "tipo": "vortgrupo", "kerno": correlative_ast, "priskriboj": []
        }

    # Recursively build and attach nested rilata_subfrazo nodes
    for n_start, n_end in nested_spans:
        n_corr = clause_words[n_start]
        n_words = clause_words[n_start + 1:n_end]
        nested_node = _build_relative_clause_node(n_corr, n_words)

        # Find antecedent: last content word before n_start that is not
        # itself inside another nested span
        def _inner_head(slot):
            if not slot:
                return None
            if slot.get("tipo") == "vortgrupo":
                return slot.get("kerno")
            return slot

        def _attach(role):
            slot = inner[role]
            if slot.get("tipo") != "vortgrupo":
                inner[role] = {
                    "tipo": "vortgrupo",
                    "kerno": slot,
                    "priskriboj": [nested_node],
                }
            else:
                slot.setdefault("priskriboj", []).append(nested_node)

        attached = False
        for i in range(n_start - 1, -1, -1):
            if i in nested_set:
                continue
            ast = clause_words[i]
            vs = ast.get("vortspeco", "")
            if vs in ("substantivo", "propra_nomo", "pronomo", "nekonata"):
                if ast is _inner_head(inner["subjekto"]):
                    _attach("subjekto")
                    attached = True
                elif ast is _inner_head(inner["objekto"]):
                    _attach("objekto")
                    attached = True
                break
        if not attached:
            inner.setdefault("aliaj", []).append(nested_node)

    return {
        "tipo": "rilata_subfrazo",
        "rilata_pronomo": correlative_ast,
        "subjekto": inner["subjekto"],
        "verbo": inner["verbo"],
        "objekto": inner["objekto"],
        "aliaj": inner.get("aliaj", []),
    }


def _attach_relative_clauses(sentence_ast: dict, word_asts: list,
                               relative_spans: list) -> dict:
    """
    For each relative clause span: build a rilata_subfrazo, attach it to the
    antecedent noun's priskriboj, and strip the clause words from aliaj.
    """
    if not relative_spans:
        return sentence_ast

    def _head(slot):
        # subjekto/objekto can be either a vortgrupo (with 'kerno') or a bare vorto
        if not slot:
            return None
        if slot.get("tipo") == "vortgrupo":
            return slot.get("kerno")
        return slot

    # All indices covered by any relative clause
    relative_indices: set = set()
    for start, end in relative_spans:
        relative_indices.update(range(start, end))

    for start, end in relative_spans:
        correlative_ast = word_asts[start]
        clause_words = word_asts[start + 1:end]
        rilata_node = _build_relative_clause_node(correlative_ast, clause_words)

        # Antecedent = last content word strictly before *start* that is not
        # itself inside a relative clause
        antecedent_role = None
        for i in range(start - 1, -1, -1):
            if i in relative_indices:
                continue
            ast = word_asts[i]
            vs = ast.get("vortspeco", "")
            if vs in ("substantivo", "propra_nomo", "pronomo", "nekonata"):
                if ast is _head(sentence_ast.get("subjekto")):
                    antecedent_role = "subjekto"
                elif ast is _head(sentence_ast.get("objekto")):
                    antecedent_role = "objekto"
                break

        if antecedent_role:
            slot = sentence_ast[antecedent_role]
            # If slot is a bare vorto, wrap it in a vortgrupo so we can attach priskriboj
            if slot.get("tipo") != "vortgrupo":
                sentence_ast[antecedent_role] = {
                    "tipo": "vortgrupo",
                    "kerno": slot,
                    "priskriboj": [rilata_node],
                }
            else:
                slot.setdefault("priskriboj", []).append(rilata_node)
        else:
            sentence_ast["aliaj"].append(rilata_node)

    # Strip relative clause words from aliaj (now inside rilata_subfrazo nodes)
    relative_ids = {id(word_asts[i]) for i in relative_indices}
    sentence_ast["aliaj"] = [
        w for w in sentence_ast["aliaj"]
        if not (isinstance(w, dict) and id(w) in relative_ids)
    ]
    return sentence_ast


# =============================================================================


def parse_subordinate_clauses(sentence_ast: dict, word_asts: list) -> dict:
    """
    Parse subordinate clauses from aliaj[] and create nested frazo nodes.

    Handles:
    - ke-clauses (complement): attach to objekto
    - Adverbial clauses (kiam/se/ĉar): keep in aliaj[] but as frazo
    - Relative clauses are handled separately by _attach_relative_clauses.

    Args:
        sentence_ast: The sentence AST with subject, verb, object, aliaj
        word_asts: List of all word ASTs (for reference)

    Returns:
        Modified sentence_ast with subordinate clauses parsed as nested frazo
    """
    aliaj = sentence_ast.get("aliaj", [])
    if not aliaj:
        return sentence_ast

    new_aliaj = []
    i = 0

    while i < len(aliaj):
        word = aliaj[i]
        radiko = word.get("radiko", "").lower()

        # Check if this is a subordinating conjunction
        if radiko in SUBORDINATING_CONJUNCTIONS:
            clause_type = SUBORDINATING_CONJUNCTIONS[radiko]

            # Relative clauses are handled separately by _attach_relative_clauses.
            if clause_type == "relative":
                new_aliaj.append(word)
                i += 1
                continue

            # Collect words for subordinate clause
            clause_words = []
            depth = 1  # Track nesting depth for nested subordinates
            j = i + 1

            while j < len(aliaj) and depth > 0:
                next_word = aliaj[j]
                next_radiko = next_word.get("radiko", "").lower()

                # Check if we hit another subordinating conjunction (nested)
                if next_radiko in SUBORDINATING_CONJUNCTIONS:
                    depth += 1

                # Check if we hit a verb in conditional mood (marks end of se-clause)
                if clause_type == "conditional" and next_word.get("modo") == "kondicionalo":
                    clause_words.append(next_word)
                    j += 1
                    # se-clause ends after conditional verb
                    break

                # Check if we hit a comma or conjunction (might mark end of clause)
                if next_word.get("vortspeco") == "konjunkcio" and next_radiko in ["kaj", "sed", "aŭ"]:
                    # Coordinate conjunction marks end of subordinate clause
                    break

                clause_words.append(next_word)
                j += 1

            # Parse collected words as a subordinate clause
            if clause_words:
                # Reconstruct sentence text from clause words
                clause_text = " ".join(w.get("plena_vorto", "") for w in clause_words)

                try:
                    # Recursively parse subordinate clause
                    # Use parse_clause() to avoid infinite recursion on same text
                    sub_frazo = parse_clause(clause_words)
                    sub_frazo["clause_type"] = clause_type

                    # Attach based on clause type
                    if clause_type == "complement":
                        # ke-clause becomes the object
                        sentence_ast["objekto"] = sub_frazo
                    else:
                        # Adverbial clauses (temporal, conditional, causal, etc.)
                        # Keep in aliaj[] but as frazo, not individual words
                        new_aliaj.append(sub_frazo)

                    # Skip past all the words we consumed
                    i = j
                    continue

                except Exception as e:
                    # If parsing fails, fall back to keeping individual words
                    new_aliaj.append(word)
                    i += 1
            else:
                # No words collected, keep conjunction
                new_aliaj.append(word)
                i += 1
        else:
            # Not a subordinating conjunction, keep word
            new_aliaj.append(word)
            i += 1

    sentence_ast["aliaj"] = new_aliaj
    return sentence_ast


def parse_clause(word_asts: list) -> dict:
    """
    Parse a list of word ASTs into a frazo structure (subordinate clause).

    This is similar to the main parse() function but operates on pre-parsed words
    instead of raw text, and doesn't recursively handle subordinates (to avoid infinite recursion).

    Args:
        word_asts: List of word AST dictionaries

    Returns:
        frazo AST dictionary
    """
    # Build basic frazo structure
    frazo = {
        "tipo": "frazo",
        "subjekto": None,
        "verbo": None,
        "objekto": None,
        "aliaj": []
    }

    # Find subject, verb, object using same rules as main parser
    for i, ast in enumerate(word_asts):
        # PP-governance check: if the preceding token is a prepozicio,
        # this noun belongs to a prepositional phrase, NOT to a clause
        # role. Skip it as a subject/object candidate. This kills the
        # fronted-PP misattribution bug (`En Volterra, li skribis…` had
        # been treating Volterra as the subject).
        is_pp_governed = (
            i > 0 and word_asts[i-1].get("vortspeco") == "prepozicio"
        )
        if ast["vortspeco"] == "verbo" and not frazo["verbo"]:
            frazo["verbo"] = ast
            # Check for negation: `ne` immediately before the verb, OR
            # any neni-prefixed correlative (`neniam`, `nenie`, `nenial`,
            # `neniu`, …) immediately before. The previous check only
            # handled `ne` and silently missed `Bach neniam aŭdis…`
            # (true negation in the source). neni-correlatives carry the
            # absolute-negation meaning equivalent to `ne` + verb here.
            if i > 0:
                prev = word_asts[i-1]
                if (prev.get("radiko") == "ne"
                        or (prev.get("vortspeco") == "korelativo"
                            and prev.get("korelativo_prefikso") == "neni")):
                    ast["negita"] = True
        elif ast["vortspeco"] in ["substantivo", "pronomo", "propra_nomo", "korelativo", "nekonata"] and ast["kazo"] == "akuzativo" and not frazo["objekto"] and not is_pp_governed:
            frazo["objekto"] = {"tipo": "vortgrupo", "kerno": ast, "priskriboj": []}
        elif ast["vortspeco"] in ["substantivo", "pronomo", "propra_nomo", "korelativo", "nekonata"] and ast["kazo"] == "nominativo" and not frazo["subjekto"] and not is_pp_governed:
            frazo["subjekto"] = {"tipo": "vortgrupo", "kerno": ast, "priskriboj": []}

    # Associate adjectives with their nouns
    for i, ast in enumerate(word_asts):
        if ast["vortspeco"] == "adjektivo":
            if frazo["objekto"] and ast["kazo"] == frazo["objekto"]["kerno"]["kazo"] and ast["nombro"] == frazo["objekto"]["kerno"]["nombro"]:
                frazo["objekto"]["priskriboj"].append(ast)
            elif frazo["subjekto"] and ast["kazo"] == frazo["subjekto"]["kerno"]["kazo"] and ast["nombro"] == frazo["subjekto"]["kerno"]["nombro"]:
                frazo["subjekto"]["priskriboj"].append(ast)
            else:
                frazo["aliaj"].append(ast)
        elif ast["vortspeco"] == "artikolo":
            # Find noun this article modifies
            for j in range(i + 1, len(word_asts)):
                next_ast = word_asts[j]
                if next_ast["vortspeco"] in ["substantivo", "pronomo"]:
                    if frazo["objekto"] and next_ast == frazo["objekto"]["kerno"]:
                        frazo["objekto"]["artikolo"] = "la"
                        break
                    elif frazo["subjekto"] and next_ast == frazo["subjekto"]["kerno"]:
                        frazo["subjekto"]["artikolo"] = "la"
                        break
                elif next_ast["vortspeco"] != "adjektivo":
                    break

    # Clean up unassociated words
    placed_words = []
    if frazo["verbo"]:
        placed_words.append(frazo["verbo"])
    if frazo["subjekto"]:
        placed_words.append(frazo["subjekto"]["kerno"])
        placed_words.extend(frazo["subjekto"]["priskriboj"])
    if frazo["objekto"]:
        placed_words.append(frazo["objekto"]["kerno"])
        placed_words.extend(frazo["objekto"]["priskriboj"])

    for ast in word_asts:
        if ast["vortspeco"] != 'artikolo' and ast not in placed_words:
            frazo["aliaj"].append(ast)

    return frazo


@lru_cache(maxsize=10000)
def parse(text: str):
    """
    Parses an Esperanto sentence and returns a structured, morpheme-based AST.

    NOTE: Cached with LRU cache (10K entries) for performance.
    """
    # Preprocess: normalize punctuation
    text = preprocess_text(text)

    # Simple tokenizer: split by space, remove all punctuation EXCEPT:
    # - Apostrophes for elision (l', hund')
    # - Hyphens connecting words (Esperanto-klubo, hundo-domo)
    # Remove common punctuation marks: . , ! ? : ; " ( ) [ ] { }
    import string
    import re
    # First, preserve elision apostrophes by converting "letter'" to a safe form
    # Match: word character followed by apostrophe (straight or curly)
    text = re.sub(r"(\w)([''])", r"\1ZZZELISIONZZZ", text)
    # Preserve compound word hyphens: word-word patterns
    # Match: letter(s) + hyphen + letter(s) (captures compound words)
    # Use ZZZCOMPOUNDZZ (no underscores) because underscore is in string.punctuation
    text = re.sub(r"(\w)-(\w)", r"\1ZZZCOMPOUNDZZ\2", text)
    # Bug #11 prep: track token indices that have a trailing comma in the
    # original text, BEFORE punctuation stripping. Used downstream to
    # detect `Kiam X Y, Z W`-style temporal subordinate clauses where the
    # sentence-initial ki-correlative is a subordinator, not a question
    # word. The split() here aligns 1:1 with the post-strip `words` list
    # (commas don't change token count, they just sit at token edges).
    _pre_strip_tokens = text.split()
    comma_after_indices = [
        i for i, tok in enumerate(_pre_strip_tokens) if tok.endswith(',')
    ]
    # Remove all punctuation (now hyphens in compounds are protected)
    for punct in string.punctuation:
        text = text.replace(punct, ' ')
    # Restore elision apostrophes
    text = text.replace("ZZZELISIONZZZ", "'")
    # Restore compound hyphens
    text = text.replace("ZZZCOMPOUNDZZ", "-")
    words = text.split()

    if not words:
        raise ValueError("Ne povis analizi malplenan ĉenon.")

    # Step 1: Morphological analysis of all words
    # Uses Fundamento-first design for better disambiguation
    # Gracefully handle unknown words by categorizing them
    word_asts = []
    for i, w in enumerate(words):
        try:
            ast = parse_word(w)
            ast["analizstato"] = "sukceso"  # Mark as successfully parsed Esperanto

            # CRITICAL FIX: Proper noun detection with sentence position awareness
            # Must happen AFTER parse_word, using sentence context
            if w and w[0].isupper() and len(w) > 1:
                # Build skip_words from comprehensive function word set
                # Include both lowercase and capitalized forms
                skip_words = {fw.capitalize() for fw in _ALL_FUNCTION_WORDS}
                skip_words.add('La')  # Ensure article is included

                # For first word: only mark as proper noun if preceded by article
                # "La Fundamento" → proper noun
                # "Fundamento estas..." → ambiguous, might be sentence-initial
                if i == 0:
                    # Sentence-initial capitalization disambiguation —
                    # purely deterministic, no gazetteer.
                    #
                    # Capitalization at position 0 is NOT evidence (every
                    # sentence starts capitalized). So the decision rests on
                    # what the morphology layer found:
                    #   - content classification (substantivo, adjektivo,
                    #     verbo, ...): trust it — the word decomposed validly
                    #     against the root lexicon, so it has a common
                    #     reading and position explains the capital.
                    #   - 'propra_nomo' already: parse_word/categorize made a
                    #     deterministic negative-detection call; keep it.
                    #   - 'nekonata': fall back to negative detection — a
                    #     root that is not a known Esperanto morpheme is a
                    #     proper noun.
                    current_vortspeco = ast.get("vortspeco", "")
                    if current_vortspeco == 'nekonata':
                        root = ast.get("radiko", "").lower()
                        surface = ast.get("plena_vorto", "")
                        # A known ROOT is not the same thing as a known WORD.
                        # Rules 2-7: a content word must carry a grammatical
                        # ending. `sam` is a Fundamento root, but `Sam` has no
                        # ending, so it is not a word — and the old check, which
                        # asked only whether the ROOT was known, therefore
                        # refused to call it a name.
                        is_known_word = (
                            (root in _FUNDAMENTO_ROOTS or root in DICTIONARY_ROOTS)
                            and _has_grammatical_ending(surface)
                        )
                        is_common_root = (
                            is_known_word or
                            root in _ALL_FUNCTION_WORDS or   # closed ending-less class
                            len(root) <= 2
                        )
                        if not is_common_root:
                            ast["vortspeco"] = "propra_nomo"
                            ast["kategorio"] = "propranomo"
                            # WHICH rule decided this? Attribution is the thesis
                            # (VISION.md): it makes per-rule precision measurable
                            # on gold, and it is where a learned tie-breaker for
                            # the residue would later declare itself.
                            ast["propra_nomo_evidence"] = (
                                "no_valid_ending"
                                if (root in _FUNDAMENTO_ROOTS or root in DICTIONARY_ROOTS)
                                else "root_not_in_lexicon"
                            )
                    elif current_vortspeco == 'adverbo':
                        # Bug #5: foreign names ending in -e (Goethe, Crusoe,
                        # Brontë, etc.) get parse_word'd as 'adverbo' because
                        # the -e ending matches adverbial morphology. At
                        # sentence-initial position, if the root is NOT in
                        # the Esperanto lexicon, this is a foreign-name
                        # misclassification — promote to propra_nomo.
                        root = ast.get("radiko", "").lower()
                        is_known_adverb_root = (
                            root in _FUNDAMENTO_ROOTS or
                            root in DICTIONARY_ROOTS or
                            root in _ALL_FUNCTION_WORDS or
                            len(root) <= 2
                        )
                        if not is_known_adverb_root:
                            ast["vortspeco"] = "propra_nomo"
                            ast["kategorio"] = "propranomo"
                            ast["propra_nomo_evidence"] = "foreign_e_ending"
                    # else: trust parse_word (content classification or an
                    # already-deterministic propra_nomo).
                # For non-initial words: capitalization is a strong signal
                # for proper nouns, BUT we must preserve unambiguous Esperanto
                # adjectives (like "Polaj" / "Hungaraj") so the agreement
                # validation pass can decide whether to keep them based on
                # whether a head noun follows.
                #
                # Rule: don't flip when parse_word returned 'adjektivo' AND
                # the surface form ends in a marked adjectival ending. The
                # agreement validation will revert names like "Maria" /
                # "Mona" with coincidental -a endings.
                elif w not in skip_words:
                    current_vortspeco = ast.get("vortspeco", "")
                    pv_lower = w.lower()
                    is_real_adjective = (
                        current_vortspeco == 'adjektivo'
                        and any(pv_lower.endswith(e)
                                for e in _ADJ_SURFACE_ENDINGS)
                    )
                    # ALL-CAPS is typographic emphasis (DEMOKRATIO,
                    # RAJTOJ), not a name — real proper nouns are
                    # Title-Case. If parse_word morphologically resolved
                    # an all-caps token to a content word, that is
                    # stronger evidence than the capital; do not flip.
                    # Narrowed to ALL-CAPS so the Title-Case proper-noun
                    # recall tradeoff is untouched (that ambiguity is the
                    # learned tie-breaker's job).
                    is_allcaps_content = (
                        len(w) > 1 and w.isupper()
                        and current_vortspeco in (
                            'substantivo', 'adjektivo', 'adverbo',
                            'verbo', 'korelativo')
                    )
                    # Don't promote a substantivo whose root is in the
                    # FUNDAMENTO / DICTIONARY lexicon. These are real
                    # Esperanto common nouns that just happen to be
                    # mid-sentence capitalised (headings, emphasis,
                    # `La unua Universitato…`). Promoting them to
                    # propra_nomo creates false-positive entity tags
                    # that downstream consumers treat as named entities.
                    root = (ast.get("radiko") or "").lower()
                    is_known_common_substantivo = (
                        current_vortspeco == 'substantivo'
                        and (root in _FUNDAMENTO_ROOTS
                             or root in DICTIONARY_ROOTS)
                    )
                    if (not is_real_adjective
                            and not is_allcaps_content
                            and not is_known_common_substantivo):
                        # Mid-sentence capitalization is a strong, purely
                        # structural proper-noun signal. Entity type is not
                        # decided here (ontology/learned layer's job).
                        ast["vortspeco"] = "propra_nomo"
                        ast["kategorio"] = "propranomo"
                        ast["propra_nomo_evidence"] = "mid_sentence_capitalization"

                # Special case: "la X" pattern → X is a referenced entity.
                # Same ALL-CAPS exception ("la DEMOKRATIO" = the democracy).
                _allcaps_content = (
                    len(w) > 1 and w.isupper()
                    and ast.get("vortspeco") in (
                        'substantivo', 'adjektivo', 'adverbo', 'verbo',
                        'korelativo'))
                if (i > 0 and words[i-1].lower() == 'la' and w not in skip_words
                        and ast.get("vortspeco") not in ('adjektivo',)
                        and not _allcaps_content):
                    ast["vortspeco"] = "propra_nomo"
                    ast["kategorio"] = "propranomo"
                    ast["propra_nomo_evidence"] = "preceded_by_la"

            word_asts.append(ast)
        except ValueError as e:
            # Word failed to parse - categorize it as non-Esperanto
            unknown_ast = categorize_unknown_word(w, str(e))
            word_asts.append(unknown_ast)
        except Exception as e:
            # Unexpected error - still create a node
            unknown_ast = categorize_unknown_word(w, f"Unexpected error: {str(e)}")
            word_asts.append(unknown_ast)

    # Validate adjective agreement: an adjective must agree with a head
    # noun. The capitalization-guard exception lets sentence-initial -a
    # words fall through to morphology (e.g., "Hungaraj princoj" works
    # because Hungaraj agrees with princoj). But a sentence-initial -a
    # word with no following agreeing noun (e.g., "Maria parolas" — verb
    # follows) is more likely a real proper noun whose -a ending is
    # coincidental. Revert those to propra_nomo so role assignment can
    # pick them as subject.
    _validate_sentence_initial_adjective_agreement(word_asts)

    # USAGE overrides a valid decomposition — the only rule that reaches the
    # residue. `Petro` decomposes cleanly to petr+o ("rock") and lands correctly
    # in a nominal slot, so morphology AND syntax both say "ordinary word". They
    # are not wrong; they are simply blind here. The corpus is not: `petro` is
    # capitalised mid-sentence 98.6% of the time, `hundo` 11.6%.
    #
    # This runs LAST, after morphology and agreement have had their say, so its
    # verdict is attributable and can be ablated. It fires only where the corpus
    # has a strong opinion; on unseen types it is silent and morphology carries on.
    _apply_capitalization_ratio(word_asts)

    # ... and usage also says NO. The weak, capitalisation-driven rules are
    # overruled where the corpus strongly says a type is a COMMON word. This is
    # where the precision was hiding: Prago P 44.3% -> 49.1%, recall unchanged.
    _veto_by_usage(word_asts)

    # Step 2: Syntactic analysis to find sentence structure
    sentence_ast = {
        "tipo": "frazo",
        "subjekto": None,
        "verbo": None,
        "objekto": None,
        "aliaj": [] # Other parts
    }

    # Detect subordinate clause boundaries to avoid assigning their words to main clause.
    # Relative clauses: tracked with precise (start, end) spans via find_relative_clause_spans.
    # Other subordinates (ke/se/ĉar/…): tracked by start position only (old behaviour).
    relative_spans = find_relative_clause_spans(word_asts)
    relative_span_set = {i for start, end in relative_spans for i in range(start, end)}

    # Bug #11: a sentence-initial ki-correlative followed later by a
    # comma opens a comma-delimited subordinate clause, not a question.
    # Example: `Kiam Zamenhof estis junulo, li lernis multajn lingvojn`
    # → Kiam-clause = positions 0..3, main clause = positions 4..7.
    #
    # Discriminator: the pre-comma segment must contain BOTH an
    # independent nominative subject candidate (a substantivo /
    # propra_nomo / pronomo — NOT the leading ki-correlative itself)
    # AND a finite verb. This catches real subordinate clauses
    # (`Kiam X V Y, ...`) and rejects:
    #   - `Kiu, kiun mi vidis hieraŭ, kreis Esperanton` (parenthetical
    #     relative — pre-comma is just `Kiu`, no independent subj+verb)
    #   - `Kio okazis, kiu surprizis ŝin?` (pre-comma `Kio okazis` has
    #     verb but no independent subj — Kio is the subject)
    if (word_asts and _is_ki_correlative(word_asts[0])
            and comma_after_indices
            and comma_after_indices[0] < len(word_asts) - 1):
        sub_end = comma_after_indices[0]  # 0-indexed token before comma
        pre_segment = word_asts[0:sub_end + 1]
        has_independent_subj = any(
            (w.get('vortspeco') in ('substantivo', 'propra_nomo', 'pronomo'))
            and w.get('kazo') == 'nominativo'
            for w in pre_segment[1:]  # skip position 0 (the ki-correlative)
        )
        has_verb = any(w.get('vortspeco') == 'verbo' for w in pre_segment)
        post_segment_size = len(word_asts) - sub_end - 1
        if has_independent_subj and has_verb and post_segment_size >= 2:
            for j in range(0, sub_end + 1):
                relative_span_set.add(j)

    other_subordinate_starts = []
    for i, ast in enumerate(word_asts):
        radiko = ast.get("radiko", "").lower()
        if radiko in SUBORDINATING_CONJUNCTIONS:
            # Position-0 ki-correlatives are question words, not relative pronouns
            if i == 0 and _is_ki_correlative(ast):
                continue
            # Fronted-PP question: `En kiu N V O?`, `Al kiu...?`, `Per kio…?`.
            # The ki-correlative is preceded by a preposition (possibly with
            # intervening articles), with no noun antecedent in between —
            # this is an interrogative pattern, not a subordinate clause.
            if _is_ki_correlative(ast):
                fronted_pp = False
                for j in range(i - 1, -1, -1):
                    vs = word_asts[j].get('vortspeco')
                    if vs in ('substantivo', 'propra_nomo', 'pronomo'):
                        break  # antecedent → real subordinate clause
                    if vs == 'prepozicio':
                        fronted_pp = True
                        break
                if fronted_pp:
                    continue
            # Relative clauses are handled by relative_spans — skip here
            if i in relative_span_set:
                continue
            other_subordinate_starts.append(i)

    # Find the main components (verb, subject noun, object noun)
    # Rule 6: Case determines grammatical function (nominative=subject, accusative=object)
    # Pronouns (pronomoj) function exactly like nouns (substantivoj) grammatically
    for i, ast in enumerate(word_asts):
        # Skip words that are inside any subordinate clause
        in_subordinate = (i in relative_span_set or
                          any(i > start for start in other_subordinate_starts))

        # PP-governance check: if the preceding token is a prepozicio,
        # this noun is the complement of that PP, NOT a clause role.
        # Fixes the fronted-PP misattribution class (`En Volterra, li…`
        # would otherwise pick Volterra as subjekto).
        is_pp_governed = (
            i > 0 and word_asts[i-1].get("vortspeco") == "prepozicio"
        )

        if ast["vortspeco"] == "verbo" and not sentence_ast["verbo"] and not in_subordinate:
            sentence_ast["verbo"] = ast
            # Check for negation: `ne` immediately before the verb, OR
            # any neni-prefixed correlative (`neniam`, `nenie`, `nenial`,
            # `neniu`, …) immediately before. Issue #78 + Bug #6.
            # neni-correlatives carry absolute-negation semantics
            # equivalent to `ne` + verb (`Bach neniam aŭdis` = `Bach never heard`).
            if i > 0:
                prev = word_asts[i-1]
                if (prev.get("radiko") == "ne"
                        or (prev.get("vortspeco") == "korelativo"
                            and prev.get("korelativo_prefikso") == "neni")):
                    ast["negita"] = True
        # Object: any noun, pronoun, proper noun, correlative, or unknown word in accusative case (-n)
        elif ast["vortspeco"] in ["substantivo", "pronomo", "propra_nomo", "korelativo", "nekonata"] and ast["kazo"] == "akuzativo" and not sentence_ast["objekto"] and not in_subordinate and not is_pp_governed:
            sentence_ast["objekto"] = {"tipo": "vortgrupo", "kerno": ast, "priskriboj": []}
        # Subject: any noun, pronoun, proper noun, correlative, or unknown word in nominative case (no -n)
        elif ast["vortspeco"] in ["substantivo", "pronomo", "propra_nomo", "korelativo", "nekonata"] and ast["kazo"] == "nominativo" and not sentence_ast["subjekto"] and not in_subordinate and not is_pp_governed:
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

    # --- Issue #691: Parse subordinate clauses as nested frazo nodes ---
    # Handle ke-clauses, temporal clauses, etc.
    sentence_ast = parse_subordinate_clauses(sentence_ast, word_asts)
    # Build rilata_subfrazo nodes for relative clauses and attach to antecedents
    if relative_spans:
        sentence_ast = _attach_relative_clauses(sentence_ast, word_asts, relative_spans)

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

    # 3. Check for ki- correlatives that are QUESTION WORDS (not relative pronouns).
    # A ki-correlative inside a relative_span is a relative pronoun, not a question word.
    if not demandotipo:
        for i, ast in enumerate(word_asts):
            if ast.get("vortspeco") == "korelativo":
                prefix = ast.get("korelativo_prefikso", "")
                if prefix == "ki" and i not in relative_span_set:
                    is_question = True
                    demandotipo = 'ki'
                    break

    if is_question:
        fraztipo = 'demando'

    # Check for command (imperative mood)
    if not is_question and sentence_ast["verbo"]:
        if sentence_ast["verbo"].get("modo") == "imperativo":
            fraztipo = 'ordono'

    # Check for conditional sentence (kondiĉa) - Issue #22
    # Conditional sentences typically:
    # 1. Start with "Se" (if) and contain conditional mood verb (-us)
    # 2. Have a main verb in conditional mood (-us)
    if not is_question and fraztipo == 'deklaro':
        has_conditional_verb = False
        starts_with_se = False

        # Check if sentence starts with "Se" (if)
        for ast in word_asts:
            if ast.get("radiko", "").lower() == "se":
                starts_with_se = True
                break

        # Check for conditional mood verbs (-us ending)
        for ast in word_asts:
            if ast.get("modo") == "kondicionalo":
                has_conditional_verb = True
                break

        # Also check main verb
        if sentence_ast["verbo"] and sentence_ast["verbo"].get("modo") == "kondicionalo":
            has_conditional_verb = True

        # Mark as conditional if we have "Se" + conditional verb, or just conditional verb
        if has_conditional_verb and (starts_with_se or (sentence_ast["verbo"] and sentence_ast["verbo"].get("modo") == "kondicionalo")):
            fraztipo = 'kondiĉa'

    sentence_ast["fraztipo"] = fraztipo
    if demandotipo:
        sentence_ast["demandotipo"] = demandotipo

    # Add parse statistics (word-level success metrics) - Pure Esperanto
    #
    # BUG (#805 / #818, fixed 2026-07-13): this counted `analizstato == "sukceso"`
    # alone. But `malsukceso` is set on exactly ONE error path, and the parser
    # reaches `vortspeco: nekonata` and `vortspeco: fremda_vorto` by OTHER paths
    # that still stamp `sukceso`. Result:
    #
    #     parse("Xyzzy plugh frobnicate.")  -> sukcesoprocento 1.0
    #     parse("The quick brown fox.")     -> sukcesoprocento 1.0, neesperantaj_vortoj 0
    #
    # The metric reported 100% success on gibberish and on English, so it was a
    # constant and told us nothing. (Downstream, `sentences.success_rate` is 0.0
    # on all 5.39M rows for a *separate* reason — the store read an English key
    # the parser never emitted. Both bugs made the same column useless.)
    #
    # Success is a property of the CLASSIFICATION OUTCOME, not of a flag that is
    # almost never set to failure. `esperantaj_vortoj` means "Esperanto words" —
    # a `fremda_vorto` is, by definition, not one.
    total_words = len(word_asts)
    successful_words = sum(
        1 for ast in word_asts
        if ast.get("analizstato") == "sukceso"
        and ast.get("vortspeco") not in NON_ESPERANTO_VORTSPECOJ
    )
    failed_words = total_words - successful_words

    # Categorize the failed words
    categories = {}
    for ast in word_asts:
        if ast.get("analizstato") == "malsukceso":
            category = ast.get("kategorio", "nekonata")
            categories[category] = categories.get(category, 0) + 1

    sentence_ast["parse_statistics"] = {
        "tutaj_vortoj": total_words,
        "esperantaj_vortoj": successful_words,
        "neesperantaj_vortoj": failed_words,
        "sukcesoprocento": successful_words / total_words if total_words > 0 else 0.0,
        "analizkategorioj": categories
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

    # AST-context reanalysis: catches sentence-initial misclassifications
    # the word-level guard can't (e.g. "Aktuale en 2008 Minesoto estis...").
    _reanalyze_sentence_initial_misclassifications(sentence_ast)

    # Annotate multi-token entity groups (consecutive propra_nomo runs).
    _annotate_multi_token_entities(sentence_ast, word_asts)

    return sentence_ast


def _validate_sentence_initial_adjective_agreement(word_asts: list) -> None:
    """Revert capitalized words classified as adjektivo via the
    capitalization-guard exception when no agreeing head noun follows.

    Esperanto adjectives MUST agree with a noun in case + number. The -a
    capitalization exception (for "Hungaraj princoj") relies on this
    invariant: a real adjective will have an agreeing noun nearby. A
    word like "Maria" (singular nominative -a) might LOOK like an
    adjective but if the next non-article word is a verb (no nominal
    agreement target), Maria is a proper name with a coincidental -a.

    Run at every position (not just sentence-initial). The same logic
    applies to "Mona Lisa" where Lisa is at index 1.

    Fundamento-skip override: a word whose radiko is in Fundamento is
    normally a rock-solid adjective ("Granda" → grand). But when followed
    by another capitalized propra-noun-like token, the multi-token-name
    reading can dominate ("Mona Lisa" — both halves are names even though
    'mon' is in Fundamento). In that case, run the validation: if no
    real (lowercase) head noun follows, revert.

    Lookahead rules (search up to 5 tokens ahead):
      - Articles (la, l')                     TRANSPARENT
      - Adjektivos / adverbs / correlatives   TRANSPARENT
      - Conjunctions                          TRANSPARENT iff followed by
                                              [adj-surface ... noun-surface];
                                              otherwise BREAK
      - Verbs / prepositions                  BREAK
      - SUBSTANTIVO (lowercase common noun)   real agreement target —
                                              kazo+nombro match → keep adj
      - propra_nomo (capitalized) with adj-   skip; not a reliable target
        like surface or default morphology    (its kazo/nombro defaults
                                              are unreliable)
      - propra_nomo with noun-like surface    treat as agreement target
        (-o/-oj/-on/-ojn)
    """
    if not word_asts:
        return

    def _is_konj_with_coordinated_adj_noun(j: int) -> bool:
        """True if word[j] is a conjunction followed by an adj-surface
        token, then any number of adj-surface / function-word tokens,
        then a real (lowercase substantivo) head noun. Catches
        "kaj Polaj princoj" but rejects "kaj Pieter Bruegel pentris" and
        "kaj Maria pentris"."""
        if j + 2 >= len(word_asts):
            return False
        adj_candidate = word_asts[j + 1]
        if not isinstance(adj_candidate, dict):
            return False
        adj_pv = (adj_candidate.get('plena_vorto') or '').lower()
        if not _surface_looks_adj(adj_pv):
            return False
        for k in range(j + 2, min(j + 6, len(word_asts))):
            nxt = word_asts[k]
            if not isinstance(nxt, dict):
                return False
            nxt_vs = nxt.get('vortspeco')
            if nxt_vs in ('verbo', 'prepozicio'):
                return False
            if nxt_vs == 'substantivo':
                return True
            # adjective / article / function-word / propra_nomo with
            # adj-surface — keep scanning for a real head noun
        return False

    def _next_is_capitalized_propra_like(i: int) -> bool:
        """True if word[i+1] starts uppercase and is propra-noun-like
        (vortspeco propra_nomo / nekonata, or content classification —
        we treat any capitalized non-function word as propra-like)."""
        if i + 1 >= len(word_asts):
            return False
        nxt = word_asts[i + 1]
        if not isinstance(nxt, dict):
            return False
        nxt_pv = nxt.get('plena_vorto') or ''
        if not nxt_pv or not nxt_pv[0].isupper():
            return False
        nxt_vs = nxt.get('vortspeco')
        return nxt_vs in (
            'propra_nomo', 'nekonata', 'adjektivo', 'substantivo', 'adverbo'
        )

    for i, ast in enumerate(word_asts):
        if not isinstance(ast, dict):
            continue
        if ast.get('vortspeco') != 'adjektivo':
            continue
        pv = ast.get('plena_vorto') or ''
        if not pv or not pv[0].isupper():
            continue

        rad = (ast.get('radiko') or '').lower()
        radiko_in_fund = rad in _FUNDAMENTO_ROOTS or rad in PROTECTED_ROOTS
        # Skip Fundamento-rooted adjectives only when the next token is
        # NOT a capitalized propra-noun-like word. With a capitalized
        # neighbor the multi-token-name reading is plausible — validate.
        if radiko_in_fund and not _next_is_capitalized_propra_like(i):
            continue

        target_kazo = ast.get('kazo', 'nominativo')
        target_nombro = ast.get('nombro', 'singularo')
        has_agreement = False
        for j in range(i + 1, min(i + 6, len(word_asts))):
            nxt = word_asts[j]
            if not isinstance(nxt, dict):
                continue
            nxt_vs = nxt.get('vortspeco')
            nxt_pv = (nxt.get('plena_vorto') or '')
            nxt_pv_lower = nxt_pv.lower()

            if nxt_vs == 'artikolo':
                continue
            if nxt_vs in ('verbo', 'prepozicio'):
                break
            if nxt_vs == 'konjunkcio':
                if _is_konj_with_coordinated_adj_noun(j):
                    continue
                break
            if nxt_vs == 'substantivo':
                if (nxt.get('kazo') == target_kazo
                        and nxt.get('nombro') == target_nombro):
                    has_agreement = True
                break
            if nxt_vs == 'propra_nomo':
                # Capitalized propra_nomo with adj-like surface: not a
                # reliable head — its default kazo/nombro just happens to
                # match singular nominative -a. Keep looking past it.
                if (nxt_pv and nxt_pv[0].isupper()
                        and _surface_looks_adj(nxt_pv_lower)):
                    continue
                # Noun-surface propra_nomo (-o/-oj/-on/-ojn) is a real
                # head — check kazo+nombro.
                if _surface_looks_noun(nxt_pv_lower):
                    if (nxt.get('kazo') == target_kazo
                            and nxt.get('nombro') == target_nombro):
                        has_agreement = True
                    break
                # Other propra_nomo surface (foreign name like 'Bach'):
                # treat as a real head; trust kazo+nombro.
                if (nxt.get('kazo') == target_kazo
                        and nxt.get('nombro') == target_nombro):
                    has_agreement = True
                break
            # adjektivo / adverbo / pronomo / korelativo: transparent.

        if not has_agreement:
            ast['vortspeco'] = 'propra_nomo'
            ast['kategorio'] = 'propranomo'
            ast['propra_nomo_evidence'] = 'adjective_unlicensed'
            ast['radiko'] = pv
            ast['_reverted_from_adjektivo'] = True


# Connector tokens permitted INSIDE a multi-token entity run (e.g.
# "Lost in Space", "Mona de la Casa", "Tower of London"). A connector
# does not start or end a run — it's only kept if a propra_nomo follows
# within the lookahead window. Coordinating words (kaj/aŭ/sed/and/or/but)
# are deliberately excluded — they separate distinct entities.
_MULTI_TOKEN_CONNECTORS = frozenset({
    # Esperanto prepositions / articles / common joiners (lowercase form)
    'de', 'da', 'en', 'al', 'kun', 'sen', 'por', 'pri', 'pro',
    'ĉe', 'sur', 'sub', 'super', 'inter', 'tra', 'trans', 'apud',
    'kontraŭ', 'la', "l'",
    # English / Romance / Germanic connectors that survive in titles
    # carried into Esperanto text (e.g. "Lost in Space", "Tower of London",
    # "Friends of the Earth", "von Neumann", "del Monte").
    'in', 'of', 'the', 'on', 'at', 'to', 'for', 'a', 'an',
    'le', 'les', 'des', 'du', 'di', 'del', 'dei', 'della',
    'der', 'die', 'das', 'von', 'van', 'el',
})


def _annotate_multi_token_entities(sentence_ast: dict, word_asts: list) -> None:
    """Mark runs of 2+ propra_nomo tokens (with optional connector words
    between them) as multi-token entities.

    Real-world examples:
      - "Bill Gates"             → 2 contiguous propra_nomo tokens
      - "Mona Lisa"              → 2 contiguous propra_nomo tokens
      - "Lost in Space"          → 2 propra + 1 connector ('in')
      - "Tower of London"        → 2 propra + 1 connector ('of')
      - "Joan of Arc"            → 2 propra + 1 connector ('of')
      - "Ludwig van Beethoven"   → 2 propra + 1 connector ('van')

    A run accumulates tokens if either:
      - vortspeco is 'propra_nomo', OR
      - vortspeco is 'nekonata' AND plena_vorto starts with uppercase

    Lowercase connector words from `_MULTI_TOKEN_CONNECTORS` are allowed
    BETWEEN propra tokens. A connector never starts or ends a run; if
    after a connector no propra token appears within 2 positions, the
    run terminates without including the connector.

    Coordinating conjunctions (kaj, aŭ, and, or, but, sed, …), verbs,
    and any other non-connector function word BREAK the run.

    Output (when at least one run has ≥2 propra tokens):
      sentence_ast['multi_token_entities'] = [
        {
          'positions':  [i, j, ...],   # indices of CAPITALIZED propra tokens
          'tokens':     [pv_i, pv_j],  # surface forms of those propra tokens
          'span':       [i, k, j, ...],# indices of ALL tokens in the span
                                       # (propra + connectors)
          'span_tokens':[pv_i, in, pv_j, ...],  # surface forms of the span
        },
        ...
      ]
    The legacy `positions` / `tokens` fields preserve the prior contract
    (propra-only). The new `span` / `span_tokens` carry the full surface.
    """

    def is_propra_token(w_ast: dict) -> bool:
        if not isinstance(w_ast, dict):
            return False
        vs = w_ast.get('vortspeco')
        pv = w_ast.get('plena_vorto') or ''
        if vs == 'propra_nomo':
            return True
        if vs == 'nekonata' and pv and pv[0].isupper():
            return True
        return False

    def is_connector_token(w_ast: dict) -> bool:
        if not isinstance(w_ast, dict):
            return False
        pv = (w_ast.get('plena_vorto') or '').lower().strip("'")
        if not pv:
            return False
        # Use surface form, not radiko/vortspeco. Connectors are matched
        # by their surface text: lowercase Esperanto prepositions or
        # foreign function words like 'in', 'of', 'the'.
        return pv in _MULTI_TOKEN_CONNECTORS or (pv + "'") in _MULTI_TOKEN_CONNECTORS

    groups = []
    n = len(word_asts)
    i = 0
    while i < n:
        if not is_propra_token(word_asts[i]):
            i += 1
            continue
        # Start of a run.
        propra_positions = [i]
        span_positions = [i]
        j = i + 1
        while j < n:
            if is_propra_token(word_asts[j]):
                propra_positions.append(j)
                span_positions.append(j)
                j += 1
                continue
            if is_connector_token(word_asts[j]):
                # Lookahead: is there a propra_token within next 2 positions
                # (allowing one more connector in between)?
                lookahead = False
                k = j + 1
                steps = 0
                while k < n and steps < 2:
                    if is_propra_token(word_asts[k]):
                        lookahead = True
                        break
                    if is_connector_token(word_asts[k]):
                        k += 1
                        steps += 1
                        continue
                    break
                if lookahead:
                    span_positions.append(j)
                    j += 1
                    continue
                # No propra ahead — terminate run here, don't include connector.
                break
            # Non-propra, non-connector — terminate run.
            break

        if len(propra_positions) >= 2:
            groups.append({
                'positions':   list(propra_positions),
                'tokens':      [word_asts[k].get('plena_vorto')
                                for k in propra_positions],
                'span':        list(span_positions),
                'span_tokens': [word_asts[k].get('plena_vorto')
                                for k in span_positions],
            })
        i = j
    if groups:
        sentence_ast['multi_token_entities'] = groups


# Vortspecos we'll reclassify into via piece 2 (sentence-initial adverb
# misclassification). We DELIBERATELY restrict to 'adverbo' only:
#
# - 'substantivo' / 'adjektivo' are handled at the word level by piece 1
#   (extended capitalization guard accepts -o/-oj/-on/-ojn and -a/-aj/-an/-ajn).
# - 'verbo' would catch Einstein/Lincoln-style misclassifications by
#   parse_word (which sometimes treats foreign names as neologism verbs)
#   — but THOSE are real propra_nomos we want to keep, so excluded.
# - Function-word vortspecos (korelativo, pronomo, …) are now handled at
#   the word level by the extended correlative-suffix logic; piece 2 doesn't
#   need to repeat them.
#
# Adverbs are uniquely the case where the parser's word-level heuristic
# can't disambiguate (Esperanto adverb vs foreign name with incidental -e
# ending), so the case-marking-of-alt-subject test is the right discriminator.
_RECLASSIFIABLE_VORTSPECOS = frozenset({'adverbo'})


def _vortgrupo_kerno_or_self(node):
    if not isinstance(node, dict):
        return None
    if node.get('tipo') == 'vortgrupo':
        return node.get('kerno')
    return node


def _reanalyze_sentence_initial_misclassifications(frazo: dict) -> None:
    """Second-pass disambiguation using full Frazo AST context.

    Targets the failure mode where a sentence-initial word is tagged
    propra_nomo only because of capitalization, not because it's actually
    a name. The discriminator uses two AST signals together:

      1. The kerno's lowercase form parses cleanly as a non-propra
         Esperanto vortspeco (adverbo / substantivo / adjektivo / etc.).
         This excludes Bach/Shakespeare/Goethe whose lowercase forms
         either fail morphology (nekonata) or aren't found in DICTIONARY.

      2. There is an alternative nominative-case subject candidate
         elsewhere in the Frazo's `aliaj` (a substantivo or propra_nomo
         in nominative case). This is the case-marking discriminator:
         "Aktuale en 2008 Minesoto estis…" has Minesoto in nominative,
         so the adverb interpretation is plausible. "Shakespeare verkis
         dramojn" has only Shakespeare in nominative, so it stays
         propra_nomo.

    When both conditions hold, we relabel the kerno's vortspeco/radiko
    to match the lowercase parse. We do NOT restructure the AST (move
    the alt subject into subjekto position) because that would require
    invasive surgery on Vortgrupo / aliaj edges that downstream consumers
    rely on. The relabel alone is enough to stop the rerank from
    treating these as candidate answers.
    """
    if not isinstance(frazo, dict):
        return
    if frazo.get('tipo') != 'frazo':
        return

    subjekto = frazo.get('subjekto')
    kerno = _vortgrupo_kerno_or_self(subjekto)
    if not isinstance(kerno, dict):
        return
    if kerno.get('vortspeco') != 'propra_nomo':
        return

    pv = kerno.get('plena_vorto') or ''
    if not pv:
        return

    # Step 1: does the lowercase form parse cleanly as a non-propra word?
    lower_ast = parse_word(pv.lower())
    new_vs = lower_ast.get('vortspeco')
    if new_vs not in _RECLASSIFIABLE_VORTSPECOS:
        return

    # Bug #5 extra guards: skip demotion when the lowercase form is
    # already flagged as foreign-origin by the morphology layer.
    #
    # Signal 1 — surface orthography: `goethe` has the `th` digraph,
    # `brontë` has `ë`. Pure Esperanto adverbs never have these.
    #
    # Signal 2 — neologism / fremda_vorto kategorio: parser already
    # flagged the lowercase form as foreign-origin (Crusoe → kategorio
    # 'neologismo'). Real adverbs (`aktuale`, `krome`) have kategorio
    # None.
    #
    # If EITHER fires, this is a real propra_nomo with incidental -e
    # ending — don't demote. (`aktuale` / `krome` pass both checks
    # and remain eligible for legitimate adverb-misclassification
    # demotion when the alt-subject signal fires.)
    if _has_foreign_orthography(pv):
        return
    if lower_ast.get('kategorio') in ('neologismo', 'fremda_vorto'):
        return
    # Reanalysis-eligibility gate. The original code restricted to
    # kategorio='propranomo' to be Shakespeare-safe — but Shakespeare-style
    # cases are already excluded by Step 1 (its lowercase parses to
    # propra_nomo via foreign-letter fast-path, not adverbo). The
    # propranomo_konata-only filter was over-cautious: it blocked
    # legitimate adverbs like "Aktuale" / "Krome" when the v3 Wikipedia-
    # extended dict happened to know an Esperanto magazine of the same
    # name. We now allow propranomo_konata too, IFF Step 1 already passed
    # — Step 2 (alt-subject case-marking) is the real discriminator.
    cur_kat = kerno.get('kategorio')
    if cur_kat not in ('propranomo', 'propranomo_konata'):
        # 'propranomo_esperantigita' or other categories — don't touch.
        return

    # Step 2: is there an alternative subject candidate in `aliaj`?
    # Subject candidates are substantivo / propra_nomo / pronomo in
    # nominative case. Pronouns count because "Krome li parolis" or
    # "Anstataŭ ili venis" are pronoun-subject sentences where the
    # sentence-initial adverb is the misclassification.
    #
    # Bug #5 exception: if aliaj contains a coordinator (kaj / aŭ / sed),
    # the "alt subject" candidate is most likely a COORDINATED subject,
    # not an alternative. `Goethe kaj Schiller verkis multajn dramojn`
    # → Schiller looks like an alt subject by case-marking, but `kaj`
    # before it signals coordination. Both Goethe and Schiller should
    # remain propra_nomo subjects; don't demote Goethe to adverbo.
    aliaj = frazo.get('aliaj') or []
    has_coordinator = any(
        isinstance(item, dict)
        and item.get('vortspeco') == 'konjunkcio'
        and (item.get('radiko') or '').lower() in ('kaj', 'aŭ', 'sed', 'kvankam')
        for item in aliaj
    )
    if has_coordinator:
        return  # coordination, not alternation — keep propra_nomo

    has_alt_subject = False
    for item in aliaj:
        target = _vortgrupo_kerno_or_self(item)
        if not isinstance(target, dict):
            continue
        vs = target.get('vortspeco')
        if vs in ('substantivo', 'propra_nomo', 'pronomo'):
            # Subject candidates are nominative; pronouns by themselves
            # imply subject role when they appear bare.
            if target.get('kazo') == 'nominativo':
                has_alt_subject = True
                break

    if not has_alt_subject:
        return  # no alternative, keep propra_nomo (Shakespeare-safe)

    # Both conditions met — relabel the kerno.
    new_radiko = lower_ast.get('radiko') or kerno.get('radiko')
    kerno['vortspeco'] = new_vs
    if new_radiko:
        kerno['radiko'] = new_radiko
    kerno['kategorio'] = lower_ast.get('kategorio')
    # Mark for downstream debugging / metrics.
    kerno['_reanalyzed_from'] = 'propra_nomo'

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
