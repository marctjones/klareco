"""A from-scratch, pure Python parser for Esperanto.

This parser is built on the 16 rules of Esperanto and does not use any
external parsing libraries like Lark. It performs morphological and syntactic
analysis to produce a detailed, Esperanto-native Abstract Syntax Tree (AST)."""
import re
import json
from pathlib import Path

try:
    from data.merged_vocabulary import MERGED_ROOTS as DICTIONARY_ROOTS
except ImportError:
    # Fallback to Gutenberg dictionary if merged vocabulary not available
    from data.extracted_vocabulary import DICTIONARY_ROOTS

# Load Fundamento roots (authoritative, tier 1 vocabulary)
# These are used to disambiguate prefix/suffix conflicts
_FUNDAMENTO_ROOTS = set()
try:
    _fundamento_path = Path(__file__).parent.parent / "data" / "vocabularies" / "fundamento_roots.json"
    if _fundamento_path.exists():
        with open(_fundamento_path, 'r', encoding='utf-8') as f:
            _fundamento_data = json.load(f)
            _FUNDAMENTO_ROOTS = set(_fundamento_data.get('roots', {}).keys())
except Exception:
    pass  # Silently fall back to empty set if file not found

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
# --- Protected Roots: Fundamento roots that look like they contain affixes
# --- These must NEVER be decomposed - they are atomic roots.
# -----------------------------------------------------------------------------

# Roots that start with prefix-like sequences but are NOT prefixed
# Example: "bona" is NOT "bo-" + "n" + "-a", it's "bon" + "-a"
PROTECTED_PREFIX_ROOTS = {
    # bo- look-alikes (bo- means "in-law")
    "bon", "bord", "bot", "botel", "bov", "bor",
    # dis- look-alikes (dis- means "dispersal")
    "disk", "disput", "displac", "dispon",
    # ek- look-alikes (ek- means "begin/sudden")
    "ekzamen", "ekzekut", "ekzempl", "ekzempler", "ekonomi", "ekzist", "ekskuz", "ekskurs",
    # eks- look-alikes (eks- means "former")
    "ekspres", "eksplod", "eksperiment", "eksport", "eksped", "ekspoz", "ekspans",
    # fi- look-alikes (fi- means "shameful")
    "fil", "fier", "figur", "fidel", "fin", "fiŝ", "fiĥ", "fiber", "fizik", "fiks",
    # for- look-alikes (for- means "away")
    "form", "fort", "forg", "forges", "formul", "formal",
    # mis- look-alikes (mis- means "wrongly")
    "mister", "misi",
    # pra- look-alikes (pra- means "primal/ancient")
    "praktik", "prav", "prateri",
    # re- look-alikes (re- means "again")
    "region", "reg", "rest", "respond", "respekt", "respir", "rezult", "rezist",
    "recenz", "recept", "redakt", "reflex", "reform", "reflekt", "registr",
    "reliĝ", "relief", "relativ", "reklam", "rekompenc", "rekord", "rekt",
    "rempar", "rent", "report", "represent", "reprodukt", "republik", "reput",
    "rezerv", "rezign", "rezon", "retir", "retor", "revizor", "revoluci",
    "redut", "refut", "rek", "rekomend", "remed", "retori", "revizi", "revu",
    # vic- look-alikes (vic- means "vice-")
    "vid", "viktim", "vilaĝ", "viol", "violet", "violon", "viper", "vigl",
}

# Roots that end with suffix-like sequences but are NOT suffixed
# Example: "rapida" is NOT "rap" + "-id" + "-a", it's "rapid" + "-a"
PROTECTED_SUFFIX_ROOTS = {
    # -ad look-alikes (-ad means "continuous action")
    "salad", "nomad", "balad", "parad", "tirad", "arkad", "fasad", "dekad",
    # -aĵ look-alikes (-aĵ means "concrete thing")
    "mesaĝ", "bagaĝ", "garaĝ", "vojaĝ", "estaĵ", "kuiraĵ",
    # -ar look-alikes (-ar means "collection")
    "altar", "bazaar", "kalendar", "polar", "poplar", "regular", "stellar",
    # -ec look-alikes (-ec means "quality")
    "sekret",
    # -eg look-alikes (-eg means "augmentative")
    "leg", "neg", "seg", "norwegian", "kolleg",
    # -ej look-alikes (-ej means "place")
    "fej",
    # -em look-alikes (-em means "tendency")
    "problem", "sistem", "poem", "teorem", "ekstrem", "stratagem",
    # -er look-alikes (-er means "particle")
    "paper", "danger", "kurier", "karakter", "minister", "jupiter",
    # -estr look-alikes (-estr means "leader")
    "maestr", "orkestr",
    # -ar look-alikes (-ar means "collection")
    "avar", "cezar", "cigar", "dolar", "familiar", "hangar", "konsular", "popular", "solar",
    # -er look-alikes (-er means "particle")
    "difer", "elster", "fajfer", "kajer", "konsider", "lucer", "maner", "miser", "moder",
    "muster", "numer", "oper", "profer", "puder", "refer", "super", "teler", "toler", "veter",
    # -et look-alikes (-et means "diminutive")
    "bilet", "diet", "poet", "kabin", "sekret", "planet", "magnet", "alfabet",
    "violon", "balot", "pilot", "kariot",
    "alumet", "biljet", "bufet", "duet", "kabinet", "kadet", "kaset", "ĵaket", "koket",
    "komplet", "kornet", "korset", "kvartet", "mulet", "oktet", "paket", "parket", "raket",
    "sonet", "stilet", "tablet", "tapet", "triket", "trompet",
    # -id look-alikes (-id means "offspring")
    "rapid", "fluid", "guid", "solid", "valid", "stupid", "morbid", "timid",
    "vivid", "arid", "horrid", "hybrid", "humid", "lucid", "placid", "rigid",
    # -ig look-alikes (-ig means "causative")
    "vestig", "orig", "prodig", "vertigo",
    # -iĝ look-alikes (-iĝ means "become")
    "refuĝ", "prestiĝ",
    # -il look-alikes (-il means "tool")
    "april", "gentil", "simil", "mobil", "facil", "fragil", "fertil", "humil",
    "civil", "docil", "agil", "subtil", "util", "viril", "stabil",
    "fibril", "fossil", "lentil", "papil", "penicil", "pupil", "reptil", "tonsil",
    "vakul", "vanil", "ventil", "vigil",
    # -in look-alikes (-in means "feminine")
    "latin", "basin", "marin", "maŝin", "vitamin", "kuzin", "origin", "domin",
    "admin", "kabin", "rabin", "rubin", "medic", "imagin", "termin", "ekzamin",
    "delfin", "kafetin", "buletein", "krokodil",
    # -ind look-alikes (-ind means "worthy")
    "hind",
    # -ing look-alikes (-ing means "holder")
    "puding", "ring", "viking", "spring", "pudding", "sterling",
    # -ism look-alikes (-ism means "doctrine")
    "organism", "optimism", "prism", "turism", "ateism",
    # -ist look-alikes (-ist means "practitioner")
    "artist", "baptist", "list", "krist",
    # -ul look-alikes (-ul means "person")
    "formul", "insul", "konsul", "stimul", "regul", "artikul", "kalkul",
    "angul", "modul", "akumul", "ampul", "kapitul", "tuberkul", "kapsul",
    "betul", "muskul",
    # -um look-alikes (-um means "indefinite")
    "album", "forum", "museum", "medium", "maksimum", "minimum", "stadium",
    "akvarium", "petroleum", "opium", "premium", "uranium", "aluminum",
    # -end look-alikes (-end means "must be done")
    "send", "tend", "legend", "blend", "trend", "dividend", "reverend",
}

# Combined set for fast lookup
PROTECTED_ROOTS = PROTECTED_PREFIX_ROOTS | PROTECTED_SUFFIX_ROOTS


# -----------------------------------------------------------------------------
# --- Layer 1: Morphological Analyzer (Fundamento-first design)
# --- See wiki: Esperanto-Parser-Design.md for architecture
# -----------------------------------------------------------------------------

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
        # Otherwise continue with regular parsing for inflected numbers

    # ==========================================================================
    # STEP 2: Correlative Check (45 entries)
    # ==========================================================================

    correl_check = lower_word
    correl_accusative = False
    if correl_check.endswith("n"):
        correl_check = correl_check[:-1]
        correl_accusative = True

    if correl_check in KNOWN_CORRELATIVES:
        ast["vortspeco"] = "korelativo"
        ast["radiko"] = correl_check
        if correl_accusative:
            ast["kazo"] = "akuzativo"

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
        """Check if s = prefix + Fundamento root. Return (prefix, root) or None."""
        for prefix in sorted_prefixes:
            if s.startswith(prefix) and len(s) > len(prefix):
                remainder = s[len(prefix):]
                # Check if remainder is a Fundamento root
                if remainder in _FUNDAMENTO_ROOTS:
                    return (prefix, remainder)
                # Check if remainder minus suffixes is Fundamento
                fund = find_fundamento_root(remainder)
                if fund:
                    return (prefix, fund)
        return None

    def check_suffix_gives_fundamento(s: str) -> tuple[str, list[str]] | None:
        """Check if stripping suffixes from s leads to a Fundamento root.

        Returns (root, [suffix1, suffix2, ...]) or None.
        Suffixes are returned in extraction order (right-to-left).
        """
        temp = s
        extracted = []
        for _ in range(3):  # Max 3 suffix layers
            for suffix in sorted_suffixes:
                if temp.endswith(suffix) and len(temp) > len(suffix) + 1:
                    potential = temp[:-len(suffix)]
                    # Check if potential is a Fundamento root
                    if potential in _FUNDAMENTO_ROOTS:
                        extracted.append(suffix)
                        return (potential, extracted)
                    # Check if we can continue stripping
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
            # E.g., after extracting "mal-" from "malbon", stem="bon" is Fundamento
            # Do NOT try to extract "bo-" from "bon"!
            if stem in _FUNDAMENTO_ROOTS or stem in PROTECTED_ROOTS:
                break

            found_prefix = False
            for prefix in sorted_prefixes:
                if stem.startswith(prefix) and len(stem) > len(prefix):
                    remainder = stem[len(prefix):]

                    # CRITICAL: Only extract prefix if:
                    # 1. The ORIGINAL stem (before extraction) is NOT a protected root
                    # 2. The remainder leads to a valid root
                    #
                    # Example: "malbon" is NOT protected, and "bon" is a valid root
                    # So we extract "mal-" + "bon"
                    #
                    # Example: "region" IS protected (looks like re-gion)
                    # So we do NOT extract - it's atomic

                    # Check if remainder leads to a Fundamento root
                    fund_root = find_fundamento_root(remainder)
                    if fund_root:
                        # Valid prefix extraction!
                        extracted_prefixes.append(prefix)
                        stem = remainder
                        found_prefix = True
                        break

                    # Also check if remainder is in KNOWN_ROOTS
                    # Note: remainder being in PROTECTED_PREFIX_ROOTS is OK here!
                    # E.g., "mal" + "bon" → "bon" is protected (starts with bo-),
                    # but that's fine because "bon" IS a valid Fundamento root.
                    if remainder in KNOWN_ROOTS:
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
                if stem.endswith(suffix) and len(stem) > len(suffix) + 1:
                    potential = stem[:-len(suffix)]

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
    # STEP 7: Identify Root (with compound word fallback)
    # ==========================================================================

    if stem in KNOWN_ROOTS or stem in _FUNDAMENTO_ROOTS or stem in PROTECTED_ROOTS:
        ast["radiko"] = stem
        return ast

    if stem in KNOWN_PARTICLES or stem in KNOWN_PREPOSITIONS:
        ast["radiko"] = stem
        return ast

    # Try compound word decomposition (root + root)
    if len(stem) >= 4:
        # Pattern 1: root1 + o + root2 (linking vowel)
        for i in range(2, len(stem) - 2):
            first = stem[:i]
            rest = stem[i:]

            if rest.startswith("o") and len(rest) > 2:
                second = rest[1:]
                if first in KNOWN_ROOTS and second in KNOWN_ROOTS:
                    ast["radiko"] = second  # Head is typically second root
                    ast["kunmetitaj_radikoj"] = [first, second]
                    return ast

            # Pattern 2: root1 + root2 (no linking vowel)
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

    # Last resort: treat stem as unknown root
    # Still set it as the root so the AST is complete
    ast["radiko"] = stem

    # If we couldn't find a valid root, raise an error
    if stem not in KNOWN_ROOTS and stem not in _FUNDAMENTO_ROOTS:
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
        "prefiksoj": [],
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
    # Uses Fundamento-first design for better disambiguation
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
