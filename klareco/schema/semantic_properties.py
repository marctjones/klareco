"""
Semantic Properties Extension for Kuzu Radiko Node (v2.1)

Extends the Radiko node with semantic classification properties needed
for schema-based summarization.

Issue: #655
Related: #664 (Pure Esperanto), #665 (Tier System 2.0)
"""

# ============================================================================
# SEMANTIC PROPERTIES TO ADD TO RADIKO NODE
# ============================================================================

SEMANTIC_PROPERTIES_SCHEMA = """
-- Extend Radiko node with semantic properties
-- Note: Kuzu syntax is: ALTER TABLE <table> ADD <column> <type>
-- No IF NOT EXISTS, no DEFAULT values (must be set via UPDATE)

ALTER TABLE Radiko ADD funda_stato STRING;
ALTER TABLE Radiko ADD estas_funda BOOLEAN;
ALTER TABLE Radiko ADD estas_funkcia BOOLEAN;
ALTER TABLE Radiko ADD estas_semantika BOOLEAN;
ALTER TABLE Radiko ADD ofteca_tavolo INT64;
ALTER TABLE Radiko ADD verba_klaso STRING;
ALTER TABLE Radiko ADD aspekta_klaso STRING;
ALTER TABLE Radiko ADD substantiva_klaso STRING;
ALTER TABLE Radiko ADD semantika_kampo STRING;
ALTER TABLE Radiko ADD graveco_biografia DOUBLE;
ALTER TABLE Radiko ADD graveco_difina DOUBLE;
ALTER TABLE Radiko ADD graveco_okazaĵa DOUBLE;
ALTER TABLE Radiko ADD mem_anotita BOOLEAN;
ALTER TABLE Radiko ADD konfido DOUBLE;
ALTER TABLE Radiko ADD fonto STRING;
"""

# ============================================================================
# SEMANTIC CLASS TAXONOMY (Pure Esperanto)
# ============================================================================

# Layer 1: Leksika Semantiko (Lexical Semantics)
# Based on VerbNet and WordNet, adapted for Esperanto

VERB_CLASSES = {
    # Creation/Destruction (kreado)
    "kreado-26": "Creation: fond, kre, produk, konstrui, starigi",
    "detruo-44": "Destruction: detru, rompi, forigi, abolici",

    # Movement (movo)
    "movo-51": "Motion: ir, veni, fali, salti, flugi, naĝi",
    "translokigo-11": "Transfer: don, send, alport, transdoni",

    # Communication (komunikado)
    "diro-37": "Saying: dir, rakont, klarig, anonc, paroli",
    "demando-40": "Question: demand, demandi, esplor, serĉi",

    # Cognition (pensado)
    "pensado-29": "Think: pens, konsider, medit, kred, imag",
    "scio-30": "Know: sci, kompren, koni, lern",

    # Perception (percepto)
    "vido-30": "See: vid, observ, rimark, ekvidi",
    "aŭdo-47": "Hear: aŭd, aŭskult",

    # Emotion (sento)
    "amo-31": "Love: am, ŝat, admir, estimestim",
    "timo-31": "Fear: tim, angoris, tremi",

    # Change (ŝanĝo)
    "ŝanĝo-45": "Change: ŝanĝ, modif, reform, konvert",
    "kreskado-26": "Grow: kresk, develop, evolui",

    # State (stato)
    "ekzisto-47": "Exist: est, ekzist, viv, mort",
    "havado-100": "Have: hav, poses, apar",
}

ASPECT_CLASSES = {
    "stato": "State: continuous, no change (esti, havi, sci)",
    "aktiveco": "Activity: ongoing, no endpoint (kuri, labori, paroli)",
    "plenumigo": "Accomplishment: has duration and endpoint (konstrui, skribi)",
    "atingaĵo": "Achievement: instantaneous change (trovi, komenci, morti)",
}

NOUN_CLASSES = {
    # People (personoj)
    "persono": "Person: homo, vir, virino, infano, maljunulo",
    "profesio": "Profession: kuracisto, instruisto, inĝeniero",
    "rolo": "Role: patro, amiko, reganto, leĝdonanto",

    # Animals/Plants (vivantaj)
    "animalo": "Animal: hundo, kato, ĉevalo, birdo",
    "planto": "Plant: arbo, floro, herbo, legomo",

    # Places (lokoj)
    "loko": "Location: loko, urbo, lando, regiono",
    "konstruaĵo": "Building: domo, palaco, ponto, preĝejo",
    "natura_loko": "Natural: monto, rivero, lago, arbaro",

    # Abstract (abstrakt)
    "koncepto": "Concept: ideo, teorio, principo, sistemo",
    "kvalito": "Quality: beleco, boneco, forto, saĝeco",
    "evento": "Event: okazaĵo, festo, milito, renkontiĝo",

    # Objects (objektoj)
    "ilo": "Tool: martelo, tranĉilo, komputilo",
    "veturilo": "Vehicle: aŭto, trajno, ŝipo, aviadilo",
    "manĝaĵo": "Food: pano, viando, legomo, frukt",
}

SEMANTIC_FIELDS = {
    "socia": "Social: politics, family, society",
    "scienca": "Science: technology, research, knowledge",
    "natura": "Nature: environment, animals, plants",
    "kultura": "Culture: art, religion, tradition",
    "ekonomia": "Economy: business, trade, money",
}

# ============================================================================
# SCHEMA IMPORTANCE TEMPLATES (Layer 4)
# ============================================================================

BIOGRAPHICAL_IMPORTANCE = {
    # Verb classes
    "kreado-26": 0.95,      # Founded, created (high importance)
    "ekzisto-47": 0.85,     # Was, lived (identification)
    "movo-51": 0.60,        # Went, moved (medium)
    "pensado-29": 0.70,     # Thought, believed (motivation)

    # Noun classes
    "persono": 1.0,         # Person identification
    "profesio": 0.90,       # Profession
    "loko": 0.75,           # Birth/death place
    "evento": 0.80,         # Major life events
}

DEFINITIONAL_IMPORTANCE = {
    # Verb classes
    "ekzisto-47": 1.0,      # Is (category assignment)
    "havado-100": 0.90,     # Has (essential property)
    "kreado-26": 0.60,      # Created by (origin)

    # Noun classes
    "koncepto": 1.0,        # Category
    "kvalito": 0.90,        # Essential quality
    "ilo": 0.80,            # Function/purpose
}

EVENT_IMPORTANCE = {
    # Verb classes
    "kreado-26": 1.0,       # Main action
    "movo-51": 0.85,        # Movement/arrival
    "diro-37": 0.75,        # Announcement/speech

    # Noun classes
    "persono": 0.90,        # Participants
    "loko": 0.85,           # Location
    "evento": 1.0,          # Event type
}

# ============================================================================
# FOUNDATIONAL STATUS CATEGORIES (Dimension 1)
# ============================================================================

FUNDA_STATOJ = {
    "fundamento_kerno": {
        "description": "One of 917 Fundamento roots",
        "priority": 1.0,
        "examples": ["hund", "tabl", "bel", "ir", "parol"]
    },
    "vortaro_agnoskita": {
        "description": "Recognized in ReVo/PIV (~9,000 roots)",
        "priority": 0.7,
        "examples": ["komputik", "televid", "retpoŝt"]
    },
    "neologismo": {
        "description": "Modern coinage, not in dictionaries",
        "priority": 0.3,
        "examples": ["blogoj", "tviter", "guglo"]
    }
}

# ============================================================================
# FUNCTION WORD CATEGORIZATION
# ============================================================================

PURE_GRAMMATICAL_FUNCTION_WORDS = [
    # Articles, pronouns, determiners (no semantic content)
    "la", "mi", "vi", "li", "ŝi", "ĝi", "ni", "ili",
    "tiu", "ĉi", "tio", "ĉio", "neni",
    # Grammatical particles
    "-n", "-j", "-jn",  # Endings (not roots, but tracked)
]

SEMANTIC_FUNCTION_WORDS = [
    # These NEED embeddings (semantic content)
    # Prepositions (spatial/temporal meaning)
    "de", "al", "en", "sur", "sub", "antaŭ", "post", "inter", "ĉe", "apud",
    "ĝis", "tra", "trans", "super", "kontraŭ", "laŭ", "sen", "krom",

    # Conjunctions (logical relations)
    "kaj", "aŭ", "sed", "ĉar", "se", "kvankam", "dum", "kiam",
    "tial", "do", "tamen", "eĉ", "ankaŭ",

    # Causal/manner
    "pro", "per", "anstataŭ", "spite", "kiel", "kvazaŭ",

    # Quantifiers (semantic meaning)
    "multe", "iom", "sufiĉe", "tro", "pli", "plej", "malpli", "malplej",
]

# ============================================================================
# EXAMPLE ANNOTATIONS (For Phase 0 - 50 roots)
# ============================================================================

PHASE_0_EXAMPLES = {
    "fond": {
        "verba_klaso": "kreado-26",
        "aspekta_klaso": "plenumigo",
        "semantika_kampo": "socia",
        "graveco_biografia": 0.95,
        "graveco_difina": 0.30,
        "graveco_okazaĵa": 0.90,
        "funda_stato": "fundamento_kerno",
        "ofteca_tavolo": 0,
    },
    "hund": {
        "substantiva_klaso": "animalo",
        "semantika_kampo": "natura",
        "graveco_biografia": 0.40,
        "graveco_difina": 0.60,
        "graveco_okazaĵa": 0.30,
        "funda_stato": "fundamento_kerno",
        "ofteca_tavolo": 1,
    },
    "est": {
        "verba_klaso": "ekzisto-47",
        "aspekta_klaso": "stato",
        "semantika_kampo": "socia",
        "graveco_biografia": 0.85,
        "graveco_difina": 1.0,
        "graveco_okazaĵa": 0.70,
        "funda_stato": "fundamento_kerno",
        "ofteca_tavolo": 0,
    },
}
