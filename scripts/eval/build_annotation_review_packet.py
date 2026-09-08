#!/usr/bin/env python3
"""Build a non-Esperanto-speaker-reviewable packet for the parser's own
dependency output: English translation + interlinear gloss + dependency
tree, per sentence.

VERSION: v1.0
COMPATIBLE WITH: klareco.parser's live AST shape (vortoj[].{radiko,
  vortspeco, kazo, nombro, tempo, kapo, rolo})
DEPENDENCIES: none beyond klareco.parser; ROOT_GLOSSES/translations
  supplied by the caller, not looked up automatically (no bilingual
  EO-EN dictionary is wired into the repo -- checked 2026-09-07,
  data/raw/eo/dictionaries/apertium_lexicon.json is a monolingual
  morphological-analyzer lexicon, not a bidix)
STAGE: Evaluation

Why this exists: klareco#920 needs an independently reviewed 200-
sentence gold set, and the user does not read Esperanto. Both existing
Esperanto UD treebanks (Prago, Cairo -- confirmed 2026-09-07 via
universaldependencies.org/eo to be the ENTIRE published set, so there is
no independent treebank to substitute in) ship an English `text_en` per
sentence for exactly this reason: an accurate translation plus the
dependency structure is enough to judge most attachment decisions
without fluency, because a wrong head produces a tree shape that
contradicts the plain English meaning. Demonstrated by hand on one
sentence before this script existed (2026-09-07): "Tiam senkonscie li
lasis la bildon de sinjorino Harding enveni en liajn pensojn" ("...he
let the image of Mrs. Harding enter into his thoughts") -- the parser
attaches `enveni` ("to enter") under `Harding` as if it read "Harding,
who enters", when the English immediately shows it belongs under
`lasis` ("let"). No Esperanto was needed to see that was wrong.

This script does NOT invoke a translation model and does NOT contain a
dictionary -- both are supplied by the caller as plain data (see
`SENTENCES` in this file for the shape), because translation is not a
deterministic function of the AST and a wrong root gloss would silently
mislead a reviewer who cannot check it themselves. Every root actually
used in this trial batch's sentences was glossed by hand.

Pipeline Position: chosen sentences (from a pilot queue) + hand-supplied
    translations/glosses -> review packet JSON -> HTML review artifact
Usage: python scripts/eval/build_annotation_review_packet.py \
    --output /path/to/packet.json
Inputs: this file's own SENTENCES/ROOT_GLOSSES/POS_LABELS/REL_LABELS
    tables; klareco.parser.parse
Outputs: one JSON file: a list of per-sentence records, each with the
    Esperanto text, translation, an interlinear gloss row per word, and
    dependency edges (id, head, relation) for tree rendering
Quality Checks: every non-punctuation root in every sentence must have
    a gloss (raises KeyError loudly otherwise -- CLAUDE.md: a silently
    missing gloss is worse than a crash, since it would look reviewable
    but mislead)
Last Updated: 2026-09-07
Related Issues: #920
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from klareco.parser import parse  # noqa: E402

# -----------------------------------------------------------------------
# Trial batch: 15 sentences drawn from data/test_sets/parser_pilot_v1
# (document-disjoint, 7-18 words, no dramatis-personae fragments), plus
# one anchor sentence (#16) already used by hand to demonstrate the
# method finds real bugs. `note` records an honest translation caveat
# where one exists -- a reviewer should see MY uncertainty, not just a
# confident-looking translation.
# -----------------------------------------------------------------------
SENTENCES = [
    {"id": 1, "source": "vikifontaro", "document_title": "Toŝio - Japanaj Rakontoj, 1923.pdf",
     "text": "Ŝi ŝpinadis fadenon per radŝpinilo, antaŭ ol ŝi teksadis la ŝtofojn.",
     "text_en": "She used to spin thread with a spinning wheel, before she would weave the fabrics.",
     "note": None},
    {"id": 2, "source": "gutenberg", "document_title": "La Reĝo de la Montoj",
     "text": "Ili elmetis sian kaptaĵon, kaj mia arĝenta horloĝo trafis denove sukceson, kiu flatis mian memestimon.",
     "text_en": "They laid out their catch, and my silver watch once again met with success, which flattered my self-esteem.",
     "note": None},
    {"id": 3, "source": "libera_folio", "document_title": "Eventa Servo helpas trovi retajn renkontiĝojn",
     "text": "Libera Folio: Kiel kaj kial Eventa Servo ekestis?",
     "text_en": "Libera Folio: How and why did Eventa Servo (\"Event Service\") come into being?",
     "note": None},
    {"id": 4, "source": "libera_folio", "document_title": "UN rimarkis Esperanton en la lingvotago",
     "text": "La venonta speciala tago de UN kiun ni nun priatentas estas Internacia Virina Tago (8 marto).",
     "text_en": "The next special UN day that we are now paying attention to is International Women's Day (March 8).",
     "note": None},
    {"id": 5, "source": "gutenberg", "document_title": "La Botistoj",
     "text": "Ho, kiam Esko de botisto edzigxas, tiam la mondo renversigxos.",
     "text_en": "Oh, when Esko, the bootmaker's [son/apprentice?], gets married, then the world will turn upside down.",
     "note": "\"Esko de botisto\" is genitive-like (\"Esko of a/the bootmaker\") -- "
             "I'm not confident whether it means Esko IS the bootmaker or is "
             "someone attached to one; flagging rather than guessing."},
    {"id": 6, "source": "libera_folio", "document_title": "”La komitato ne montris sufiĉe da respondeco”",
     "text": "Kiam ĝi komencas dissolviĝi, tio havas katastrofan domenan efikon je la tuta E-movado.",
     "text_en": "When it starts to dissolve, that has a catastrophic effect on the whole "
                "field/domain of the Esperanto movement.",
     "note": "\"domenan\" = domain-related, not \"domino\" -- I mistranslated this "
             "as \"domino effect\" on first pass and corrected it before shipping "
             "this packet; leaving the correction visible on purpose."},
    {"id": 7, "source": "vikifontaro", "document_title": "Voltaire - Tri Verkoj de Volter, 1956, Lanti.pdf",
     "text": "Ili invadis la Ŝtatojn de l’bona Nabusan.",
     "text_en": "They invaded the States of the good Nabusan.",
     "note": "\"Nabusan\" is an unfamiliar proper name to me (a Voltaire character "
             "or place) -- kept untranslated rather than guessed."},
    {"id": 8, "source": "libera_folio", "document_title": "Kongresa flago transdonita al Afriko",
     "text": "La ĝenerala direktoro de UEA, Martin Schäffer, listigis mankojn, kiuj laŭ li "
             "troviĝas en la proponata sistemo:",
     "text_en": "The general director of UEA, Martin Schäffer, listed shortcomings which, "
                "according to him, are found in the proposed system:",
     "note": None},
    {"id": 9, "source": "libera_folio", "document_title": "Konata kanada muzikisto koncertis en Esperantujo",
     "text": "Kiam li ne alparolis la publikon rekte, la kantisto aŭdigis recitadon, kiun li mem surbendigis.",
     "text_en": "When he did not address the audience directly, the singer played a recitation "
                "which he himself had recorded (onto tape).",
     "note": None},
    {"id": 10, "source": "vikifontaro", "document_title": "R. U. R./Antaŭludo",
     "text": "Ĉi tiun alian vojon, kiun povis sekvi evoluo de la vivo, mi trovis hodiaŭ\".",
     "text_en": "This other path, which the evolution of life could have followed, I found today.",
     "note": None},
    {"id": 11, "source": "libera_folio", "document_title": "Kie estas la Duolinganoj?",
     "text": "Certe estas ekzemploj de esperantistoj kiuj lernis pro Duolingo.",
     "text_en": "There are certainly examples of Esperantists who learned because of Duolingo.",
     "note": None},
    {"id": 12, "source": "vikifontaro", "document_title": "Fervoja Esperantisto - Oktobro 1912.pdf",
     "text": "Tiaj informoj, eble en formo de cirkuleroj, estu dissendotaj kvarfoje en unu jaro.",
     "text_en": "Such information, perhaps in the form of circulars, should be sent out four times a year.",
     "note": None},
    {"id": 13, "source": "gutenberg", "document_title": "Paĝoj el la Flandra Literaturo",
     "text": "Pluvo ronkas, defluegas, Frapegante ĉiuflanken; Arboj devas sin kurbigi, Kaj kriegas, kaj plendegas.",
     "text_en": "Rain roars, gushes down, pounding on every side; trees must bend themselves, "
                "and cry out, and complain loudly.",
     "note": "Poetic verse -- word order is looser than prose by convention."},
    {"id": 14, "source": "libera_folio", "document_title": "Martin Wiese: Barakte kun la tempo, sed ne vane",
     "text": "Miaj fingromovoj estas iomete malpli precizaj, sed mi povas kompensi tion per "
             "multa ekzercado kaj koncentriĝo.",
     "text_en": "My finger movements are a bit less precise, but I can compensate for that "
                "with a lot of practice and concentration.",
     "note": None},
    {"id": 15, "source": "gutenberg", "document_title": "La Granda Admiralo: Unua vojaĝo ĉirkaŭ la mondo",
     "text": "Elcano, unu el 1a plej grandaj ribelantoj en San Julian!",
     "text_en": "Elcano, one of the greatest rebels in San Julián!",
     "note": "\"1a\" almost certainly should be \"la\" (\"the\") -- looks like an "
             "OCR/scan artifact from the original digitization, not a real word; "
             "kept in the text exactly as extracted rather than silently fixed."},
    {"id": 16, "source": "gutenberg", "document_title": "(anchor case, found by hand before this script existed)",
     "text": "Tiam senkonscie li lasis la bildon de sinjorino Harding enveni en liajn pensojn.",
     "text_en": "Then unconsciously he let the image of Mrs. Harding enter into his thoughts.",
     "note": "Known parser bug: the tree below attaches \"enveni\" (\"to enter\") under "
             "\"Harding\" instead of under \"lasis\" (\"let\") -- see if you can spot "
             "why that's wrong just from the English."},
]

# Root -> short English gloss, for every content-word root in SENTENCES
# above. Grammatical (function) roots are glossed too so the interlinear
# line is complete; hand-written, not dictionary-sourced (see docstring).
ROOT_GLOSSES = {
    "1a": "[sic, likely OCR error for \"la\"]", "8": "8", "ali": "other", "antaŭ": "before",
    "arb": "tree", "arĝent": "silver", "atent": "attention", "aŭd": "hear/play(sound)",
    "bild": "image/picture", "bon": "good", "bot": "boot", "cert": "certain(ly)",
    "cirkuler": "circular(letter)", "de": "of/from", "dev": "must", "direktor": "director",
    "domen": "domain/field", "dot": "-able(dowry-suffix homograph, see note)", "ebl": "possibly",
    "edz": "marry/spouse", "efik": "effect", "ekzempl": "example", "ekzerc": "practice/exercise",
    "el": "out of/one of", "en": "in",
    "esk": "Esko(name-like root)", "esperant": "Esperanto", "est": "be", "estim": "esteem",
    "event": "event", "evolu": "evolution", "faden": "thread", "flank": "side",
    "flat": "flatter", "flu": "flow", "foj": "time(occurrence)", "foli": "leaf/page",
    "form": "form", "frap": "knock/pound", "grand": "big/great", "hard": "[Harding, surname]",
    "hav": "have", "ho": "oh", "hodiaŭ": "today", "horloĝ": "clock/watch",
    "ili": "they", "inform": "information", "internaci": "international", "invad": "invade",
    "iom": "somewhat", "jar": "year", "je": "at(generic preposition)", "juli": "[Julian, name]",
    "kaj": "and", "kant": "song/singer", "kapt": "catch", "katastrof": "catastrophe",
    "kial": "why", "kiam": "when", "kiel": "how", "kiu": "who/which",
    "komenc": "begin", "kompens": "compensate", "koncentriĝ": "concentration",
    "konsci": "conscious", "kri": "shout", "kurb": "bend/curve", "l": "[elided \"la\"]",
    "la": "the", "las": "let/leave", "laŭ": "according to", "lern": "learn",
    "li": "he", "liber": "free", "list": "list", "mank": "lack/shortcoming",
    "mart": "March", "mem": "-self(reflexive, see note)", "met": "put/place",
    "mi": "I/my", "mond": "world", "mov": "move", "mult": "many/much",
    "ne": "not", "ni": "we", "nov": "new", "nun": "now", "ol": "than",
    "parol": "speak", "pens": "thought", "per": "by means of", "plej": "most",
    "plend": "complain", "pli": "more", "pluv": "rain", "pov": "can/be able",
    "preciz": "precise", "pro": "because of", "propon": "propose", "publik": "public",
    "recit": "recitation", "rekt": "direct(ly)", "renvers": "overturn/reverse",
    "ribel": "rebel", "ronk": "roar/snore", "sed": "but", "sekv": "follow",
    "serv": "service", "si": "-self(reflexive pronoun)", "sinjor": "Mr./Mrs./sir",
    "sistem": "system", "solv": "dissolve/solve", "special": "special",
    "sukces": "success", "surbendig": "record onto tape", "tag": "day",
    "teks": "weave", "tia": "such", "tiam": "then(at that time)", "tio": "that(thing)", "tiu": "that(one)",
    "traf": "hit/meet with", "trov": "find", "tut": "whole", "ue": "[UEA, org.]",
    "unu": "one", "ven": "come", "vir": "woman(vir-ino = woman)", "viv": "life",
    "voj": "path/way", "ĉi": "this-here(deictic particle)", "ĝeneral": "general",
    "ĝi": "it", "ŝi": "she", "ŝpin": "spin", "ŝtat": "state(country)", "ŝtof": "fabric",
    "Duolingo": "Duolingo", "Elcano": "Elcano(name)", "Martin": "Martin(name)",
    "Nabusan": "Nabusan(name, unfamiliar to me)", "San": "San(place-name particle)",
    "Schäffer": "Schäffer(surname)", "UN": "UN(United Nations)",
}

POS_LABELS = {
    "substantivo": "noun", "verbo": "verb", "adjektivo": "adj", "adverbo": "adv",
    "pronomo": "pron", "prepozicio": "prep", "korelativo": "correlative", "artikolo": "article",
    "konjunkcio": "conj", "numero": "num", "propra_nomo": "proper-noun", "partiklo": "particle",
    "fremda_vorto": "[unrecognized]", "nekonata": "[UNKNOWN -- vocabulary gap]",
    "ekzemplo": "[single-letter fragment]",
}

REL_LABELS = {
    "root": "is the main predicate of the sentence",
    "nsubj": "is the subject of", "csubj": "is the clausal subject of",
    "obj": "is the direct object of", "iobj": "is the indirect object of",
    "obl": "is an oblique (adverbial-like) argument of",
    "nmod": "modifies (as a noun-attached modifier)", "amod": "modifies (as an adjective)",
    "advmod": "modifies (as an adverb)", "det": "is the determiner of",
    "case": "marks the case/role of", "cc": "coordinates with", "conj": "is conjoined with",
    "mark": "introduces the clause of", "acl": "is a clause modifying",
    "advcl": "is an adverbial clause attached to", "ccomp": "is a clausal complement of",
    "xcomp": "is an open clausal complement of", "cop": "is the copula of",
    "aux": "is an auxiliary of", "appos": "is in apposition to", "nummod": "modifies (as a number)",
    "discourse": "is a discourse particle attached to", "parataxis": "is loosely juxtaposed with",
    "punct": "punctuates", "fixed": "is part of a fixed expression with",
    "vocative": "is a vocative address to", "orphan": "is an orphaned dependent of",
    "nmod:poss": "possessively modifies",
}


def build_word_row(w: dict) -> dict:
    root = (w.get("radiko") or "").strip()
    if root not in ROOT_GLOSSES:
        raise KeyError(
            f"no hand-written gloss for root {root!r} (surface {w.get('plena_vorto')!r}); "
            "add one to ROOT_GLOSSES before shipping this packet -- a missing gloss "
            "would look reviewable but silently mislead a non-Esperanto-speaking reviewer")
    # Esperanto only inflects nominals (and their agreeing articles/adjectives/
    # pronouns/correlatives/numbers) for case and number -- verbs, adverbs,
    # prepositions etc. carry a `kazo` field in the AST as a structural
    # default, not a real grammatical marking, and showing it would look like
    # a genuine tag to a reviewer who can't tell the difference.
    _DECLINABLE = {"substantivo", "adjektivo", "pronomo", "artikolo", "propra_nomo",
                   "korelativo", "numero"}
    tags = []
    if w.get("vortspeco") in _DECLINABLE:
        if w.get("kazo") == "akuzativo":
            tags.append("acc")
        if w.get("nombro") == "pluralo":
            tags.append("pl")
    if w.get("tempo"):
        tags.append({"pasinteco": "past", "estanteco": "present", "estonteco": "future",
                     "kondicionalo": "conditional", "vola": "imperative"}.get(w["tempo"], w["tempo"]))
    return {
        "id": w.get("id"), "surface": w.get("plena_vorto"), "root": root,
        "gloss": ROOT_GLOSSES[root],
        "pos": POS_LABELS.get(w.get("vortspeco"), w.get("vortspeco")),
        "tags": tags, "head": w.get("kapo"), "relation": w.get("rolo"),
        "relation_label": REL_LABELS.get(w.get("rolo"), w.get("rolo")),
        "flagged_ambiguous": bool(w.get("alligo_ambigua")),
    }


def build_packet(sentences: list[dict]) -> list[dict]:
    packet = []
    for item in sentences:
        ast = parse(item["text"])
        words = [w for w in ast.get("vortoj", []) if isinstance(w, dict)
                 and w.get("analizstato") != "interpunkcio"]
        rows = [build_word_row(w) for w in words]
        packet.append({
            "id": item["id"], "source": item["source"], "document_title": item["document_title"],
            "text": item["text"], "text_en": item["text_en"], "note": item.get("note"),
            "words": rows,
        })
    return packet


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    packet = build_packet(SENTENCES)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {len(packet)} sentences to {args.output}")


if __name__ == "__main__":
    main()
