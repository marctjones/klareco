"""
Canonical signatures and grammar-driven tokens for Esperanto ASTs.

These helpers convert parsed ASTs into deterministic slot signatures and
grammar-aware token sequences. They lean on Esperanto's regular endings so
retrieval can use structure first, with minimal learned components.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

Word = Dict[str, object]
Sentence = Dict[str, object]


@dataclass(frozen=True)
class SlotSignature:
    """Canonical representation of a core slot (subject, verb, object)."""

    role: str
    root: Optional[str]
    pos: Optional[str]
    number: Optional[str]
    case: Optional[str]
    tense: Optional[str]
    mood: Optional[str]
    modifiers: List[str]

    def to_string(self) -> str:
        """
        Build a deterministic string for indexing / comparison.

        Example:
        SUBJ:root=hund|pos=substantivo|num=singularo|case=nominativo|mods=
        VERB:root=vid|pos=verbo|tense=prezenco
        """
        parts = [
            f"role={self.role}",
            f"root={self.root or ''}",
            f"pos={self.pos or ''}",
        ]
        if self.number:
            parts.append(f"num={self.number}")
        if self.case:
            parts.append(f"case={self.case}")
        if self.tense:
            parts.append(f"tense={self.tense}")
        if self.mood:
            parts.append(f"mood={self.mood}")

        mods = ",".join(sorted(self.modifiers)) if self.modifiers else ""
        parts.append(f"mods={mods}")
        return f"{self.role.upper()}:" + "|".join(parts)


def _word_tokens(word: Word) -> List[str]:
    """
    Convert a word AST node into grammar tokens (prefix/root/suffix/ending/role).
    Ordering is deterministic to keep token streams stable.
    """
    tokens: List[str] = []
    prefikso = word.get("prefikso")
    if prefikso:
        tokens.append(f"pref:{prefikso}")

    root = word.get("radiko")
    if root:
        tokens.append(f"root:{root}")

    for suf in word.get("sufiksoj", []) or []:
        tokens.append(f"suf:{suf}")

    vortspeco = word.get("vortspeco")
    ending = _ending_from_word(word)
    if ending:
        tokens.append(f"ending:{ending}")

    if vortspeco:
        tokens.append(f"pos:{vortspeco}")

    # Number / case (useful for nouns/adjectives)
    if word.get("nombro"):
        tokens.append(f"num:{word['nombro']}")
    if word.get("kazo"):
        tokens.append(f"case:{word['kazo']}")

    # Tense/mood for verbs
    if word.get("tempo"):
        tokens.append(f"tense:{word['tempo']}")
    if word.get("modo"):
        tokens.append(f"mood:{word['modo']}")

    return tokens


def _ending_from_word(word: Word) -> Optional[str]:
    """
    Infer the grammatical ending token from word attributes.
    """
    vortspeco = word.get("vortspeco")
    if vortspeco == "verbo":
        # Prefer explicit tense/mood markers
        if word.get("tempo") in ("prezenco", "pasinteco", "futuro", "kondiĉa"):
            mapping = {"prezenco": "as", "pasinteco": "is", "futuro": "os", "kondiĉa": "us"}
            return mapping[word["tempo"]]
        if word.get("modo") == "kondiĉa":
            return "us"
        if word.get("modo") == "imperativo":
            return "u"
        if word.get("modo") == "infinitivo":
            return "i"
    elif vortspeco == "substantivo":
        return "o"
    elif vortspeco == "adjektivo":
        return "a"
    elif vortspeco == "adverbo":
        return "e"
    return None


def canonicalize_sentence(ast: Sentence) -> Dict[str, SlotSignature]:
    """
    Build canonical slot signatures for subject/verb/object from a sentence AST.
    """
    if ast.get("tipo") != "frazo":
        raise ValueError("Expected sentence AST (tipo='frazo').")

    return {
        "subjekto": _slot_from_vortgrupo(ast.get("subjekto"), "subj"),
        "verbo": _slot_from_word(ast.get("verbo"), "verb"),
        "objekto": _slot_from_vortgrupo(ast.get("objekto"), "obj"),
    }


def signature_string(ast: Sentence) -> str:
    """
    Deterministic, compact signature string for indexing or comparison.
    """
    slots = canonicalize_sentence(ast)
    ordered_roles = ["subjekto", "verbo", "objekto"]
    parts = []
    for role in ordered_roles:
        sig = slots.get(role)
        if sig:
            parts.append(sig.to_string())
    return ";".join(parts)


def tokens_for_sentence(ast: Sentence) -> List[str]:
    """
    Flatten grammar tokens for all core words in a sentence (subj/verb/obj + modifiers).
    """
    tokens: List[str] = []
    if ast.get("subjekto"):
        tokens.extend(tokens_for_vortgrupo(ast["subjekto"]))
    if ast.get("verbo"):
        tokens.extend(_word_tokens(ast["verbo"]))
    if ast.get("objekto"):
        tokens.extend(tokens_for_vortgrupo(ast["objekto"]))
    for alia in ast.get("aliaj", []) or []:
        tokens.extend(_word_tokens(alia))
    return tokens


def tokens_for_vortgrupo(vg: Dict[str, object]) -> List[str]:
    """Tokenize a vortgrupo (noun phrase)."""
    if not vg:
        return []
    tokens: List[str] = []
    kerno = vg.get("kerno")
    if kerno:
        tokens.extend(_word_tokens(kerno))
    for adj in vg.get("priskriboj", []) or []:
        tokens.extend(_word_tokens(adj))
    return tokens


def _slot_from_word(word: Optional[Word], role: str) -> SlotSignature:
    if not word:
        return SlotSignature(
            role=role,
            root=None,
            pos=None,
            number=None,
            case=None,
            tense=None,
            mood=None,
        modifiers=[],
    )

    pos = word.get("vortspeco")
    number = None if pos == "verbo" else word.get("nombro")
    case = None if pos == "verbo" else word.get("kazo")

    return SlotSignature(
        role=role,
        root=word.get("radiko"),
        pos=pos,
        number=number,
        case=case,
        tense=word.get("tempo"),
        mood=word.get("modo"),
        modifiers=[],
    )


def _slot_from_vortgrupo(vg: Optional[Dict[str, object]], role: str) -> SlotSignature:
    if not vg:
        return SlotSignature(
            role=role,
            root=None,
            pos=None,
            number=None,
            case=None,
            tense=None,
            mood=None,
            modifiers=[],
        )

    kerno = vg.get("kerno", {})
    modifiers = []
    for adj in vg.get("priskriboj", []) or []:
        if not isinstance(adj, dict):
            continue
        radiko = adj.get("radiko")
        pref = adj.get("prefikso") or ""
        if radiko:
            modifiers.append(f"{pref}{radiko}")

    return SlotSignature(
        role=role,
        root=kerno.get("radiko"),
        pos=kerno.get("vortspeco"),
        number=kerno.get("nombro"),
        case=kerno.get("kazo"),
        tense=None,
        mood=None,
        modifiers=modifiers,
    )
