"""The ontology — CURATED, not hand-seeded. And it is finally not empty.

CLAUDE.md, on this project's own ontology:

    "`ontology_nodes` and `ontology_edges` are EMPTY and `verb_klaso` is 0%
     populated ... the 'always query the ontology' rule is currently UNFOLLOWABLE,
     and a couple of paths fall back to hardcoded lists. That is ACKNOWLEDGED DEBT."

and, with real honesty:

    "even when loaded, the ontology is hand-seeded and THIN (`kreado-26` =
     ["fond","kre","produk","far"]; `persono` = ["homo","vir","infan","kuracist"])
     ... Lexical synonymy is a genuine learned residue we are currently FAKING
     WITH A LIST. Don't oversell it."

**It does not have to be faked, and it is not a learned residue.** ReVo ships all
of it, curated by lexicographers, GPL-2.0:

    8,709  HYPERNYM edges      <ref tip="super">   — a real taxonomy
    2,984  synonym edges       <ref tip="sin">
   22,770  DOMAIN labels       <uzo tip="fak">     — 78 distinct: ZOO BOT MED GEOG…
      133  TYPED ENTITY LISTS  <ref lst="voko:…">
               voko:urboj (309 cities) · voko:personaj_nomoj (293)
               voko:ŝtatoj (166) · voko:ĉefurboj (215) · voko:malsanoj (172)
   40,230  SENSES              <snc>               — the sense inventory

`voko:urboj` and `voko:personaj_nomoj` ARE the `loko` and `persono` classes —
attested and curated, not four hand-picked roots.

PLUS the SEMANTIC TYPE HIERARCHY from voko-akrido, which is already doing work:
`best` (animate) ⊂ `subst`, `pers` ⊂ `best`, `parc` (kinship) ⊂ `pers`. That is
what took morpheme ambiguity from 32% to 0.285%.

WHAT THIS DOES *NOT* SOLVE — and the honesty matters
----------------------------------------------------
`hundo` has THREE senses in ReVo: the genus, the domestic animal, and an insult
for an aggressive man. **Which one is meant is not a grammatical question**, and
no rule in this module answers it.

Bick measured it on Esperanto (Arbobanko): 3.8% of noun lemmas, 2.4% of
adjectives and 2.2% of verbs are semantically ambiguous IN THE CORPUS, with the
lexicon's unrealized potential ~3x higher.

Some of it will fall to the selectional restrictions we already have. The rest is
world knowledge, and it becomes an **OR-node** — not a guess.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

_DICT = Path(__file__).parent.parent / 'data' / 'raw' / 'eo' / 'dictionaries'

# The `voko:` lists that map onto the ontology classes CLAUDE.md names.
ENTITY_CLASSES: dict[str, tuple[str, ...]] = {
    'persono': ('voko:personaj_nomoj', 'voko:historiaj_personoj'),
    'loko': ('voko:urboj', 'voko:ĉefurboj', 'voko:ŝtatoj', 'voko:riveroj',
             'voko:insuloj', 'voko:montoj', 'voko:maroj'),
    'lingvo': ('voko:lingvoj',),
    'malsano': ('voko:malsanoj',),
    'besto': ('voko:zoologiaj_genroj', 'voko:zoologiaj_familioj'),
    'planto': ('voko:botanikaj_genroj', 'voko:botanikaj_familioj'),
}


class Ontology:
    def __init__(self) -> None:
        p = _DICT / 'revo_ontology.json'
        if not p.exists():
            raise FileNotFoundError(
                f'{p} missing.\n'
                '  Acquire it: python scripts/acquire/acquire_revo_ontology.py\n'
                'Refusing to fall back to a hardcoded list — CLAUDE.md calls that '
                'acknowledged debt, and the curated data exists.')
        d = json.loads(p.read_text(encoding='utf-8'))
        self.roots: dict[str, dict] = d['roots']
        self.lists: dict[str, list[str]] = d['lists']

        # root -> the entity classes it belongs to
        self._class_of: dict[str, set[str]] = {}
        for klaso, lst_names in ENTITY_CLASSES.items():
            for name in lst_names:
                for r in self.lists.get(name, ()):
                    self._class_of.setdefault(r, set()).add(klaso)

    # -- senses ------------------------------------------------------------
    def senses(self, root: str) -> list[str]:
        """The root's own senses. `hund` has three, and only ONE is a dog."""
        return self.roots.get(root, {}).get('senses', [])

    def is_polysemous(self, root: str) -> bool:
        return len(self.senses(root)) > 1

    def senses_of_form(self, word: str) -> list[str]:
        """Senses keyed to the WORD, not the morpheme — `hundejo` has its own."""
        for v in self.roots.values():
            f = v.get('formoj', {})
            if word in f:
                return f[word]
        return []

    # -- taxonomy ----------------------------------------------------------
    def hypernyms(self, root: str) -> list[str]:
        return self.roots.get(root, {}).get('hypernyms', [])

    def synonyms(self, root: str) -> list[str]:
        """Curated, from ReVo. NOT a hand-written list."""
        return self.roots.get(root, {}).get('synonyms', [])

    def domains(self, root: str) -> list[str]:
        """ZOO, BOT, MED, GEOG… 78 of them, on 22,770 roots."""
        return self.roots.get(root, {}).get('domains', [])

    # -- entity classes ----------------------------------------------------
    def classes(self, root: str) -> set[str]:
        """`persono`, `loko`, … — from ReVo's curated `voko:` lists.

        This is what the Decision Checklist in CLAUDE.md tells us to query
        instead of hardcoding a gazetteer. It is no longer unfollowable.
        """
        return self._class_of.get(root, set())

    def is_a(self, root: str, klaso: str) -> bool:
        return klaso in self.classes(root)

    def members(self, klaso: str) -> set[str]:
        out: set[str] = set()
        for name in ENTITY_CLASSES.get(klaso, ()):
            out |= set(self.lists.get(name, ()))
        return out


@lru_cache(maxsize=1)
def ontology() -> Ontology:
    return Ontology()
