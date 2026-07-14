"""A PARSE FOREST — every reading the grammar licenses, packed, with attribution.

    "Since the grammar licenses multiple AST trees and words can have ambiguous
     meanings, shouldn't we be able to have a representation of the AST that
     supports including ALL of the possible meanings?"

Yes. It is called a **shared packed parse forest** (SPPF) — an AND-OR graph — and
it is what Earley/GLR/CYK parsers actually produce. Our parser did not: it
returned one tree and threw the rest away *silently*, which is how `Esperanton`
became `esper+ant`. It did not fail. It committed.

WHY PACK RATHER THAN LIST
-------------------------
Church & Patil (1982): the number of trees a broad-coverage grammar licenses grows
as the CATALAN NUMBERS. Five prepositional phrases -> 42 trees. Ten -> 16,796.
Enumeration is hopeless.

But wherever readings AGREE you can SHARE the substructure, and wherever they
DIFFER you insert one OR-node. Exponentially many trees; polynomially many nodes.

ONE MECHANISM, THREE KINDS OF AMBIGUITY
---------------------------------------
The same OR-node serves at every level of the tree:

    MORPHEME    papero  = paper|o   OR   pap|er|o  ("element of a pope")
    ATTACHMENT  "kun teleskopo" attaches to VIDIS  OR  to VIRON
    SENSE       a leaf's meaning, once a sense inventory exists

AND IT MAKES THE THESIS MECHANICALLY CHECKABLE
----------------------------------------------
Every OR-node records WHO COLLAPSED IT:

    fonto = 'regulo'  — a deterministic rule chose (selectional restriction,
                        Occam, case, agreement). `kialo` says which.
    fonto = 'modelo'  — a learned component chose. It may ONLY choose among
                        readings the grammar already licensed; it can never add
                        one.
    fonto = None      — NOTHING could choose. **This is the residue.**

So the boundary this project exists to find is not argued. It is COUNTED:
`residue(forest)` returns the OR-nodes that survived determinism. A learned
component is evaluated by exactly one question — how many of those did it
collapse, and how often was it right? Ablate it and the rules still run.

MEASURED TODAY (300K-sentence sample, morpheme level only):
    the grammar leaves      8.6% of running-text TOKENS with 2+ readings
    deterministic ranking resolves  83.5%  of that
    UNRESOLVED OR-nodes:    1.4% of tokens   <- the residue, with zero ML
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from klareco.morphology import Analysis, analyze


@dataclass
class Elekto:
    """One option inside an OR-node."""
    valoro: Any                      # the reading itself
    poentaro: float = 0.0            # deterministic score
    kialo: str = ''                  # why it won/lost, in words
    malobservoj: list[str] = field(default_factory=list)   # selectional violations


@dataclass
class Alternativoj:
    """An OR-node: the grammar licensed more than one reading here.

    `elektita` is the index of the chosen option — or None, meaning NOTHING could
    choose, and this node IS part of the residue.
    """
    nivelo: str                      # 'morfemo' | 'alligo' | 'senco'
    opcioj: list[Elekto]
    elektita: int | None = None
    fonto: str | None = None         # 'regulo' | 'modelo' | None
    kialo: str = ''

    @property
    def solvita(self) -> bool:
        return self.elektita is not None

    @property
    def elekto(self) -> Elekto | None:
        return self.opcioj[self.elektita] if self.elektita is not None else None

    def to_dict(self) -> dict:
        return {
            'tipo': 'alternativoj',
            'nivelo': self.nivelo,
            'elektita': self.elektita,
            'fonto': self.fonto,
            'kialo': self.kialo,
            'opcioj': [
                {'valoro': (repr(o.valoro) if isinstance(o.valoro, Analysis)
                            else o.valoro),
                 'poentaro': o.poentaro,
                 'malobservoj': o.malobservoj}
                for o in self.opcioj
            ],
        }


# ---------------------------------------------------------------------------
# Morpheme level — the one we can build today, because morphology.py already
# enumerates and ranks.
# ---------------------------------------------------------------------------

# How far apart two scores must be for the ranker to claim it DECIDED. Below
# this, the readings are effectively tied and we must NOT pretend otherwise —
# claiming a decision we did not make is exactly the arbitrariness this module
# exists to abolish.
_DECISION_MARGIN = 0.5


def morpheme_node(word: str) -> Alternativoj | Analysis | None:
    """Analyse a word. Return a plain reading if the grammar left no choice, and
    an OR-node if it did."""
    readings = analyze(word)
    if not readings:
        return None
    if len(readings) == 1:
        return readings[0]

    opts = [
        Elekto(valoro=r, poentaro=r.score,
               malobservoj=list(r.violations),
               kialo=('selectional violation: ' + '; '.join(r.violations))
               if r.violations else '')
        for r in readings
    ]
    node = Alternativoj(nivelo='morfemo', opcioj=opts)

    margin = readings[0].score - readings[1].score
    if margin >= _DECISION_MARGIN:
        node.elektita = 0
        node.fonto = 'regulo'
        if readings[1].violations:
            node.kialo = (f'the runner-up violates a selectional restriction '
                          f'({readings[1].violations[0]})')
        else:
            node.kialo = (f'fewer morphemes (Occam): '
                          f'{len(readings[0].morphemes)} vs '
                          f'{len(readings[1].morphemes)}')
    else:
        # THE RESIDUE. The grammar licensed both; determinism cannot separate
        # them; and we say so, rather than picking one and calling it a parse.
        node.fonto = None
        node.kialo = (f'TIED at {readings[0].score:+.1f} — the grammar licenses '
                      f'{len(readings)} readings and no deterministic rule '
                      f'separates them')
    return node


def residue(nodes: list) -> list[Alternativoj]:
    """The OR-nodes NOTHING could collapse.

    This is the boundary, enumerated rather than argued. A learned component's
    entire job is this list, and its entire evaluation is: how many did you
    collapse, and how often were you right?
    """
    return [n for n in nodes
            if isinstance(n, Alternativoj) and not n.solvita]


def forest_for_sentence(words: list[str]) -> dict:
    """A packed forest for a sentence, at the morpheme level.

    Attachment-level and sense-level OR-nodes slot into the same structure — the
    node type is identical, only `nivelo` changes. That is the point of one
    mechanism.
    """
    nodes = [morpheme_node(w) for w in words]
    nodes = [n for n in nodes if n is not None]
    ors = [n for n in nodes if isinstance(n, Alternativoj)]
    unresolved = residue(nodes)
    return {
        'tipo': 'arbaro',                    # forest
        'nodoj': nodes,
        'statistiko': {
            'vortoj': len(nodes),
            'ambiguaj': len(ors),            # the grammar left a choice
            'solvitaj_de_reguloj': sum(1 for n in ors if n.fonto == 'regulo'),
            'restaĵo': len(unresolved),      # THE RESIDUE
        },
    }
