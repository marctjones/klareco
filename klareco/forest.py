"""Morphological candidate groups and deterministic ranking decisions.

This module groups independent word analyses. It is not a shared packed syntax
forest and does not enumerate jointly valid sentence trees. Unresolved groups
measure what this ranking policy leaves open, not an irreducible language limit.
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
    """A group containing more than one generated reading.

    `elektita` is the chosen index, or None when this ranking policy abstains.
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
                {'valoro': (o.valoro.to_dict() if isinstance(o.valoro, Analysis)
                            else o.valoro),
                 'poentaro': o.poentaro,
                 'kialo': o.kialo,
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
    """Return one generated reading, or a group when the search found several."""
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
        node.fonto = None
        node.kialo = (f'TIED within the heuristic margin: top scores '
                      f'{readings[0].score:+.1f} and {readings[1].score:+.1f}; '
                      f'bounded search returned {len(readings)} readings')
    return node


def residue(nodes: list) -> list[Alternativoj]:
    """Groups left unresolved by the current deterministic ranking policy."""
    return [n for n in nodes
            if isinstance(n, Alternativoj) and not n.solvita]


def forest_for_sentence(words: list[str]) -> dict:
    """Independent morphological groups for words with generated readings."""
    nodes = [morpheme_node(w) for w in words]
    nodes = [n for n in nodes if n is not None]
    ors = [n for n in nodes if isinstance(n, Alternativoj)]
    unresolved = residue(nodes)
    return {
        'tipo': 'arbaro',                    # forest
        'nodoj': nodes,
        'statistiko': {
            'vortoj': len(nodes),
            'ambiguaj': len(ors),
            'solvitaj_de_reguloj': sum(1 for n in ors if n.fonto == 'regulo'),
            'restaĵo': len(unresolved),
        },
    }
