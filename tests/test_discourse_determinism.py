"""#896: the discourse planner must be deterministic (no RNG in the answer path)."""
import ast as pyast
from pathlib import Path

from klareco.rag.discourse_planner import DiscoursePlanner, DiscourseRelation


def test_marker_assignment_is_deterministic():
    p = DiscoursePlanner()
    rels = [(0, 1, DiscourseRelation.CAUSE),
            (1, 2, DiscourseRelation.SEQUENCE),
            (2, 3, DiscourseRelation.ELABORATION)]
    runs = {tuple(p._assign_markers([None] * 4, rels)) for _ in range(20)}
    assert len(runs) == 1, f"non-deterministic markers: {runs}"


def test_no_random_import_in_discourse_planner():
    src = (Path(__file__).resolve().parents[1] /
           'klareco' / 'rag' / 'discourse_planner.py').read_text()
    tree = pyast.parse(src)
    imported = {n.name for node in pyast.walk(tree)
                if isinstance(node, pyast.Import) for n in node.names}
    assert 'random' not in imported, "random must not be imported (#896)"
