"""
Path-finding + explanation over entity_facts (#761).

Treat entity_facts as a graph: entities are nodes, slots are typed
edges. Find connecting chains between two entities (or an entity and
a target value) to surface explanations of how the system arrived at
an answer.

Uses NetworkX so we get free BFS / shortest-path / centrality. The
graph for our scale (~5K facts, ~few k unique entities) is trivial.
"""
from __future__ import annotations

import logging
from typing import Optional

import duckdb
import networkx as nx

logger = logging.getLogger(__name__)


def _build_fact_graph(conn) -> nx.DiGraph:
    """Build a directed multigraph from entity_facts.

    Edges: entity_radiko --slot--> value_radiko
    Edge attrs: slot, source_sid, confidence
    """
    g = nx.DiGraph()
    rows = conn.execute("""
        SELECT entity_radiko, slot, value, value_radiko,
               source_sid, confidence
        FROM entity_facts
    """).fetchall()
    for er, slot, val, val_r, sid, conf in rows:
        # If the edge already exists, keep the higher-confidence one
        if g.has_edge(er, val_r):
            existing = g[er][val_r]
            if existing.get('confidence', 0) >= conf:
                continue
        g.add_edge(er, val_r,
                   slot=slot, source_sid=int(sid),
                   confidence=float(conf), value=val)
    return g


_GRAPH_CACHE: Optional[nx.DiGraph] = None


def _graph(conn) -> nx.DiGraph:
    """Build-once, reuse. Caller can pass `force_rebuild=True` later
    when entity_facts changes."""
    global _GRAPH_CACHE
    if _GRAPH_CACHE is None:
        _GRAPH_CACHE = _build_fact_graph(conn)
        logger.info(f'fact-graph: {_GRAPH_CACHE.number_of_nodes()} nodes, '
                    f'{_GRAPH_CACHE.number_of_edges()} edges')
    return _GRAPH_CACHE


def paths_between(conn, a: str, b: str, max_hops: int = 4,
                   max_paths: int = 5) -> list[list[dict]]:
    """Find paths from entity `a` to entity `b` (or value `b`) using
    BFS over the fact graph.

    Returns a list of paths, each path a list of edge dicts:
        [{'from': 'zamenhof', 'to': 'bjalistok',
          'slot': 'birth_place', 'source_sid': 311744, 'confidence': 0.85},
         {'from': 'bjalistok', 'to': 'pollando', ...}]

    Empty list if no path exists within max_hops.

    Searches the UNDIRECTED graph for path candidates (most relations
    are conceptually symmetric for explanation purposes — Zamenhof's
    birthplace is Bjalistok, and Bjalistok is where Zamenhof was born).
    """
    a = a.lower()
    b = b.lower()
    g = _graph(conn)
    if a not in g or b not in g:
        return []

    # Operate on undirected view for explanation
    ug = g.to_undirected(as_view=True)
    try:
        # Get up to max_paths shortest paths
        gen = nx.shortest_simple_paths(ug, a, b)
        paths_nodes = []
        for i, p in enumerate(gen):
            if i >= max_paths or len(p) - 1 > max_hops:
                break
            paths_nodes.append(p)
    except nx.NetworkXNoPath:
        return []

    # Convert node-paths to edge-dict-paths
    out_paths: list[list[dict]] = []
    for nodes in paths_nodes:
        edges = []
        for u, v in zip(nodes[:-1], nodes[1:]):
            # We have undirected nodes; recover the actual edge direction
            if g.has_edge(u, v):
                ed = g[u][v]
                edges.append({'from': u, 'to': v, **ed})
            elif g.has_edge(v, u):
                ed = g[v][u]
                edges.append({'from': v, 'to': u, '_reverse': True, **ed})
            else:
                # Shouldn't happen — undirected said there was an edge
                continue
        out_paths.append(edges)
    return out_paths


def explain(conn, a: str, b: str, max_hops: int = 3) -> str:
    """Produce a natural-language Esperanto explanation of how `a`
    and `b` are connected. Returns 'Mi ne trovis ligon...' if no path.
    """
    paths = paths_between(conn, a, b, max_hops=max_hops, max_paths=1)
    if not paths:
        return f'Mi ne trovis ligon inter {a.capitalize()} kaj {b.capitalize()}.'
    p = paths[0]
    parts = []
    for edge in p:
        slot = edge.get('slot', 'rilatas al')
        if edge.get('_reverse'):
            parts.append(f"{edge['from'].capitalize()} estas {slot} "
                         f"de {edge['to'].capitalize()}")
        else:
            parts.append(f"{edge['from'].capitalize()} {slot} "
                         f"{edge['to'].capitalize()}")
    return ' → '.join(parts) + '.'
