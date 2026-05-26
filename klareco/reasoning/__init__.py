"""
klareco.reasoning — symbolic-AI layer over entity_facts.

Public API:
    apply_rules(conn, rules=ALL_RULES, max_iterations=10) → counts
        Run forward-chaining over entity_facts; insert derived rows
        with pattern_name='inferred:<rule_name>'.
    paths_between(conn, a, b, max_hops=4) → list of paths
        Find chains of facts connecting two entities.
"""
from klareco.reasoning.inference import apply_rules, ALL_RULES, Rule
from klareco.reasoning.paths import paths_between

__all__ = ['apply_rules', 'ALL_RULES', 'Rule', 'paths_between']
