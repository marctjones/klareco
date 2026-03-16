#!/usr/bin/env python3
"""Tests for FactExtractor."""

import pytest
from klareco.parser import parse
from klareco.rag.fact_extractor import FactExtractor, RelationType


def test_extract_is_a_fact():
    """Test IS-A fact extraction from simple copula."""
    extractor = FactExtractor()

    ast = parse("Esperanto estas planlingvo")
    facts = extractor.extract(ast)

    assert len(facts) == 1
    fact = facts[0]
    assert fact.entity == "esperant"
    assert fact.relation == RelationType.IS_A
    # "planlingvo" is compound (plan+lingv), extractor returns root
    assert fact.arguments['type'] == "lingv"


def test_extract_created_by_fact():
    """Test CREATED-BY fact extraction."""
    extractor = FactExtractor()

    ast = parse("Zamenhof kreis Esperanton")
    facts = extractor.extract(ast)

    assert len(facts) == 1
    fact = facts[0]
    assert fact.entity == "esperant"
    assert fact.relation == RelationType.CREATED_BY
    assert fact.arguments['agent'] == "zamenhof"


def test_extract_has_fact():
    """Test HAS fact extraction."""
    extractor = FactExtractor()

    ast = parse("Esperanto havas parolantojn")
    facts = extractor.extract(ast)

    assert len(facts) == 1
    fact = facts[0]
    assert fact.entity == "esperant"
    assert fact.relation == RelationType.HAS
    # "parolantojn" has root "parol" (suffix -ant- is separate)
    assert fact.arguments['property'] == "parol"


def test_extract_temporal_modifiers():
    """Test temporal modifier extraction."""
    extractor = FactExtractor()

    ast = parse("Zamenhof kreis Esperanton en 1887")
    facts = extractor.extract(ast)

    assert len(facts) == 1
    fact = facts[0]
    assert 'time' in fact.modifiers
    # Should extract "1887" from the temporal modifier


def test_no_facts_from_empty_ast():
    """Test handling of empty/invalid AST."""
    extractor = FactExtractor()

    facts = extractor.extract({})
    assert len(facts) == 0

    facts = extractor.extract(None)
    assert len(facts) == 0


def test_extract_from_participial_noun_simple():
    """Test extraction from 'kreinto de X' pattern (participial noun)."""
    extractor = FactExtractor()

    ast = parse("La kreinto de Esperanto sciis multe")
    facts = extractor.extract(ast)

    # Should extract CREATED-BY fact from "kreinto de Esperanto"
    created_by_facts = [f for f in facts if f.relation == RelationType.CREATED_BY]
    assert len(created_by_facts) > 0

    fact = created_by_facts[0]
    assert fact.entity == 'esperant'
    assert fact.confidence >= 0.9  # High confidence for participial patterns


def test_extract_from_participial_noun_with_agent():
    """Test extraction from compound participial noun with agent (Issue #681)."""
    extractor = FactExtractor()

    # This is the actual sentence from Q1 test (Sentence 5)
    ast = parse("La kreinto-iniciatinto de Esperanto, Ludoviko Lazaro Zamenhof, sciis")
    facts = extractor.extract(ast)

    # Should extract CREATED-BY fact with Zamenhof as agent
    created_by_facts = [f for f in facts if f.relation == RelationType.CREATED_BY]
    assert len(created_by_facts) > 0

    fact = created_by_facts[0]
    assert fact.entity == 'esperant', f"Expected entity='esperant', got '{fact.entity}'"
    assert fact.arguments.get('agent') == 'zamenhof', \
        f"Expected agent='zamenhof', got '{fact.arguments.get('agent')}'"
    assert fact.confidence >= 0.9


def test_extract_from_participial_noun_founded():
    """Test extraction from 'fondinto de X' pattern."""
    extractor = FactExtractor()

    ast = parse("La fondinto de la asocio estis tre aktiva")
    facts = extractor.extract(ast)

    founded_facts = [f for f in facts if f.relation == RelationType.FOUNDED]
    assert len(founded_facts) > 0

    fact = founded_facts[0]
    assert fact.entity == 'asoci'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
