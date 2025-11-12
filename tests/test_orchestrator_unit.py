"""
Unit tests for Orchestrator.

Tests the orchestrator's core functionality:
- Expert registration
- Intent-based routing
- Capability-based fallback routing
- Expert selection logic
"""

import pytest
from klareco.parser import parse
from klareco.orchestrator import Orchestrator, create_orchestrator_with_experts
from klareco.experts.math_expert import MathExpert
from klareco.experts.date_expert import DateExpert


class TestOrchestrator:
    """Test suite for Orchestrator."""

    def setup_method(self):
        """Initialize orchestrator before each test."""
        self.orchestrator = Orchestrator()

    def test_initialization(self):
        """Test orchestrator initialization."""
        assert self.orchestrator is not None
        assert len(self.orchestrator.experts) == 0
        assert len(self.orchestrator.intent_to_expert) == 0

    def test_register_expert(self):
        """Test registering a single expert."""
        expert = MathExpert()
        self.orchestrator.register_expert('calculation_request', expert)

        assert len(self.orchestrator.experts) == 1
        assert 'Math Tool Expert' in self.orchestrator.experts
        assert self.orchestrator.intent_to_expert['calculation_request'] == 'Math Tool Expert'

    def test_register_multiple_experts(self):
        """Test registering multiple experts."""
        math_expert = MathExpert()
        date_expert = DateExpert()

        self.orchestrator.register_expert('calculation_request', math_expert)
        self.orchestrator.register_expert('temporal_query', date_expert)

        assert len(self.orchestrator.experts) == 2
        assert len(self.orchestrator.intent_to_expert) == 2

    def test_list_experts(self):
        """Test listing registered experts."""
        math_expert = MathExpert()
        self.orchestrator.register_expert('calculation_request', math_expert)

        experts = self.orchestrator.list_experts()
        assert 'Math Tool Expert' in experts
        assert len(experts) == 1

    def test_list_intents(self):
        """Test listing registered intents."""
        math_expert = MathExpert()
        self.orchestrator.register_expert('calculation_request', math_expert)

        intents = self.orchestrator.list_intents()
        assert 'calculation_request' in intents
        assert len(intents) == 1

    def test_route_with_registered_expert(self):
        """Test routing with a registered expert."""
        math_expert = MathExpert()
        self.orchestrator.register_expert('calculation_request', math_expert)

        ast = parse("Kiom estas du plus tri?")
        response = self.orchestrator.route(ast)

        assert response['intent'] == 'calculation_request'
        assert response['expert'] == 'Math Tool Expert'
        assert 'answer' in response
        assert response['result'] == 5

    def test_route_fallback_no_intent_match(self):
        """Test fallback routing when no expert registered for intent."""
        # Register expert but NOT for the intent that will be classified
        math_expert = MathExpert()
        self.orchestrator.register_expert('some_other_intent', math_expert)

        # This will be classified as 'calculation_request' but no expert registered
        # Should fall back to capability-based routing
        ast = parse("Kiom estas du plus tri?")
        response = self.orchestrator.route(ast)

        # Should still get routed to MathExpert via fallback
        assert response['expert'] == 'Math Tool Expert'
        assert response['result'] == 5

    def test_route_no_expert_available(self):
        """Test routing when no expert can handle the query."""
        ast = parse("Saluton!")
        response = self.orchestrator.route(ast)

        assert response['confidence'] == 0.0
        assert response['expert'] is None
        assert 'error' in response

    def test_fallback_selects_highest_confidence(self):
        """Test that fallback selects expert with highest confidence."""
        math_expert = MathExpert()
        date_expert = DateExpert()

        # Register both but not for the specific intent
        self.orchestrator.register_expert('some_intent_1', math_expert)
        self.orchestrator.register_expert('some_intent_2', date_expert)

        # Math query - should select MathExpert
        ast = parse("Kiom estas du plus tri?")
        response = self.orchestrator.route(ast)

        assert response['expert'] == 'Math Tool Expert'

    def test_create_orchestrator_with_experts(self):
        """Test factory function creates orchestrator with all experts."""
        orchestrator = create_orchestrator_with_experts()

        assert len(orchestrator.experts) == 3
        assert 'Math Tool Expert' in orchestrator.list_experts()
        assert 'Date/Time Tool Expert' in orchestrator.list_experts()
        assert 'Grammar Tool Expert' in orchestrator.list_experts()

        assert 'calculation_request' in orchestrator.list_intents()
        assert 'temporal_query' in orchestrator.list_intents()
        assert 'grammar_query' in orchestrator.list_intents()

    def test_route_adds_metadata(self):
        """Test that route adds orchestration metadata to response."""
        math_expert = MathExpert()
        self.orchestrator.register_expert('calculation_request', math_expert)

        ast = parse("Kiom estas du plus tri?")
        response = self.orchestrator.route(ast)

        # Check for orchestration metadata
        assert 'intent' in response
        assert 'intent_confidence' in response
        assert response['intent_confidence'] == 1.0  # Symbolic gating is deterministic

    def test_expert_cannot_handle_fallback(self):
        """Test fallback when registered expert cannot handle query."""
        math_expert = MathExpert()
        date_expert = DateExpert()

        # Register MathExpert for temporal_query (wrong!)
        self.orchestrator.register_expert('temporal_query', math_expert)
        # Register DateExpert for different intent
        self.orchestrator.register_expert('calculation_request', date_expert)

        # Temporal query - will route to MathExpert by intent,
        # but MathExpert can't handle it, should fallback to DateExpert
        ast = parse("Kiu tago estas hodiaŭ?")
        response = self.orchestrator.route(ast)

        # Should use DateExpert via fallback
        assert response['expert'] == 'Date/Time Tool Expert'


class TestOrchestratorExecutionLoop:
    """Test suite for orchestrator execution loop (placeholder tests)."""

    def test_execute_loop_exists(self):
        """Test that execute_loop method exists."""
        orchestrator = Orchestrator()
        assert hasattr(orchestrator, 'execute_loop')

    def test_execute_loop_single_step(self):
        """Test execution loop with single step."""
        orchestrator = create_orchestrator_with_experts()
        ast = parse("Kiom estas du plus tri?")

        steps = orchestrator.execute_loop(ast, goal="answer_question", max_steps=1)

        assert len(steps) >= 1
        assert steps[0]['expert'] == 'Math Tool Expert'

    def test_goal_achieved_stops_loop(self):
        """Test that loop stops when goal is achieved."""
        orchestrator = create_orchestrator_with_experts()
        ast = parse("Kiom estas du plus tri?")

        steps = orchestrator.execute_loop(ast, goal="answer_question", max_steps=10)

        # Should stop after 1 step for simple query
        assert len(steps) == 1
