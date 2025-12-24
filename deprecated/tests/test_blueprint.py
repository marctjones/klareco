"""
Unit tests for Blueprint system
"""

import pytest
from klareco.blueprint import (
    Blueprint, BlueprintStep, BlueprintGenerator,
    StepStatus, create_blueprint_generator
)


class TestBlueprintStep:
    """Test BlueprintStep dataclass"""

    def test_create_step(self):
        """Test creating a blueprint step"""
        step = BlueprintStep(
            step_id=0,
            description="Test step",
            ast={'tipo': 'frazo'},
            intent='general_query'
        )

        assert step.step_id == 0
        assert step.description == "Test step"
        assert step.status == StepStatus.PENDING
        assert step.result is None
        assert step.error is None
        assert step.dependencies == []

    def test_step_with_dependencies(self):
        """Test step with dependencies"""
        step = BlueprintStep(
            step_id=1,
            description="Dependent step",
            ast={'tipo': 'frazo'},
            intent='calculation_request',
            dependencies=[0]
        )

        assert step.dependencies == [0]

    def test_step_repr(self):
        """Test step string representation"""
        step = BlueprintStep(
            step_id=0,
            description="Test",
            ast={},
            intent='general_query'
        )

        assert "Step 0" in str(step)
        assert "Test" in str(step)
        assert "pending" in str(step)


class TestBlueprint:
    """Test Blueprint dataclass"""

    def create_sample_blueprint(self):
        """Helper to create a sample blueprint"""
        return Blueprint(
            blueprint_id="TEST-001",
            original_query="Test query",
            original_ast={'tipo': 'frazo'},
            steps=[
                BlueprintStep(
                    step_id=0,
                    description="Step 1",
                    ast={},
                    intent='general_query',
                    dependencies=[]
                ),
                BlueprintStep(
                    step_id=1,
                    description="Step 2",
                    ast={},
                    intent='general_query',
                    dependencies=[0]
                ),
                BlueprintStep(
                    step_id=2,
                    description="Step 3",
                    ast={},
                    intent='general_query',
                    dependencies=[1]
                )
            ]
        )

    def test_create_blueprint(self):
        """Test creating a blueprint"""
        bp = self.create_sample_blueprint()

        assert bp.blueprint_id == "TEST-001"
        assert bp.original_query == "Test query"
        assert len(bp.steps) == 3

    def test_get_ready_steps_initial(self):
        """Test getting ready steps at start"""
        bp = self.create_sample_blueprint()

        ready = bp.get_ready_steps()

        # Only step 0 should be ready (no dependencies)
        assert len(ready) == 1
        assert ready[0].step_id == 0

    def test_get_ready_steps_after_completion(self):
        """Test getting ready steps after completing one"""
        bp = self.create_sample_blueprint()

        # Complete step 0
        bp.steps[0].status = StepStatus.COMPLETED

        ready = bp.get_ready_steps()

        # Now step 1 should be ready
        assert len(ready) == 1
        assert ready[0].step_id == 1

    def test_is_complete(self):
        """Test completion check"""
        bp = self.create_sample_blueprint()

        assert not bp.is_complete()

        # Mark all completed
        for step in bp.steps:
            step.status = StepStatus.COMPLETED

        assert bp.is_complete()

    def test_has_failed(self):
        """Test failure check"""
        bp = self.create_sample_blueprint()

        assert not bp.has_failed()

        # Mark one as failed
        bp.steps[1].status = StepStatus.FAILED

        assert bp.has_failed()

    def test_get_step(self):
        """Test getting step by ID"""
        bp = self.create_sample_blueprint()

        step = bp.get_step(1)
        assert step is not None
        assert step.step_id == 1

        step = bp.get_step(99)
        assert step is None

    def test_get_results(self):
        """Test getting results"""
        bp = self.create_sample_blueprint()

        # Complete some steps with results
        bp.steps[0].status = StepStatus.COMPLETED
        bp.steps[0].result = "Result 1"

        bp.steps[1].status = StepStatus.COMPLETED
        bp.steps[1].result = "Result 2"

        results = bp.get_results()

        assert len(results) == 2
        assert results[0] == "Result 1"
        assert results[1] == "Result 2"

    def test_blueprint_repr(self):
        """Test blueprint string representation"""
        bp = self.create_sample_blueprint()

        assert "TEST-001" in str(bp)
        assert "0/3" in str(bp)

        bp.steps[0].status = StepStatus.COMPLETED
        assert "1/3" in str(bp)


class TestBlueprintGenerator:
    """Test BlueprintGenerator"""

    def test_create_generator(self):
        """Test creating generator"""
        gen = create_blueprint_generator()
        assert gen is not None
        assert gen.blueprint_counter == 0

    def test_simple_query_no_decomposition(self):
        """Test that simple queries don't get decomposed"""
        gen = create_blueprint_generator()

        ast = {
            'tipo': 'frazo',
            'subjekto': {'tipo': 'vorto', 'radiko': 'hund'},
            'verbo': {'tipo': 'vorto', 'radiko': 'kur'},
            'objekto': None,
            'aliaj': []
        }

        blueprint = gen.generate(ast, "La hundo kuras.")

        assert blueprint is None

    def test_compound_query_with_kaj(self):
        """Test that compound queries with 'kaj' get decomposed"""
        gen = create_blueprint_generator()

        ast = {
            'tipo': 'frazo',
            'subjekto': {'tipo': 'vorto', 'radiko': 'mi'},
            'verbo': {'tipo': 'vorto', 'radiko': 'vol'},
            'objekto': None,
            'aliaj': [
                {'tipo': 'vorto', 'radiko': 'manĝ'},
                {'tipo': 'vorto', 'radiko': 'kaj'},
                {'tipo': 'vorto', 'radiko': 'dorm'}
            ]
        }

        blueprint = gen.generate(ast, "Mi volas manĝi kaj dormi.")

        assert blueprint is not None
        assert len(blueprint.steps) == 2
        assert "manĝi" in blueprint.steps[0].description
        assert "dormi" in blueprint.steps[1].description

    def test_contains_conjunction(self):
        """Test conjunction detection"""
        gen = create_blueprint_generator()

        ast_with_kaj = {
            'tipo': 'frazo',
            'aliaj': [{'tipo': 'vorto', 'radiko': 'kaj'}]
        }

        assert gen._contains_conjunction(ast_with_kaj, 'kaj')

        ast_without_kaj = {
            'tipo': 'frazo',
            'aliaj': [{'tipo': 'vorto', 'radiko': 'hund'}]
        }

        assert not gen._contains_conjunction(ast_without_kaj, 'kaj')

    def test_contains_comparison(self):
        """Test comparison detection"""
        gen = create_blueprint_generator()

        ast_with_comparison = {
            'tipo': 'frazo',
            'aliaj': [{'tipo': 'vorto', 'radiko': 'pli'}]
        }

        assert gen._contains_comparison(ast_with_comparison)

    def test_blueprint_id_increment(self):
        """Test that blueprint IDs increment"""
        gen = create_blueprint_generator()

        ast = {
            'tipo': 'frazo',
            'aliaj': [{'tipo': 'vorto', 'radiko': 'kaj'}]
        }

        bp1 = gen.generate(ast, "Test 1 kaj Test 2")
        bp2 = gen.generate(ast, "Test 3 kaj Test 4")

        assert bp1.blueprint_id == "BP-0001"
        assert bp2.blueprint_id == "BP-0002"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
