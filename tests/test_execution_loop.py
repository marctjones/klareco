"""
Unit tests for Execution Loop
"""

import pytest
from unittest.mock import Mock, MagicMock
from klareco.execution_loop import ExecutionLoop, create_execution_loop
from klareco.blueprint import Blueprint, BlueprintStep, StepStatus


class TestExecutionLoop:
    """Test ExecutionLoop"""

    def create_mock_orchestrator(self):
        """Create a mock orchestrator for testing"""
        mock = Mock()
        mock.route = MagicMock(return_value={
            'answer': 'Test answer',
            'confidence': 0.9,
            'expert': 'TestExpert'
        })
        return mock

    def create_simple_blueprint(self):
        """Create a simple blueprint for testing"""
        return Blueprint(
            blueprint_id="TEST-001",
            original_query="Test query",
            original_ast={'tipo': 'frazo'},
            steps=[
                BlueprintStep(
                    step_id=0,
                    description="Step 1",
                    ast={'tipo': 'frazo', 'test': 'step1'},
                    intent='general_query',
                    dependencies=[]
                )
            ]
        )

    def create_multi_step_blueprint(self):
        """Create a multi-step blueprint"""
        return Blueprint(
            blueprint_id="TEST-002",
            original_query="Multi-step query",
            original_ast={'tipo': 'frazo'},
            steps=[
                BlueprintStep(
                    step_id=0,
                    description="Step 1",
                    ast={'tipo': 'frazo'},
                    intent='general_query',
                    dependencies=[]
                ),
                BlueprintStep(
                    step_id=1,
                    description="Step 2",
                    ast={'tipo': 'frazo'},
                    intent='general_query',
                    dependencies=[0]
                ),
                BlueprintStep(
                    step_id=2,
                    description="Step 3",
                    ast={'tipo': 'frazo'},
                    intent='general_query',
                    dependencies=[1]
                )
            ]
        )

    def test_create_execution_loop(self):
        """Test creating execution loop"""
        orchestrator = self.create_mock_orchestrator()
        loop = create_execution_loop(orchestrator, max_iterations=10)

        assert loop is not None
        assert loop.orchestrator == orchestrator
        assert loop.max_iterations == 10

    def test_execute_simple_blueprint(self):
        """Test executing a simple single-step blueprint"""
        orchestrator = self.create_mock_orchestrator()
        loop = ExecutionLoop(orchestrator, max_iterations=10)

        blueprint = self.create_simple_blueprint()
        result = loop.execute(blueprint)

        assert result['success'] is True
        assert result['blueprint_id'] == "TEST-001"
        assert result['steps_completed'] == 1
        assert result['steps_failed'] == 0
        assert result['error'] is None
        assert orchestrator.route.called

    def test_execute_multi_step_blueprint(self):
        """Test executing a multi-step blueprint"""
        orchestrator = self.create_mock_orchestrator()
        loop = ExecutionLoop(orchestrator, max_iterations=10)

        blueprint = self.create_multi_step_blueprint()
        result = loop.execute(blueprint)

        assert result['success'] is True
        assert result['steps_completed'] == 3
        assert result['steps_failed'] == 0
        assert orchestrator.route.call_count == 3

    def test_execute_with_failure(self):
        """Test execution with step failure"""
        orchestrator = self.create_mock_orchestrator()
        # Make orchestrator return error
        orchestrator.route = MagicMock(return_value={
            'answer': None,
            'error': 'Test error'
        })

        loop = ExecutionLoop(orchestrator, max_iterations=10)
        blueprint = self.create_simple_blueprint()

        result = loop.execute(blueprint)

        assert result['success'] is False
        assert result['steps_failed'] == 1
        assert 'failed' in result['error'].lower()

    def test_execute_max_iterations(self):
        """Test that max iterations prevents infinite loops"""
        orchestrator = self.create_mock_orchestrator()
        loop = ExecutionLoop(orchestrator, max_iterations=5)

        # Create blueprint with circular dependency (impossible to complete)
        blueprint = Blueprint(
            blueprint_id="TEST-CIRCULAR",
            original_query="Circular",
            original_ast={},
            steps=[
                BlueprintStep(
                    step_id=0,
                    description="Step 1",
                    ast={},
                    intent='general_query',
                    dependencies=[1]  # Depends on step 1
                ),
                BlueprintStep(
                    step_id=1,
                    description="Step 2",
                    ast={},
                    intent='general_query',
                    dependencies=[0]  # Depends on step 0 (circular!)
                )
            ]
        )

        result = loop.execute(blueprint)

        assert result['success'] is False
        assert 'iterations' in result['error'].lower() or 'deadlock' in result['error'].lower()
        assert result['iterations'] >= 1

    def test_build_success_result(self):
        """Test building success result"""
        orchestrator = self.create_mock_orchestrator()
        loop = ExecutionLoop(orchestrator)

        blueprint = self.create_simple_blueprint()
        blueprint.steps[0].status = StepStatus.COMPLETED
        blueprint.steps[0].result = {'answer': 'Test result'}

        result = loop._build_success_result(blueprint, iterations=2)

        assert result['success'] is True
        assert result['blueprint_id'] == blueprint.blueprint_id
        assert result['steps_completed'] == 1
        assert result['iterations'] == 2
        assert 'Test result' in result['final_answer']

    def test_build_failure_result(self):
        """Test building failure result"""
        orchestrator = self.create_mock_orchestrator()
        loop = ExecutionLoop(orchestrator)

        blueprint = self.create_simple_blueprint()
        blueprint.steps[0].status = StepStatus.FAILED

        result = loop._build_failure_result(blueprint, iterations=1, error="Test error")

        assert result['success'] is False
        assert result['steps_failed'] == 1
        assert result['error'] == "Test error"

    def test_execute_step_success(self):
        """Test executing a single step successfully"""
        orchestrator = self.create_mock_orchestrator()
        loop = ExecutionLoop(orchestrator)

        step = BlueprintStep(
            step_id=0,
            description="Test",
            ast={'tipo': 'frazo'},
            intent='general_query'
        )

        loop._execute_step(step)

        assert step.status == StepStatus.COMPLETED
        assert step.result is not None
        assert step.error is None

    def test_execute_step_failure(self):
        """Test executing a step that fails"""
        orchestrator = self.create_mock_orchestrator()
        orchestrator.route = MagicMock(side_effect=Exception("Test exception"))

        loop = ExecutionLoop(orchestrator)

        step = BlueprintStep(
            step_id=0,
            description="Test",
            ast={'tipo': 'frazo'},
            intent='general_query'
        )

        loop._execute_step(step)

        assert step.status == StepStatus.FAILED
        assert step.error is not None
        assert "Test exception" in step.error

    def test_execute_respects_dependencies(self):
        """Test that execution respects step dependencies"""
        orchestrator = self.create_mock_orchestrator()
        loop = ExecutionLoop(orchestrator)

        blueprint = self.create_multi_step_blueprint()
        result = loop.execute(blueprint)

        # All steps should complete in order
        assert blueprint.steps[0].status == StepStatus.COMPLETED
        assert blueprint.steps[1].status == StepStatus.COMPLETED
        assert blueprint.steps[2].status == StepStatus.COMPLETED

        # Verify orchestrator was called for each step
        assert orchestrator.route.call_count == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
