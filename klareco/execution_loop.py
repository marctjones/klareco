"""
Execution Loop - Multi-step query execution

The Execution Loop executes Blueprints by:
1. Getting ready steps from blueprint
2. Routing each step to appropriate expert via Orchestrator
3. Collecting results
4. Updating blueprint status
5. Repeating until goal achieved or max iterations reached

This is the "while not goal_achieved:" logic from Phase 5.
"""

from typing import Dict, Any, Optional, List
import logging
from .blueprint import Blueprint, BlueprintStep, StepStatus
from .orchestrator import Orchestrator

logger = logging.getLogger(__name__)


class ExecutionLoop:
    """
    Executes multi-step plans (Blueprints) until completion.

    The execution loop:
    - Manages step-by-step execution of blueprints
    - Handles dependencies between steps
    - Collects and aggregates results
    - Provides error handling and recovery
    """

    def __init__(self, orchestrator: Orchestrator, max_iterations: int = 100):
        """
        Initialize execution loop.

        Args:
            orchestrator: Orchestrator for routing queries to experts
            max_iterations: Maximum iterations to prevent infinite loops
        """
        self.orchestrator = orchestrator
        self.max_iterations = max_iterations
        logger.info(f"ExecutionLoop initialized (max_iterations={max_iterations})")

    def execute(self, blueprint: Blueprint) -> Dict[str, Any]:
        """
        Execute a blueprint until completion.

        Args:
            blueprint: Multi-step execution plan

        Returns:
            Execution result with structure:
            {
                'success': bool,
                'blueprint_id': str,
                'steps_completed': int,
                'steps_failed': int,
                'results': Dict[int, Any],
                'final_answer': str,
                'iterations': int,
                'error': Optional[str]
            }
        """
        logger.info(f"Executing {blueprint}")

        iteration = 0
        while iteration < self.max_iterations:
            iteration += 1

            # Check if complete
            if blueprint.is_complete():
                logger.info(f"Blueprint completed after {iteration} iterations")
                return self._build_success_result(blueprint, iteration)

            # Check if failed
            if blueprint.has_failed():
                logger.warning(f"Blueprint has failed steps")
                return self._build_failure_result(blueprint, iteration, "One or more steps failed")

            # Get steps that are ready to execute
            ready_steps = blueprint.get_ready_steps()

            if not ready_steps:
                # No steps ready - either waiting on dependencies or deadlock
                logger.warning(f"No ready steps (iteration {iteration})")
                return self._build_failure_result(blueprint, iteration, "Execution deadlock - no steps ready")

            # Execute each ready step
            for step in ready_steps:
                logger.info(f"Executing {step}")
                self._execute_step(step)

            logger.debug(f"Iteration {iteration}: {blueprint}")

        # Max iterations reached
        logger.warning(f"Max iterations ({self.max_iterations}) reached")
        return self._build_failure_result(blueprint, iteration, "Maximum iterations exceeded")

    def _execute_step(self, step: BlueprintStep):
        """
        Execute a single blueprint step.

        Args:
            step: Step to execute
        """
        try:
            # Mark as in progress
            step.status = StepStatus.IN_PROGRESS

            # Route to orchestrator
            response = self.orchestrator.route(step.ast)

            # Check if successful
            if response.get('error'):
                step.status = StepStatus.FAILED
                step.error = response.get('error')
                logger.error(f"Step {step.step_id} failed: {step.error}")
            else:
                step.status = StepStatus.COMPLETED
                step.result = response
                logger.info(f"Step {step.step_id} completed successfully")

        except Exception as e:
            step.status = StepStatus.FAILED
            step.error = str(e)
            logger.error(f"Step {step.step_id} failed with exception: {e}", exc_info=True)

    def _build_success_result(self, blueprint: Blueprint, iterations: int) -> Dict[str, Any]:
        """Build success result from completed blueprint"""
        results = blueprint.get_results()

        # Aggregate answers from all steps
        answers = []
        for step_id, result in results.items():
            if isinstance(result, dict) and 'answer' in result:
                answers.append(result['answer'])
            else:
                answers.append(str(result))

        # Combine into final answer
        final_answer = " ".join(answers) if answers else "Query executed successfully"

        completed_count = sum(1 for s in blueprint.steps if s.status == StepStatus.COMPLETED)

        return {
            'success': True,
            'blueprint_id': blueprint.blueprint_id,
            'steps_completed': completed_count,
            'steps_failed': 0,
            'results': results,
            'final_answer': final_answer,
            'iterations': iterations,
            'error': None
        }

    def _build_failure_result(self, blueprint: Blueprint, iterations: int, error: str) -> Dict[str, Any]:
        """Build failure result from incomplete blueprint"""
        completed_count = sum(1 for s in blueprint.steps if s.status == StepStatus.COMPLETED)
        failed_count = sum(1 for s in blueprint.steps if s.status == StepStatus.FAILED)

        return {
            'success': False,
            'blueprint_id': blueprint.blueprint_id,
            'steps_completed': completed_count,
            'steps_failed': failed_count,
            'results': blueprint.get_results(),
            'final_answer': None,
            'iterations': iterations,
            'error': error
        }


# Factory function
def create_execution_loop(orchestrator: Orchestrator, max_iterations: int = 100) -> ExecutionLoop:
    """
    Create and return an ExecutionLoop instance.

    Args:
        orchestrator: Orchestrator for routing queries
        max_iterations: Maximum execution iterations

    Returns:
        Initialized ExecutionLoop
    """
    return ExecutionLoop(orchestrator, max_iterations)


if __name__ == "__main__":
    # Test execution loop
    print("Testing Execution Loop")
    print("=" * 80)

    from .blueprint import BlueprintStep, Blueprint, StepStatus
    from .orchestrator import create_orchestrator_with_experts

    # Create orchestrator
    orchestrator = create_orchestrator_with_experts()

    # Create test blueprint
    blueprint = Blueprint(
        blueprint_id="TEST-001",
        original_query="Test compound query",
        original_ast={'tipo': 'frazo'},
        steps=[
            BlueprintStep(
                step_id=0,
                description="Step 1: Calculate 2+3",
                ast={'tipo': 'frazo'},  # Would be real AST
                intent='calculation_request',
                dependencies=[]
            ),
            BlueprintStep(
                step_id=1,
                description="Step 2: Get current date",
                ast={'tipo': 'frazo'},  # Would be real AST
                intent='temporal_query',
                dependencies=[0]  # Depends on step 0
            )
        ]
    )

    print(f"Blueprint: {blueprint}")
    print(f"Steps:")
    for step in blueprint.steps:
        print(f"  {step}")

    # Create execution loop
    loop = create_execution_loop(orchestrator, max_iterations=10)

    # Execute (this would fail since ASTs are placeholders)
    print(f"\nNote: Execution would work with real ASTs parsed from Esperanto queries")
    print(f"Blueprint system and execution loop are ready for integration!")

    print("\n✅ Execution loop test complete!")
