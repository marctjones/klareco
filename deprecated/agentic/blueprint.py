"""
Blueprint System - Multi-step planning for complex queries

Blueprints represent decomposed, multi-step plans that can be executed
sequentially to answer complex queries. This is the core of Phase 5's
planning capabilities.

A Blueprint is a symbolic composition of ASTs representing sub-queries
that need to be executed to answer a complex question.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class StepStatus(Enum):
    """Status of a blueprint step"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class BlueprintStep:
    """
    A single step in a multi-step plan.

    Each step represents a sub-query that needs to be executed,
    with dependencies on previous steps.
    """
    step_id: int
    description: str
    ast: Dict[str, Any]  # The AST for this sub-query
    intent: str  # Expected intent (calculation_request, factoid_question, etc.)
    dependencies: List[int] = field(default_factory=list)  # Step IDs this depends on
    status: StepStatus = StepStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None

    def __repr__(self) -> str:
        return f"Step {self.step_id}: {self.description} [{self.status.value}]"


@dataclass
class Blueprint:
    """
    A multi-step execution plan.

    Represents a decomposed complex query as a series of steps
    that can be executed by the orchestrator.
    """
    blueprint_id: str
    original_query: str
    original_ast: Dict[str, Any]
    steps: List[BlueprintStep]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_ready_steps(self) -> List[BlueprintStep]:
        """
        Get steps that are ready to execute.

        A step is ready if:
        - Status is PENDING
        - All dependencies are COMPLETED

        Returns:
            List of steps ready for execution
        """
        ready = []
        completed_ids = {step.step_id for step in self.steps if step.status == StepStatus.COMPLETED}

        for step in self.steps:
            if step.status == StepStatus.PENDING:
                # Check if all dependencies are met
                deps_met = all(dep_id in completed_ids for dep_id in step.dependencies)
                if deps_met:
                    ready.append(step)

        return ready

    def is_complete(self) -> bool:
        """Check if all steps are completed or skipped"""
        return all(
            step.status in [StepStatus.COMPLETED, StepStatus.SKIPPED]
            for step in self.steps
        )

    def has_failed(self) -> bool:
        """Check if any step has failed"""
        return any(step.status == StepStatus.FAILED for step in self.steps)

    def get_step(self, step_id: int) -> Optional[BlueprintStep]:
        """Get step by ID"""
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None

    def get_results(self) -> Dict[int, Any]:
        """Get results from all completed steps"""
        return {
            step.step_id: step.result
            for step in self.steps
            if step.status == StepStatus.COMPLETED and step.result is not None
        }

    def __repr__(self) -> str:
        completed = sum(1 for s in self.steps if s.status == StepStatus.COMPLETED)
        return f"Blueprint({self.blueprint_id}): {completed}/{len(self.steps)} steps complete"


class BlueprintGenerator:
    """
    Generates execution blueprints from complex queries.

    This is the planning component that decomposes complex queries
    into executable steps.
    """

    def __init__(self):
        """Initialize blueprint generator"""
        self.blueprint_counter = 0
        logger.info("BlueprintGenerator initialized")

    def generate(self, ast: Dict[str, Any], query_text: str) -> Optional[Blueprint]:
        """
        Generate a blueprint from a query AST.

        Args:
            ast: Parsed query AST
            query_text: Original query text

        Returns:
            Blueprint if query needs multi-step execution, None otherwise
        """
        # Check if this query needs decomposition
        if not self._needs_decomposition(ast):
            logger.debug("Query does not need decomposition")
            return None

        # Generate blueprint ID
        self.blueprint_counter += 1
        blueprint_id = f"BP-{self.blueprint_counter:04d}"

        # Decompose query into steps
        steps = self._decompose_query(ast, query_text)

        if not steps or len(steps) <= 1:
            logger.debug("Could not decompose into multiple steps")
            return None

        blueprint = Blueprint(
            blueprint_id=blueprint_id,
            original_query=query_text,
            original_ast=ast,
            steps=steps,
            metadata={
                'decomposition_method': 'symbolic',
                'num_steps': len(steps)
            }
        )

        logger.info(f"Generated {blueprint}")
        return blueprint

    def _needs_decomposition(self, ast: Dict[str, Any]) -> bool:
        """
        Check if a query needs multi-step decomposition.

        Queries that need decomposition:
        - Compound questions with "kaj" (and)
        - Comparative questions requiring multiple facts
        - Questions with temporal sequences
        - Questions requiring intermediate calculations

        Args:
            ast: Parsed query AST

        Returns:
            True if query should be decomposed
        """
        if ast.get('tipo') != 'frazo':
            return False

        # Check for compound conjunction "kaj" (and)
        if self._contains_conjunction(ast, 'kaj'):
            return True

        # Check for comparative structures
        if self._contains_comparison(ast):
            return True

        # Check for temporal sequences
        if self._contains_temporal_sequence(ast):
            return True

        return False

    def _contains_conjunction(self, ast: Dict[str, Any], conjunction: str) -> bool:
        """Check if AST contains a specific conjunction"""
        if ast.get('tipo') == 'vorto':
            radiko = ast.get('radiko', '').lower()
            return radiko == conjunction
        elif ast.get('tipo') == 'vortgrupo':
            return any(self._contains_conjunction(v, conjunction) for v in ast.get('vortoj', []))
        elif ast.get('tipo') == 'frazo':
            for key in ['subjekto', 'verbo', 'objekto']:
                if ast.get(key) and self._contains_conjunction(ast[key], conjunction):
                    return True
            return any(self._contains_conjunction(v, conjunction) for v in ast.get('aliaj', []))
        return False

    def _contains_comparison(self, ast: Dict[str, Any]) -> bool:
        """Check if AST contains comparative structures"""
        comparative_roots = {'pli', 'malpli', 'plej', 'malplej', 'kompari'}
        return self._contains_any_root(ast, comparative_roots)

    def _contains_temporal_sequence(self, ast: Dict[str, Any]) -> bool:
        """Check if AST contains temporal sequence indicators"""
        temporal_sequence_roots = {'antaŭ', 'post', 'dum', 'tiam', 'poste', 'unue'}
        return self._contains_any_root(ast, temporal_sequence_roots)

    def _contains_any_root(self, ast: Dict[str, Any], roots: set) -> bool:
        """Check if AST contains any of the specified roots"""
        if ast.get('tipo') == 'vorto':
            radiko = ast.get('radiko', '').lower()
            return radiko in roots
        elif ast.get('tipo') == 'vortgrupo':
            return any(self._contains_any_root(v, roots) for v in ast.get('vortoj', []))
        elif ast.get('tipo') == 'frazo':
            for key in ['subjekto', 'verbo', 'objekto']:
                if ast.get(key) and self._contains_any_root(ast[key], roots):
                    return True
            return any(self._contains_any_root(v, roots) for v in ast.get('aliaj', []))
        return False

    def _decompose_query(self, ast: Dict[str, Any], query_text: str) -> List[BlueprintStep]:
        """
        Decompose a complex query into steps.

        This is a simplified symbolic decomposition. In Phase 5+,
        this would use more sophisticated analysis.

        Args:
            ast: Parsed query AST
            query_text: Original query text

        Returns:
            List of execution steps
        """
        steps = []

        # For now, implement simple conjunction splitting
        if self._contains_conjunction(ast, 'kaj'):
            # Split on "kaj" conjunction
            # This is a simplified version - full implementation would
            # properly parse and split the AST

            parts = query_text.split(' kaj ')
            if len(parts) >= 2:
                for i, part in enumerate(parts):
                    steps.append(BlueprintStep(
                        step_id=i,
                        description=f"Sub-query {i+1}: {part.strip()}",
                        ast={'tipo': 'teksto', 'enhavo': part.strip()},  # Placeholder AST
                        intent='general_query',  # Would be classified properly
                        dependencies=[]
                    ))

        # If no steps generated, create single step
        if not steps:
            steps.append(BlueprintStep(
                step_id=0,
                description=query_text,
                ast=ast,
                intent='general_query',
                dependencies=[]
            ))

        return steps


# Factory function
def create_blueprint_generator() -> BlueprintGenerator:
    """Create and return a BlueprintGenerator instance"""
    return BlueprintGenerator()


if __name__ == "__main__":
    # Test blueprint system
    print("Testing Blueprint System")
    print("=" * 80)

    # Create generator
    generator = create_blueprint_generator()

    # Test simple query (no decomposition)
    simple_ast = {
        'tipo': 'frazo',
        'subjekto': {'tipo': 'vorto', 'radiko': 'hund'},
        'verbo': {'tipo': 'vorto', 'radiko': 'kur'},
        'objekto': None,
        'aliaj': []
    }

    simple_query = "La hundo kuras."
    blueprint = generator.generate(simple_ast, simple_query)
    print(f"\nSimple query: {simple_query}")
    print(f"Needs decomposition: {blueprint is not None}")

    # Test compound query (needs decomposition)
    compound_ast = {
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

    compound_query = "Mi volas manĝi kaj dormi."
    blueprint = generator.generate(compound_ast, compound_query)
    print(f"\nCompound query: {compound_query}")
    print(f"Needs decomposition: {blueprint is not None}")

    if blueprint:
        print(f"Blueprint: {blueprint}")
        print(f"Steps:")
        for step in blueprint.steps:
            print(f"  {step}")

        # Test execution flow
        print(f"\nReady steps: {blueprint.get_ready_steps()}")
        print(f"Is complete: {blueprint.is_complete()}")

        # Mark first step complete
        blueprint.steps[0].status = StepStatus.COMPLETED
        blueprint.steps[0].result = "First sub-query result"

        print(f"\nAfter completing step 0:")
        print(f"Ready steps: {blueprint.get_ready_steps()}")
        print(f"Is complete: {blueprint.is_complete()}")
        print(f"Results: {blueprint.get_results()}")

    print("\n✅ Blueprint system test complete!")
