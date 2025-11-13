"""
Orchestrator - Main coordination layer for Klareco.

The Orchestrator:
1. Receives parsed ASTs from the Front Door
2. Uses Gating Network to classify intent
3. Routes queries to appropriate Experts
4. Manages multi-step execution loops
5. Returns structured responses

This is the heart of the agentic AI system.
"""

import logging
from typing import Dict, Any, List, Optional

from .gating_network import GatingNetwork
from .experts.base import Expert

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Main orchestration layer for Klareco.

    Coordinates between the Gating Network and Experts to handle queries.
    """

    def __init__(self, gating_network: Optional[GatingNetwork] = None):
        """
        Initialize Orchestrator.

        Args:
            gating_network: Gating network for intent classification
                           If None, creates default symbolic gating network
        """
        self.gating_network = gating_network or GatingNetwork(mode='symbolic')
        self.experts: Dict[str, Expert] = {}
        self.intent_to_expert: Dict[str, str] = {}

        logger.info(f"Orchestrator initialized with {self.gating_network}")

    def register_expert(self, intent: str, expert: Expert):
        """
        Register an expert for a specific intent.

        Args:
            intent: Intent string (e.g., 'factoid_question')
            expert: Expert instance to handle this intent
        """
        self.experts[expert.name] = expert
        self.intent_to_expert[intent] = expert.name

        logger.info(f"Registered {expert.name} for intent '{intent}'")

    def register_experts(self, intent_expert_map: Dict[str, Expert]):
        """
        Register multiple experts at once.

        Args:
            intent_expert_map: Mapping of intent -> Expert
        """
        for intent, expert in intent_expert_map.items():
            self.register_expert(intent, expert)

    def _detect_compound_question(self, ast: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """
        Detect if AST represents a compound question (multiple questions with 'kaj').

        Args:
            ast: Parsed query AST

        Returns:
            List of sub-query ASTs if compound, None otherwise

        Example:
            "Kio estas Esperanto kaj kiu kreis ĝin?"
            → ["Kio estas Esperanto?", "Kiu kreis ĝin?"]
        """
        if not ast or ast.get('tipo') != 'frazo':
            return None

        # Look for 'kaj' (and) in the sentence
        aliaj = ast.get('aliaj', [])
        has_kaj = False

        for word in aliaj:
            if word.get('tipo') == 'vorto':
                if word.get('radiko', '').lower() == 'kaj':
                    has_kaj = True
                    break

        # If no 'kaj', not a compound question
        if not has_kaj:
            return None

        # For now, we don't split the AST (that would require complex parsing)
        # Instead, we mark it as compound and let the execution loop handle it
        # Phase 5 will add proper AST splitting
        logger.info("Detected compound question with 'kaj', but AST splitting not yet implemented")
        return None

    def route(self, ast: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route a query to the appropriate expert.

        This is the main entry point for query handling.

        Args:
            ast: Parsed query AST

        Returns:
            Response from the expert, with structure:
            {
                'answer': str,           # The answer/result
                'confidence': float,     # Confidence (0.0-1.0)
                'expert': str,           # Expert that handled it
                'intent': str,           # Classified intent
                'sources': list,         # (optional) Source documents
                'explanation': str,      # (optional) Explanation
                'metadata': dict         # (optional) Additional metadata
            }

        Raises:
            ValueError: If no expert is registered for the classified intent
        """
        logger.info("Routing query...")

        # 1. Classify intent
        classification = self.gating_network.classify(ast)
        intent = classification['intent']
        confidence = classification['confidence']

        logger.info(f"Classified as '{intent}' (confidence: {confidence:.2f})")

        # 2. Select expert by intent
        expert_name = self.intent_to_expert.get(intent)
        expert = None

        if expert_name:
            expert = self.experts[expert_name]
            logger.info(f"Selected expert by intent: {expert}")

            # 3. Check if expert can handle
            if not expert.can_handle(ast):
                logger.warning(f"{expert} cannot handle this query, trying fallback")
                expert = None

        # 4. Fallback: try all experts if no intent match or expert can't handle
        if not expert:
            logger.info("Using fallback routing: checking all experts")
            expert = self._find_expert_by_capability(ast)

        # 5. No expert found
        if not expert:
            logger.warning(f"No expert can handle this query (intent: {intent})")
            return {
                'answer': f"Pardonu, mi ne povas respondi tiun tipon de demando. (Intent: {intent})",
                'confidence': 0.0,
                'expert': None,
                'intent': intent,
                'error': 'no_expert_available'
            }

        # 6. Execute
        logger.info(f"Executing with {expert}...")
        response = expert.execute(ast)

        # 7. Add orchestration metadata
        response['intent'] = intent
        response['intent_confidence'] = confidence

        logger.info(f"Response generated (confidence: {response.get('confidence', 0):.2f})")

        return response

    def _find_expert_by_capability(self, ast: Dict[str, Any]) -> Optional[Expert]:
        """
        Find an expert that can handle the query by checking all experts.

        This is a fallback mechanism when intent-based routing fails.

        Args:
            ast: Parsed query AST

        Returns:
            Expert that can handle the query, or None if none found
        """
        candidates = []

        for expert in self.experts.values():
            if expert.can_handle(ast):
                confidence = expert.estimate_confidence(ast)
                candidates.append((expert, confidence))
                logger.debug(f"{expert.name} can handle (confidence: {confidence:.2f})")

        if not candidates:
            return None

        # Select expert with highest confidence
        candidates.sort(key=lambda x: x[1], reverse=True)
        selected_expert = candidates[0][0]
        logger.info(f"Selected expert by capability: {selected_expert} (confidence: {candidates[0][1]:.2f})")

        return selected_expert

    def execute_loop(
        self,
        ast: Dict[str, Any],
        goal: str,
        max_steps: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Execute a multi-step reasoning loop.

        Continues executing queries until the goal is achieved or max_steps reached.

        Args:
            ast: Initial query AST
            goal: Goal description (e.g., "answer_question", "solve_problem")
            max_steps: Maximum number of execution steps

        Returns:
            List of execution steps, each a response dict

        Note:
            This is a foundational implementation. Phase 5 will add:
            - Goal completion checking
            - Next action planning
            - Structured blueprints
        """
        logger.info(f"Starting execution loop (goal: {goal}, max_steps: {max_steps})")

        steps = []
        current_ast = ast

        for step_num in range(1, max_steps + 1):
            logger.info(f"Execution step {step_num}/{max_steps}")

            # Execute current query
            response = self.route(current_ast)
            steps.append(response)

            # Check if goal achieved (simple heuristic for now)
            if self._goal_achieved(goal, steps):
                logger.info(f"Goal achieved after {step_num} steps")
                break

            # Plan next action (placeholder for Phase 5)
            current_ast = self._plan_next(response, goal)

            if not current_ast:
                logger.info("No next action planned, stopping")
                break

        logger.info(f"Execution loop complete: {len(steps)} steps")
        return steps

    def _goal_achieved(self, goal: str, steps: List[Dict[str, Any]]) -> bool:
        """
        Check if goal has been achieved.

        Args:
            goal: Goal description
            steps: Execution steps so far

        Returns:
            True if goal achieved, False otherwise

        Note:
            This is a Level 1 implementation using simple heuristics.
            Phase 5 will add neural goal completion checking.
        """
        # Goal: answer_question - check if we got a confident answer
        if goal == "answer_question":
            if len(steps) >= 1:
                last_step = steps[-1]
                # Consider goal achieved if:
                # 1. We got an answer with confidence > 0.5, OR
                # 2. We got an error (can't proceed further)
                if last_step.get('error'):
                    return True  # Stop on error
                if last_step.get('confidence', 0) > 0.5:
                    return True  # Got confident answer
            return False

        # Goal: compound_question - check if all sub-questions answered
        if goal == "compound_question":
            # For compound questions, we need at least 2 steps
            if len(steps) >= 2:
                # Check if all steps have confident answers
                all_confident = all(
                    step.get('confidence', 0) > 0.5 or step.get('error')
                    for step in steps
                )
                return all_confident
            return False

        # Goal: solve_problem - check if we have a final solution
        if goal == "solve_problem":
            if len(steps) >= 1:
                last_step = steps[-1]
                # Look for calculation results or final answers
                if last_step.get('expert') == 'Math Expert':
                    return True  # Math problems are single-step
                if last_step.get('confidence', 0) > 0.7:
                    return True
            return False

        # Default: single-step goals achieved after one confident response
        return len(steps) >= 1 and steps[-1].get('confidence', 0) > 0.5

    def _plan_next(
        self,
        response: Dict[str, Any],
        goal: str
    ) -> Optional[Dict[str, Any]]:
        """
        Plan the next action based on current response and goal.

        This is a Level 1 symbolic implementation that handles:
        - Compound questions (detected at runtime)
        - Sequential tool invocations
        - Error recovery

        Args:
            response: Response from previous step
            goal: Overall goal

        Returns:
            AST for next query, or None if no next action

        Note:
            Phase 5 will add Blueprint-based planning with neural components.
        """
        # If previous step had an error, stop
        if response.get('error'):
            logger.info("Previous step had error, stopping execution loop")
            return None

        # If previous step had low confidence, stop
        if response.get('confidence', 0) < 0.3:
            logger.info("Previous step had low confidence, stopping execution loop")
            return None

        # For compound questions, check if there's a pending sub-query
        # This is set during initial routing if compound structure detected
        if goal == "compound_question":
            pending_subqueries = response.get('_pending_subqueries', [])
            if pending_subqueries:
                next_ast = pending_subqueries[0]
                # Pass remaining subqueries forward
                next_ast['_pending_subqueries'] = pending_subqueries[1:]
                logger.info(f"Processing next sub-query ({len(pending_subqueries)} remaining)")
                return next_ast

        # For other goals, no next action by default
        # Phase 5 will add Blueprint-based planning here
        return None

    def list_experts(self) -> List[str]:
        """
        List all registered experts.

        Returns:
            List of expert names
        """
        return list(self.experts.keys())

    def list_intents(self) -> List[str]:
        """
        List all intents that have registered experts.

        Returns:
            List of intent strings
        """
        return list(self.intent_to_expert.keys())

    def __repr__(self) -> str:
        return f"Orchestrator(experts={len(self.experts)}, intents={len(self.intent_to_expert)})"


def create_orchestrator_with_experts() -> Orchestrator:
    """
    Factory function to create an Orchestrator with all available experts registered.

    Returns:
        Orchestrator with all available experts: Math, Date, Grammar, and RAG (if available)

    Example:
        >>> orchestrator = create_orchestrator_with_experts()
        >>> ast = parse("Kiom estas du plus tri?")
        >>> response = orchestrator.route(ast)
        >>> print(response['answer'])
        La rezulto estas: 5
    """
    from .experts import MathExpert, DateExpert, GrammarExpert
    from .experts.dictionary_expert import DictionaryExpert
    from .experts.rag_expert import create_rag_expert

    # Create orchestrator (uses symbolic gating network by default)
    orchestrator = Orchestrator()

    # Register symbolic tool experts (always available)
    orchestrator.register_expert('calculation_request', MathExpert())
    orchestrator.register_expert('temporal_query', DateExpert())
    orchestrator.register_expert('grammar_query', GrammarExpert())
    orchestrator.register_expert('dictionary_query', DictionaryExpert())

    # Register RAG expert (if corpus and model are available)
    try:
        rag_expert = create_rag_expert()
        orchestrator.register_expert('factoid_question', rag_expert)
        logger.info("RAG Expert successfully loaded and registered for factoid questions")
    except Exception as e:
        logger.warning(
            f"Could not load RAG Expert (corpus/model unavailable): {e}. "
            f"Factoid questions will not be answered."
        )

    logger.info(
        f"Created orchestrator with {len(orchestrator.experts)} experts: "
        f"{', '.join(orchestrator.list_experts())}"
    )

    return orchestrator
