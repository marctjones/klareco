"""
Goals System - Strategic planning and long-term objectives

The Goals system enables Klareco to:
- Maintain long-term objectives
- Prioritize queries based on goals
- Track goal progress and completion
- Pursue strategic directions autonomously

Goals are checked before each query to determine if the current
query advances any active goals.

This is part of Phase 7 of the Klareco development roadmap.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class GoalStatus(Enum):
    """Status of a goal"""
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    SUSPENDED = "suspended"


class GoalPriority(Enum):
    """Priority level of a goal"""
    CRITICAL = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    MINIMAL = 1


@dataclass
class CompletionCriteria:
    """
    Criteria for determining if a goal is complete.

    Can be based on:
    - Number of actions taken
    - Information gathered
    - Time elapsed
    - External conditions
    """
    criteria_type: str  # "action_count", "info_gathered", "time_based", "custom"
    target: Any  # Target value
    current: Any = None  # Current value
    description: str = ""

    def is_met(self) -> bool:
        """Check if criteria is met"""
        if self.criteria_type == "action_count":
            return int(self.current or 0) >= int(self.target)
        elif self.criteria_type == "info_gathered":
            return bool(self.current)
        elif self.criteria_type == "time_based":
            if self.current:
                return datetime.now() >= self.current
            return False
        elif self.criteria_type == "custom":
            # For custom criteria, target should be a callable
            if callable(self.target):
                return self.target(self.current)
            return False
        return False


@dataclass
class Goal:
    """
    A strategic goal that guides system behavior.

    Goals represent long-term objectives that the system
    can pursue across multiple queries.
    """
    goal_id: str
    description: str
    priority: GoalPriority
    status: GoalStatus = GoalStatus.ACTIVE
    completion_criteria: List[CompletionCriteria] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    progress_log: List[Dict[str, Any]] = field(default_factory=list)

    def is_complete(self) -> bool:
        """Check if all completion criteria are met"""
        if not self.completion_criteria:
            return False
        return all(c.is_met() for c in self.completion_criteria)

    def update_progress(self, action: str, details: Dict[str, Any]):
        """Log progress towards goal"""
        self.progress_log.append({
            'timestamp': datetime.now(),
            'action': action,
            'details': details
        })

        logger.debug(f"Goal {self.goal_id}: {action}")

        # Check if complete
        if self.status == GoalStatus.ACTIVE and self.is_complete():
            self.status = GoalStatus.COMPLETED
            self.completed_at = datetime.now()
            logger.info(f"Goal {self.goal_id} completed!")

    def __repr__(self) -> str:
        return f"Goal({self.goal_id}: {self.description[:50]}... [{self.status.value}, P{self.priority.value}])"


class GoalsSystem:
    """
    Manages system goals and goal-directed behavior.

    The goals system:
    - Maintains active and completed goals
    - Evaluates queries against goals (pre-query check)
    - Tracks progress towards goals
    - Suggests goal-advancing actions
    """

    def __init__(self):
        """Initialize goals system"""
        self.goals: Dict[str, Goal] = {}
        self.goal_counter = 0

        logger.info("GoalsSystem initialized")

    def add_goal(self, description: str, priority: GoalPriority,
                 completion_criteria: Optional[List[CompletionCriteria]] = None,
                 metadata: Optional[Dict[str, Any]] = None) -> Goal:
        """
        Add a new goal.

        Args:
            description: Goal description
            priority: Priority level
            completion_criteria: List of completion criteria
            metadata: Optional metadata

        Returns:
            Created goal
        """
        self.goal_counter += 1
        goal_id = f"GOAL-{self.goal_counter:04d}"

        goal = Goal(
            goal_id=goal_id,
            description=description,
            priority=priority,
            completion_criteria=completion_criteria or [],
            metadata=metadata or {}
        )

        self.goals[goal_id] = goal

        logger.info(f"Added {goal}")
        return goal

    def get_goal(self, goal_id: str) -> Optional[Goal]:
        """Get goal by ID"""
        return self.goals.get(goal_id)

    def get_active_goals(self) -> List[Goal]:
        """Get all active goals, sorted by priority"""
        active = [g for g in self.goals.values() if g.status == GoalStatus.ACTIVE]
        return sorted(active, key=lambda g: g.priority.value, reverse=True)

    def get_completed_goals(self) -> List[Goal]:
        """Get all completed goals"""
        return [g for g in self.goals.values() if g.status == GoalStatus.COMPLETED]

    def pre_query_check(self, query_ast: Dict[str, Any], query_text: str) -> Dict[str, Any]:
        """
        Check if query advances any goals (pre-query goal check).

        This is called before query execution to determine if the
        query helps achieve any active goals.

        Args:
            query_ast: Parsed query AST
            query_text: Original query text

        Returns:
            Analysis result with relevant goals and recommendations
        """
        active_goals = self.get_active_goals()

        if not active_goals:
            return {
                'relevant_goals': [],
                'should_prioritize': False,
                'recommendations': []
            }

        # Analyze query relevance to each goal
        relevant_goals = []
        for goal in active_goals:
            relevance_score = self._compute_relevance(query_text, goal)

            if relevance_score > 0.3:  # Threshold for relevance
                relevant_goals.append({
                    'goal': goal,
                    'relevance': relevance_score
                })

        # Sort by relevance * priority
        relevant_goals.sort(
            key=lambda g: g['relevance'] * g['goal'].priority.value,
            reverse=True
        )

        # Generate recommendations
        recommendations = []
        if relevant_goals:
            top_goal = relevant_goals[0]['goal']
            recommendations.append(
                f"This query may advance goal: {top_goal.description}"
            )

        return {
            'relevant_goals': relevant_goals,
            'should_prioritize': len(relevant_goals) > 0,
            'recommendations': recommendations
        }

    def _compute_relevance(self, query_text: str, goal: Goal) -> float:
        """
        Compute relevance score between query and goal.

        This is a simple keyword-based approach. In a full implementation,
        this would use semantic similarity via GNN embeddings.

        Args:
            query_text: Query text
            goal: Goal to compare against

        Returns:
            Relevance score (0.0 to 1.0)
        """
        query_lower = query_text.lower()
        goal_lower = goal.description.lower()

        # Simple keyword overlap
        query_words = set(query_lower.split())
        goal_words = set(goal_lower.split())

        if not goal_words:
            return 0.0

        overlap = len(query_words & goal_words)
        return overlap / len(goal_words)

    def update_goal_progress(self, goal_id: str, action: str, details: Dict[str, Any]):
        """
        Update progress for a goal.

        Args:
            goal_id: Goal ID
            action: Action taken
            details: Action details
        """
        goal = self.get_goal(goal_id)
        if goal:
            goal.update_progress(action, details)

    def suspend_goal(self, goal_id: str, reason: Optional[str] = None):
        """Suspend a goal"""
        goal = self.get_goal(goal_id)
        if goal:
            goal.status = GoalStatus.SUSPENDED
            if reason:
                goal.metadata['suspension_reason'] = reason
            logger.info(f"Suspended goal {goal_id}")

    def resume_goal(self, goal_id: str):
        """Resume a suspended goal"""
        goal = self.get_goal(goal_id)
        if goal and goal.status == GoalStatus.SUSPENDED:
            goal.status = GoalStatus.ACTIVE
            logger.info(f"Resumed goal {goal_id}")

    def fail_goal(self, goal_id: str, reason: Optional[str] = None):
        """Mark goal as failed"""
        goal = self.get_goal(goal_id)
        if goal:
            goal.status = GoalStatus.FAILED
            if reason:
                goal.metadata['failure_reason'] = reason
            logger.info(f"Failed goal {goal_id}")

    def __len__(self) -> int:
        return len(self.goals)

    def __repr__(self) -> str:
        active_count = len(self.get_active_goals())
        completed_count = len(self.get_completed_goals())
        return f"GoalsSystem({active_count} active, {completed_count} completed)"


# Factory function
def create_goals_system() -> GoalsSystem:
    """Create and return a GoalsSystem instance"""
    return GoalsSystem()


if __name__ == "__main__":
    # Test goals system
    print("Testing Goals System")
    print("=" * 80)

    # Create goals system
    goals = create_goals_system()

    print(f"\n{goals}\n")

    # Add some goals
    print("Adding goals...")

    goal1 = goals.add_goal(
        description="Learn about Esperanto grammar",
        priority=GoalPriority.HIGH,
        completion_criteria=[
            CompletionCriteria(
                criteria_type="action_count",
                target=5,
                current=0,
                description="Ask 5 grammar questions"
            )
        ]
    )

    goal2 = goals.add_goal(
        description="Build vocabulary in common topics",
        priority=GoalPriority.MEDIUM,
        completion_criteria=[
            CompletionCriteria(
                criteria_type="action_count",
                target=10,
                current=0,
                description="Learn 10 new words"
            )
        ]
    )

    print(f"{goals}\n")

    # Test pre-query check
    print("Testing pre-query check...")

    test_queries = [
        "Kiel funkcias la akuzativo en Esperanto?",  # Grammar question
        "Kio estas la vetero hodiaŭ?",  # Weather question (unrelated)
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        result = goals.pre_query_check({}, query)
        print(f"  Relevant goals: {len(result['relevant_goals'])}")
        print(f"  Should prioritize: {result['should_prioritize']}")
        if result['recommendations']:
            print(f"  Recommendation: {result['recommendations'][0]}")

    # Update progress
    print("\nUpdating goal progress...")
    goals.update_goal_progress(
        goal1.goal_id,
        "grammar_query",
        {'question': 'about accusative'}
    )

    # Update criteria
    goal1.completion_criteria[0].current = 1

    print(f"  Goal progress: {len(goal1.progress_log)} actions")
    print(f"  Completion: {goal1.is_complete()}")

    # Complete goal
    print("\nCompleting goal...")
    goal1.completion_criteria[0].current = 5
    goals.update_goal_progress(
        goal1.goal_id,
        "goal_completed",
        {}
    )

    print(f"{goals}")

    print("\n✅ Goals system test complete!")
