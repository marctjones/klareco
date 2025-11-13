"""
Unit tests for Goals and Values Systems
"""

import pytest
from datetime import datetime
from klareco.goals import (
    Goal, GoalsSystem, GoalStatus, GoalPriority,
    CompletionCriteria, create_goals_system
)
from klareco.values import (
    Value, ValuesSystem, ValueCategory, ValueConflict,
    create_values_system
)


class TestGoalsSystem:
    """Test Goals System"""

    def test_create_goals_system(self):
        """Test creating goals system"""
        goals = create_goals_system()
        assert len(goals) == 0

    def test_add_goal(self):
        """Test adding a goal"""
        goals = create_goals_system()

        goal = goals.add_goal(
            description="Test goal",
            priority=GoalPriority.HIGH
        )

        assert goal.goal_id == "GOAL-0001"
        assert goal.status == GoalStatus.ACTIVE
        assert len(goals) == 1

    def test_get_active_goals(self):
        """Test getting active goals"""
        goals = create_goals_system()

        goals.add_goal("Goal 1", GoalPriority.HIGH)
        goals.add_goal("Goal 2", GoalPriority.LOW)

        active = goals.get_active_goals()
        assert len(active) == 2
        assert active[0].priority == GoalPriority.HIGH  # Sorted by priority

    def test_completion_criteria(self):
        """Test goal completion criteria"""
        criteria = CompletionCriteria(
            criteria_type="action_count",
            target=5,
            current=0
        )

        assert not criteria.is_met()

        criteria.current = 5
        assert criteria.is_met()

    def test_goal_completion(self):
        """Test goal completion"""
        goals = create_goals_system()

        goal = goals.add_goal(
            description="Test goal",
            priority=GoalPriority.MEDIUM,
            completion_criteria=[
                CompletionCriteria("action_count", 3, 0)
            ]
        )

        assert not goal.is_complete()

        goal.completion_criteria[0].current = 3
        assert goal.is_complete()

    def test_pre_query_check(self):
        """Test pre-query goal check"""
        goals = create_goals_system()

        goals.add_goal(
            description="Learn about grammar",
            priority=GoalPriority.HIGH
        )

        result = goals.pre_query_check({}, "Tell me about grammar")

        assert 'relevant_goals' in result
        assert 'should_prioritize' in result


class TestValuesSystem:
    """Test Values System"""

    def test_create_values_system(self):
        """Test creating values system"""
        values = create_values_system()
        assert len(values) > 0  # Has default values

    def test_default_values(self):
        """Test that default values are added"""
        values = create_values_system()

        # Should have accuracy, helpfulness, clarity, respect
        assert len(values) >= 4

    def test_add_value(self):
        """Test adding a value"""
        values = create_values_system()
        initial_count = len(values)

        value = values.add_value(
            name="Test Value",
            description="Test description",
            weight=0.75,
            category=ValueCategory.TECHNICAL
        )

        assert len(values) == initial_count + 1
        assert value.weight == 0.75

    def test_weight_clamping(self):
        """Test that weights are clamped to [0, 1]"""
        values = create_values_system()

        value1 = values.add_value(
            name="Too High",
            description="Test",
            weight=1.5,
            category=ValueCategory.TECHNICAL
        )

        value2 = values.add_value(
            name="Too Low",
            description="Test",
            weight=-0.5,
            category=ValueCategory.TECHNICAL
        )

        assert value1.weight == 1.0
        assert value2.weight == 0.0

    def test_post_retrieval_reflection(self):
        """Test post-retrieval reflection"""
        values = create_values_system()

        result = values.post_retrieval_reflection({}, "Help me understand this")

        assert 'relevant_values' in result
        assert 'weighting_instructions' in result
        assert len(result['relevant_values']) > 0

    def test_value_conflict_resolution(self):
        """Test value conflict resolution"""
        value1 = Value(
            value_id="V1",
            name="Value 1",
            description="First value",
            weight=0.9,
            category=ValueCategory.ETHICAL
        )

        value2 = Value(
            value_id="V2",
            name="Value 2",
            description="Second value",
            weight=0.7,
            category=ValueCategory.TECHNICAL
        )

        conflict = ValueConflict(
            value1=value1,
            value2=value2,
            resolution_strategy="highest_weight"
        )

        resolved = conflict.resolve()
        assert resolved == value1  # Higher weight wins


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
