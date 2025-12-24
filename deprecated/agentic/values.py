"""
Values System - Ethical and motivational framework

The Values system enables Klareco to:
- Maintain ethical guidelines and preferences
- Weight decisions based on values
- Resolve conflicts between competing values
- Generate value-aligned responses

Values are applied during response generation through a
"Post-Retrieval Reflection" step that produces "Weighting Instructions"
for the Writer Loop.

This is part of Phase 7 of the Klareco development roadmap.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ValueCategory(Enum):
    """Category of value"""
    ETHICAL = "ethical"
    EDUCATIONAL = "educational"
    SOCIAL = "social"
    TECHNICAL = "technical"
    PERSONAL = "personal"


@dataclass
class Value:
    """
    A value that guides system behavior.

    Values represent principles, preferences, and ethical guidelines
    that influence how the system responds to queries.
    """
    value_id: str
    name: str
    description: str
    weight: float  # 0.0 to 1.0
    category: ValueCategory
    keywords: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"Value({self.value_id}: {self.name}, weight={self.weight:.2f})"


@dataclass
class ValueConflict:
    """
    Represents a conflict between values.

    When two values suggest different responses, the conflict
    is resolved based on weights and context.
    """
    value1: Value
    value2: Value
    resolution_strategy: str  # "highest_weight", "context_dependent", "merge"
    resolved_value: Optional[Value] = None

    def resolve(self, context: Optional[Dict[str, Any]] = None) -> Value:
        """
        Resolve the conflict.

        Args:
            context: Optional context for context-dependent resolution

        Returns:
            Winning value
        """
        if self.resolution_strategy == "highest_weight":
            self.resolved_value = self.value1 if self.value1.weight >= self.value2.weight else self.value2
        elif self.resolution_strategy == "context_dependent":
            # In full implementation, this would use context to decide
            # For now, fall back to highest weight
            self.resolved_value = self.value1 if self.value1.weight >= self.value2.weight else self.value2
        elif self.resolution_strategy == "merge":
            # Create merged value with average weight
            self.resolved_value = Value(
                value_id=f"MERGED-{self.value1.value_id}-{self.value2.value_id}",
                name=f"{self.value1.name} + {self.value2.name}",
                description=f"Merged: {self.value1.description} & {self.value2.description}",
                weight=(self.value1.weight + self.value2.weight) / 2,
                category=self.value1.category
            )

        return self.resolved_value


class ValuesSystem:
    """
    Manages system values and value-aligned behavior.

    The values system:
    - Maintains core values and their weights
    - Applies values during response generation
    - Resolves conflicts between values
    - Generates weighting instructions for responses
    """

    def __init__(self):
        """Initialize values system"""
        self.values: Dict[str, Value] = {}
        self.value_counter = 0

        logger.info("ValuesSystem initialized")

        # Add default values
        self._add_default_values()

    def _add_default_values(self):
        """Add default system values"""

        # Ethical values
        self.add_value(
            name="Accuracy",
            description="Provide accurate, truthful information",
            weight=0.9,
            category=ValueCategory.ETHICAL,
            keywords=['truth', 'accurate', 'correct', 'factual']
        )

        self.add_value(
            name="Helpfulness",
            description="Be helpful and constructive",
            weight=0.85,
            category=ValueCategory.SOCIAL,
            keywords=['helpful', 'assist', 'support', 'aid']
        )

        self.add_value(
            name="Clarity",
            description="Communicate clearly and understandably",
            weight=0.8,
            category=ValueCategory.EDUCATIONAL,
            keywords=['clear', 'understandable', 'simple', 'explain']
        )

        self.add_value(
            name="Respect",
            description="Treat users with respect and consideration",
            weight=0.9,
            category=ValueCategory.ETHICAL,
            keywords=['respect', 'polite', 'considerate', 'kind']
        )

    def add_value(self, name: str, description: str, weight: float,
                  category: ValueCategory, keywords: Optional[List[str]] = None,
                  metadata: Optional[Dict[str, Any]] = None) -> Value:
        """
        Add a new value.

        Args:
            name: Value name
            description: Description of the value
            weight: Importance weight (0.0 to 1.0)
            category: Value category
            keywords: Keywords associated with this value
            metadata: Optional metadata

        Returns:
            Created value
        """
        self.value_counter += 1
        value_id = f"VALUE-{self.value_counter:04d}"

        value = Value(
            value_id=value_id,
            name=name,
            description=description,
            weight=min(max(weight, 0.0), 1.0),  # Clamp to [0, 1]
            category=category,
            keywords=keywords or [],
            metadata=metadata or {}
        )

        self.values[value_id] = value

        logger.debug(f"Added {value}")
        return value

    def get_value(self, value_id: str) -> Optional[Value]:
        """Get value by ID"""
        return self.values.get(value_id)

    def get_values_by_category(self, category: ValueCategory) -> List[Value]:
        """Get all values in a category"""
        return [v for v in self.values.values() if v.category == category]

    def post_retrieval_reflection(self, query_ast: Dict[str, Any],
                                   query_text: str,
                                   retrieved_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform post-retrieval reflection to generate weighting instructions.

        This is called after retrieving information but before generating
        the final response. It determines which values are most relevant
        and generates instructions for value-aligned response generation.

        Args:
            query_ast: Parsed query AST
            query_text: Original query text
            retrieved_context: Retrieved information/context

        Returns:
            Reflection result with weighting instructions
        """
        # Identify relevant values
        relevant_values = self._identify_relevant_values(query_text)

        # Check for conflicts
        conflicts = self._detect_conflicts(relevant_values)

        # Resolve conflicts
        resolved_values = relevant_values.copy()
        for conflict in conflicts:
            resolved = conflict.resolve()
            # Remove conflicting values and add resolved
            resolved_values = [v for v in resolved_values
                             if v.value_id not in [conflict.value1.value_id, conflict.value2.value_id]]
            resolved_values.append(resolved)

        # Generate weighting instructions
        instructions = self._generate_weighting_instructions(resolved_values, query_text)

        return {
            'relevant_values': relevant_values,
            'conflicts': conflicts,
            'resolved_values': resolved_values,
            'weighting_instructions': instructions
        }

    def _identify_relevant_values(self, query_text: str) -> List[Value]:
        """
        Identify values relevant to the query.

        Uses keyword matching to find relevant values.
        In full implementation, would use semantic similarity.

        Args:
            query_text: Query text

        Returns:
            List of relevant values
        """
        query_lower = query_text.lower()
        relevant = []

        for value in self.values.values():
            # Check if any keyword appears in query
            if any(keyword in query_lower for keyword in value.keywords):
                relevant.append(value)

        # If no specific matches, include top weighted values
        if not relevant:
            all_values = sorted(self.values.values(), key=lambda v: v.weight, reverse=True)
            relevant = all_values[:2]  # Top 2 by default

        return sorted(relevant, key=lambda v: v.weight, reverse=True)

    def _detect_conflicts(self, values: List[Value]) -> List[ValueConflict]:
        """
        Detect conflicts between values.

        For now, implements a simple heuristic: ethical values
        may conflict with other categories.

        Args:
            values: List of values to check

        Returns:
            List of detected conflicts
        """
        conflicts = []

        ethical_values = [v for v in values if v.category == ValueCategory.ETHICAL]
        other_values = [v for v in values if v.category != ValueCategory.ETHICAL]

        # Check for ethical vs non-ethical conflicts
        for ethical in ethical_values:
            for other in other_values:
                # Simple conflict detection (placeholder)
                if abs(ethical.weight - other.weight) < 0.1:
                    conflicts.append(ValueConflict(
                        value1=ethical,
                        value2=other,
                        resolution_strategy="highest_weight"
                    ))

        return conflicts

    def _generate_weighting_instructions(self, values: List[Value], query_text: str) -> str:
        """
        Generate instructions for value-aligned response generation.

        These instructions guide the response writer to incorporate
        the relevant values.

        Args:
            values: Resolved relevant values
            query_text: Original query

        Returns:
            Weighting instructions as text
        """
        if not values:
            return "Generate a response that is helpful and accurate."

        # Build instructions from top values
        instructions_parts = []

        for value in values[:3]:  # Top 3 values
            if value.category == ValueCategory.ETHICAL:
                instructions_parts.append(
                    f"Ensure the response is {value.name.lower()} ({value.description.lower()})"
                )
            elif value.category == ValueCategory.EDUCATIONAL:
                instructions_parts.append(
                    f"Make the response {value.name.lower()} ({value.description.lower()})"
                )
            else:
                instructions_parts.append(
                    f"Emphasize {value.name.lower()} ({value.description.lower()})"
                )

        return ". ".join(instructions_parts) + "."

    def update_value_weight(self, value_id: str, new_weight: float):
        """Update the weight of a value"""
        value = self.get_value(value_id)
        if value:
            value.weight = min(max(new_weight, 0.0), 1.0)
            logger.info(f"Updated {value_id} weight to {value.weight:.2f}")

    def __len__(self) -> int:
        return len(self.values)

    def __repr__(self) -> str:
        return f"ValuesSystem({len(self.values)} values)"


# Factory function
def create_values_system() -> ValuesSystem:
    """Create and return a ValuesSystem instance"""
    return ValuesSystem()


if __name__ == "__main__":
    # Test values system
    print("Testing Values System")
    print("=" * 80)

    # Create values system
    values = create_values_system()

    print(f"\n{values}")
    print(f"Default values: {len(values.values)}\n")

    # Show default values
    print("Default values:")
    for value in values.values.values():
        print(f"  - {value.name}: {value.description} (weight={value.weight:.2f})")

    # Test post-retrieval reflection
    print("\nTesting post-retrieval reflection...")

    test_queries = [
        "How accurate is this information?",
        "Can you help me understand?",
        "General question"
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        result = values.post_retrieval_reflection({}, query)
        print(f"  Relevant values: {[v.name for v in result['relevant_values']]}")
        print(f"  Conflicts: {len(result['conflicts'])}")
        print(f"  Instructions: {result['weighting_instructions']}")

    # Add custom value
    print("\nAdding custom value...")
    custom = values.add_value(
        name="Conciseness",
        description="Be brief and to the point",
        weight=0.75,
        category=ValueCategory.TECHNICAL,
        keywords=['brief', 'concise', 'short']
    )

    print(f"  Added: {custom}")
    print(f"  Total values: {len(values)}")

    print("\n✅ Values system test complete!")
