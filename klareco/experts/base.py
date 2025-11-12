"""
Base Expert interface.

All experts must implement this interface to work with the Orchestrator.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class Expert(ABC):
    """
    Base class for all experts in the Klareco system.

    Experts are specialized components that handle specific types of queries.
    Each expert knows its domain and can estimate its confidence in handling
    a given query.
    """

    def __init__(self, name: str):
        """
        Initialize expert.

        Args:
            name: Human-readable name for this expert
        """
        self.name = name

    @abstractmethod
    def can_handle(self, ast: Dict[str, Any]) -> bool:
        """
        Check if this expert can handle the given query.

        Args:
            ast: Parsed query AST

        Returns:
            True if this expert can handle the query, False otherwise
        """
        pass

    @abstractmethod
    def estimate_confidence(self, ast: Dict[str, Any]) -> float:
        """
        Estimate confidence in handling this query.

        Args:
            ast: Parsed query AST

        Returns:
            Confidence score between 0.0 and 1.0
        """
        pass

    @abstractmethod
    def execute(self, ast: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the query and return a response.

        Args:
            ast: Parsed query AST

        Returns:
            Response dictionary with at minimum:
            - 'answer': The answer/result
            - 'confidence': Confidence in the answer (0.0-1.0)
            - 'expert': Name of this expert

            May also include:
            - 'sources': List of source documents (for RAG-based experts)
            - 'explanation': Explanation of how answer was derived
            - 'metadata': Additional metadata about the execution
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
