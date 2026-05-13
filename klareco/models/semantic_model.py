"""
SemanticModel Base Interface

All M1-M4 models must implement this interface to ensure consistent
data contracts and integration with the EnrichedAST pipeline.

Design Principles:
1. Immutable enrichment - enrich() returns NEW EnrichedAST via clone()
2. Dependency tracking - get_dependencies() enables automatic orchestration
3. Explainability - explain() generates human-readable explanations
4. Capability declaration - get_capabilities() for dynamic model discovery
5. Input validation - validate_input() checks dependencies before running
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from enum import Enum
from pathlib import Path

# Import EnrichedAST (will be available after #459 is complete)
try:
    from klareco.enriched_ast import EnrichedAST, AnnotationMetadata
except ImportError:
    # Fallback for development
    EnrichedAST = Any
    AnnotationMetadata = Any


class ModelCapability(Enum):
    """
    What kinds of enrichments can this model provide?

    Used for dynamic model discovery and capability checking.
    """
    SEMANTIC_SIMILARITY = "semantic"
    SELECTIONAL_PREFERENCE = "selectional"
    TAXONOMIC_RELATIONS = "taxonomic"
    DISCOURSE_COHERENCE = "discourse"
    THEMATIC_ROLES = "thematic"
    COREFERENCE = "coreference"
    REASONING = "reasoning"


class SemanticModel(ABC):
    """
    Base interface for all Klareco semantic models.

    All models must implement:
    1. enrich() - add annotations to EnrichedAST
    2. get_metadata() - provide model information
    3. get_capabilities() - declare what model can do

    Optional methods:
    - train() - for trainable models
    - validate() - for quality checks
    - explain() - for human-readable explanations
    """

    def __init__(self, model_path: Optional[str] = None, device: str = 'cpu'):
        """
        Initialize the model.

        Args:
            model_path: Path to trained model checkpoint (optional)
            device: Device to run on ('cpu' or 'cuda')
        """
        self.model_path = model_path
        self.device = device
        self._model = None
        self._metadata = None

    @abstractmethod
    def enrich(self, ast: EnrichedAST, **kwargs) -> EnrichedAST:
        """
        Add this model's annotations to an EnrichedAST.

        CRITICAL: This method MUST return a NEW EnrichedAST via clone(),
        never mutate the input AST directly.

        Args:
            ast: The EnrichedAST to enrich
            **kwargs: Model-specific options (e.g., context for discourse model)

        Returns:
            NEW EnrichedAST with annotations added (immutable progression)

        Raises:
            ValueError: If ast is missing required fields
            Runtime Error: If model not loaded

        Example:
            def enrich(self, ast: EnrichedAST, **kwargs) -> EnrichedAST:
                self.validate_input(ast)
                enriched = ast.clone()  # Clone first!
                enriched.selectional = SelectionalAnnotation(...)
                enriched.stages_applied.add("selectional")
                return enriched
        """
        pass

    @abstractmethod
    def get_metadata(self) -> AnnotationMetadata:
        """
        Return metadata about this model.

        Returns:
            AnnotationMetadata with model name, version, parameters, etc.

        Example:
            return AnnotationMetadata(
                model_name="selectional_preference",
                model_version="1.0.0",
                confidence=0.0,  # Will be set per-inference
                compute_time_ms=0.0,
                timestamp="",
                parameters={
                    "embedding_dim": 64,
                    "hidden_dim": 128,
                    "total_params": "10M"
                }
            )
        """
        pass

    @abstractmethod
    def get_capabilities(self) -> List[ModelCapability]:
        """
        Declare what this model can do.

        Returns:
            List of ModelCapability enums

        Example:
            return [
                ModelCapability.SELECTIONAL_PREFERENCE,
                ModelCapability.THEMATIC_ROLES
            ]
        """
        pass

    def get_dependencies(self) -> List[str]:
        """
        Return list of other models this model depends on.

        Used for automatic dependency resolution and orchestration.

        Returns:
            List of model names (e.g., ["stage1_semantic", "selectional"])

        Example:
            def get_dependencies(self) -> List[str]:
                return ["stage1_semantic"]  # Needs root embeddings
        """
        return []

    def validate_input(self, ast: EnrichedAST) -> bool:
        """
        Check if input AST has required fields for this model.

        Args:
            ast: The EnrichedAST to validate

        Returns:
            True if valid

        Raises:
            ValueError: If missing required dependencies

        Example:
            If model requires stage1_semantic but ast doesn't have it applied,
            raises ValueError with helpful message.
        """
        required = self.get_dependencies()
        for dep in required:
            if dep not in ast.stages_applied:
                raise ValueError(
                    f"Model {self.get_metadata().model_name} requires '{dep}' "
                    f"to be applied first. Available stages: {ast.stages_applied}"
                )
        return True

    def explain(self, ast: EnrichedAST) -> str:
        """
        Generate human-readable explanation of this model's annotations.

        Args:
            ast: Enriched AST with this model's annotations

        Returns:
            English explanation string

        Example:
            "Selectional Preference Analysis:
              Triple: (hund, manĝ, viand)
              Subject-Verb compatibility: 0.850
              Verb-Object compatibility: 0.920
              Overall plausibility: 0.880
              → Highly plausible combination"
        """
        return f"No explanation available for {self.get_metadata().model_name}"

    @classmethod
    def load(cls, model_path: str, **kwargs) -> 'SemanticModel':
        """
        Load a trained model from disk.

        Args:
            model_path: Path to model checkpoint
            **kwargs: Model-specific loading options (e.g., device='cuda')

        Returns:
            Initialized model ready for inference

        Example:
            model = SelectionalPreferenceModel.load(
                "models/selectional_preference/best_model.pt",
                device='cuda'
            )
        """
        instance = cls(model_path=model_path, **kwargs)
        instance._load_checkpoint()
        return instance

    def _load_checkpoint(self):
        """
        Internal method to load model weights.

        Subclasses should override this to load their specific checkpoint format.
        """
        if self.model_path and Path(self.model_path).exists():
            # Subclasses implement specific loading logic
            pass


# =============================================================================
# Example Mock Implementation (for testing before M1-M4 are trained)
# =============================================================================

class MockSelectionalModel(SemanticModel):
    """
    Mock implementation of M1 for testing infrastructure.

    Returns placeholder scores until real model is trained.
    """

    def enrich(self, ast: EnrichedAST, **kwargs) -> EnrichedAST:
        """Add mock selectional preference annotations."""
        from klareco.enriched_ast import SelectionalAnnotation, AnnotationMetadata
        import time

        self.validate_input(ast)

        # Clone for immutability
        enriched = ast.clone()

        # Mock scores (replace with real model when trained)
        enriched.selectional = SelectionalAnnotation(
            subject_verb_score=0.75,
            verb_object_score=0.80,
            triple_plausibility=0.77,
            meta=AnnotationMetadata(
                model_name="mock_selectional",
                model_version="0.1.0",
                confidence=0.5,
                compute_time_ms=1.0,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
                parameters={"mode": "mock"}
            )
        )

        enriched.stages_applied.add("selectional")
        return enriched

    def get_metadata(self) -> AnnotationMetadata:
        """Return mock model metadata."""
        from klareco.enriched_ast import AnnotationMetadata
        import time

        return AnnotationMetadata(
            model_name="mock_selectional",
            model_version="0.1.0",
            confidence=0.5,
            compute_time_ms=0.0,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            parameters={"mode": "mock", "note": "Replace with trained model"}
        )

    def get_capabilities(self) -> List[ModelCapability]:
        """Declare capabilities."""
        return [ModelCapability.SELECTIONAL_PREFERENCE]

    def get_dependencies(self) -> List[str]:
        """Declare dependencies."""
        return ["stage1_semantic"]  # Needs root embeddings

    def explain(self, ast: EnrichedAST) -> str:
        """Explain mock annotations."""
        if not ast.selectional:
            return "No selectional annotations"

        return (
            f"Mock Selectional Preference:\n"
            f"  Subject-Verb: {ast.selectional.subject_verb_score:.3f}\n"
            f"  Verb-Object: {ast.selectional.verb_object_score:.3f}\n"
            f"  Triple Plausibility: {ast.selectional.triple_plausibility:.3f}\n"
            f"  (These are placeholder values until M1 is trained)"
        )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'SemanticModel',
    'ModelCapability',
    'MockSelectionalModel',
]
