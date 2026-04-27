"""
AST Annotator Base Class - Standard Interface for Semantic Annotations

This module defines the standard protocol for all Klareco models to:
1. Consume structured ASTs (with deterministic features already computed)
2. Add semantic annotations without re-learning deterministic features
3. Pass annotated ASTs to downstream models

Design Principles:
- NEVER re-parse grammar (case/tense/gender/number already in AST from M0)
- NEVER re-learn deterministic Esperanto features
- ALWAYS read previous annotations (root_embedding, etc.)
- ALWAYS add new annotations to existing AST structure
- Models focus on SEMANTIC meaning, not grammatical structure

Example Pipeline:
    ast = parser.parse(text)              # M0: Deterministic features
    ast = root_embeddings.annotate(ast)   # Add: root_embedding (64d)
    ast = compositional.annotate(ast)     # Add: word_embedding (128d)
    ast = m1_selectional.annotate(ast)    # Add: M1_plausibility
    ast = entity_classifier.annotate(ast) # Add: entity_type
    ast = taxonomy.annotate(ast)          # Add: hypernyms (90% deterministic!)
    ast = coreference.annotate(ast)       # Add: coref_cluster (80% deterministic!)
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any, List
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Utility Functions for AST Serialization
# ============================================================================

def convert_tensors_to_lists(ast: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively convert all torch.Tensor annotations to lists for JSON serialization.

    Use this when saving ASTs to disk or debugging.

    Args:
        ast: AST dictionary (may contain tensors in annotations)

    Returns:
        AST with all tensors converted to lists (JSON-serializable)

    Example:
        >>> ast = parser.parse("La hundo kuras.")
        >>> ast = root_embeddings.annotate(ast)  # Adds tensor embeddings
        >>> ast_serializable = convert_tensors_to_lists(ast)
        >>> with open('debug.json', 'w') as f:
        ...     json.dump(ast_serializable, f)  # Now works!
    """
    import torch
    import copy

    ast_copy = copy.deepcopy(ast)

    def _convert_node(node):
        if isinstance(node, dict):
            if 'annotations' in node and isinstance(node['annotations'], dict):
                for key, value in node['annotations'].items():
                    if isinstance(value, torch.Tensor):
                        node['annotations'][key] = value.cpu().detach().numpy().tolist()

            # Recursively convert children
            for key, value in node.items():
                if key == 'annotations':
                    continue  # Already processed
                if isinstance(value, dict):
                    _convert_node(value)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            _convert_node(item)

        return node

    return _convert_node(ast_copy)


def get_annotation_summary(ast: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Get summary of all annotations in an AST (for debugging).

    Args:
        ast: AST dictionary

    Returns:
        Dictionary mapping annotation keys to list of node paths where they appear

    Example:
        >>> summary = get_annotation_summary(ast)
        >>> print(summary)
        {
            'root_embedding': ['verbo', 'subjekto.kerno', 'objekto'],
            'M1_plausibility': ['<root>'],
            'entity_type': ['subjekto.kerno']
        }
    """
    import torch

    annotations = {}

    def _collect_annotations(node, path='<root>'):
        if isinstance(node, dict):
            if 'annotations' in node and isinstance(node['annotations'], dict):
                for key, value in node['annotations'].items():
                    if key not in annotations:
                        annotations[key] = []

                    # Add type info
                    if isinstance(value, torch.Tensor):
                        type_info = f"{path} (tensor: {list(value.shape)})"
                    elif isinstance(value, list):
                        type_info = f"{path} (list: len={len(value)})"
                    elif isinstance(value, float):
                        type_info = f"{path} (float: {value:.4f})"
                    else:
                        type_info = f"{path} ({type(value).__name__})"

                    annotations[key].append(type_info)

            # Recursively collect from children
            for key, value in node.items():
                if key == 'annotations':
                    continue
                if isinstance(value, dict):
                    _collect_annotations(value, f"{path}.{key}")
                elif isinstance(value, list):
                    for i, item in enumerate(value):
                        if isinstance(item, dict):
                            _collect_annotations(item, f"{path}.{key}[{i}]")

    _collect_annotations(ast)
    return annotations


class ASTAnnotator(ABC):
    """
    Abstract base class for all models that annotate ASTs.

    All Klareco models (root embeddings, M1, M2.1, M2.2, etc.) should inherit
    from this class and implement the annotate() method.

    The annotate() method:
    - Takes an AST (dict) with existing annotations
    - Returns the same AST with new annotations added
    - NEVER modifies deterministic features (kazo, nombro, genro, tempo)
    - NEVER re-computes features already in the AST
    """

    def __init__(self, model_name: str):
        """
        Initialize the annotator.

        Args:
            model_name: Human-readable name for this model (e.g., "RootEmbeddings", "M1Selectional")
        """
        self.model_name = model_name
        self._validate_setup()

    @abstractmethod
    def annotate(self, ast: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Add semantic annotations to an AST.

        This method MUST:
        1. Read structure from AST (subjekto, verbo, objekto, aliaj)
        2. Read deterministic features (kazo, nombro, genro, tempo) - never re-compute!
        3. Read previous annotations (root_embedding, word_embedding, etc.)
        4. Add new semantic annotations relevant to this model
        5. Return the AST with new annotations added

        This method MUST NOT:
        - Modify deterministic features (kazo, nombro, genro, tempo, vortspeco)
        - Re-parse grammatical structure
        - Remove existing annotations
        - Change AST structure (subjekto/verbo/objekto)

        Args:
            ast: AST dictionary with structure:
                {
                    'tipo': 'frazo',
                    'subjekto': {...},
                    'verbo': {'radiko': '...', 'kazo': '...', 'nombro': '...', ...},
                    'objekto': {...},
                    'aliaj': [...],
                    'annotations': {  # Previous annotations
                        'root_embedding': [...],
                        'word_embedding': [...],
                        ...
                    }
                }
            context: Optional context dict with:
                {
                    'previous_sentences': [...],  # For coreference
                    'document_id': '...',         # For retrieval
                    'query': '...',               # For relevance scoring
                    ...
                }

        Returns:
            AST with new annotations added to 'annotations' dict

        Example:
            >>> ast = parser.parse("La hundo vidis la katon.")
            >>> # AST has deterministic features: kazo='n', nombro='singular', etc.
            >>> ast = root_embeddings.annotate(ast)
            >>> # AST now has: annotations['root_embedding'] = [...]
            >>> ast = m1_selectional.annotate(ast)
            >>> # AST now has: annotations['M1_plausibility'] = 0.87
        """
        pass

    def _validate_setup(self):
        """
        Validate that the model is correctly configured.

        Override this method to add model-specific validation (e.g., check
        that vocabulary files exist, model checkpoint is loaded, etc.)
        """
        pass

    def _ensure_annotations_dict(self, ast: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure AST has an 'annotations' dict for storing semantic annotations.

        Args:
            ast: AST dictionary

        Returns:
            AST with 'annotations' dict (creates if missing)
        """
        if 'annotations' not in ast:
            ast['annotations'] = {}
        return ast

    def _get_annotation(self, ast: Dict[str, Any], key: str, default: Any = None) -> Any:
        """
        Get a previous annotation from the AST.

        Args:
            ast: AST dictionary
            key: Annotation key (e.g., 'root_embedding', 'M1_plausibility')
            default: Default value if annotation not found

        Returns:
            Annotation value or default
        """
        return ast.get('annotations', {}).get(key, default)

    def _get_annotation_tensor(self, ast: Dict[str, Any], key: str, default: Any = None):
        """
        Get annotation as tensor (convert from list if needed).

        Use this when reading embeddings from previous annotators.
        Handles both tensor and list representations transparently.

        Args:
            ast: AST dictionary
            key: Annotation key (e.g., 'root_embedding')
            default: Default value if annotation not found

        Returns:
            torch.Tensor or default

        Example:
            # Previous annotator added embedding (could be tensor or list)
            root_emb = self._get_annotation_tensor(ast['verbo'], 'root_embedding')
            # Always returns tensor (converts from list if needed)
        """
        import torch

        value = self._get_annotation(ast, key, default)

        if value is None:
            return default

        if isinstance(value, torch.Tensor):
            return value
        elif isinstance(value, (list, tuple)):
            return torch.tensor(value, dtype=torch.float32)
        else:
            return value

    def _add_annotation(self, ast: Dict[str, Any], key: str, value: Any, keep_tensor: bool = True) -> Dict[str, Any]:
        """
        Add a new annotation to the AST.

        Supports two modes:
        1. keep_tensor=True (default): Keep torch.Tensor as-is for pipeline efficiency
        2. keep_tensor=False: Convert to list for JSON serialization/debugging

        Args:
            ast: AST dictionary
            key: Annotation key (e.g., 'root_embedding', 'M1_plausibility')
            value: Annotation value (can be tensor, float, string, list, etc.)
            keep_tensor: If True, keep tensors as-is. If False, convert to list.

        Returns:
            AST with new annotation added

        Example:
            # During inference: Keep tensors for efficiency
            self._add_annotation(ast, 'root_embedding', torch.tensor([...]), keep_tensor=True)
            # Downstream models can read tensor directly (zero conversion overhead)

            # For serialization/debugging: Convert to list
            self._add_annotation(ast, 'root_embedding', torch.tensor([...]), keep_tensor=False)
            # Can now save to JSON: json.dump(ast, f)
        """
        import torch  # Import here to avoid hard dependency

        ast = self._ensure_annotations_dict(ast)

        if isinstance(value, torch.Tensor):
            if keep_tensor:
                # Keep as tensor for downstream models
                ast['annotations'][key] = value
            else:
                # Convert to list for JSON serialization
                ast['annotations'][key] = value.cpu().detach().numpy().tolist()
        else:
            ast['annotations'][key] = value

        return ast

    def _read_deterministic_feature(self, word_ast: Dict[str, Any], feature: str) -> Any:
        """
        Read a deterministic feature from a word AST.

        Deterministic features are computed by M0 parser and MUST NOT be
        re-computed or modified by semantic models.

        Args:
            word_ast: Word-level AST node
            feature: Feature name ('kazo', 'nombro', 'genro', 'tempo', 'vortspeco')

        Returns:
            Feature value (e.g., 'nominativo', 'singularo', 'maskulino', 'estanto')

        Raises:
            KeyError: If feature not found (indicates M0 parser bug)
        """
        DETERMINISTIC_FEATURES = {'kazo', 'nombro', 'genro', 'tempo', 'vortspeco', 'radiko'}

        if feature not in DETERMINISTIC_FEATURES:
            logger.warning(f"Feature '{feature}' is not a deterministic feature. "
                          f"Use _get_annotation() for semantic features.")

        if feature not in word_ast:
            raise KeyError(f"Deterministic feature '{feature}' not found in word AST. "
                          f"This indicates a bug in M0 parser. Word: {word_ast}")

        return word_ast[feature]

    def annotate_batch(self, asts: List[Dict[str, Any]], context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Annotate a batch of ASTs (for efficiency).

        Default implementation: annotate each AST individually.
        Override this method for models that benefit from batch processing
        (e.g., neural models with GPU acceleration).

        Args:
            asts: List of AST dictionaries
            context: Optional context dict

        Returns:
            List of annotated ASTs
        """
        return [self.annotate(ast, context) for ast in asts]

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(model_name='{self.model_name}')>"


class DeterministicAnnotator(ASTAnnotator):
    """
    Base class for deterministic annotators (M2.1 Taxonomy, M2.2 Coreference).

    Deterministic annotators:
    - Use rule-based logic (no learned parameters)
    - Query knowledge bases (ReVo, ConceptNet, Fundamento)
    - Apply Esperanto grammar rules
    - Only use learned models as FALLBACK for OOV/ambiguous cases

    Example: M2.1 Taxonomy
        - 90% deterministic: ReVo definitions, ConceptNet IS-A, affix rules
        - 10% learned: Fallback for OOV words not in ReVo

    Example: M2.2 Coreference
        - 80% deterministic: Gender/number/case matching from Esperanto grammar
        - 20% learned: Disambiguation when multiple candidates match
    """

    def __init__(self, model_name: str, fallback_model: Optional[ASTAnnotator] = None):
        """
        Initialize deterministic annotator.

        Args:
            model_name: Human-readable name
            fallback_model: Optional learned model for OOV/ambiguous cases
        """
        super().__init__(model_name)
        self.fallback_model = fallback_model

    def annotate(self, ast: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Annotate using deterministic rules, fallback to learned model if needed.

        Subclasses should override:
        - _annotate_deterministic(): Rule-based annotation
        - _annotate_fallback(): Learned model annotation (optional)
        """
        # Try deterministic annotation first
        ast, success = self._annotate_deterministic(ast, context)

        if not success:
            if self.fallback_model is not None:
                logger.debug(f"{self.model_name}: Deterministic annotation failed, using fallback model")
                ast = self._annotate_fallback(ast, context)
            else:
                raise RuntimeError(
                    f"{self.model_name}: No fallback model configured but deterministic annotation failed"
                )

        return ast

    @abstractmethod
    def _annotate_deterministic(self, ast: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> tuple[Dict[str, Any], bool]:
        """
        Apply deterministic rules to annotate AST.

        Returns:
            (annotated_ast, success) where success=True if annotation was deterministic
        """
        pass

    def _annotate_fallback(self, ast: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Use learned fallback model to annotate AST (for OOV/ambiguous cases).

        Default: Use fallback_model.annotate()
        Override if custom fallback logic needed.
        """
        if self.fallback_model is None:
            raise RuntimeError(f"{self.model_name}: No fallback model configured but deterministic annotation failed")

        return self.fallback_model.annotate(ast, context)
