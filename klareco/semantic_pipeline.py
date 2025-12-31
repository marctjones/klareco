"""
SemanticPipeline: Staged model pipeline for EnrichedAST enrichment.

This module provides the pipeline that chains models together, where each model
reads from and writes to an EnrichedAST. This is the runtime component of the
"AST as thought" architecture.

Pipeline Flow:
    Text → Parser (Stage 0) → Semantic Model (Stage 1) → Grammatical (Stage 2) → ...

Each stage adds learned representations to the EnrichedAST without modifying
the deterministic parser output.

Usage:
    pipeline = SemanticPipeline.load()
    enriched = pipeline.for_retrieval("La hundo kuras rapide.")
    embedding = enriched.get_effective_embedding()
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Callable

from klareco.enriched_ast import EnrichedAST
from klareco.parser import parse

try:
    import torch
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

# Default model paths
DEFAULT_ROOT_MODEL = Path("models/root_embeddings/best_model.pt")
DEFAULT_AFFIX_MODEL = Path("models/affix_transforms_v2/best_model.pt")

# Function words excluded from embeddings (handled by AST layer)
FUNCTION_WORDS = {
    # Conjunctions
    'kaj', 'aŭ', 'sed', 'nek', 'do', 'tamen', 'ĉar', 'ke', 'se',
    # Prepositions
    'al', 'de', 'en', 'el', 'kun', 'per', 'por', 'pri', 'sen', 'sur', 'sub', 'ĉe', 'tra', 'ĉirkaŭ',
    # Pronouns
    'mi', 'vi', 'li', 'ŝi', 'ĝi', 'ni', 'ili', 'si', 'oni',
    # Correlatives
    'kiu', 'kio', 'kia', 'kie', 'kiel', 'kiam', 'kiom', 'kial',
    'tiu', 'tio', 'tia', 'tie', 'tiel', 'tiam', 'tiom', 'tial',
    'ĉiu', 'ĉio', 'ĉia', 'ĉie', 'ĉiel', 'ĉiam', 'ĉiom', 'ĉial',
    'neniu', 'nenio', 'nenia', 'nenie', 'neniel', 'neniam', 'neniom', 'nenial',
    'iu', 'io', 'ia', 'ie', 'iel', 'iam', 'iom', 'ial',
    # Particles
    'la', 'ne', 'tre', 'nur', 'ankaŭ', 'eĉ', 'ja', 'jen', 'jes', 'plej', 'pli', 'tro',
}


class LowRankTransform:
    """Low-rank transformation for affixes.

    The affix model stores:
    - down.weight: (rank, embedding_dim)
    - up.weight: (embedding_dim, rank)

    Transform: x @ down.T @ up.T = x @ (embedding_dim, rank) @ (rank, embedding_dim)
    """

    def __init__(self, down: 'torch.Tensor', up: 'torch.Tensor'):
        """
        Args:
            down: Down projection (rank, embedding_dim)
            up: Up projection (embedding_dim, rank)
        """
        # Store transposed for efficient computation
        self.down_T = down.T  # (embedding_dim, rank)
        self.up_T = up.T      # (rank, embedding_dim)

    def __call__(self, x: 'torch.Tensor') -> 'torch.Tensor':
        """Apply low-rank transform: x @ down.T @ up.T"""
        return x @ self.down_T @ self.up_T


class SemanticModel:
    """
    Stage 1 semantic model: root embeddings + affix transforms.

    Loads trained models and applies them to EnrichedAST.
    """

    def __init__(
        self,
        root_embeddings: 'torch.Tensor',
        root_to_idx: Dict[str, int],
        embedding_dim: int,
        prefix_transforms: Optional[Dict[str, LowRankTransform]] = None,
        suffix_transforms: Optional[Dict[str, LowRankTransform]] = None,
    ):
        self.root_embeddings = root_embeddings
        self.root_to_idx = root_to_idx
        self.embedding_dim = embedding_dim
        self.prefix_transforms = prefix_transforms or {}
        self.suffix_transforms = suffix_transforms or {}

    @classmethod
    def load(
        cls,
        root_model_path: Path = DEFAULT_ROOT_MODEL,
        affix_model_path: Optional[Path] = DEFAULT_AFFIX_MODEL,
    ) -> 'SemanticModel':
        """
        Load trained models from checkpoints.

        Args:
            root_model_path: Path to root embeddings model
            affix_model_path: Path to affix transforms model (optional)

        Returns:
            SemanticModel instance ready for inference
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for semantic model")

        # Load root embeddings
        logger.info(f"Loading root embeddings from {root_model_path}")
        checkpoint = torch.load(root_model_path, map_location='cpu', weights_only=False)
        root_embeddings = checkpoint['model_state_dict']['embeddings.weight']
        root_to_idx = checkpoint['root_to_idx']
        embedding_dim = root_embeddings.shape[1]
        logger.info(f"Loaded {len(root_to_idx)} roots, dim={embedding_dim}")

        # Load affix transforms if available
        prefix_transforms = {}
        suffix_transforms = {}

        if affix_model_path and affix_model_path.exists():
            logger.info(f"Loading affix transforms from {affix_model_path}")
            affix_checkpoint = torch.load(affix_model_path, map_location='cpu', weights_only=False)

            rank = affix_checkpoint['rank']
            prefixes = affix_checkpoint['prefixes']
            suffixes = affix_checkpoint['suffixes']
            state = affix_checkpoint['model_state_dict']

            # Load prefix transforms
            # Checkpoint format: prefix_transforms.{name}.down.weight, prefix_transforms.{name}.up.weight
            for prefix in prefixes:
                down_key = f'prefix_transforms.{prefix}.down.weight'
                up_key = f'prefix_transforms.{prefix}.up.weight'
                if down_key in state and up_key in state:
                    # down: (embedding_dim, rank), up: (rank, embedding_dim)
                    # Transform: x @ down @ up
                    down = state[down_key]
                    up = state[up_key]
                    prefix_transforms[prefix] = LowRankTransform(down, up)

            # Load suffix transforms
            for suffix in suffixes:
                down_key = f'suffix_transforms.{suffix}.down.weight'
                up_key = f'suffix_transforms.{suffix}.up.weight'
                if down_key in state and up_key in state:
                    down = state[down_key]
                    up = state[up_key]
                    suffix_transforms[suffix] = LowRankTransform(down, up)

            logger.info(f"Loaded {len(prefix_transforms)} prefixes, {len(suffix_transforms)} suffixes (rank={rank})")

        return cls(
            root_embeddings=root_embeddings,
            root_to_idx=root_to_idx,
            embedding_dim=embedding_dim,
            prefix_transforms=prefix_transforms,
            suffix_transforms=suffix_transforms,
        )

    def embed_word(self, word_ast: Dict[str, Any]) -> Optional['torch.Tensor']:
        """
        Compute embedding for a single word AST.

        Applies: root_embedding + prefix_transform + suffix_transforms

        Args:
            word_ast: Word AST from parser

        Returns:
            Word embedding tensor, or None if root unknown
        """
        root = word_ast.get('radiko', '').lower()

        # Skip function words
        if root in FUNCTION_WORDS:
            return None

        # Check if root is known
        if root not in self.root_to_idx:
            return None

        # Get root embedding
        idx = self.root_to_idx[root]
        emb = self.root_embeddings[idx].clone()

        # Apply prefix transforms
        prefixes = word_ast.get('prefiksoj', [])
        for prefix in prefixes:
            if prefix in self.prefix_transforms:
                emb = self.prefix_transforms[prefix](emb)

        # Apply suffix transforms
        suffixes = word_ast.get('sufiksoj', [])
        for suffix in suffixes:
            if suffix in self.suffix_transforms:
                emb = self.suffix_transforms[suffix](emb)

        return emb

    def __call__(self, ast: EnrichedAST) -> EnrichedAST:
        """
        Apply Stage 1 semantic model to EnrichedAST.

        Computes embeddings for content words and pools to sentence embedding.

        Args:
            ast: EnrichedAST with Stage 0 (parser) output

        Returns:
            New EnrichedAST with Stage 1 enrichments
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for semantic model")

        content_embeddings = {}
        known_roots = set()
        unknown_roots = set()
        word_embeddings = []

        # Process all content words
        for word_ast in ast.get_all_words():
            root = word_ast.get('radiko', '').lower()

            # Skip function words
            if root in FUNCTION_WORDS:
                continue

            # Try to embed
            emb = self.embed_word(word_ast)

            if emb is not None:
                word_id = f"{root}_{len(content_embeddings)}"
                content_embeddings[word_id] = emb
                word_embeddings.append(emb)
                known_roots.add(root)
            else:
                unknown_roots.add(root)

        # Pool to sentence embedding (mean of word embeddings)
        if word_embeddings:
            sentence_embedding = torch.stack(word_embeddings).mean(dim=0)
        else:
            # Fallback: zero embedding
            sentence_embedding = torch.zeros(self.embedding_dim)

        # Return enriched AST
        return ast.with_stage1(
            sentence_embedding=sentence_embedding,
            content_embeddings=content_embeddings,
            known_roots=known_roots,
            unknown_roots=unknown_roots,
        )


class GrammaticalModel:
    """
    Stage 2 grammatical model (placeholder for future implementation).

    Will apply transformations for negation, tense, mood, etc.
    """

    def __call__(self, ast: EnrichedAST) -> EnrichedAST:
        """Apply Stage 2 grammatical transforms."""
        # TODO: Implement after Stage 2 training (#157, #158)
        # For now, just pass through with Stage 2 marker
        return ast.with_stage2(
            grammatical_embedding=ast.sentence_embedding,
        )


class DiscourseModel:
    """
    Stage 3 discourse model (placeholder for future implementation).

    Will handle coreference chains and discourse relations.
    """

    def __call__(
        self,
        ast: EnrichedAST,
        context: Optional[List[EnrichedAST]] = None
    ) -> EnrichedAST:
        """Apply Stage 3 discourse model."""
        # TODO: Implement after Stage 3 design
        return ast.with_stage3()


class SemanticPipeline:
    """
    Pipeline that chains models, where each reads/writes to EnrichedAST.

    This is the main entry point for semantic processing.

    Example:
        pipeline = SemanticPipeline.load()

        # For retrieval (fast, Stage 1 only)
        enriched = pipeline.for_retrieval("La hundo kuras rapide.")

        # For Q&A (full pipeline)
        enriched = pipeline.for_qa("Kio estas Esperanto?", context=[...])
    """

    def __init__(
        self,
        semantic: Optional[SemanticModel] = None,
        grammatical: Optional[GrammaticalModel] = None,
        discourse: Optional[DiscourseModel] = None,
    ):
        """
        Initialize pipeline with models.

        Models are loaded lazily if not provided.
        """
        self._semantic = semantic
        self._grammatical = grammatical
        self._discourse = discourse

        # Stage hooks for debugging (#107)
        self.stage_hooks: Dict[str, List[Callable]] = {
            'stage0': [],
            'stage1': [],
            'stage2': [],
            'stage3': [],
        }

    @classmethod
    def load(
        cls,
        root_model_path: Path = DEFAULT_ROOT_MODEL,
        affix_model_path: Optional[Path] = DEFAULT_AFFIX_MODEL,
        load_stage2: bool = False,
        load_stage3: bool = False,
    ) -> 'SemanticPipeline':
        """
        Load pipeline with trained models.

        Args:
            root_model_path: Path to root embeddings
            affix_model_path: Path to affix transforms
            load_stage2: If True, load grammatical model (future)
            load_stage3: If True, load discourse model (future)

        Returns:
            SemanticPipeline ready for inference
        """
        semantic = SemanticModel.load(root_model_path, affix_model_path)
        grammatical = GrammaticalModel() if load_stage2 else None
        discourse = DiscourseModel() if load_stage3 else None

        return cls(
            semantic=semantic,
            grammatical=grammatical,
            discourse=discourse,
        )

    @property
    def semantic(self) -> SemanticModel:
        """Lazy-load semantic model."""
        if self._semantic is None:
            self._semantic = SemanticModel.load()
        return self._semantic

    @property
    def grammatical(self) -> GrammaticalModel:
        """Lazy-load grammatical model."""
        if self._grammatical is None:
            self._grammatical = GrammaticalModel()
        return self._grammatical

    @property
    def discourse(self) -> DiscourseModel:
        """Lazy-load discourse model."""
        if self._discourse is None:
            self._discourse = DiscourseModel()
        return self._discourse

    def add_hook(self, stage: str, hook: Callable[[EnrichedAST], None]) -> None:
        """
        Add a hook to be called after a stage completes.

        Useful for debugging and visualization (#107).

        Args:
            stage: 'stage0', 'stage1', 'stage2', or 'stage3'
            hook: Callable that receives the EnrichedAST
        """
        if stage in self.stage_hooks:
            self.stage_hooks[stage].append(hook)

    def _run_hooks(self, stage: str, ast: EnrichedAST) -> None:
        """Run all hooks for a stage."""
        for hook in self.stage_hooks.get(stage, []):
            try:
                hook(ast)
            except Exception as e:
                logger.warning(f"Hook error in {stage}: {e}")

    def __call__(
        self,
        text: str,
        context: Optional[List[EnrichedAST]] = None,
        stages: Optional[List[str]] = None,
    ) -> EnrichedAST:
        """
        Process text through pipeline stages.

        Args:
            text: Input Esperanto text
            context: Previous sentences for discourse (Stage 3)
            stages: Which stages to run ('semantic', 'grammatical', 'discourse')
                   If None, runs semantic only.

        Returns:
            EnrichedAST with requested enrichments
        """
        if stages is None:
            stages = ['semantic']

        # Stage 0: Parse (always runs)
        try:
            raw_ast = parse(text)
            ast = EnrichedAST.from_parser_output(raw_ast, text)
            self._run_hooks('stage0', ast)
        except Exception as e:
            logger.error(f"Parse error: {e}")
            # Return minimal AST for robustness
            ast = EnrichedAST(original_text=text, stages_applied={'stage0'})
            return ast

        # Stage 1: Semantic
        if 'semantic' in stages:
            try:
                ast = self.semantic(ast)
                self._run_hooks('stage1', ast)
            except Exception as e:
                logger.error(f"Semantic model error: {e}")

        # Stage 2: Grammatical
        if 'grammatical' in stages:
            try:
                ast = self.grammatical(ast)
                self._run_hooks('stage2', ast)
            except Exception as e:
                logger.error(f"Grammatical model error: {e}")

        # Stage 3: Discourse
        if 'discourse' in stages and context is not None:
            try:
                ast = self.discourse(ast, context)
                self._run_hooks('stage3', ast)
            except Exception as e:
                logger.error(f"Discourse model error: {e}")

        return ast

    def for_retrieval(self, text: str) -> EnrichedAST:
        """
        Quick path: semantic only for indexing/retrieval.

        This is the most common use case - fast embedding for search.
        """
        return self(text, stages=['semantic'])

    def for_qa(
        self,
        text: str,
        context: Optional[List[EnrichedAST]] = None
    ) -> EnrichedAST:
        """
        Full path: all stages for question answering.

        Uses the complete pipeline including grammatical and discourse.
        """
        return self(text, context, stages=['semantic', 'grammatical', 'discourse'])

    def for_analysis(self, text: str) -> EnrichedAST:
        """
        Analysis path: semantic + grammatical for sentence analysis.

        Useful for understanding sentence semantics without discourse context.
        """
        return self(text, stages=['semantic', 'grammatical'])

    def batch_for_retrieval(
        self,
        texts: List[str],
        show_progress: bool = False,
    ) -> List[EnrichedAST]:
        """
        Batch processing for corpus indexing.

        Args:
            texts: List of Esperanto texts
            show_progress: If True, log progress

        Returns:
            List of EnrichedASTs with semantic embeddings
        """
        results = []
        total = len(texts)

        for i, text in enumerate(texts):
            if show_progress and (i + 1) % 1000 == 0:
                logger.info(f"Processed {i + 1}/{total} sentences")

            results.append(self.for_retrieval(text))

        return results
