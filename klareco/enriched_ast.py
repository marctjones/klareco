"""
EnrichedAST: A structured container that accumulates semantic meaning through the pipeline.

This dataclass wraps the parser's Dict-based AST and provides slots for each stage's
enrichments. As the AST flows through the pipeline, each stage adds its learned
representations without modifying the deterministic parser output.

Stage Progression:
- Stage 0 (Parser): Deterministic morphology, syntax, grammar features
- Stage 1 (Semantic): Root embeddings + affix transforms → sentence embedding
- Stage 2 (Grammatical): Negation, tense, mood effects → refined embedding
- Stage 3 (Discourse): Coreference chains, discourse relations
- Stage 4 (Reasoning): AST-to-AST inference (future)

Design Principles:
- Immutable progression: Each stage creates a new EnrichedAST via clone()
- Lazy enrichment: Stages can be applied on-demand at query time
- Serializable: Full round-trip to/from dict for storage
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
import copy

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class EnrichedAST:
    """
    A container for parser AST + learned enrichments from each pipeline stage.

    The original parser output is preserved in `parser_ast`, while learned
    representations are added to stage-specific slots.
    """

    # =========================================================================
    # Stage 0: Parser Output (Deterministic)
    # =========================================================================

    # The original parser Dict output, preserved unchanged
    parser_ast: Dict[str, Any] = field(default_factory=dict)

    # Convenience accessors for common parser fields
    # These are copied from parser_ast for easier access
    tipo: str = "frazo"
    fraztipo: str = "deklaro"  # deklaro, demando, ordono
    negita: bool = False

    # Original text (for debugging/display)
    original_text: str = ""

    # =========================================================================
    # Stage 1: Semantic Model (Learned)
    # =========================================================================

    # Per-word content embeddings: word_id -> embedding tensor
    # Only content words get embeddings (not function words)
    content_embeddings: Dict[str, Any] = field(default_factory=dict)

    # Pooled sentence embedding (64d from Stage 1)
    sentence_embedding: Optional[Any] = None

    # Which roots were found vs missing
    known_roots: Set[str] = field(default_factory=set)
    unknown_roots: Set[str] = field(default_factory=set)

    # =========================================================================
    # Stage 2: Grammatical Model (Learned)
    # =========================================================================

    # Refined embedding after grammatical transforms
    grammatical_embedding: Optional[Any] = None

    # Individual transform effects (for explainability)
    negation_effect: Optional[Any] = None
    tense_effect: Optional[Any] = None
    mood_effect: Optional[Any] = None

    # =========================================================================
    # Stage 3: Discourse Model (Learned)
    # =========================================================================

    # Coreference links: pronoun_id -> antecedent_id
    coreference_links: Dict[str, str] = field(default_factory=dict)

    # Discourse relation to previous sentence
    discourse_relation: Optional[str] = None

    # =========================================================================
    # Stage 4: Reasoning Model (Future)
    # =========================================================================

    # Inferred ASTs from reasoning
    inferences: List['EnrichedAST'] = field(default_factory=list)

    # =========================================================================
    # Pipeline Metadata
    # =========================================================================

    # Track which stages have been applied
    stages_applied: Set[str] = field(default_factory=set)

    # Source information (for corpus tracking)
    source_id: Optional[str] = None
    tier: Optional[int] = None

    # =========================================================================
    # Factory Methods
    # =========================================================================

    @classmethod
    def from_parser_output(cls, ast: Dict[str, Any], text: str = "") -> 'EnrichedAST':
        """
        Create an EnrichedAST from parser.parse() output.

        Args:
            ast: The Dict output from parser.parse()
            text: Original text (optional, for debugging)

        Returns:
            EnrichedAST with Stage 0 fields populated
        """
        return cls(
            parser_ast=ast,
            tipo=ast.get("tipo", "frazo"),
            fraztipo=ast.get("fraztipo", "deklaro"),
            negita=ast.get("negita", False),
            original_text=text,
            stages_applied={"stage0"}
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnrichedAST':
        """
        Deserialize from a dict (e.g., from JSON storage).

        Handles tensor conversion if torch is available.
        """
        # Handle embeddings - convert lists back to tensors if torch available
        content_embeddings = {}
        if "content_embeddings" in data and TORCH_AVAILABLE:
            for word_id, emb in data["content_embeddings"].items():
                if isinstance(emb, list):
                    content_embeddings[word_id] = torch.tensor(emb)
                else:
                    content_embeddings[word_id] = emb

        sentence_embedding = None
        if "sentence_embedding" in data and data["sentence_embedding"] is not None:
            if TORCH_AVAILABLE and isinstance(data["sentence_embedding"], list):
                sentence_embedding = torch.tensor(data["sentence_embedding"])
            else:
                sentence_embedding = data["sentence_embedding"]

        grammatical_embedding = None
        if "grammatical_embedding" in data and data["grammatical_embedding"] is not None:
            if TORCH_AVAILABLE and isinstance(data["grammatical_embedding"], list):
                grammatical_embedding = torch.tensor(data["grammatical_embedding"])
            else:
                grammatical_embedding = data["grammatical_embedding"]

        return cls(
            parser_ast=data.get("parser_ast", {}),
            tipo=data.get("tipo", "frazo"),
            fraztipo=data.get("fraztipo", "deklaro"),
            negita=data.get("negita", False),
            original_text=data.get("original_text", ""),
            content_embeddings=content_embeddings,
            sentence_embedding=sentence_embedding,
            known_roots=set(data.get("known_roots", [])),
            unknown_roots=set(data.get("unknown_roots", [])),
            grammatical_embedding=grammatical_embedding,
            coreference_links=data.get("coreference_links", {}),
            discourse_relation=data.get("discourse_relation"),
            stages_applied=set(data.get("stages_applied", [])),
            source_id=data.get("source_id"),
            tier=data.get("tier"),
        )

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(self, include_embeddings: bool = True) -> Dict[str, Any]:
        """
        Serialize to a dict suitable for JSON storage.

        Args:
            include_embeddings: If True, include tensor data as lists.
                               If False, omit embeddings (for metadata-only storage).
        """
        result = {
            "parser_ast": self.parser_ast,
            "tipo": self.tipo,
            "fraztipo": self.fraztipo,
            "negita": self.negita,
            "original_text": self.original_text,
            "known_roots": list(self.known_roots),
            "unknown_roots": list(self.unknown_roots),
            "coreference_links": self.coreference_links,
            "discourse_relation": self.discourse_relation,
            "stages_applied": list(self.stages_applied),
            "source_id": self.source_id,
            "tier": self.tier,
        }

        if include_embeddings:
            # Convert tensors to lists for JSON serialization
            content_emb_dict = {}
            for word_id, emb in self.content_embeddings.items():
                if TORCH_AVAILABLE and hasattr(emb, 'tolist'):
                    content_emb_dict[word_id] = emb.tolist()
                elif hasattr(emb, 'tolist'):  # numpy
                    content_emb_dict[word_id] = emb.tolist()
                else:
                    content_emb_dict[word_id] = emb
            result["content_embeddings"] = content_emb_dict

            if self.sentence_embedding is not None:
                if TORCH_AVAILABLE and hasattr(self.sentence_embedding, 'tolist'):
                    result["sentence_embedding"] = self.sentence_embedding.tolist()
                elif hasattr(self.sentence_embedding, 'tolist'):
                    result["sentence_embedding"] = self.sentence_embedding.tolist()
                else:
                    result["sentence_embedding"] = self.sentence_embedding
            else:
                result["sentence_embedding"] = None

            if self.grammatical_embedding is not None:
                if TORCH_AVAILABLE and hasattr(self.grammatical_embedding, 'tolist'):
                    result["grammatical_embedding"] = self.grammatical_embedding.tolist()
                elif hasattr(self.grammatical_embedding, 'tolist'):
                    result["grammatical_embedding"] = self.grammatical_embedding.tolist()
                else:
                    result["grammatical_embedding"] = self.grammatical_embedding
            else:
                result["grammatical_embedding"] = None

        return result

    # =========================================================================
    # Pipeline Methods
    # =========================================================================

    def clone(self) -> 'EnrichedAST':
        """
        Create a deep copy for immutable pipeline stages.

        Each stage should clone before modifying to preserve the original.
        """
        return copy.deepcopy(self)

    def with_stage1(
        self,
        sentence_embedding: Any,
        content_embeddings: Optional[Dict[str, Any]] = None,
        known_roots: Optional[Set[str]] = None,
        unknown_roots: Optional[Set[str]] = None
    ) -> 'EnrichedAST':
        """
        Create a new EnrichedAST with Stage 1 enrichments added.

        This is the preferred way to add stage enrichments - it creates
        a new instance rather than mutating in place.
        """
        result = self.clone()
        result.sentence_embedding = sentence_embedding
        if content_embeddings is not None:
            result.content_embeddings = content_embeddings
        if known_roots is not None:
            result.known_roots = known_roots
        if unknown_roots is not None:
            result.unknown_roots = unknown_roots
        result.stages_applied.add("stage1")
        return result

    def with_stage2(
        self,
        grammatical_embedding: Any,
        negation_effect: Optional[Any] = None,
        tense_effect: Optional[Any] = None,
        mood_effect: Optional[Any] = None
    ) -> 'EnrichedAST':
        """
        Create a new EnrichedAST with Stage 2 enrichments added.
        """
        result = self.clone()
        result.grammatical_embedding = grammatical_embedding
        result.negation_effect = negation_effect
        result.tense_effect = tense_effect
        result.mood_effect = mood_effect
        result.stages_applied.add("stage2")
        return result

    def with_stage3(
        self,
        coreference_links: Optional[Dict[str, str]] = None,
        discourse_relation: Optional[str] = None
    ) -> 'EnrichedAST':
        """
        Create a new EnrichedAST with Stage 3 enrichments added.
        """
        result = self.clone()
        if coreference_links is not None:
            result.coreference_links = coreference_links
        if discourse_relation is not None:
            result.discourse_relation = discourse_relation
        result.stages_applied.add("stage3")
        return result

    # =========================================================================
    # Parser AST Accessors
    # =========================================================================

    @property
    def subjekto(self) -> Optional[Dict[str, Any]]:
        """Get the subject word group from parser AST."""
        return self.parser_ast.get("subjekto")

    @property
    def verbo(self) -> Optional[Dict[str, Any]]:
        """Get the verb from parser AST."""
        return self.parser_ast.get("verbo")

    @property
    def objekto(self) -> Optional[Dict[str, Any]]:
        """Get the object word group from parser AST."""
        return self.parser_ast.get("objekto")

    @property
    def aliaj(self) -> List[Dict[str, Any]]:
        """Get the other parts from parser AST."""
        return self.parser_ast.get("aliaj", [])

    @property
    def tempo(self) -> Optional[str]:
        """Get tense from verb if present."""
        verbo = self.verbo
        if verbo:
            return verbo.get("tempo")
        return None

    @property
    def modo(self) -> Optional[str]:
        """Get mood from verb if present."""
        verbo = self.verbo
        if verbo:
            return verbo.get("modo", "indikativo")
        return None

    @property
    def parse_statistics(self) -> Dict[str, Any]:
        """Get parse statistics from parser AST."""
        return self.parser_ast.get("parse_statistics", {})

    @property
    def demandotipo(self) -> Optional[str]:
        """Get question type (ĉu, ki) if this is a question."""
        return self.parser_ast.get("demandotipo")

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_all_words(self) -> List[Dict[str, Any]]:
        """Extract all word ASTs from the parser output."""
        words = []

        # Subject
        if self.subjekto:
            kerno = self.subjekto.get("kerno")
            if kerno:
                words.append(kerno)
            for adj in self.subjekto.get("priskriboj", []):
                words.append(adj)

        # Verb
        if self.verbo:
            words.append(self.verbo)

        # Object
        if self.objekto:
            kerno = self.objekto.get("kerno")
            if kerno:
                words.append(kerno)
            for adj in self.objekto.get("priskriboj", []):
                words.append(adj)

        # Other parts
        words.extend(self.aliaj)

        return words

    def get_content_words(self) -> List[Dict[str, Any]]:
        """
        Get only content words (excluding function words).

        Content words are those that should have learned embeddings:
        - substantivo (nouns)
        - verbo (verbs)
        - adjektivo (adjectives)
        - adverbo (content adverbs)
        """
        content_types = {"substantivo", "verbo", "adjektivo", "adverbo"}
        return [w for w in self.get_all_words()
                if w.get("vortspeco") in content_types]

    def has_stage(self, stage: str) -> bool:
        """Check if a stage has been applied."""
        return stage in self.stages_applied

    def get_effective_embedding(self) -> Optional[Any]:
        """
        Get the most refined embedding available.

        Returns grammatical_embedding if Stage 2 applied,
        otherwise sentence_embedding if Stage 1 applied,
        otherwise None.
        """
        if self.grammatical_embedding is not None:
            return self.grammatical_embedding
        return self.sentence_embedding

    # =========================================================================
    # Display
    # =========================================================================

    def __repr__(self) -> str:
        """Show which stages have been applied."""
        stages = sorted(self.stages_applied) if self.stages_applied else ["none"]
        emb_status = "with_embedding" if self.sentence_embedding is not None else "no_embedding"
        text_preview = self.original_text[:30] + "..." if len(self.original_text) > 30 else self.original_text
        return (
            f"EnrichedAST("
            f"stages={stages}, "
            f"{emb_status}, "
            f"fraztipo='{self.fraztipo}', "
            f"negita={self.negita}, "
            f"text='{text_preview}')"
        )
