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
from typing import Dict, List, Optional, Any, Set, Tuple
import copy
import json

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# =============================================================================
# Annotation Metadata
# =============================================================================

@dataclass
class AnnotationMetadata:
    """
    Metadata for every model annotation.

    Enables version tracking, reproducibility, and debugging.
    """
    model_name: str              # "selectional_preference_v1"
    model_version: str           # "1.0.0"
    confidence: float            # 0.0-1.0
    compute_time_ms: float       # How long did inference take?
    timestamp: str               # When was this annotated?
    parameters: Dict[str, Any] = field(default_factory=dict)  # Model hyperparams


# =============================================================================
# M1: Selectional Preference Annotations
# =============================================================================

@dataclass
class SelectionalAnnotation:
    """
    M1: Selectional Preference Model (10M params)

    Learns (subject, verb, object) compatibility from corpus statistics.
    Enables filtering implausible answers like "libroj manĝas viandon".

    Memory: ~200 bytes per annotation
    """
    # Core compatibility scores (0.0-1.0)
    subject_verb_score: float           # P(subj|verb) compatibility
    verb_object_score: float            # P(obj|verb) compatibility
    triple_plausibility: float          # P(subj, verb, obj) joint plausibility

    # Detailed breakdown (optional, for explainability)
    subject_verb_alternatives: List[Tuple[str, float]] = field(default_factory=list)
    verb_object_alternatives: List[Tuple[str, float]] = field(default_factory=list)

    # Metadata
    meta: Optional[AnnotationMetadata] = None


# =============================================================================
# M2: Taxonomic Relations Annotations
# =============================================================================

@dataclass
class TaxonomicAnnotation:
    """
    M2: Taxonomic Relations Model (10M params)

    Pure IS-A relationships (no co-occurrence noise).
    Enables query expansion with semantically related concepts.

    Memory: ~500 bytes per annotation
    """
    # Hierarchical relationships per content word
    # Key: root, Value: List of (parent/child, similarity)
    hypernyms: Dict[str, List[Tuple[str, float]]] = field(default_factory=dict)  # hund → [(mamul, 0.95), (best, 0.92)]
    hyponyms: Dict[str, List[Tuple[str, float]]] = field(default_factory=dict)   # best → [(hund, 0.92), (kat, 0.91)]

    # Cluster membership (for semantic grouping)
    concept_clusters: Dict[str, str] = field(default_factory=dict)  # hund → "animals.mammals"

    # Similarity to query (if used for retrieval)
    taxonomic_similarity: Optional[float] = None

    # Metadata
    meta: Optional[AnnotationMetadata] = None


# =============================================================================
# M3: Discourse Coherence Annotations
# =============================================================================

@dataclass
class DiscourseAnnotation:
    """
    M3: Discourse Coherence Model (30-50M params)

    Sentence-level coherence for passage ranking.
    Learns which sentences naturally follow each other.

    Memory: ~300 bytes (without contextual embedding), ~2KB (with)
    """
    # Sentence-level coherence (0.0-1.0)
    coherence_with_previous: Optional[float] = None  # Coherence with prev sentence
    coherence_with_next: Optional[float] = None      # Coherence with next sentence

    # Coreference resolution
    coreferences: Dict[str, str] = field(default_factory=dict)  # pronoun_id -> antecedent

    # Discourse relations
    discourse_relation: Optional[str] = None  # "elaboration", "contrast", "cause", etc.
    relation_confidence: Optional[float] = None

    # Thematic role labels (richer than M1)
    thematic_roles: Dict[str, str] = field(default_factory=dict)  # word_id -> "AGENT", "PATIENT", etc.

    # Contextual embedding (heavier, optional - only computed when needed)
    discourse_embedding: Optional[Any] = None  # 256d contextual vector

    # Metadata
    meta: Optional[AnnotationMetadata] = None


# =============================================================================
# M4: Multi-Model Orchestration Annotations
# =============================================================================

@dataclass
class MultiModelAnnotation:
    """
    M4: Multi-Model Orchestration (0 params, coordination only)

    Combines outputs from M1-M3 models.
    Provides explainable ranking with model contributions.

    Memory: ~300 bytes
    """
    # Combined scores from all models
    model_scores: Dict[str, float] = field(default_factory=dict)  # model_name -> score
    combined_score: float = 0.0

    # Ranking explanation
    score_breakdown: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Which models contributed to final decision
    active_models: Set[str] = field(default_factory=set)

    # Final decision trace
    decision_trace: List[str] = field(default_factory=list)

    # Metadata
    meta: Optional[AnnotationMetadata] = None


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
    # M1: Selectional Preference Model (10M params)
    # =========================================================================

    selectional: Optional[SelectionalAnnotation] = None

    # =========================================================================
    # M2: Taxonomic Relations Model (10M params)
    # =========================================================================

    taxonomic: Optional[TaxonomicAnnotation] = None

    # =========================================================================
    # M3: Discourse Coherence Model (30-50M params)
    # =========================================================================

    discourse: Optional[DiscourseAnnotation] = None

    # =========================================================================
    # M4: Multi-Model Orchestration
    # =========================================================================

    multi_model: Optional[MultiModelAnnotation] = None

    # =========================================================================
    # Generic Extension Mechanism
    # =========================================================================

    # For future models or experimental annotations
    extensions: Dict[str, Any] = field(default_factory=dict)

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

        # M1: Selectional Preference
        selectional = None
        if "selectional" in data:
            sel = data["selectional"]
            selectional = SelectionalAnnotation(
                subject_verb_score=sel["subject_verb_score"],
                verb_object_score=sel["verb_object_score"],
                triple_plausibility=sel["triple_plausibility"],
                subject_verb_alternatives=sel.get("subject_verb_alternatives", []),
                verb_object_alternatives=sel.get("verb_object_alternatives", []),
            )

        # M2: Taxonomic Relations
        taxonomic = None
        if "taxonomic" in data:
            tax = data["taxonomic"]
            taxonomic = TaxonomicAnnotation(
                hypernyms={k: [tuple(v) for v in vals] for k, vals in tax["hypernyms"].items()},
                hyponyms={k: [tuple(v) for v in vals] for k, vals in tax["hyponyms"].items()},
                concept_clusters=tax["concept_clusters"],
                taxonomic_similarity=tax.get("taxonomic_similarity"),
            )

        # M3: Discourse Coherence
        discourse = None
        if "discourse" in data:
            disc = data["discourse"]
            discourse = DiscourseAnnotation(
                coherence_with_previous=disc.get("coherence_with_previous"),
                coherence_with_next=disc.get("coherence_with_next"),
                coreferences=disc.get("coreferences", {}),
                discourse_relation=disc.get("discourse_relation"),
                relation_confidence=disc.get("relation_confidence"),
                thematic_roles=disc.get("thematic_roles", {}),
            )

        # M4: Multi-Model Orchestration
        multi_model = None
        if "multi_model" in data:
            mm = data["multi_model"]
            multi_model = MultiModelAnnotation(
                model_scores=mm["model_scores"],
                combined_score=mm["combined_score"],
                score_breakdown=mm.get("score_breakdown", {}),
                active_models=set(mm.get("active_models", [])),
                decision_trace=mm.get("decision_trace", []),
            )

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
            selectional=selectional,
            taxonomic=taxonomic,
            discourse=discourse,
            multi_model=multi_model,
            extensions=data.get("extensions", {}),
            stages_applied=set(data.get("stages_applied", [])),
            source_id=data.get("source_id"),
            tier=data.get("tier"),
        )

    @classmethod
    def from_json(cls, json_str: str) -> 'EnrichedAST':
        """
        Deserialize from JSON string.

        Args:
            json_str: JSON string from to_json()

        Returns:
            EnrichedAST with all annotations restored
        """
        return cls.from_dict(json.loads(json_str))

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(self, include_embeddings: bool = True, include: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Serialize to a dict suitable for JSON storage.

        Args:
            include_embeddings: If True, include tensor data as lists.
                               If False, omit embeddings (for metadata-only storage).
            include: Optional list of field names to include (default: all).
                    Useful for selective serialization in training data.
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

        # M1-M4 Annotations (always included if present)
        if self.selectional:
            result["selectional"] = {
                "subject_verb_score": self.selectional.subject_verb_score,
                "verb_object_score": self.selectional.verb_object_score,
                "triple_plausibility": self.selectional.triple_plausibility,
                "subject_verb_alternatives": self.selectional.subject_verb_alternatives,
                "verb_object_alternatives": self.selectional.verb_object_alternatives,
            }

        if self.taxonomic:
            result["taxonomic"] = {
                "hypernyms": {k: list(v) for k, v in self.taxonomic.hypernyms.items()},
                "hyponyms": {k: list(v) for k, v in self.taxonomic.hyponyms.items()},
                "concept_clusters": self.taxonomic.concept_clusters,
                "taxonomic_similarity": self.taxonomic.taxonomic_similarity,
            }

        if self.discourse:
            result["discourse"] = {
                "coherence_with_previous": self.discourse.coherence_with_previous,
                "coherence_with_next": self.discourse.coherence_with_next,
                "coreferences": self.discourse.coreferences,
                "discourse_relation": self.discourse.discourse_relation,
                "relation_confidence": self.discourse.relation_confidence,
                "thematic_roles": self.discourse.thematic_roles,
            }

        if self.multi_model:
            result["multi_model"] = {
                "model_scores": self.multi_model.model_scores,
                "combined_score": self.multi_model.combined_score,
                "score_breakdown": self.multi_model.score_breakdown,
                "active_models": list(self.multi_model.active_models),
                "decision_trace": self.multi_model.decision_trace,
            }

        result["extensions"] = self.extensions

        # Filter by include list if provided
        if include is not None:
            result = {k: v for k, v in result.items() if k in include}

        return result

    def to_json(self, **kwargs) -> str:
        """
        Serialize to JSON string.

        Args:
            **kwargs: Passed to json.dumps() (e.g., indent=2 for pretty printing)
        """
        return json.dumps(self.to_dict(), ensure_ascii=False, **kwargs)

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

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def memory_footprint(self) -> Dict[str, int]:
        """
        Calculate memory usage by component in bytes.

        Returns:
            Dict mapping component name to bytes used
        """
        footprint = {}

        # Parser AST (roughly estimate as length of JSON string)
        footprint['parser_ast'] = len(str(self.parser_ast).encode())

        # Stage 1: Sentence embedding
        if self.sentence_embedding is not None:
            if TORCH_AVAILABLE and hasattr(self.sentence_embedding, 'numel'):
                footprint['sentence_embedding'] = self.sentence_embedding.numel() * 4  # 4 bytes per float32
            else:
                footprint['sentence_embedding'] = 256  # Estimate

        # M1: Selectional Preference
        footprint['selectional'] = 200 if self.selectional else 0

        # M2: Taxonomic Relations
        if self.taxonomic:
            # Estimate based on number of entries
            hyper_size = sum(len(v) for v in self.taxonomic.hypernyms.values()) * 20
            hypo_size = sum(len(v) for v in self.taxonomic.hyponyms.values()) * 20
            cluster_size = len(self.taxonomic.concept_clusters) * 50
            footprint['taxonomic'] = min(hyper_size + hypo_size + cluster_size, 500)
        else:
            footprint['taxonomic'] = 0

        # M3: Discourse (300 bytes default, 2KB with embedding)
        if self.discourse is not None:
            discourse_emb_size = 0
            if self.discourse.discourse_embedding is not None:
                if TORCH_AVAILABLE and hasattr(self.discourse.discourse_embedding, 'numel'):
                    discourse_emb_size = self.discourse.discourse_embedding.numel() * 4  # 4 bytes per float
                else:
                    discourse_emb_size = 1024  # Estimate
            footprint['discourse'] = 300 + discourse_emb_size
        else:
            footprint['discourse'] = 0

        # M4: Multi-Model Orchestration
        footprint['multi_model'] = 300 if self.multi_model else 0

        # Extensions
        footprint['extensions'] = len(str(self.extensions).encode('utf-8'))

        # Total
        footprint['total'] = sum(v for k, v in footprint.items() if k != 'total')

        return footprint

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
