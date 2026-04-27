"""
Immutable state dataclasses for the Klareco orchestration pipeline.

Immutability contract
---------------------
- QueryContext, SymbolicLayer, LatentLayer are frozen dataclasses.
  Their fields cannot be reassigned after construction.
- Numpy arrays placed in LatentLayer are locked writeable=False by
  QueryContext.apply(), preventing in-place mutation.
- flags is a MappingProxyType: a read-only view of a dict.
- ContextDelta is NOT frozen — it is a transient builder consumed
  immediately by the orchestrator and never stored after apply().

Copy-on-write semantics
-----------------------
dataclasses.replace() performs a shallow copy: unchanged fields share
the same object references.  Stages that do not touch passage_asts, for
example, incur zero cost for those arrays.  Only the fields named in the
delta are "written" (replaced with new objects).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Optional

try:
    import numpy as np
    _NUMPY = True
except ImportError:
    _NUMPY = False


# ---------------------------------------------------------------------------
# Leaf types — items stored in SymbolicLayer collections
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParsedPassage:
    """A retrieved corpus sentence together with its pre-built AST from Kuzu."""
    sentence_id: str
    text: str
    ast: Optional[dict]
    score: float
    source_doc: str
    source_type: str


@dataclass(frozen=True)
class FactFragment:
    """
    A semantic fact extracted from a passage, encoded as an AST-native triple.

    arguments is a tuple of (role, value) pairs rather than a dict so the
    object remains hashable and structurally immutable.
    """
    relation: str
    entity: str
    arguments: tuple           # ((role, value), …)
    confidence: float
    source_passage_id: str
    ast_node: Optional[dict] = None


@dataclass(frozen=True)
class Segment:
    """One sentence of the generated answer, with its citation ids."""
    text: str
    citation_ids: tuple = ()   # ("1", "2", …)


@dataclass(frozen=True)
class CitationRecord:
    """Stable citation pointer attached to a source sentence."""
    id: str                    # "1", "2", … — displayed inline as [1]
    sentence_id: str
    snippet: str
    doc_title: str
    doc_source: str


# ---------------------------------------------------------------------------
# Dual-layer state
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SymbolicLayer:
    """
    Everything expressible as Esperanto AST or AST-derived structure.
    All sequences are tuples for structural immutability.
    """
    question_ast: Optional[dict] = None
    question_type: str = 'nekonata'    # 'kiu'|'kio'|'kie'|'kiam'|'kiel'|'kial'|'nekonata'
    passage_asts: tuple = ()          # tuple[ParsedPassage]
    fact_fragments: tuple = ()        # tuple[FactFragment]
    answer_segments: tuple = ()       # tuple[Segment]
    citations: tuple = ()             # tuple[CitationRecord]
    final_text: str = ''


@dataclass(frozen=True)
class LatentLayer:
    """
    Dense matrix representations with no clean Esperanto AST encoding.
    Arrays stored here are locked writeable=False by QueryContext.apply().
    """
    question_embedding: Optional[Any] = None   # np.ndarray | None
    passage_embeddings: tuple = ()             # tuple[np.ndarray]
    relevance_matrix: Optional[Any] = None     # np.ndarray | None
    stage_attention: tuple = ()               # tuple[(stage_name, np.ndarray)]


# ---------------------------------------------------------------------------
# Stage performance measurement (defined here to avoid circular imports)
# ---------------------------------------------------------------------------

@dataclass
class StageMetrics:
    """Per-stage performance measurements written into every trace entry."""
    stage_name: str
    timing_ms: float              # set by orchestrator after the stage returns
    confidence_before: float
    confidence_after: float
    symbolic_coverage: float      # fraction of stage input that has an AST
    stage_specific: dict = field(default_factory=dict)

    @property
    def confidence_delta(self) -> float:
        return self.confidence_after - self.confidence_before


# ---------------------------------------------------------------------------
# Delta — what a stage produces (NOT frozen; transient builder)
# ---------------------------------------------------------------------------

@dataclass
class ContextDelta:
    """
    Output of a pipeline stage: field updates to apply to the current context.

    symbolic maps SymbolicLayer field names → new values.
    latent   maps LatentLayer  field names → new values.

    Keys must match fields on their respective layer; the orchestrator
    validates this in debug mode.  metrics.timing_ms is filled in by the
    orchestrator (stages set it to 0.0 as a placeholder).
    """
    symbolic: dict = field(default_factory=dict)
    latent: dict = field(default_factory=dict)
    metrics: Optional[StageMetrics] = None
    flags: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# The context itself
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QueryContext:
    """
    Immutable snapshot of pipeline state at a stage boundary.

    Stages receive a QueryContext and return a ContextDelta.  The
    orchestrator calls apply(delta) to produce the next snapshot, storing
    both ctx_before and the delta in the trace.  Every intermediate state
    is therefore reachable without storing redundant copies.
    """
    question: str
    symbolic: SymbolicLayer = field(default_factory=SymbolicLayer)
    latent: LatentLayer = field(default_factory=LatentLayer)
    confidence: float = 0.0
    flags: MappingProxyType = field(
        default_factory=lambda: MappingProxyType({})
    )

    def apply(self, delta: ContextDelta) -> QueryContext:
        """
        Return a new QueryContext with delta applied.

        Unchanged layers share their object references from self.
        New numpy arrays in delta.latent are locked writeable=False.
        """
        new_symbolic = (
            dataclasses.replace(self.symbolic, **delta.symbolic)
            if delta.symbolic else self.symbolic
        )

        new_latent = self.latent
        if delta.latent:
            _lock_arrays(delta.latent)
            new_latent = dataclasses.replace(self.latent, **delta.latent)

        new_confidence = (
            delta.metrics.confidence_after
            if (delta.metrics and delta.metrics.confidence_after is not None)
            else self.confidence
        )

        new_flags = MappingProxyType({**self.flags, **delta.flags})

        return dataclasses.replace(
            self,
            symbolic=new_symbolic,
            latent=new_latent,
            confidence=new_confidence,
            flags=new_flags,
        )

    def flag(self, key: str, default: Any = None) -> Any:
        """Read a pipeline flag; returns default if not set."""
        return self.flags.get(key, default)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lock_arrays(latent_updates: dict) -> None:
    """Mark all numpy arrays in a latent delta dict as non-writeable."""
    if not _NUMPY:
        return
    for val in latent_updates.values():
        if isinstance(val, np.ndarray):
            val.flags.writeable = False
        elif isinstance(val, tuple):
            for item in val:
                if isinstance(item, np.ndarray):
                    item.flags.writeable = False
