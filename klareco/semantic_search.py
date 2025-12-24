"""
Semantic search using signature-based retrieval.

Searches for sentences by semantic roles (agent, action, patient)
rather than just keyword matching.

Example:
    Query: "Kiu vidas la katon?" (Who sees the cat?)
    Signature: (None, 'vid', 'kat')  # wildcard agent, 'vid' action, 'kat' patient

    Matches sentences where cat is the PATIENT (being seen),
    not where cat is the AGENT (doing the seeing).
"""

import json
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

from klareco.semantic_signatures import (
    extract_signature,
    signature_to_string,
    signature_from_string,
    match_signature,
)


class SemanticIndex:
    """
    Index for semantic signature-based search.

    Loads pre-built index from disk and provides search functionality.
    """

    def __init__(self, index_dir: Path):
        """
        Load semantic index from directory.

        Args:
            index_dir: Directory containing signatures.json and metadata.jsonl
        """
        self.index_dir = Path(index_dir)
        self.signatures: Dict[str, List[int]] = {}
        self.metadata: Dict[int, dict] = {}

        self._load_index()

    def _load_index(self):
        """Load index files from disk."""
        signatures_file = self.index_dir / "signatures.json"
        metadata_file = self.index_dir / "metadata.jsonl"

        if not signatures_file.exists():
            raise FileNotFoundError(f"Signatures file not found: {signatures_file}")

        # Load signatures
        with open(signatures_file, 'r', encoding='utf-8') as f:
            self.signatures = json.load(f)

        # Load metadata (streaming to save memory for large corpora)
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                for line in f:
                    entry = json.loads(line)
                    self.metadata[entry['id']] = entry

    def search(
        self,
        query_signature: Tuple[Optional[str], Optional[str], Optional[str]],
        k: int = 10,
        min_score: float = 0.0,
    ) -> List[dict]:
        """
        Search by semantic signature.

        Args:
            query_signature: (agent, action, patient) tuple
                             Use None for wildcards
            k: Maximum number of results
            min_score: Minimum match score (0.0 to 1.0)

        Returns:
            List of matching results with scores
        """
        matches = []

        for sig_str, sentence_ids in self.signatures.items():
            candidate_sig = signature_from_string(sig_str)
            score = match_signature(query_signature, candidate_sig)

            if score > min_score:
                for sid in sentence_ids:
                    meta = self.metadata.get(sid, {})
                    matches.append({
                        'sentence_id': sid,
                        'signature': sig_str,
                        'score': score,
                        'text': meta.get('text', ''),
                        'source': meta.get('source', ''),
                        'agent': meta.get('agent'),
                        'action': meta.get('action'),
                        'patient': meta.get('patient'),
                    })

        # Sort by score descending, then by sentence_id for determinism
        matches.sort(key=lambda x: (-x['score'], x['sentence_id']))

        return matches[:k]

    def search_by_role(
        self,
        agent: Optional[str] = None,
        action: Optional[str] = None,
        patient: Optional[str] = None,
        k: int = 10,
    ) -> List[dict]:
        """
        Convenience method to search by named roles.

        Args:
            agent: Agent/subject root (or None for wildcard)
            action: Action/verb root (or None for wildcard)
            patient: Patient/object root (or None for wildcard)
            k: Maximum number of results

        Returns:
            List of matching results
        """
        return self.search((agent, action, patient), k=k)

    def search_from_ast(
        self,
        query_ast: dict,
        k: int = 10,
    ) -> List[dict]:
        """
        Search using signature extracted from a query AST.

        Args:
            query_ast: Parsed query AST
            k: Maximum number of results

        Returns:
            List of matching results
        """
        sig = extract_signature(query_ast)
        return self.search(sig, k=k)

    def get_sentence(self, sentence_id: int) -> Optional[dict]:
        """
        Get full metadata for a sentence by ID.

        Args:
            sentence_id: Sentence ID

        Returns:
            Metadata dict or None if not found
        """
        return self.metadata.get(sentence_id)

    def stats(self) -> dict:
        """Get index statistics."""
        return {
            'unique_signatures': len(self.signatures),
            'total_sentences': len(self.metadata),
            'avg_sentences_per_sig': (
                len(self.metadata) / len(self.signatures)
                if self.signatures else 0
            ),
        }


def search_semantic(
    query_sig: Tuple[Optional[str], Optional[str], Optional[str]],
    index: SemanticIndex,
    k: int = 10,
) -> List[dict]:
    """
    Convenience function for semantic search.

    Args:
        query_sig: (agent, action, patient) tuple
        index: Loaded SemanticIndex
        k: Number of results

    Returns:
        List of matching results
    """
    return index.search(query_sig, k=k)


# Default index instance (lazy loaded)
_default_index: Optional[SemanticIndex] = None


def get_semantic_index(index_dir: Optional[Path] = None) -> SemanticIndex:
    """
    Get the default semantic index (singleton).

    Args:
        index_dir: Optional custom index directory

    Returns:
        SemanticIndex instance
    """
    global _default_index

    if index_dir:
        return SemanticIndex(index_dir)

    if _default_index is None:
        default_path = Path(__file__).parent.parent / "data" / "semantic_index"
        if default_path.exists():
            _default_index = SemanticIndex(default_path)
        else:
            raise FileNotFoundError(
                f"Default semantic index not found at {default_path}. "
                "Run scripts/build_semantic_index.py first."
            )

    return _default_index


def reset_index():
    """Reset the singleton index (mainly for testing)."""
    global _default_index
    _default_index = None
