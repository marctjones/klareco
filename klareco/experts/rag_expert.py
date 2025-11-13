"""
RAG (Retrieval-Augmented Generation) Expert - Handles factoid queries via semantic search.

This expert uses the RAG retriever to find relevant information from the indexed corpus
and present it as context for answering factoid questions.

Examples:
- "Kio estas Esperanto?" → Retrieves relevant sentences about Esperanto
- "Kiu verkis Alice's Adventures in Wonderland?" → Finds information about the author
- "Kiam estis kreita Esperanto?" → Retrieves historical information

Currently returns raw retrieved sentences. In Phase 5, will incorporate neural decoder
for generating natural language answers from retrieved context.
"""

from typing import Dict, Any, List, Optional
import logging

from .base import Expert
from klareco.rag.retriever import KlarecoRetriever


logger = logging.getLogger(__name__)


class RAGExpert(Expert):
    """
    Expert for handling factoid queries using RAG retrieval.

    Uses semantic search over the indexed corpus to find relevant information.
    """

    # Question words that typically indicate factoid queries
    FACTOID_QUESTION_WORDS = {
        'kiu',     # who/which
        'kio',     # what
        'kie',     # where
        'kiam',    # when
        'kial',    # why
        'kiel',    # how
    }

    def __init__(
        self,
        retriever: KlarecoRetriever,
        k: int = 5,
        min_score_threshold: float = 0.5
    ):
        """
        Initialize RAG Expert.

        Args:
            retriever: Initialized KlarecoRetriever instance
            k: Number of results to retrieve per query
            min_score_threshold: Minimum similarity score to include result
        """
        super().__init__("RAG Expert")
        self.retriever = retriever
        self.k = k
        self.min_score_threshold = min_score_threshold

    def can_handle(self, ast: Dict[str, Any]) -> bool:
        """
        Check if this is a factoid question that RAG can handle.

        Looks for:
        - Question words (kiu, kio, kie, kiam, kial, kiel)
        - Question structure with interrogative pronouns

        Args:
            ast: Parsed query AST

        Returns:
            True if this appears to be a factoid question
        """
        if not ast or ast.get('tipo') != 'frazo':
            return False

        # Extract all words from AST
        words = self._extract_all_words(ast)
        words_lower = [w.lower() for w in words]

        # Check for factoid question words
        has_question_word = any(
            qword in word
            for word in words_lower
            for qword in self.FACTOID_QUESTION_WORDS
        )

        return has_question_word

    def estimate_confidence(self, ast: Dict[str, Any]) -> float:
        """
        Estimate confidence in handling this query.

        High confidence if:
        - Clear factoid question word present
        - Query is well-formed and parseable

        Args:
            ast: Parsed query AST

        Returns:
            Confidence score 0.0-1.0
        """
        if not self.can_handle(ast):
            return 0.0

        words = self._extract_all_words(ast)
        words_lower = [w.lower() for w in words]

        # Check for word presence (order-independent)
        has_kio = 'kio' in words_lower
        has_kiu = 'kiu' in words_lower
        has_estas = 'estas' in words_lower
        has_estis = 'estis' in words_lower
        has_kie = 'kie' in words_lower
        has_kiam = 'kiam' in words_lower
        has_kial = 'kial' in words_lower
        has_kiel = 'kiel' in words_lower

        # High confidence for direct information queries
        if has_kio and has_estas:  # "what is"
            return 0.95

        if has_kiu and (has_estas or has_estis):  # "who is/was"
            return 0.95

        if has_kie:  # "where"
            return 0.90

        if has_kiam:  # "when"
            return 0.90

        if has_kial or has_kiel:  # "why/how"
            return 0.85

        # Moderate confidence for other question words
        has_question_word = any(
            qword in words_lower for qword in self.FACTOID_QUESTION_WORDS
        )

        if has_question_word:
            return 0.75

        return 0.5

    def execute(self, ast: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute factoid query via RAG retrieval.

        Args:
            ast: Parsed query AST

        Returns:
            Response with retrieved information
        """
        # Validate AST
        if not ast or ast.get('tipo') != 'frazo':
            return {
                'answer': 'Mi ne povas procezi malplenan aŭ malvalidan demandon.',
                'confidence': 0.0,
                'expert': self.name,
                'error': 'Invalid or empty AST'
            }

        try:
            # Retrieve relevant sentences using hybrid approach
            # (keyword filter + semantic rerank for better entity matching)
            results = self.retriever.retrieve_hybrid(
                ast,
                k=self.k,
                return_scores=True
            )

            # Filter by score threshold
            filtered_results = [
                r for r in results
                if r.get('score', 0.0) >= self.min_score_threshold
            ]

            if not filtered_results:
                return {
                    'answer': 'Mi ne trovis rilatan informon en la korpuso.',
                    'confidence': 0.0,
                    'expert': self.name,
                    'retrieved_count': 0
                }

            # Format answer with retrieved sentences
            answer = self._format_answer(filtered_results)

            # Compute confidence from retrieval scores
            confidence = self._compute_answer_confidence(filtered_results)

            return {
                'answer': answer,
                'confidence': confidence,
                'expert': self.name,
                'sources': filtered_results,  # Pass through all metadata (text, source_name, line, etc.)
                'retrieved_count': len(filtered_results),
                'explanation': f'Trovis {len(filtered_results)} rilatan frazon en la korpuso'
            }

        except Exception as e:
            logger.error(f"RAG retrieval failed: {e}", exc_info=True)
            return {
                'answer': f'Eraro dum serĉo en la korpuso: {str(e)}',
                'confidence': 0.0,
                'expert': self.name,
                'error': str(e)
            }

    def _extract_all_words(self, ast: Dict[str, Any]) -> List[str]:
        """Extract all words from AST recursively."""
        words = []

        if isinstance(ast, dict):
            if ast.get('tipo') == 'vorto':
                word = ast.get('plena_vorto', '') or ast.get('radiko', '')
                if word:
                    words.append(word)

            # Recursively extract from all fields
            for value in ast.values():
                if isinstance(value, (dict, list)):
                    words.extend(self._extract_all_words(value))

        elif isinstance(ast, list):
            for item in ast:
                words.extend(self._extract_all_words(item))

        return words

    def _format_answer(self, results: List[Dict[str, Any]]) -> str:
        """
        Format retrieved results into an answer.

        Currently returns a simple list of retrieved sentences.
        In Phase 5, this will incorporate a neural decoder to generate
        natural language answers from the retrieved context.

        Args:
            results: List of retrieval results

        Returns:
            Formatted answer string
        """
        if not results:
            return "Mi ne trovis rilatan informon."

        # Phase 4: Simple concatenation of top results
        # Phase 5: Neural generation from retrieved context
        answer_parts = [
            "Jen kion mi trovis:",  # "Here's what I found:"
            ""
        ]

        for i, result in enumerate(results[:3], 1):  # Show top 3
            score_indicator = "★" * min(5, int(result['score']))
            answer_parts.append(f"{i}. {result['text']} {score_indicator}")

        if len(results) > 3:
            answer_parts.append(f"\n(kaj {len(results) - 3} plia{'j' if len(results) - 3 > 1 else ''} rezulto{'j' if len(results) - 3 > 1 else ''})")

        return "\n".join(answer_parts)

    def _compute_answer_confidence(self, results: List[Dict[str, Any]]) -> float:
        """
        Compute answer confidence from retrieval scores.

        Args:
            results: List of retrieval results with scores

        Returns:
            Confidence score 0.0-1.0
        """
        if not results:
            return 0.0

        # Use top result's score as primary confidence indicator
        top_score = results[0].get('score', 0.0)

        # Normalize score to 0-1 range (assuming scores typically in [0, 2] range)
        # Tree-LSTM embeddings with cosine similarity can exceed 1.0
        normalized_score = min(1.0, top_score / 2.0)

        # Boost confidence if multiple high-scoring results
        if len(results) >= 3:
            avg_top3_score = sum(r.get('score', 0.0) for r in results[:3]) / 3
            normalized_avg = min(1.0, avg_top3_score / 2.0)
            # Blend top score and average
            normalized_score = 0.7 * normalized_score + 0.3 * normalized_avg

        return max(0.0, min(1.0, normalized_score))


def create_rag_expert(
    index_dir: str = "data/corpus_index",
    model_path: str = "models/tree_lstm/checkpoint_epoch_12.pt",
    k: int = 5,
    min_score_threshold: float = 0.5,
    device: str = 'cpu'
) -> RAGExpert:
    """
    Convenience function to create RAG expert with default configuration.

    Args:
        index_dir: Directory with indexed corpus
        model_path: Path to encoder checkpoint
        k: Number of results to retrieve
        min_score_threshold: Minimum similarity score
        device: 'cpu' or 'cuda'

    Returns:
        Initialized RAGExpert
    """
    retriever = KlarecoRetriever(
        index_dir=index_dir,
        model_path=model_path,
        mode='tree_lstm',
        device=device
    )

    return RAGExpert(
        retriever=retriever,
        k=k,
        min_score_threshold=min_score_threshold
    )


# Export
__all__ = ['RAGExpert', 'create_rag_expert']
