"""
Factoid QA Expert - Neural decoder for factual questions

This expert uses RAG (Retrieval-Augmented Generation) to answer factual questions:
1. Retrieve relevant documents from corpus using GNN-based semantic search
2. Use LLM to generate answer from retrieved context

Part of Phase 5: Planning & Advanced Experts
"""

from typing import Dict, Any, Optional, List
import logging
from pathlib import Path
from ..llm_provider import get_llm_provider

logger = logging.getLogger(__name__)


class FactoidQAExpert:
    """
    Expert for answering factual questions using RAG.

    Pipeline:
    1. Encode query AST using GNN (symbolic structure → semantic embedding)
    2. Retrieve relevant documents from corpus
    3. Use LLM to generate answer from context
    """

    def __init__(self, llm_provider=None, rag_system=None, corpus_path=None):
        """
        Initialize Factoid QA Expert.

        Args:
            llm_provider: Optional LLM provider (auto-detected if None)
            rag_system: Optional RAG system (created if None)
            corpus_path: Optional path to corpus (uses default if None)
        """
        self.name = "Factoid_QA_Expert"
        self.capabilities = ["factual_questions", "knowledge_retrieval", "rag"]
        self.llm_provider = llm_provider or get_llm_provider()

        # Initialize RAG system
        self.rag_system = rag_system
        if self.rag_system is None:
            self.rag_system = self._initialize_rag(corpus_path)

        logger.info(f"{self.name} initialized")

    def _initialize_rag(self, corpus_path: Optional[str]):
        """
        Initialize RAG system with hybrid retrieval.

        Args:
            corpus_path: Path to corpus file

        Returns:
            Initialized RAG system or None if unavailable
        """
        try:
            from ..rag.hybrid_rag import HybridRAG

            # Use default corpus if not specified
            if corpus_path is None:
                corpus_path = Path(__file__).parent.parent.parent / "data" / "esperanto_corpus.jsonl"

            if not Path(corpus_path).exists():
                logger.warning(f"Corpus not found at {corpus_path}, RAG disabled")
                return None

            rag = HybridRAG(corpus_path=str(corpus_path))
            logger.info(f"RAG system initialized with corpus: {corpus_path}")
            return rag

        except ImportError as e:
            logger.warning(f"RAG dependencies not available: {e}")
            return None
        except Exception as e:
            logger.error(f"Error initializing RAG: {e}")
            return None

    def can_handle(self, ast: Dict[str, Any]) -> bool:
        """
        Check if this expert can handle the AST.

        Factoid questions typically:
        - Start with question words (kio, kiu, kie, kiam, kial, kiel, kiom)
        - Ask about facts, people, places, events
        - Request specific information

        Args:
            ast: Parsed AST

        Returns:
            True if this expert can handle the request
        """
        # Extract all words from AST
        words = self._extract_words(ast)

        # Check for question words
        question_words = {
            'kio',   # what
            'kiu',   # who/which
            'kie',   # where
            'kiam',  # when
            'kial',  # why
            'kiel',  # how
            'kiom',  # how much/many
            'ĉu'     # whether/if
        }

        for word in words:
            radiko = word.get('radiko', '').lower()
            plena_vorto = word.get('plena_vorto', '').lower()

            # Check if it's a question word
            if radiko in question_words or plena_vorto in question_words:
                logger.debug(f"Question word detected: {radiko or plena_vorto}")
                return True

        return False

    def handle(self, ast: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> str:
        """
        Answer factual question using RAG + LLM.

        Args:
            ast: Parsed AST of the question
            context: Optional context

        Returns:
            Generated answer
        """
        logger.info(f"{self.name} handling factual question")

        # Extract question from AST
        question = self._extract_question(ast, context)

        logger.debug(f"Question: {question}")

        # Retrieve relevant documents
        retrieved_docs = self._retrieve_documents(ast, question)

        # Generate answer using LLM
        answer = self._generate_answer(question, retrieved_docs)

        return answer

    def _extract_words(self, ast: Dict[str, Any]) -> list:
        """
        Extract all words from AST recursively.

        Args:
            ast: AST node

        Returns:
            List of word dictionaries
        """
        words = []

        if ast.get('tipo') == 'vorto':
            words.append(ast)
        elif ast.get('tipo') == 'vortgrupo':
            for node in ast.get('vortoj', []):
                words.extend(self._extract_words(node))
        elif ast.get('tipo') == 'frazo':
            for key in ['subjekto', 'verbo', 'objekto']:
                if ast.get(key):
                    words.extend(self._extract_words(ast[key]))
            for node in ast.get('aliaj', []):
                words.extend(self._extract_words(node))

        return words

    def _extract_question(self, ast: Dict[str, Any], context: Optional[Dict[str, Any]]) -> str:
        """
        Extract question text from AST or context.

        Args:
            ast: Parsed AST
            context: Optional context

        Returns:
            Question text
        """
        # Check context first
        if context and 'original_text' in context:
            return context['original_text']

        # Reconstruct from AST
        from ..deparser import deparse
        return deparse(ast)

    def _retrieve_documents(self, ast: Dict[str, Any], question: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents using RAG.

        Args:
            ast: Parsed AST (for GNN encoding)
            question: Question text

        Returns:
            List of retrieved documents with scores
        """
        if self.rag_system is None:
            logger.warning("RAG system not available, no documents retrieved")
            return []

        try:
            # Use hybrid retrieval (BM25 + semantic)
            results = self.rag_system.retrieve(
                query_ast=ast,
                query_text=question,
                top_k=3  # Retrieve top 3 documents
            )

            logger.info(f"Retrieved {len(results)} documents")
            return results

        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []

    def _generate_answer(self, question: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Generate answer using LLM with retrieved context.

        Args:
            question: Question text
            retrieved_docs: Retrieved documents

        Returns:
            Generated answer
        """
        # Build context from retrieved documents
        if retrieved_docs:
            context_parts = []
            for i, doc in enumerate(retrieved_docs, 1):
                text = doc.get('text', '')
                score = doc.get('score', 0.0)
                source = doc.get('source', 'Unknown')
                context_parts.append(f"[Source {i} (relevance: {score:.2f}) - {source}]\n{text}")

            context = "\n\n".join(context_parts)

            system_prompt = (
                "You are a helpful assistant that answers questions based on provided context. "
                "Use the context to provide accurate, factual answers. "
                "If the context doesn't contain enough information, say so honestly. "
                "Cite sources when possible using [Source N] notation."
            )

            user_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

        else:
            # No retrieved documents, answer from general knowledge
            system_prompt = (
                "You are a helpful assistant that answers factual questions. "
                "Provide accurate, concise answers. "
                "If you're uncertain, express your uncertainty."
            )

            user_prompt = question

        logger.debug(f"Generating answer using {self.llm_provider.provider_type.value}")

        try:
            answer = self.llm_provider.generate(
                prompt=user_prompt,
                system=system_prompt,
                max_tokens=300,
                temperature=0.1  # Low temperature for factual accuracy
            )

            logger.info(f"Answer generated successfully ({len(answer)} chars)")
            return answer

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"[Error generating answer: {str(e)}]"


# Factory function for creating expert instance
def create_factoid_qa_expert(llm_provider=None, rag_system=None, corpus_path=None) -> FactoidQAExpert:
    """
    Create and return a Factoid QA Expert instance.

    Args:
        llm_provider: Optional LLM provider (auto-detected if None)
        rag_system: Optional RAG system (created if None)
        corpus_path: Optional path to corpus

    Returns:
        Initialized FactoidQAExpert
    """
    return FactoidQAExpert(llm_provider, rag_system, corpus_path)


if __name__ == "__main__":
    # Test the expert
    import sys
    from ..parser import parse_esperanto

    # Example usage
    expert = create_factoid_qa_expert()

    # Test questions
    test_questions = [
        "Kio estas Esperanto?",  # What is Esperanto?
        "Kiu kreis Esperanton?",  # Who created Esperanto?
        "Kiam estis kreita Esperanto?"  # When was Esperanto created?
    ]

    for question in test_questions:
        print(f"\nQuestion: {question}")
        ast = parse_esperanto(question)
        print(f"Can handle: {expert.can_handle(ast)}")

        if expert.can_handle(ast):
            context = {'original_text': question}
            answer = expert.handle(ast, context)
            print(f"Answer: {answer}")
