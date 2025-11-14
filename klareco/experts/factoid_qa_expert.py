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
from .base import Expert

logger = logging.getLogger(__name__)


class FactoidQAExpert(Expert):
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
        super().__init__(name="Factoid_QA_Expert")
        self.capabilities = ["factual_questions", "knowledge_retrieval", "rag"]
        self.llm_provider = llm_provider or get_llm_provider()

        # Initialize RAG system
        self.rag_system = rag_system
        if self.rag_system is None:
            self.rag_system = self._initialize_rag(corpus_path)

        logger.info(f"{self.name} initialized")

    def _initialize_rag(self, corpus_path: Optional[str]):
        """
        Initialize RAG system with KlarecoRetriever.

        Args:
            corpus_path: Path to corpus directory (not used - uses indexed corpus)

        Returns:
            Initialized KlarecoRetriever or None if unavailable
        """
        try:
            from ..rag.retriever import KlarecoRetriever

            # Use default index directory
            index_dir = Path(__file__).parent.parent.parent / "data" / "corpus_index"

            # Try to find latest model checkpoint
            model_dir = Path(__file__).parent.parent.parent / "models" / "tree_lstm"

            # Try epochs in descending order: 20, 12, 5, 4, 3, 2, 1
            for epoch in [20, 12, 5, 4, 3, 2, 1]:
                model_path = model_dir / f"checkpoint_epoch_{epoch}.pt"
                if model_path.exists():
                    logger.info(f"Found model checkpoint: checkpoint_epoch_{epoch}.pt")
                    break
            else:
                # No checkpoint found
                model_path = model_dir / "checkpoint_epoch_20.pt"  # Will fail below

            if not index_dir.exists():
                logger.warning(f"Index directory not found at {index_dir}, RAG disabled")
                return None

            if not model_path.exists():
                logger.warning(f"Model checkpoint not found at {model_path}, RAG disabled")
                return None

            retriever = KlarecoRetriever(
                index_dir=str(index_dir),
                model_path=str(model_path),
                mode='tree_lstm',
                device='cpu'
            )
            logger.info(f"RAG system (KlarecoRetriever) initialized with index: {index_dir}")
            return retriever

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

    def estimate_confidence(self, ast: Dict[str, Any]) -> float:
        """
        Estimate confidence in handling this query.

        Args:
            ast: Parsed query AST

        Returns:
            Confidence score 0.0-1.0
        """
        if not self.can_handle(ast):
            return 0.0

        # High confidence for questions with question words
        words = self._extract_words(ast)
        question_words = {'kio', 'kiu', 'kie', 'kiam', 'kial', 'kiel', 'kiom', 'ĉu'}

        for word in words:
            radiko = word.get('radiko', '').lower()
            if radiko in question_words:
                # Very high confidence if RAG system is available
                if self.rag_system:
                    return 0.90
                else:
                    return 0.50  # Lower if no RAG available

        return 0.70

    def execute(self, ast: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Answer factual question using RAG + LLM.

        Args:
            ast: Parsed AST of the question
            context: Optional context

        Returns:
            Response dictionary with:
            - 'answer': The generated answer
            - 'confidence': Confidence in the answer
            - 'expert': Name of this expert
            - 'sources': Retrieved source documents
            - 'question': The extracted question
        """
        logger.info(f"{self.name} handling factual question")

        try:
            # Extract question from AST
            question = self._extract_question(ast, context)
            logger.debug(f"Question: {question}")

            # Check if RAG system is available
            if not self.rag_system:
                return {
                    'answer': 'RAG system ne disponeblas. (RAG system not available.)',
                    'confidence': 0.0,
                    'expert': self.name,
                    'error': 'RAG system not initialized',
                    'question': question
                }

            # Retrieve relevant documents
            retrieval_result = self._retrieve_documents(ast, question)

            # Handle dict or list return (for backwards compatibility)
            if isinstance(retrieval_result, dict):
                retrieved_docs = retrieval_result.get('results', [])
                stage1_info = retrieval_result.get('stage1_info', {})
            elif isinstance(retrieval_result, list):
                retrieved_docs = retrieval_result
                stage1_info = {}
            else:
                # Unexpected type, treat as empty
                retrieved_docs = []
                stage1_info = {}

            if not retrieved_docs:
                return {
                    'answer': 'Mi ne trovis rilatan informon. (I did not find relevant information.)',
                    'confidence': 0.3,
                    'expert': self.name,
                    'sources': [],
                    'question': question,
                    'stage1_stats': stage1_info
                }

            # Generate answer using LLM if available, otherwise format retrieved docs
            if self.llm_provider and hasattr(self.llm_provider, '_claude_code_callback') \
               and self.llm_provider._claude_code_callback:
                # LLM callback available - use it to generate answer
                logger.debug("Using LLM generation with callback")
                answer = self._generate_answer(question, retrieved_docs)
                confidence = 0.90
            else:
                # No LLM callback - just format retrieved documents
                logger.debug("No LLM callback, formatting retrieved docs")
                answer = self._format_retrieved_docs(question, retrieved_docs)
                confidence = 0.75

            return {
                'answer': answer,
                'confidence': confidence,
                'expert': self.name,
                'sources': retrieved_docs,
                'question': question,
                'num_sources': len(retrieved_docs),
                'stage1_stats': stage1_info
            }

        except Exception as e:
            logger.error(f"Error executing factoid QA: {e}", exc_info=True)
            return {
                'answer': f'Eraro: {str(e)}',
                'confidence': 0.0,
                'expert': self.name,
                'error': str(e)
            }

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
            # Use hybrid retrieval: keyword filter + semantic rerank
            # Stage 1: Find keyword candidates (BM25-like)
            # Stage 2: Rerank by Tree-LSTM semantic similarity
            result = self.rag_system.retrieve_hybrid(
                ast=ast,
                k=5,  # Return top 5 after reranking
                keyword_candidates=100,  # Consider top 100 keyword matches
                return_scores=True,
                return_stage1_info=True  # Get stage1 stats for display
            )

            # Handle different return formats
            if isinstance(result, dict):
                # Dict with results and stage1 info
                results = result.get('results', [])
                stage1_info = result.get('stage1', {})
            elif isinstance(result, list):
                # Just a list of results (backwards compatibility)
                results = result
                stage1_info = {}
            else:
                # Unexpected return type
                logger.warning(f"Unexpected retrieval result type: {type(result)}")
                results = []
                stage1_info = {}

            if results and stage1_info:
                logger.info(f"Retrieved {len(results)} documents via hybrid search "
                           f"(stage1: {stage1_info.get('total_candidates', '?')} candidates)")
            elif results:
                logger.info(f"Retrieved {len(results)} documents")

            # Return both results and stage1 info
            return {
                'results': results,
                'stage1_info': stage1_info
            }

        except Exception as e:
            logger.error(f"Error retrieving documents: {e}", exc_info=True)
            # Return empty result dict instead of empty list
            return {
                'results': [],
                'stage1_info': {}
            }

    def _format_retrieved_docs(self, question: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Format retrieved documents as answer (without LLM generation).

        Args:
            question: Original question
            retrieved_docs: Retrieved documents from RAG

        Returns:
            Formatted answer from retrieved docs
        """
        if not retrieved_docs:
            return "Mi ne trovis informon pri tio. (I didn't find information about that.)"

        # Take the top result
        top_doc = retrieved_docs[0]
        text = top_doc.get('text', '')
        score = top_doc.get('score', 0.0)
        source = top_doc.get('source_name', 'Unknown')

        # Format answer showing the most relevant sentence
        answer = f"Laŭ la trovita teksto:\n\"{text}\"\n\n(Fonto: {source}, simileco: {score:.3f})"

        return answer

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
