"""
Summarize Expert - Neural decoder for text summarization

This expert uses LLMs to generate concise summaries of long text.
Unlike symbolic experts, this requires genuine creative/generative capabilities.

Part of Phase 5: Planning & Advanced Experts
"""

from typing import Dict, Any, Optional
import logging
from ..llm_provider import get_llm_provider

logger = logging.getLogger(__name__)


class SummarizeExpert:
    """
    Expert for summarizing text using LLMs.

    This expert:
    1. Extracts text from AST (symbolic)
    2. Uses LLM to generate summary (neural)
    3. Returns formatted summary

    The LLM provider is auto-detected (Claude Code when available).
    """

    def __init__(self, llm_provider=None):
        """
        Initialize Summarize Expert.

        Args:
            llm_provider: Optional LLM provider (auto-detected if None)
        """
        self.name = "Summarize_Expert"
        self.capabilities = ["summarization", "text_condensation", "key_points"]
        self.llm_provider = llm_provider or get_llm_provider()

        logger.info(f"{self.name} initialized with provider: {self.llm_provider.provider_type.value}")

    def can_handle(self, ast: Dict[str, Any]) -> bool:
        """
        Check if this expert can handle the AST.

        Summarization requests typically contain keywords like:
        - resumo/resumu (summarize)
        - mallongigo (shorten)
        - ĉefaj punktoj (main points)

        Args:
            ast: Parsed AST

        Returns:
            True if this expert can handle the request
        """
        # Extract all words from AST
        words = self._extract_words(ast)

        # Keywords indicating summarization request
        summarize_keywords = {
            'resumo', 'resumu', 'resumigi', 'resumo',  # summarize
            'mallongigo', 'mallongigu',  # shorten
            'ĉefaj', 'punktoj',  # main points
            'kompenso', 'esenci',  # essence
            'koncizi', 'koncize'  # concise
        }

        for word in words:
            radiko = word.get('radiko', '').lower()
            if radiko in summarize_keywords:
                logger.debug(f"Summarize keyword detected: {radiko}")
                return True

        return False

    def handle(self, ast: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a summary using LLM.

        Args:
            ast: Parsed AST
            context: Optional context (may contain text to summarize)

        Returns:
            Generated summary
        """
        logger.info(f"{self.name} handling summarization request")

        # Extract text to summarize from context or AST
        text_to_summarize = self._extract_target_text(ast, context)

        if not text_to_summarize:
            return "Mi ne trovis tekston por resumigi. (I didn't find text to summarize.)"

        # Determine summary style from AST
        summary_style = self._determine_style(ast)

        # Generate summary using LLM
        summary = self._generate_summary(text_to_summarize, summary_style)

        return summary

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

    def _extract_target_text(self, ast: Dict[str, Any], context: Optional[Dict[str, Any]]) -> str:
        """
        Extract the text that needs to be summarized.

        This could come from:
        1. Context (if provided)
        2. A referenced document
        3. The AST itself (less common)

        Args:
            ast: Parsed AST
            context: Optional context

        Returns:
            Text to summarize
        """
        # Check context first
        if context and 'text_to_summarize' in context:
            return context['text_to_summarize']

        # For demo purposes, extract from context['original_text']
        if context and 'original_text' in context:
            return context['original_text']

        # Fall back to reconstructing from AST
        # This is less ideal but provides a default
        from ..deparser import deparse
        return deparse(ast)

    def _determine_style(self, ast: Dict[str, Any]) -> str:
        """
        Determine the summary style from AST.

        Styles:
        - brief: Very short summary (1-2 sentences)
        - moderate: Standard summary (3-5 sentences)
        - detailed: Comprehensive summary with key points
        - bullet_points: Bulleted list of main points

        Args:
            ast: Parsed AST

        Returns:
            Summary style identifier
        """
        words = self._extract_words(ast)

        # Check for style indicators
        for word in words:
            radiko = word.get('radiko', '').lower()

            if radiko in ['mallonga', 'konciza', 'kurta']:  # short, concise, brief
                return 'brief'
            elif radiko in ['detala', 'ampleksa']:  # detailed, comprehensive
                return 'detailed'
            elif radiko in ['punkto', 'listo']:  # points, list
                return 'bullet_points'

        # Default to moderate
        return 'moderate'

    def _generate_summary(self, text: str, style: str) -> str:
        """
        Generate summary using LLM.

        Args:
            text: Text to summarize
            style: Summary style (brief, moderate, detailed, bullet_points)

        Returns:
            Generated summary
        """
        # Create style-specific system prompt
        style_prompts = {
            'brief': "Generate a very concise summary in 1-2 sentences.",
            'moderate': "Generate a clear summary in 3-5 sentences, capturing the main ideas.",
            'detailed': "Generate a comprehensive summary that covers all key points and important details.",
            'bullet_points': "Generate a bulleted list of the main points (4-6 bullets)."
        }

        system_prompt = (
            "You are a helpful assistant that creates clear, accurate summaries. "
            + style_prompts.get(style, style_prompts['moderate'])
        )

        user_prompt = f"Please summarize the following text:\n\n{text}"

        logger.debug(f"Generating {style} summary using {self.llm_provider.provider_type.value}")

        try:
            summary = self.llm_provider.generate(
                prompt=user_prompt,
                system=system_prompt,
                max_tokens=500,
                temperature=0.3  # Lower temperature for more focused summaries
            )

            logger.info(f"Summary generated successfully ({len(summary)} chars)")
            return summary

        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return f"[Error generating summary: {str(e)}]"


# Factory function for creating expert instance
def create_summarize_expert(llm_provider=None) -> SummarizeExpert:
    """
    Create and return a Summarize Expert instance.

    Args:
        llm_provider: Optional LLM provider (auto-detected if None)

    Returns:
        Initialized SummarizeExpert
    """
    return SummarizeExpert(llm_provider)


if __name__ == "__main__":
    # Test the expert
    import sys
    from ..parser import parse_esperanto

    # Example usage
    expert = create_summarize_expert()

    # Test AST (request to summarize)
    test_query = "Resumu la tekston."  # "Summarize the text"
    ast = parse_esperanto(test_query)

    print(f"Can handle: {expert.can_handle(ast)}")

    if len(sys.argv) > 1:
        # Summarize provided text
        text_to_summarize = " ".join(sys.argv[1:])
        context = {'text_to_summarize': text_to_summarize}
        result = expert.handle(ast, context)
        print(f"\nSummary:\n{result}")
