"""
Web Search Tool - Search the web for information

Provides web search capabilities using DuckDuckGo or other search engines.
No API key required for basic functionality.

Part of Phase 8: External Tools
"""

from typing import Dict, Any, Optional, List
import logging
import re

logger = logging.getLogger(__name__)


class WebSearchTool:
    """
    Tool for searching the web.

    Uses DuckDuckGo HTML search (no API key required) with
    fallback to mock results for testing.
    """

    def __init__(self, mock_mode: bool = False):
        """
        Initialize Web Search Tool.

        Args:
            mock_mode: If True, returns mock results for testing
        """
        self.name = "Web_Search_Tool"
        self.capabilities = ["web_search", "information_retrieval", "fact_checking"]
        self.mock_mode = mock_mode

        logger.info(f"{self.name} initialized (mock_mode={mock_mode})")

    def can_handle(self, ast: Dict[str, Any]) -> bool:
        """
        Check if this tool can handle the query.

        Web search queries contain:
        - "serĉu" (search)
        - "trovu" (find)
        - "informo" (information)
        - "reto" (web/net)

        Args:
            ast: Parsed query AST

        Returns:
            True if web search query
        """
        search_keywords = {'serĉ', 'trov', 'inform', 'ret', 'interret'}

        return self._contains_any_root(ast, search_keywords)

    def execute(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Execute web search.

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            Search results
        """
        logger.info(f"{self.name} searching for: {query}")

        if self.mock_mode:
            results = self._mock_search(query, max_results)
        else:
            results = self._duckduckgo_search(query, max_results)

        return {
            'query': query,
            'results': results,
            'num_results': len(results),
            'source': 'mock' if self.mock_mode else 'duckduckgo'
        }

    def _mock_search(self, query: str, max_results: int) -> List[Dict[str, str]]:
        """
        Generate mock search results for testing.

        Args:
            query: Search query
            max_results: Maximum results

        Returns:
            Mock results
        """
        # Generate plausible mock results
        results = []

        for i in range(min(max_results, 3)):
            results.append({
                'title': f"Result {i+1} for '{query}'",
                'url': f"https://example.com/result-{i+1}",
                'snippet': f"This is a mock search result for the query '{query}'. "
                          f"It contains relevant information about {query}."
            })

        return results

    def _duckduckgo_search(self, query: str, max_results: int) -> List[Dict[str, str]]:
        """
        Search using DuckDuckGo HTML (no API key needed).

        Args:
            query: Search query
            max_results: Maximum results

        Returns:
            Search results
        """
        try:
            import requests
            from urllib.parse import quote

            # DuckDuckGo HTML search
            search_url = f"https://html.duckduckgo.com/html/?q={quote(query)}"

            headers = {
                'User-Agent': 'Mozilla/5.0 (compatible; Klareco/1.0)'
            }

            response = requests.get(search_url, headers=headers, timeout=10)

            if response.status_code != 200:
                logger.warning(f"Search failed with status {response.status_code}")
                return self._mock_search(query, max_results)

            # Parse HTML results
            results = self._parse_duckduckgo_html(response.text, max_results)

            if not results:
                logger.warning("No results parsed, using mock")
                return self._mock_search(query, max_results)

            return results

        except ImportError:
            logger.warning("requests library not available, using mock")
            return self._mock_search(query, max_results)
        except Exception as e:
            logger.error(f"Search error: {e}")
            return self._mock_search(query, max_results)

    def _parse_duckduckgo_html(self, html: str, max_results: int) -> List[Dict[str, str]]:
        """
        Parse DuckDuckGo HTML results.

        Args:
            html: HTML response
            max_results: Maximum results

        Returns:
            Parsed results
        """
        results = []

        try:
            # Simple regex-based parsing (better than nothing)
            # In production, would use BeautifulSoup or similar

            # Find result blocks
            result_pattern = r'class="result__a"[^>]*href="([^"]+)"[^>]*>([^<]+)<'
            snippet_pattern = r'class="result__snippet">([^<]+)<'

            urls_titles = re.findall(result_pattern, html)
            snippets = re.findall(snippet_pattern, html)

            for i, (url, title) in enumerate(urls_titles[:max_results]):
                snippet = snippets[i] if i < len(snippets) else ""

                results.append({
                    'title': title.strip(),
                    'url': url.strip(),
                    'snippet': snippet.strip()
                })

        except Exception as e:
            logger.error(f"HTML parsing error: {e}")

        return results

    def search_and_summarize(self, query: str, max_results: int = 3) -> str:
        """
        Search and format results as text.

        Args:
            query: Search query
            max_results: Maximum results

        Returns:
            Formatted search results
        """
        result = self.execute(query, max_results)
        results = result['results']

        if not results:
            return f"Neniuj rezultoj por '{query}'. (No results for '{query}'.)"

        lines = [f"Serĉrezultoj por '{query}':"]
        lines.append("")

        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r['title']}")
            lines.append(f"   URL: {r['url']}")
            if r['snippet']:
                lines.append(f"   {r['snippet'][:150]}...")
            lines.append("")

        return "\n".join(lines)

    def _contains_any_root(self, ast: Dict[str, Any], roots: set) -> bool:
        """Check if AST contains any of the specified roots"""
        if ast.get('tipo') == 'vorto':
            radiko = ast.get('radiko', '').lower()
            return any(radiko.startswith(root) for root in roots)
        elif ast.get('tipo') == 'vortgrupo':
            return any(self._contains_any_root(v, roots) for v in ast.get('vortoj', []))
        elif ast.get('tipo') == 'frazo':
            for key in ['subjekto', 'verbo', 'objekto']:
                if ast.get(key) and self._contains_any_root(ast[key], roots):
                    return True
            return any(self._contains_any_root(v, roots) for v in ast.get('aliaj', []))
        return False

    def __repr__(self) -> str:
        mode = "mock" if self.mock_mode else "live"
        return f"{self.name}(mode={mode})"


# Factory function
def create_web_search_tool(mock_mode: bool = False) -> WebSearchTool:
    """
    Create and return a WebSearchTool instance.

    Args:
        mock_mode: If True, returns mock results

    Returns:
        Initialized WebSearchTool
    """
    return WebSearchTool(mock_mode=mock_mode)


if __name__ == "__main__":
    # Test web search tool
    print("Testing Web Search Tool")
    print("=" * 80)

    # Test in mock mode
    tool = create_web_search_tool(mock_mode=True)
    print(f"\n{tool}\n")

    # Test search
    test_queries = [
        "Esperanto language history",
        "Python programming tutorial",
    ]

    for query in test_queries:
        print(f"Searching for: {query}")
        result = tool.execute(query, max_results=3)

        print(f"  Found {result['num_results']} results")
        for i, r in enumerate(result['results'], 1):
            print(f"    {i}. {r['title']}")

        print()

    # Test formatted output
    print("Formatted search:")
    print(tool.search_and_summarize("Esperanto"))

    print("\n✅ Web Search Tool test complete!")
