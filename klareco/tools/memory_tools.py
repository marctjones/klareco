"""
Memory Tools - Read and Write tools for accessing memory

These tools allow experts to interact with the memory system:
- Memory_Read_Tool: Query and retrieve memories
- Memory_Write_Tool: Store new memories

Part of Phase 6: Memory System
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging
from ..memory import MemorySystem, MemoryType, MemoryEntry

logger = logging.getLogger(__name__)


class MemoryReadTool:
    """
    Tool for reading from memory system.

    Allows experts to query both STM and LTM to retrieve
    relevant context and facts.
    """

    def __init__(self, memory_system: MemorySystem):
        """
        Initialize Memory Read Tool.

        Args:
            memory_system: The memory system to query
        """
        self.memory_system = memory_system
        self.name = "Memory_Read_Tool"
        self.capabilities = ["memory_retrieval", "context_lookup", "fact_recall"]

        logger.info(f"{self.name} initialized")

    def can_handle(self, ast: Dict[str, Any]) -> bool:
        """
        Check if this tool can handle the query.

        Memory read queries typically contain:
        - "memori" (remember)
        - "kio mi diris" (what did I say)
        - "antaŭe" (before)
        - "rakont" (tell)

        Args:
            ast: Parsed query AST

        Returns:
            True if this tool can handle it
        """
        # Check for memory-related keywords
        memory_keywords = {'memor', 'antaŭ', 'rakont', 'dir'}

        return self._contains_any_root(ast, memory_keywords)

    def execute(self, ast: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute memory read.

        Args:
            ast: Parsed query AST
            context: Optional execution context

        Returns:
            Response with retrieved memories
        """
        logger.info(f"{self.name} executing memory read")

        # Determine query type
        query_type = self._determine_query_type(ast)

        # Execute appropriate query
        if query_type == "recent":
            memories = self._recall_recent(context)
        elif query_type == "temporal":
            memories = self._recall_temporal(context)
        elif query_type == "fact":
            memories = self._recall_facts(context)
        else:
            memories = self._recall_recent(context)

        # Format response
        answer = self._format_memories(memories)

        return {
            'answer': answer,
            'confidence': 0.9,
            'expert': self.name,
            'memories_found': len(memories),
            'metadata': {
                'query_type': query_type,
                'memory_count': len(memories)
            }
        }

    def _determine_query_type(self, ast: Dict[str, Any]) -> str:
        """Determine type of memory query"""
        # Check for temporal keywords
        if self._contains_any_root(ast, {'antaŭ', 'hieraŭ', 'pasint'}):
            return "temporal"

        # Check for fact keywords
        if self._contains_any_root(ast, {'fakto', 'vero', 'informo'}):
            return "fact"

        # Default to recent
        return "recent"

    def _recall_recent(self, context: Optional[Dict[str, Any]], n: int = 5) -> List[MemoryEntry]:
        """Recall recent memories"""
        return self.memory_system.recall_recent(n=n)

    def _recall_temporal(self, context: Optional[Dict[str, Any]]) -> List[MemoryEntry]:
        """Recall memories from specific time period"""
        # Get memories from last 24 hours
        since = datetime.now() - timedelta(hours=24)
        return self.memory_system.ltm.search(since=since, limit=10)

    def _recall_facts(self, context: Optional[Dict[str, Any]]) -> List[MemoryEntry]:
        """Recall facts from memory"""
        return self.memory_system.ltm.search(memory_type=MemoryType.FACT, limit=10)

    def _format_memories(self, memories: List[MemoryEntry]) -> str:
        """Format memories as readable text"""
        if not memories:
            return "Mi ne trovas memoraĵojn. (I don't find any memories.)"

        lines = []
        for i, mem in enumerate(memories, 1):
            time_str = mem.timestamp.strftime("%Y-%m-%d %H:%M")
            lines.append(f"{i}. [{time_str}] {mem.text}")

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
        return f"{self.name}(memory={self.memory_system})"


class MemoryWriteTool:
    """
    Tool for writing to memory system.

    Allows experts to store new facts, events, and important
    information for future retrieval.
    """

    def __init__(self, memory_system: MemorySystem):
        """
        Initialize Memory Write Tool.

        Args:
            memory_system: The memory system to write to
        """
        self.memory_system = memory_system
        self.name = "Memory_Write_Tool"
        self.capabilities = ["memory_storage", "fact_storage", "event_logging"]

        logger.info(f"{self.name} initialized")

    def can_handle(self, ast: Dict[str, Any]) -> bool:
        """
        Check if this tool can handle the query.

        Memory write commands typically contain:
        - "memoru" (remember)
        - "konservu" (save/preserve)
        - "registru" (record)

        Args:
            ast: Parsed query AST

        Returns:
            True if this tool can handle it
        """
        memory_write_keywords = {'memor', 'konserv', 'registr', 'skrib'}

        return self._contains_any_root(ast, memory_write_keywords)

    def execute(self, ast: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute memory write.

        Args:
            ast: Parsed query AST
            context: Optional execution context with data to store

        Returns:
            Response confirming storage
        """
        logger.info(f"{self.name} executing memory write")

        # Extract what to remember
        if context and 'data_to_remember' in context:
            data = context['data_to_remember']
            text = data.get('text', '')
            data_ast = data.get('ast', ast)
            memory_type = MemoryType(data.get('type', 'fact'))
            persistent = data.get('persistent', True)
        else:
            # Store the query itself as a fact
            from ..deparser import deparse
            text = deparse(ast)
            data_ast = ast
            memory_type = MemoryType.FACT
            persistent = True

        # Store memory
        self.memory_system.remember(
            memory_type=memory_type,
            ast=data_ast,
            text=text,
            persistent=persistent
        )

        answer = f"Mi memoris: {text}"

        return {
            'answer': answer,
            'confidence': 1.0,
            'expert': self.name,
            'metadata': {
                'memory_type': memory_type.value,
                'persistent': persistent
            }
        }

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
        return f"{self.name}(memory={self.memory_system})"


# Factory functions
def create_memory_read_tool(memory_system: MemorySystem) -> MemoryReadTool:
    """Create Memory Read Tool"""
    return MemoryReadTool(memory_system)


def create_memory_write_tool(memory_system: MemorySystem) -> MemoryWriteTool:
    """Create Memory Write Tool"""
    return MemoryWriteTool(memory_system)


if __name__ == "__main__":
    # Test memory tools
    print("Testing Memory Tools")
    print("=" * 80)

    from ..memory import create_memory_system

    # Create memory system
    memory = create_memory_system()

    # Add some test data
    memory.remember(
        MemoryType.USER_QUERY,
        {'tipo': 'frazo'},
        "Kio estas Esperanto?"
    )

    memory.remember(
        MemoryType.FACT,
        {'tipo': 'fakto'},
        "Esperanto estis kreita de Zamenhof.",
        persistent=True
    )

    # Create tools
    read_tool = create_memory_read_tool(memory)
    write_tool = create_memory_write_tool(memory)

    print(f"\nTools created:")
    print(f"  {read_tool}")
    print(f"  {write_tool}")

    # Test read
    print("\nTesting memory read...")
    test_ast = {'tipo': 'frazo', 'verbo': {'tipo': 'vorto', 'radiko': 'memor'}}

    if read_tool.can_handle(test_ast):
        result = read_tool.execute(test_ast)
        print(f"Result: {result['answer']}")

    # Test write
    print("\nTesting memory write...")
    test_ast = {'tipo': 'frazo', 'verbo': {'tipo': 'vorto', 'radiko': 'memor'}}
    context = {
        'data_to_remember': {
            'text': 'La ĉielo estas blua.',
            'ast': {'tipo': 'fakto'},
            'type': 'fact',
            'persistent': True
        }
    }

    if write_tool.can_handle(test_ast):
        result = write_tool.execute(test_ast, context)
        print(f"Result: {result['answer']}")

    print(f"\nMemory system now: {memory}")

    print("\n✅ Memory tools test complete!")
