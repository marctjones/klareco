"""
Unit tests for External Tools (Phase 8)
"""

import pytest
from klareco.experts.dictionary_expert import DictionaryExpert, create_dictionary_expert
from klareco.tools.web_search_tool import WebSearchTool, create_web_search_tool
from klareco.tools.code_interpreter_tool import CodeInterpreterTool, create_code_interpreter_tool


class TestDictionaryExpert:
    """Test Dictionary Expert"""

    def test_create_dictionary_expert(self):
        """Test creating dictionary expert"""
        expert = create_dictionary_expert()

        assert expert is not None
        assert expert.name == "Dictionary_Expert"
        assert len(expert.vocabulary) > 0

    def test_custom_vocabulary(self):
        """Test with custom vocabulary"""
        vocab = {'test': 'testo', 'word': 'vorto'}
        expert = DictionaryExpert(vocabulary=vocab)

        assert len(expert.vocabulary) == 2
        assert expert.vocabulary['test'] == 'testo'

    def test_lookup_word(self):
        """Test word lookup"""
        expert = create_dictionary_expert()

        # Direct lookup
        result = expert._lookup_word('hund')
        assert result == 'dog'

        # With ending
        result = expert._lookup_word('hundo')
        assert result == 'dog'

        # Not found
        result = expert._lookup_word('nonexistent')
        assert result is None

    def test_can_handle_definition_question(self):
        """Test detection of definition questions"""
        expert = create_dictionary_expert()

        # "Kio estas X?" pattern
        ast = {
            'tipo': 'frazo',
            'subjekto': {'tipo': 'vorto', 'radiko': 'kio'},
            'verbo': {'tipo': 'vorto', 'radiko': 'est'},
            'aliaj': [{'tipo': 'vorto', 'radiko': 'hund'}]
        }

        assert expert.can_handle(ast)

    def test_can_handle_define_command(self):
        """Test detection of define commands"""
        expert = create_dictionary_expert()

        ast = {
            'tipo': 'frazo',
            'verbo': {'tipo': 'vorto', 'radiko': 'difin'},
            'objekto': {'tipo': 'vorto', 'radiko': 'kat'}
        }

        assert expert.can_handle(ast)

    def test_execute_found(self):
        """Test execution with found word"""
        expert = create_dictionary_expert()

        ast = {
            'tipo': 'frazo',
            'subjekto': {'tipo': 'vorto', 'radiko': 'kio'},
            'verbo': {'tipo': 'vorto', 'radiko': 'est'},
            'aliaj': [{'tipo': 'vorto', 'radiko': 'hund'}]
        }

        result = expert.execute(ast)

        assert result['confidence'] > 0.9
        assert 'dog' in result['answer'].lower()
        assert result['word'] == 'hund'
        assert result['definition'] == 'dog'

    def test_execute_not_found(self):
        """Test execution with word not in vocabulary"""
        expert = create_dictionary_expert()

        ast = {
            'tipo': 'frazo',
            'aliaj': [{'tipo': 'vorto', 'radiko': 'unknownword'}]
        }

        result = expert.execute(ast, context={'target_word': 'unknownword'})

        assert result['confidence'] < 0.5
        assert 'ne trovis' in result['answer'].lower()

    def test_estimate_confidence(self):
        """Test confidence estimation"""
        expert = create_dictionary_expert()

        # Can handle
        ast = {
            'tipo': 'frazo',
            'verbo': {'tipo': 'vorto', 'radiko': 'difin'}
        }
        confidence = expert.estimate_confidence(ast)
        assert confidence == 0.9

        # Cannot handle
        ast = {
            'tipo': 'frazo',
            'verbo': {'tipo': 'vorto', 'radiko': 'kur'}
        }
        confidence = expert.estimate_confidence(ast)
        assert confidence == 0.0


class TestWebSearchTool:
    """Test Web Search Tool"""

    def test_create_web_search_tool(self):
        """Test creating web search tool"""
        tool = create_web_search_tool(mock_mode=True)

        assert tool is not None
        assert tool.name == "Web_Search_Tool"
        assert tool.mock_mode is True

    def test_mock_search(self):
        """Test mock search"""
        tool = WebSearchTool(mock_mode=True)

        result = tool.execute("test query", max_results=3)

        assert result['num_results'] == 3
        assert result['source'] == 'mock'
        assert len(result['results']) == 3

        # Check result structure
        for r in result['results']:
            assert 'title' in r
            assert 'url' in r
            assert 'snippet' in r

    def test_search_and_summarize(self):
        """Test formatted search output"""
        tool = WebSearchTool(mock_mode=True)

        output = tool.search_and_summarize("test query", max_results=2)

        assert 'Serĉrezultoj' in output
        assert 'test query' in output
        assert 'URL:' in output

    def test_can_handle(self):
        """Test query detection"""
        tool = WebSearchTool(mock_mode=True)

        # Search query
        ast = {
            'tipo': 'frazo',
            'verbo': {'tipo': 'vorto', 'radiko': 'serĉ'}
        }
        assert tool.can_handle(ast)

        # Non-search query
        ast = {
            'tipo': 'frazo',
            'verbo': {'tipo': 'vorto', 'radiko': 'manĝ'}
        }
        assert not tool.can_handle(ast)

    def test_max_results_limit(self):
        """Test max results limit"""
        tool = WebSearchTool(mock_mode=True)

        result = tool.execute("test", max_results=10)

        # Mock returns max 3
        assert result['num_results'] <= 3


class TestCodeInterpreterTool:
    """Test Code Interpreter Tool"""

    def test_create_code_interpreter_tool(self):
        """Test creating code interpreter tool"""
        tool = create_code_interpreter_tool(timeout=5)

        assert tool is not None
        assert tool.name == "Code_Interpreter_Tool"
        assert tool.timeout == 5

    def test_simple_execution(self):
        """Test simple code execution"""
        tool = CodeInterpreterTool(timeout=5)

        code = "print('Hello')"
        result = tool.execute(code)

        assert result['success'] is True
        assert 'Hello' in result['output']
        assert result['error'] is None
        assert result['execution_time'] > 0

    def test_calculation(self):
        """Test calculation"""
        tool = CodeInterpreterTool(timeout=5)

        code = "print(2 + 2)"
        result = tool.execute(code)

        assert result['success'] is True
        assert '4' in result['output']

    def test_blocked_import(self):
        """Test blocked import detection"""
        tool = CodeInterpreterTool(timeout=5)

        code = "import os\nprint(os.listdir())"
        result = tool.execute(code)

        assert result['success'] is False
        assert 'Blocked' in result['error']

    def test_syntax_error(self):
        """Test syntax error handling"""
        tool = CodeInterpreterTool(timeout=5)

        code = "print('missing paren'"
        result = tool.execute(code)

        assert result['success'] is False
        assert result['error'] is not None

    def test_timeout(self):
        """Test execution timeout"""
        tool = CodeInterpreterTool(timeout=1)

        code = "import time\ntime.sleep(5)\nprint('done')"
        result = tool.execute(code)

        assert result['success'] is False
        assert 'timeout' in result['error'].lower()

    def test_output_truncation(self):
        """Test output truncation for large output"""
        tool = CodeInterpreterTool(timeout=5, max_output_size=100)

        code = "for i in range(1000):\n    print(i)"
        result = tool.execute(code)

        assert len(result['output']) <= 200  # Some margin for truncation message

    def test_can_handle(self):
        """Test query detection"""
        tool = CodeInterpreterTool()

        # Code execution query
        ast = {
            'tipo': 'frazo',
            'verbo': {'tipo': 'vorto', 'radiko': 'kalkul'}
        }
        assert tool.can_handle(ast)

        # Non-code query
        ast = {
            'tipo': 'frazo',
            'verbo': {'tipo': 'vorto', 'radiko': 'manĝ'}
        }
        assert not tool.can_handle(ast)

    def test_execute_and_format(self):
        """Test formatted execution output"""
        tool = CodeInterpreterTool(timeout=5)

        output = tool.execute_and_format("print('test')")

        assert 'Sukcese plenumita' in output
        assert 'test' in output

    def test_validation(self):
        """Test code validation"""
        tool = CodeInterpreterTool()

        # Valid code
        error = tool._validate_code("print('hello')")
        assert error is None

        # Blocked function
        error = tool._validate_code("eval('1+1')")
        assert error is not None
        assert 'Blocked' in error

        # Too long
        error = tool._validate_code("x = 1\n" * 10000)
        assert error is not None
        assert 'too long' in error.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
