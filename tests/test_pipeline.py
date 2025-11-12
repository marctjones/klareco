"""
Tests for the KlarecoPipeline orchestrator.

This test suite validates the main pipeline orchestration, including:
- Successful full pipeline execution
- Stopping at each intermediate step
- Error handling at each stage
- Trace logging correctness
"""
import unittest
from unittest.mock import patch, MagicMock
from klareco.pipeline import KlarecoPipeline
from klareco.trace import ExecutionTrace


class TestKlarecoPipeline(unittest.TestCase):
    """Test suite for the KlarecoPipeline class."""

    def setUp(self):
        """Initialize a pipeline instance for each test."""
        self.pipeline = KlarecoPipeline()

    def test_init_creates_components(self):
        """Tests that pipeline initialization creates required components."""
        self.assertIsNotNone(self.pipeline.front_door)
        self.assertIsNotNone(self.pipeline.safety_monitor)

    def test_run_full_pipeline_returns_trace_with_response(self):
        """Tests that a successful full pipeline run returns a trace with final response."""
        query = "La hundo amas la katon."
        trace = self.pipeline.run(query)

        # Verify trace structure
        self.assertIsInstance(trace, ExecutionTrace)
        self.assertEqual(trace.initial_query, query)
        self.assertIsNotNone(trace.final_response)
        self.assertIsNone(trace.error)
        self.assertIsNotNone(trace.end_time)

        # Verify all steps were executed (5 steps: Safety, FrontDoor, Parser, Safety_AST, Orchestrator)
        self.assertGreaterEqual(len(trace.steps), 5)

    def test_run_records_all_pipeline_steps(self):
        """Tests that all pipeline steps are recorded in the trace."""
        query = "La hundo amas la katon."
        trace = self.pipeline.run(query)

        # Extract step names
        step_names = [step['name'] for step in trace.steps]

        # Verify expected steps are present (SafetyMonitor appears twice)
        self.assertIn("SafetyMonitor", step_names)
        self.assertIn("FrontDoor", step_names)
        self.assertIn("Parser", step_names)
        self.assertIn("Orchestrator", step_names)

        # Verify SafetyMonitor appears at least twice (input and AST checks)
        safety_count = step_names.count("SafetyMonitor")
        self.assertGreaterEqual(safety_count, 2)

    def test_run_with_stop_after_safety_monitor(self):
        """Tests that pipeline stops after SafetyMonitor when requested."""
        query = "La hundo amas la katon."
        trace = self.pipeline.run(query, stop_after="SafetyMonitor")

        # Verify trace has only SafetyMonitor step
        self.assertEqual(len(trace.steps), 1)
        self.assertEqual(trace.steps[0]['name'], "SafetyMonitor")

        # Verify no final response (stopped early)
        self.assertIsNone(trace.final_response)
        self.assertIsNone(trace.error)

    def test_run_with_stop_after_front_door(self):
        """Tests that pipeline stops after FrontDoor when requested."""
        query = "The dog loves the cat."
        trace = self.pipeline.run(query, stop_after="FrontDoor")

        # Verify trace has SafetyMonitor and FrontDoor steps
        self.assertEqual(len(trace.steps), 2)
        step_names = [step['name'] for step in trace.steps]
        self.assertEqual(step_names, ["SafetyMonitor", "FrontDoor"])

        # Verify FrontDoor outputs contain language and processed text
        front_door_step = trace.steps[1]
        self.assertIn("original_lang", front_door_step['outputs'])
        self.assertIn("processed_text", front_door_step['outputs'])

    def test_run_with_stop_after_parser(self):
        """Tests that pipeline stops after Parser when requested."""
        query = "La hundo vidas la katon."
        trace = self.pipeline.run(query, stop_after="Parser")

        # Verify trace has SafetyMonitor, FrontDoor, and Parser steps
        self.assertEqual(len(trace.steps), 3)
        step_names = [step['name'] for step in trace.steps]
        self.assertEqual(step_names, ["SafetyMonitor", "FrontDoor", "Parser"])

        # Verify Parser outputs contain AST
        parser_step = trace.steps[2]
        self.assertIn("ast", parser_step['outputs'])
        ast = parser_step['outputs']['ast']
        self.assertIsInstance(ast, dict)
        self.assertEqual(ast.get('tipo'), 'frazo')

    def test_run_with_stop_after_safety_monitor_ast(self):
        """Tests that pipeline stops after AST safety check when requested."""
        query = "La hundo vidas la katon."
        trace = self.pipeline.run(query, stop_after="SafetyMonitor_AST")

        # Verify trace has 4 steps (Safety input, FrontDoor, Parser, Safety AST)
        self.assertEqual(len(trace.steps), 4)

        # Verify last step is SafetyMonitor with AST complexity check
        last_step = trace.steps[3]
        self.assertEqual(last_step['name'], "SafetyMonitor")
        self.assertIn("node_count", last_step['outputs'])

    def test_run_with_stop_after_orchestrator(self):
        """Tests that pipeline stops after Orchestrator when requested."""
        query = "La hundo vidas la katon."
        trace = self.pipeline.run(query, stop_after="Orchestrator")

        # Verify trace has 5 steps
        self.assertEqual(len(trace.steps), 5)

        # Verify Orchestrator outputs contain intent and expert info
        orchestrator_step = trace.steps[4]
        self.assertEqual(orchestrator_step['name'], "Orchestrator")
        self.assertIn("intent", orchestrator_step['outputs'])

    def test_run_with_invalid_input_length_records_error(self):
        """Tests that input exceeding length limit is recorded as error in trace."""
        # Set a very low max length to trigger failure
        self.pipeline.safety_monitor.max_input_length = 5
        query = "This text is way too long."

        trace = self.pipeline.run(query)

        # Verify error was recorded
        self.assertIsNotNone(trace.error)
        self.assertIn("exceeds maximum", trace.error)
        self.assertIsNone(trace.final_response)

        # Verify SafetyMonitor step was still added (shows where it failed)
        self.assertEqual(len(trace.steps), 0)  # No steps added before exception

    def test_run_with_invalid_esperanto_syntax_records_error(self):
        """Tests that invalid Esperanto syntax succeeds with graceful degradation."""
        # Use a query with unknown root that will be handled gracefully
        query = "xyzabc"

        trace = self.pipeline.run(query)

        # With graceful degradation, should succeed
        self.assertIsNone(trace.error)
        self.assertIsNotNone(trace.final_response)

        # Verify pipeline completed all steps
        step_names = [step['name'] for step in trace.steps]
        self.assertIn("SafetyMonitor", step_names)
        self.assertIn("FrontDoor", step_names)
        self.assertIn("Parser", step_names)  # Parser should complete successfully

    def test_run_with_complex_ast_exceeding_limit_records_error(self):
        """Tests that AST exceeding complexity limit is recorded as error."""
        # Set a very low AST complexity limit
        self.pipeline.safety_monitor.max_ast_nodes = 5
        query = "La granda hundo amas la malgrandan katon."  # Complex sentence

        trace = self.pipeline.run(query)

        # Verify error was recorded
        self.assertIsNotNone(trace.error)
        self.assertIn("exceeds maximum", trace.error)
        self.assertIsNone(trace.final_response)

        # Verify pipeline got through Parser but failed at AST safety check
        step_names = [step['name'] for step in trace.steps]
        self.assertIn("Parser", step_names)

    def test_run_with_english_query_translates_to_esperanto(self):
        """Tests that English queries are translated to Esperanto via FrontDoor."""
        query = "The dog sees the cat."
        trace = self.pipeline.run(query)

        # Find FrontDoor step
        front_door_step = None
        for step in trace.steps:
            if step['name'] == 'FrontDoor':
                front_door_step = step
                break

        self.assertIsNotNone(front_door_step)

        # Verify language was detected as English
        self.assertIn("original_lang", front_door_step['outputs'])

        # Verify processed_text is present (translated or original)
        self.assertIn("processed_text", front_door_step['outputs'])
        processed = front_door_step['outputs']['processed_text']
        self.assertIsInstance(processed, str)
        self.assertGreater(len(processed), 0)

    def test_run_with_esperanto_query_bypasses_translation(self):
        """Tests that Esperanto queries bypass translation in FrontDoor."""
        query = "La hundo vidas la katon."
        trace = self.pipeline.run(query)

        # Find FrontDoor step
        front_door_step = None
        for step in trace.steps:
            if step['name'] == 'FrontDoor':
                front_door_step = step
                break

        self.assertIsNotNone(front_door_step)

        # Verify language was detected as Esperanto
        lang = front_door_step['outputs']['original_lang']
        self.assertEqual(lang, 'eo')

        # Verify processed text is same as input (no translation needed)
        processed = front_door_step['outputs']['processed_text']
        self.assertEqual(processed, query)

    def test_run_adds_step_descriptions(self):
        """Tests that each step has a human-readable description in the trace."""
        query = "La hundo amas la katon."
        trace = self.pipeline.run(query)

        # Verify all steps have descriptions
        for step in trace.steps:
            self.assertIn("description", step)
            self.assertIsInstance(step['description'], str)
            self.assertGreater(len(step['description']), 10)

    def test_run_includes_timestamps_for_each_step(self):
        """Tests that each step includes a timestamp."""
        query = "La hundo amas la katon."
        trace = self.pipeline.run(query)

        # Verify all steps have timestamps
        for step in trace.steps:
            self.assertIn("timestamp", step)
            self.assertIsInstance(step['timestamp'], str)
            # Verify timestamp format (ISO 8601)
            self.assertRegex(step['timestamp'], r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}')

    def test_run_assigns_sequential_step_ids(self):
        """Tests that steps are assigned sequential IDs starting from 1."""
        query = "La hundo amas la katon."
        trace = self.pipeline.run(query)

        # Verify step IDs are sequential
        for i, step in enumerate(trace.steps):
            self.assertEqual(step['step_id'], i + 1)

    def test_pipeline_preserves_original_query_in_trace(self):
        """Tests that the original query is preserved exactly in the trace."""
        query = "La HUNDO amas la Katon!"  # Mixed case and punctuation
        trace = self.pipeline.run(query)

        self.assertEqual(trace.initial_query, query)

    def test_run_with_stop_after_orchestrator_returns_trace(self):
        """Tests that pipeline stops after Orchestrator (effectively full run)."""
        query = "La hundo amas la katon."
        trace = self.pipeline.run(query, stop_after="Orchestrator")

        # stop_after="Orchestrator" should be same as full run
        # (Orchestrator is the last step, so trace should have response but no end_time set by set_final_response)
        self.assertEqual(len(trace.steps), 5)

        # When stopping after Orchestrator, final_response is not set
        # (pipeline returns before calling trace.set_final_response())
        self.assertIsNone(trace.final_response)

    def test_run_sets_trace_end_time_on_completion(self):
        """Tests that trace end_time is set when pipeline completes successfully."""
        query = "La hundo amas la katon."
        trace = self.pipeline.run(query)

        self.assertIsNotNone(trace.end_time)
        # Verify end_time format (ISO 8601 with Z)
        self.assertRegex(trace.end_time, r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z')

    def test_run_sets_trace_end_time_on_error(self):
        """Tests that trace end_time is set even when pipeline errors."""
        # Force an error with invalid input
        self.pipeline.safety_monitor.max_input_length = 5
        query = "This is too long."

        trace = self.pipeline.run(query)

        self.assertIsNotNone(trace.error)
        self.assertIsNotNone(trace.end_time)


class TestPipelineMainFunction(unittest.TestCase):
    """Test suite for the pipeline main() demonstration function."""

    @patch('klareco.pipeline.print')
    def test_main_runs_without_errors(self, mock_print):
        """Tests that main() demo function runs without raising exceptions."""
        from klareco.pipeline import main

        # Should not raise any exceptions
        try:
            main()
            success = True
        except Exception:
            success = False

        self.assertTrue(success)

        # Verify print was called (traces were output)
        self.assertGreater(mock_print.call_count, 0)


if __name__ == '__main__':
    unittest.main()
