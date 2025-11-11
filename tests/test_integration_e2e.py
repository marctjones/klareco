"""
End-to-End Integration Tests for the Klareco Pipeline.

This test suite validates the complete pipeline workflow from input to output,
testing real sentences through all components.
"""
import unittest
import json
import os
from klareco.pipeline import KlarecoPipeline


class TestPipelineIntegrationE2E(unittest.TestCase):
    """End-to-end integration tests for the full pipeline."""

    @classmethod
    def setUpClass(cls):
        """Load test corpus once for all tests."""
        corpus_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'test_corpus.json')

        if os.path.exists(corpus_path):
            with open(corpus_path, 'r', encoding='utf-8') as f:
                cls.test_corpus = json.load(f)
        else:
            cls.test_corpus = [
                "La hundo vidas la katon.",
                "Mi amas la grandan hundon.",
                "La bona kato manĝas.",
            ]

    def setUp(self):
        """Initialize pipeline for each test."""
        self.pipeline = KlarecoPipeline()

    def test_e2e_simple_sentence_completes_successfully(self):
        """Tests that a simple Esperanto sentence completes full pipeline."""
        query = "La hundo vidas la katon."

        trace = self.pipeline.run(query)

        # Verify successful completion
        self.assertIsNone(trace.error)
        self.assertIsNotNone(trace.final_response)
        self.assertEqual(trace.initial_query, query)

        # Verify all pipeline stages executed
        step_names = [step['name'] for step in trace.steps]
        self.assertIn("FrontDoor", step_names)
        self.assertIn("Parser", step_names)
        self.assertIn("IntentClassifier", step_names)
        self.assertIn("Responder", step_names)

    def test_e2e_pronoun_subject_sentence_completes_successfully(self):
        """Tests that sentences with pronoun subjects work end-to-end."""
        query = "mi vidas la hundon."

        trace = self.pipeline.run(query)

        self.assertIsNone(trace.error)
        self.assertIsNotNone(trace.final_response)

        # Verify Parser correctly handled pronoun
        parser_step = None
        for step in trace.steps:
            if step['name'] == 'Parser':
                parser_step = step
                break

        self.assertIsNotNone(parser_step)
        ast = parser_step['outputs']['ast']

        # Verify pronoun was parsed as subject
        self.assertIsNotNone(ast.get('subjekto'))
        self.assertEqual(ast['subjekto']['kerno']['vortspeco'], 'pronomo')

    def test_e2e_complex_sentence_with_adjectives_completes(self):
        """Tests that complex sentences with multiple adjectives complete."""
        query = "Malgrandaj hundoj vidas la grandan katon."

        trace = self.pipeline.run(query)

        self.assertIsNone(trace.error)
        self.assertIsNotNone(trace.final_response)

        # Verify Parser handled adjective agreement
        parser_step = None
        for step in trace.steps:
            if step['name'] == 'Parser':
                parser_step = step
                break

        self.assertIsNotNone(parser_step)
        ast = parser_step['outputs']['ast']

        # Verify both subject and object have modifiers
        self.assertIsNotNone(ast.get('subjekto'))
        self.assertIsNotNone(ast.get('objekto'))
        self.assertGreater(len(ast['subjekto'].get('priskriboj', [])), 0)
        self.assertGreater(len(ast['objekto'].get('priskriboj', [])), 0)

    def test_e2e_english_input_translates_and_processes(self):
        """Tests that English input is translated and processed end-to-end."""
        query = "The dog sees the cat."

        trace = self.pipeline.run(query)

        # Should complete (may succeed or fail depending on translation quality)
        # At minimum, should not raise unhandled exception
        self.assertIsNotNone(trace)

        # Verify FrontDoor was invoked
        step_names = [step['name'] for step in trace.steps]
        self.assertIn("FrontDoor", step_names)

        # If successful, verify translation occurred
        if trace.error is None:
            front_door_step = None
            for step in trace.steps:
                if step['name'] == 'FrontDoor':
                    front_door_step = step
                    break

            self.assertIsNotNone(front_door_step)
            # Language should be detected
            self.assertIn('original_lang', front_door_step['outputs'])

    def test_e2e_article_preservation_through_pipeline(self):
        """Tests that articles are preserved through parsing and deparsing."""
        query = "La hundo amas la katon."

        trace = self.pipeline.run(query)

        self.assertIsNone(trace.error)

        # Get parsed AST
        parser_step = None
        for step in trace.steps:
            if step['name'] == 'Parser':
                parser_step = step
                break

        self.assertIsNotNone(parser_step)
        ast = parser_step['outputs']['ast']

        # Verify articles are tracked in AST
        if ast.get('subjekto'):
            # Should have artikolo field
            self.assertIn('artikolo', ast['subjekto'])
        if ast.get('objekto'):
            self.assertIn('artikolo', ast['objekto'])

    def test_e2e_accusative_case_marking_preserved(self):
        """Tests that accusative case marking is preserved through pipeline."""
        query = "Mi amas hundon."

        trace = self.pipeline.run(query)

        self.assertIsNone(trace.error)

        # Get parsed AST
        parser_step = None
        for step in trace.steps:
            if step['name'] == 'Parser':
                parser_step = step
                break

        self.assertIsNotNone(parser_step)
        ast = parser_step['outputs']['ast']

        # Verify object is in accusative case
        self.assertIsNotNone(ast.get('objekto'))
        self.assertEqual(ast['objekto']['kerno']['kazo'], 'akuzativo')

    def test_e2e_pipeline_trace_includes_timestamps(self):
        """Tests that pipeline trace includes timestamps for all steps."""
        query = "La hundo vidas la katon."

        trace = self.pipeline.run(query)

        # Verify trace has start and end times
        self.assertIsNotNone(trace.start_time)
        self.assertIsNotNone(trace.end_time)

        # Verify each step has timestamp
        for step in trace.steps:
            self.assertIn('timestamp', step)
            self.assertIsNotNone(step['timestamp'])

    def test_e2e_intent_classification_from_ast(self):
        """Tests that intent is classified from parsed AST."""
        query = "La hundo vidas la katon."

        trace = self.pipeline.run(query)

        self.assertIsNone(trace.error)

        # Find IntentClassifier step
        intent_step = None
        for step in trace.steps:
            if step['name'] == 'IntentClassifier':
                intent_step = step
                break

        self.assertIsNotNone(intent_step)
        self.assertIn('intent', intent_step['outputs'])

        # For simple statement with subject-verb-object, should classify as SimpleStatement
        intent = intent_step['outputs']['intent']
        self.assertEqual(intent, 'SimpleStatement')

    def test_e2e_responder_generates_output_from_intent(self):
        """Tests that Responder generates output based on classified intent."""
        query = "La hundo vidas la katon."

        trace = self.pipeline.run(query)

        self.assertIsNone(trace.error)

        # Find Responder step
        responder_step = None
        for step in trace.steps:
            if step['name'] == 'Responder':
                responder_step = step
                break

        self.assertIsNotNone(responder_step)
        self.assertIn('response_text', responder_step['outputs'])

        # Response should not be empty
        response = responder_step['outputs']['response_text']
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)

    def test_e2e_multiple_corpus_sentences(self):
        """Tests that multiple sentences from test corpus process successfully."""
        # Test first 10 sentences from corpus (or all if fewer)
        sentences_to_test = self.test_corpus[:10]

        passed = 0
        failed = 0

        for sentence in sentences_to_test:
            trace = self.pipeline.run(sentence)

            if trace.error:
                failed += 1
            else:
                passed += 1

        # At least 50% should pass (lenient for expanding corpus)
        success_rate = passed / len(sentences_to_test)
        self.assertGreaterEqual(success_rate, 0.5,
            f"Only {passed}/{len(sentences_to_test)} sentences passed ({success_rate:.0%})")


class TestPipelineStopAfterE2E(unittest.TestCase):
    """Integration tests for pipeline stop_after functionality."""

    def setUp(self):
        """Initialize pipeline for each test."""
        self.pipeline = KlarecoPipeline()

    def test_e2e_stop_after_front_door_returns_partial_trace(self):
        """Tests that stopping after FrontDoor returns trace with only FrontDoor step."""
        query = "La hundo vidas la katon."

        trace = self.pipeline.run(query, stop_after="FrontDoor")

        # Should have SafetyMonitor + FrontDoor steps only
        step_names = [step['name'] for step in trace.steps]
        self.assertIn("SafetyMonitor", step_names)
        self.assertIn("FrontDoor", step_names)
        self.assertNotIn("Parser", step_names)

        # Should not have final response (stopped early)
        self.assertIsNone(trace.final_response)

    def test_e2e_stop_after_parser_allows_ast_inspection(self):
        """Tests that stopping after Parser allows AST inspection."""
        query = "La hundo vidas la katon."

        trace = self.pipeline.run(query, stop_after="Parser")

        # Should have Parser step
        step_names = [step['name'] for step in trace.steps]
        self.assertIn("Parser", step_names)
        self.assertNotIn("IntentClassifier", step_names)

        # Get Parser output
        parser_step = None
        for step in trace.steps:
            if step['name'] == 'Parser':
                parser_step = step
                break

        self.assertIsNotNone(parser_step)
        self.assertIn('ast', parser_step['outputs'])

        # AST should be valid
        ast = parser_step['outputs']['ast']
        self.assertEqual(ast['tipo'], 'frazo')


class TestPipelineErrorHandlingE2E(unittest.TestCase):
    """Integration tests for pipeline error handling."""

    def setUp(self):
        """Initialize pipeline for each test."""
        self.pipeline = KlarecoPipeline()

    def test_e2e_invalid_esperanto_records_error_in_trace(self):
        """Tests that invalid Esperanto input records error in trace."""
        query = "xyzabc"  # Not valid Esperanto

        trace = self.pipeline.run(query)

        # Should have error recorded
        self.assertIsNotNone(trace.error)
        self.assertIsNone(trace.final_response)

        # Should have attempted parsing
        step_names = [step['name'] for step in trace.steps]
        self.assertIn("FrontDoor", step_names)

    def test_e2e_very_long_input_blocked_by_safety_monitor(self):
        """Tests that extremely long input is blocked by SafetyMonitor."""
        # Set a low limit
        self.pipeline.safety_monitor.max_input_length = 10

        query = "This is a very long query that exceeds the safety limit."

        trace = self.pipeline.run(query)

        # Should have error
        self.assertIsNotNone(trace.error)
        self.assertIn("exceeds maximum", trace.error)

    def test_e2e_complex_ast_blocked_by_safety_monitor(self):
        """Tests that overly complex AST is blocked by SafetyMonitor."""
        # Set a very low AST complexity limit
        self.pipeline.safety_monitor.max_ast_nodes = 3

        query = "La granda hundo amas la katon."

        trace = self.pipeline.run(query)

        # Should have error
        self.assertIsNotNone(trace.error)
        self.assertIn("exceeds maximum", trace.error)

        # Should have parsed successfully before failing safety check
        step_names = [step['name'] for step in trace.steps]
        self.assertIn("Parser", step_names)


if __name__ == '__main__':
    unittest.main()
