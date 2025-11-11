"""
Tests for the ExecutionTrace.
"""
import unittest
import json
from klareco.trace import ExecutionTrace

class TestExecutionTrace(unittest.TestCase):

    def test_trace_initialization(self):
        """Tests that the trace is initialized correctly."""
        query = "Test query"
        trace = ExecutionTrace(initial_query=query)
        self.assertEqual(trace.initial_query, query)
        self.assertIsNotNone(trace.trace_id)
        self.assertIsNotNone(trace.start_time)
        self.assertIsNone(trace.end_time)
        self.assertEqual(trace.steps, [])
        self.assertIsNone(trace.final_response)
        self.assertIsNone(trace.error)

    def test_add_step(self):
        """Tests adding a step to the trace."""
        trace = ExecutionTrace("query")
        trace.add_step(
            "TestStep",
            inputs={"in": 1},
            outputs={"out": 2},
            description="A test step."
        )
        self.assertEqual(len(trace.steps), 1)
        step = trace.steps[0]
        self.assertEqual(step["name"], "TestStep")
        self.assertEqual(step["inputs"], {"in": 1})
        self.assertEqual(step["outputs"], {"out": 2})
        self.assertEqual(step["description"], "A test step.")
        self.assertIsNotNone(step["timestamp"])

    def test_set_final_response(self):
        """Tests setting the final response."""
        trace = ExecutionTrace("query")
        response = "Final answer."
        trace.set_final_response(response)
        self.assertEqual(trace.final_response, response)
        self.assertIsNotNone(trace.end_time)
        self.assertIsNone(trace.error)

    def test_set_error(self):
        """Tests setting an error."""
        trace = ExecutionTrace("query")
        error_msg = "Something went wrong."
        trace.set_error(error_msg)
        self.assertEqual(trace.error, error_msg)
        self.assertIsNotNone(trace.end_time)
        self.assertIsNone(trace.final_response)

    def test_to_json(self):
        """Tests serialization to JSON."""
        trace = ExecutionTrace("query")
        trace.add_step("TestStep", inputs={}, outputs={})
        trace.set_final_response("done")
        
        json_output = trace.to_json()
        self.assertIsInstance(json_output, str)
        
        data = json.loads(json_output)
        self.assertEqual(data['trace_id'], trace.trace_id)
        self.assertEqual(data['initial_query'], trace.initial_query)
        self.assertEqual(len(data['steps']), 1)
        self.assertEqual(data['final_response'], trace.final_response)

if __name__ == '__main__':
    unittest.main()
