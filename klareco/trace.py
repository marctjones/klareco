"""
The Execution Trace and Traceability Subsystem.

This module defines the structure for logging the AI's "thought process".
"""
import json
import uuid
from datetime import datetime, timezone

class ExecutionTrace:
    """
    Represents a single, complete trace of an AI's reasoning process.
    """
    def __init__(self, initial_query: str):
        self.trace_id = str(uuid.uuid4())
        # Using timezone-aware datetime (Python 3.12+ best practice)
        # datetime.utcnow() is deprecated as of Python 3.12
        self.start_time = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        self.end_time = None
        self.initial_query = initial_query
        self.steps = []
        self.final_response = None
        self.error = None

    def add_step(self, step_name: str, inputs: dict, outputs: dict, description: str = None):
        """
        Adds a step to the execution trace.

        Args:
            step_name: The name of the component or action (e.g., "FrontDoor", "Parser").
            inputs: A dictionary of inputs to the step.
            outputs: A dictionary of outputs from the step.
            description: An optional natural language description of the step.
        """
        step = {
            "step_id": len(self.steps) + 1,
            "name": step_name,
            "timestamp": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            "inputs": inputs,
            "outputs": outputs,
        }
        if description:
            step["description"] = description
        self.steps.append(step)

    def set_final_response(self, response: str):
        """Sets the final response and concludes the trace."""
        self.final_response = response
        self.end_time = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')

    def set_error(self, error_message: str):
        """Records an error and concludes the trace."""
        self.error = error_message
        self.end_time = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')

    def to_json(self, indent=2):
        """Serializes the trace to a JSON string."""
        return json.dumps(self, default=lambda o: o.__dict__, indent=indent, ensure_ascii=False)

if __name__ == '__main__':
    # Example Usage
    from klareco.front_door import FrontDoor
    from klareco.parser import parse
    from klareco.deparser import deparse

    query = "La hundo amas la katon."
    trace = ExecutionTrace(initial_query=query)

    try:
        # Step 1: Front Door (although in this case, it does nothing)
        front_door = FrontDoor()
        lang, processed_text = front_door.process(query)
        trace.add_step(
            "FrontDoor",
            inputs={"text": query},
            outputs={"original_lang": lang, "processed_text": processed_text},
            description="Identified language and translated to internal standard (Esperanto)."
        )

        # Step 2: Parser
        ast = parse(processed_text)
        trace.add_step(
            "Parser",
            inputs={"text": processed_text},
            outputs={"ast": ast},
            description="Parsed Esperanto text into an Abstract Syntax Tree."
        )

        # --- Imagine many complex AI logic steps here ---
        # For this example, we'll just deparse the same AST.
        
        # Step 3: Deparser
        final_text = deparse(ast)
        trace.add_step(
            "Deparser",
            inputs={"ast": ast},
            outputs={"text": final_text},
            description="Converted the final AST back into an Esperanto sentence."
        )

        trace.set_final_response(final_text)

    except Exception as e:
        trace.set_error(str(e))

    # Print the final trace
    print(trace.to_json())
