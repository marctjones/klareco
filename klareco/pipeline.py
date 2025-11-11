"""
The main processing pipeline for Klareco.
"""
from .trace import ExecutionTrace
from .front_door import FrontDoor
from .parser import parse as parse_esperanto
from .intent_classifier import classify_intent
from .responder import respond_to_intent
from .safety import SafetyMonitor

class KlarecoPipeline:
    """
    Orchestrates the full processing of a query, from raw text to AST,
    with complete traceability.
    """
    def __init__(self):
        self.front_door = FrontDoor()
        self.safety_monitor = SafetyMonitor()

    def run(self, query: str, stop_after: str = None) -> ExecutionTrace:
        """
        Runs the full pipeline on a given query.

        Args:
            query: The raw input text from the user.
            stop_after: If specified, the pipeline will stop after this step.

        Returns:
            A complete ExecutionTrace object.
        """
        trace = ExecutionTrace(initial_query=query)
        
        try:
            # Step 1: Input Safety Check
            self.safety_monitor.check_input_length(query)
            trace.add_step(
                "SafetyMonitor",
                inputs={"text_length": len(query)},
                outputs={"status": "OK"},
                description="Checked input length against policy."
            )
            if stop_after == "SafetyMonitor": return trace

            # Step 2: Front Door
            lang, processed_text = self.front_door.process(query)
            trace.add_step(
                "FrontDoor",
                inputs={"text": query},
                outputs={"original_lang": lang, "processed_text": processed_text},
                description="Identified language and translated to internal standard (Esperanto)."
            )
            if stop_after == "FrontDoor": return trace

            # Step 3: Parser
            ast = parse_esperanto(processed_text)
            trace.add_step(
                "Parser",
                inputs={"text": processed_text},
                outputs={"ast": ast},
                description="Parsed Esperanto text into an Abstract Syntax Tree."
            )
            if stop_after == "Parser": return trace

            # Step 4: AST Safety Check
            self.safety_monitor.check_ast_complexity(ast)
            node_count = self.safety_monitor._count_ast_nodes(ast)
            trace.add_step(
                "SafetyMonitor",
                inputs={"ast": ast},
                outputs={"status": "OK", "node_count": node_count},
                description="Checked AST complexity against policy."
            )
            if stop_after == "SafetyMonitor_AST": return trace # Differentiate from input safety check
            
            # Step 5: Intent Classifier
            intent = classify_intent(ast)
            trace.add_step(
                "IntentClassifier",
                inputs={"ast": ast},
                outputs={"intent": intent},
                description="Classified the user's intent from the AST."
            )
            if stop_after == "IntentClassifier": return trace

            # Step 6: Responder
            response_text = respond_to_intent(intent, ast)
            trace.add_step(
                "Responder",
                inputs={"intent": intent, "ast": ast},
                outputs={"response_text": response_text},
                description="Generated a symbolic response based on the intent."
            )
            if stop_after == "Responder": return trace

            # Set the final response
            trace.set_final_response(response_text)

        except Exception as e:
            # Ensure any failure is logged to the trace
            trace.set_error(str(e))

        return trace

def main():
    """Main function for demonstrating the pipeline."""
    pipeline = KlarecoPipeline()
    
    # --- Example 1: English Query ---
    query_en = "The dog loves the cat."
    trace_en = pipeline.run(query_en)
    print("--- English Query Trace ---")
    print(trace_en.to_json())

    # --- Example 2: Esperanto Query ---
    query_eo = "mi vidas la hundon."
    trace_eo = pipeline.run(query_eo)
    print("\n--- Esperanto Query Trace ---")
    print(trace_eo.to_json())

    # --- Example 3: Failing Query (Input Length) ---
    pipeline.safety_monitor.max_input_length = 10
    query_fail_len = "This is too long."
    trace_fail_len = pipeline.run(query_fail_len)
    print("\n--- Failing Query (Length) Trace ---")
    print(trace_fail_len.to_json())

if __name__ == '__main__':
    main()
