"""
The main processing pipeline for Klareco.
"""
import logging
from .logging_config import setup_logging
from .trace import ExecutionTrace
from .front_door import FrontDoor
from .parser import parse as parse_esperanto
from .safety import SafetyMonitor
from .orchestrator import create_orchestrator_with_experts

# Setup logging at the module level
setup_logging()

class KlarecoPipeline:
    """
    Orchestrates the full processing of a query, from raw text to AST,
    with complete traceability.
    """
    def __init__(self, use_orchestrator=True):
        """
        Initialize the pipeline.

        Args:
            use_orchestrator: If True, use the expert system orchestrator.
                             If False, use legacy intent classifier (for testing).
        """
        self.front_door = FrontDoor()
        self.safety_monitor = SafetyMonitor()
        self.use_orchestrator = use_orchestrator

        if use_orchestrator:
            self.orchestrator = create_orchestrator_with_experts()
            logging.info("KlarecoPipeline initialized with Expert System Orchestrator.")
        else:
            # Legacy mode for backwards compatibility
            from .intent_classifier import classify_intent
            from .responder import respond_to_intent
            self.classify_intent = classify_intent
            self.respond_to_intent = respond_to_intent
            logging.info("KlarecoPipeline initialized in legacy mode.")

    def run(self, query: str, stop_after: str = None) -> ExecutionTrace:
        """
        Runs the full pipeline on a given query.

        Args:
            query: The raw input text from the user.
            stop_after: If specified, the pipeline will stop after this step.

        Returns:
            A complete ExecutionTrace object.
        """
        logging.info(f"Starting pipeline run for query: '{query}'")
        trace = ExecutionTrace(initial_query=query)
        
        try:
            # Step 1: Input Safety Check
            logging.info("Step 1: SafetyMonitor - Checking input length.")
            self.safety_monitor.check_input_length(query)
            trace.add_step(
                "SafetyMonitor",
                inputs={"text_length": len(query)},
                outputs={"status": "OK"},
                description="Checked input length against policy."
            )
            if stop_after == "SafetyMonitor": return trace

            # Step 2: Front Door
            logging.info("Step 2: FrontDoor - Processing input text.")
            lang, processed_text = self.front_door.process(query)
            trace.add_step(
                "FrontDoor",
                inputs={"text": query},
                outputs={"original_lang": lang, "processed_text": processed_text},
                description="Identified language and translated to internal standard (Esperanto)."
            )
            if stop_after == "FrontDoor": return trace

            # Step 3: Parser
            logging.info("Step 3: Parser - Parsing Esperanto text to AST.")
            ast = parse_esperanto(processed_text)
            trace.add_step(
                "Parser",
                inputs={"text": processed_text},
                outputs={"ast": ast},
                description="Parsed Esperanto text into an Abstract Syntax Tree."
            )
            if stop_after == "Parser": return trace

            # Step 4: AST Safety Check
            logging.info("Step 4: SafetyMonitor - Checking AST complexity.")
            self.safety_monitor.check_ast_complexity(ast)
            node_count = self.safety_monitor._count_ast_nodes(ast)
            trace.add_step(
                "SafetyMonitor",
                inputs={"ast": ast},
                outputs={"status": "OK", "node_count": node_count},
                description="Checked AST complexity against policy."
            )
            if stop_after == "SafetyMonitor_AST": return trace # Differentiate from input safety check
            
            if self.use_orchestrator:
                # Step 5: Orchestrator (Intent Classification + Expert Routing)
                logging.info("Step 5: Orchestrator - Routing to expert system.")
                expert_response = self.orchestrator.route(ast)

                trace.add_step(
                    "Orchestrator",
                    inputs={"ast": ast},
                    outputs={
                        "intent": expert_response.get('intent'),
                        "intent_confidence": expert_response.get('intent_confidence'),
                        "expert": expert_response.get('expert'),
                        "confidence": expert_response.get('confidence'),
                        "answer": expert_response.get('answer'),
                        "full_response": expert_response
                    },
                    description=f"Routed to {expert_response.get('expert', 'no expert')} via intent '{expert_response.get('intent', 'unknown')}'"
                )
                if stop_after == "Orchestrator": return trace

                # Extract response text
                response_text = expert_response.get('answer', 'Neniu respondo.')

            else:
                # Legacy mode: Step 5 and 6 separate
                logging.info("Step 5: IntentClassifier - Classifying intent from AST.")
                intent = self.classify_intent(ast)
                trace.add_step(
                    "IntentClassifier",
                    inputs={"ast": ast},
                    outputs={"intent": intent},
                    description="Classified the user's intent from the AST."
                )
                if stop_after == "IntentClassifier": return trace

                logging.info("Step 6: Responder - Generating response.")
                response_text = self.respond_to_intent(intent, ast)
                trace.add_step(
                    "Responder",
                    inputs={"intent": intent, "ast": ast},
                    outputs={"response_text": response_text},
                    description="Generated a symbolic response based on the intent."
                )
                if stop_after == "Responder": return trace

            # Set the final response
            logging.info("Pipeline run completed successfully.")
            trace.set_final_response(response_text)

        except Exception as e:
            # Ensure any failure is logged to the trace
            logging.error(f"Pipeline failed with error: {e}", exc_info=True)
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
