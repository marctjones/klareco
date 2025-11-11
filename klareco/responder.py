"""
Symbolic-Only Responders.

This module contains functions that generate responses based on classified intents
and the morpheme-based AST.
"""
from klareco.deparser import deparse

def respond_to_intent(intent: str, ast: dict) -> str:
    """
    Selects the appropriate responder based on the intent.

    Args:
        intent: The classified intent.
        ast: The AST of the original query.

    Returns:
        A string response.
    """
    if intent == "SimpleStatement":
        return respond_simple_statement(ast)
    else:
        return "Mi ne komprenas vian intencon." # "I don't understand your intent."

def respond_simple_statement(ast: dict) -> str:
    """
    Responds to a simple statement by echoing it back in a conversational way.

    Args:
        ast: The AST of the simple statement.

    Returns:
        An acknowledgement string.
    """
    try:
        # Use the deparser to reconstruct the original sentence from the AST
        original_sentence = deparse(ast)
        response = f"Vi diras, ke {original_sentence.lower()}." # "You say that [sentence]."
        # Remove the trailing period from the deparsed sentence if it's already there
        if response.endswith('..'):
            response = response[:-1]
        return response
    except Exception as e:
        return f"Mi ne povis respondi al via deklaro pro eraro: {e}" # "I could not respond to your statement due to an error."

if __name__ == '__main__':
    # Example Usage
    from klareco.parser import parse
    import json

    # Test Case 1: Simple Statement
    text_1 = "La hundo amas la katon."
    ast_1 = parse(text_1)
    intent_1 = "SimpleStatement"
    response_1 = respond_to_intent(intent_1, ast_1)
    print(f"Text: '{text_1}'\nAST: {json.dumps(ast_1, indent=2, ensure_ascii=False)}\nIntent: {intent_1}\nResponse: {response_1}\n")

    # Test Case 2: Simple Statement with pronoun
    text_2 = "mi vidas la hundon."
    ast_2 = parse(text_2)
    intent_2 = "SimpleStatement"
    response_2 = respond_to_intent(intent_2, ast_2)
    print(f"Text: '{text_2}'\nAST: {json.dumps(ast_2, indent=2, ensure_ascii=False)}\nIntent: {intent_2}\nResponse: {response_2}\n")

    # Test Case 3: Unknown intent
    unknown_intent = "UnknownIntent"
    response_unknown = respond_to_intent(unknown_intent, {})
    print(f"Intent: {unknown_intent}\nResponse: {response_unknown}\n")