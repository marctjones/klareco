"""
Level 1 Symbolic Intent Classifier.

Classifies the user's intent based on the morpheme-based AST.
"""

def classify_intent(ast: dict) -> str:
    """
    Classifies the user's intent based on the morpheme-based AST.

    This is a simple, rule-based classifier that inspects the morphemes.

    Args:
        ast: The Abstract Syntax Tree of the user's query.

    Returns:
        A string representing the classified intent.
    """
    if ast.get('type') != 'sentence':
        return "Unknown"

    subject = ast.get('subject')
    verb = ast.get('verb')
    obj = ast.get('object')

    # Rule for SimpleStatement:
    # Check if we have a valid subject, verb, and object, all of type 'word'.
    # This is a basic check for a declarative sentence structure.
    if (subject and subject.get('type') == 'word' and subject.get('root')
        and verb and verb.get('type') == 'word' and verb.get('root') and verb.get('endings')
        and obj and obj.get('type') == 'word' and obj.get('root')):
        
        # Further checks could be added here, e.g., verb tense, presence of accusative on object.
        # For now, this is sufficient for a basic statement.
        return "SimpleStatement"

    # Placeholder for other intents, e.g., Questions
    # if verb and verb.get('root') == 'demand': # If we had a root for 'ask'
    #     return "Question"

    return "Unknown"

if __name__ == '__main__':
    # Example Usage with the new AST structure
    from klareco.parser import parse
    import json

    # Test Case 1: Simple Statement
    text_1 = "La hundo amas la katon."
    ast_1 = parse(text_1)
    intent_1 = classify_intent(ast_1)
    print(f"AST 1 -> Intent: {intent_1}")

    # Test Case 2: Simple Statement with pronoun
    text_2 = "mi vidas la hundon."
    ast_2 = parse(text_2)
    intent_2 = classify_intent(ast_2)
    print(f"AST 2 -> Intent: {intent_2}")

    # Test Case 3: Unknown structure
    ast_3 = {"type": "invalid_ast"}
    intent_3 = classify_intent(ast_3)
    print(f"AST 3 -> Intent: {intent_3}")