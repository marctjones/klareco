"""
Language identification services using fasttext.
"""
import fasttext
import os

# Load the model
# The model file is expected to be at models/lid.176.bin
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'lid.176.bin')

# Check if model exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"FastText model not found at {MODEL_PATH}. "
        "Please run scripts/download_fasttext_model.py to download it."
    )

# Pre-load the model to avoid loading it on every call
try:
    model = fasttext.load_model(MODEL_PATH)
except ValueError as e:
    # Provide a more helpful error message if the model is corrupted
    raise RuntimeError(
        f"Failed to load FastText model at {MODEL_PATH}. "
        f"The model file might be corrupted. Please try downloading it again. Original error: {e}"
    )


def identify_language(text: str) -> str | None:
    """
    Identifies the language of a given text using fasttext.

    Args:
        text: The text to analyze.

    Returns:
        The ISO 639-1 language code (e.g., 'en', 'eo') or None if detection fails.
    """
    if not isinstance(text, str):
        raise TypeError("Input must be a string.")
    
    if not text.strip():
        return None

    # fasttext predicts a label like '__label__en'
    # We need to extract the 'en' part.
    # The predict method returns a tuple of lists: (labels, probabilities)
    predictions = model.predict(text.replace("\n", " "), k=1)
    if predictions[0]:
        label = predictions[0][0]
        lang_code = label.replace('__label__', '')
        return lang_code
    return None

if __name__ == '__main__':
    # Example usage
    texts_to_test = [
        "This is an English sentence.",
        "Ĉi tio estas frazo en Esperanto.",
        "Ceci est une phrase en français.",
        "Esto es una frase en español.",
        "12345",
        "",
        "    ",
    ]
    for t in texts_to_test:
        try:
            lang = identify_language(t)
            print(f"'{t}' -> {lang}")
        except TypeError as e:
            print(f"'{t}' -> Error: {e}")
    
    # Test non-string input
    try:
        identify_language(123)
    except TypeError as e:
        print(f"'123' -> Error: {e}")
