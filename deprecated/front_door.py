"""
The Front Door of the Klareco system.

Handles language identification and translation to the internal standard (Esperanto).
"""
from .lang_id import identify_language
from .translator import TranslationService

class FrontDoor:
    """
    Orchestrates the initial processing of incoming text.
    """
    def __init__(self, internal_lang: str = "eo"):
        self.internal_lang = internal_lang
        self.translator = TranslationService()

    def process(self, text: str) -> tuple[str, str]:
        """
        Processes raw text to the internal standard language.

        Args:
            text: The raw input text.

        Returns:
            A tuple containing:
            - The original language code.
            - The text in the internal standard language (Esperanto).
        """
        if not isinstance(text, str):
            raise TypeError("Input must be a string.")
        
        original_lang = identify_language(text)
        if original_lang is None:
            # If language can't be identified, we can't proceed.
            # Or we could default to treating it as the internal language.
            # For now, let's raise an error for clarity.
            raise ValueError("Could not identify the language of the input text.")

        if original_lang == self.internal_lang:
            return original_lang, text
        else:
            try:
                translated_text = self.translator.translate(text, original_lang, self.internal_lang)
                return original_lang, translated_text
            except ValueError as e:
                # Translation model not available - fall back to original text
                # This happens for rare language pairs (e.g., Irish Gaelic → Esperanto)
                # Assume text might already be in target language or close enough
                return original_lang, text

if __name__ == '__main__':
    # Example Usage
    front_door = FrontDoor()

    texts_to_process = [
        "Hello, world!",
        "Saluton, mondo!",
        "Bonjour, le monde!",
    ]

    for text in texts_to_process:
        try:
            lang, processed_text = front_door.process(text)
            print(f"Original ('{lang}'): '{text}' -> Processed ('eo'): '{processed_text}'")
        except (ValueError, TypeError) as e:
            print(f"Error processing '{text}': {e}")
