"""
Language identification services using lingua-language-detector.
Pure Python implementation with Esperanto support.
"""
from lingua import LanguageDetectorBuilder, Language, IsoCode639_1

# Build the language detector once at module load time
# Using all languages for maximum flexibility
# This takes a moment to initialize but subsequent calls are fast
_detector = LanguageDetectorBuilder.from_all_languages().build()

# Mapping from lingua's Language enum to ISO 639-1 codes
# lingua provides this via the iso_code_639_1 property
def _get_iso_code(language: Language) -> str:
    """Convert lingua Language enum to ISO 639-1 code string."""
    iso_code = language.iso_code_639_1
    return iso_code.name.lower()


def identify_language(text: str) -> str | None:
    """
    Identifies the language of a given text using lingua.

    Args:
        text: The text to analyze.

    Returns:
        The ISO 639-1 language code (e.g., 'en', 'eo') or None if detection fails.
    """
    if not isinstance(text, str):
        raise TypeError("Input must be a string.")

    if not text.strip():
        return None

    # lingua's detect_language_of returns Language enum or None
    detected_language = _detector.detect_language_of(text)

    if detected_language is None:
        return None

    return _get_iso_code(detected_language)


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
