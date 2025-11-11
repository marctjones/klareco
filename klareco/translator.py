"""
Translation services between languages.
"""
from transformers import MarianMTModel, MarianTokenizer

class TranslationService:
    """
    A service for translating text between languages using MarianMT models.
    """
    def __init__(self):
        self.models = {}
        self.tokenizers = {}

    def _load_model(self, model_name: str):
        """Loads a translation model and tokenizer."""
        if model_name not in self.models:
            try:
                self.tokenizers[model_name] = MarianTokenizer.from_pretrained(model_name)
                self.models[model_name] = MarianMTModel.from_pretrained(model_name)
            except OSError:
                raise ValueError(f"Model '{model_name}' not found. Check model name.")
    
    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Translates a text from a source language to a target language.

        Args:
            text: The text to translate.
            source_lang: The source language code (e.g., 'en').
            target_lang: The target language code (e.g., 'eo').

        Returns:
            The translated text.
        """
        if not isinstance(text, str):
            raise TypeError("Input text must be a string.")
        if not text.strip():
            return ""

        model_name = f'Helsinki-NLP/opus-mt-{source_lang}-{target_lang}'
        self._load_model(model_name)

        tokenizer = self.tokenizers[model_name]
        model = self.models[model_name]

        # The tokenizer expects a list of sentences.
        # For simplicity, we'll treat the whole text as one sentence.
        # For production, splitting into sentences might be better.
        inputs = tokenizer([text], return_tensors="pt", padding=True)
        
        translated_ids = model.generate(**inputs)
        
        translated_text = tokenizer.batch_decode(translated_ids, skip_special_tokens=True)[0]
        
        return translated_text

if __name__ == '__main__':
    # Example Usage
    translator = TranslationService()

    # English to Esperanto
    en_text = "Hello, how are you?"
    eo_translation = translator.translate(en_text, 'en', 'eo')
    print(f"'{en_text}' (en) -> '{eo_translation}' (eo)")

    # Esperanto to English
    eo_text = "Saluton, kiel vi fartas?"
    en_translation = translator.translate(eo_text, 'eo', 'en')
    print(f"'{eo_text}' (eo) -> '{en_translation}' (en)")
