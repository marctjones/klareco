
import argparse
from transformers import MarianMTModel, MarianTokenizer

def translate(text, model_name):
    """
    Translates a given text using a specified MarianMT model.
    """
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    
    translated_tokens = model.generate(**tokenizer(text, return_tensors="pt", padding=True))
    return tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

def main():
    """
    Main function to handle command-line arguments and perform translation.
    """
    parser = argparse.ArgumentParser(description="Translate text between English and Esperanto.")
    parser.add_argument('text', type=str, help="The text to translate.")
    parser.add_argument('--lang', type=str, choices=['en-eo', 'eo-en'], required=True, 
                        help="The translation direction (en-eo or eo-en).")
    
    args = parser.parse_args()
    
    if args.lang == 'en-eo':
        model_name = 'Helsinki-NLP/opus-mt-en-eo'
    else: # eo-en
        model_name = 'Helsinki-NLP/opus-mt-eo-en'
        
    translated_text = translate(args.text, model_name)
    print(translated_text)

if __name__ == "__main__":
    main()
