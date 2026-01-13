#!/usr/bin/env python3
"""
Inline Esperanto-to-English translator for text streams.

Detects Esperanto text and either:
- Adds English translation in parentheses: "Mi amas vin (I love you)"
- Replaces Esperanto with English: "I love you"

Backends:
- 'native': Uses Klareco parser + ReVo dictionary (fast, offline, lexical)
- 'neural': Uses Helsinki-NLP/opus-mt-eo-en (better quality, requires download)

Usage:
    cat file.txt | python scripts/translate_eo_inline.py
    echo "Mi estas programisto" | python scripts/translate_eo_inline.py --backend neural
    python scripts/translate_eo_inline.py --replace < input.txt
    python scripts/translate_eo_inline.py --backend native < input.txt

Requirements:
    # For native backend (already installed):
    - Klareco parser
    - ReVo dictionary

    # For neural backend:
    pip install transformers sentencepiece
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class NativeTranslator:
    """Translate using Klareco parser + ReVo dictionary."""

    def __init__(self):
        """Initialize native translator with parser and dictionary."""
        from klareco.parser import parse
        import json

        self.parse = parse

        # Load ReVo dictionary
        revo_path = Path(__file__).parent.parent / 'data' / 'raw' / 'eo' / 'dictionaries' / 'revo' / 'revo_definitions.json'
        if revo_path.exists():
            with open(revo_path) as f:
                self.revo = json.load(f)
        else:
            print(f"Warning: ReVo dictionary not found at {revo_path}", file=sys.stderr)
            self.revo = {}

        print(f"Native translator loaded ({len(self.revo)} ReVo entries)", file=sys.stderr)

    def translate_word(self, word: str) -> Optional[str]:
        """Translate a single Esperanto word to English."""
        # Parse the word to get root
        try:
            ast = self.parse(word)

            # Extract root from parsed word
            root = None
            if ast.get('tipo') == 'frazo':
                # Try to get root from subject/verb/object
                for key in ['subjekto', 'verbo', 'objekto']:
                    node = ast.get(key)
                    if node and isinstance(node, dict):
                        if node.get('tipo') == 'vorto':
                            root = node.get('radiko')
                        elif node.get('tipo') == 'vortgrupo':
                            kerno = node.get('kerno', {})
                            root = kerno.get('radiko')
                        if root:
                            break

            # Look up root in ReVo
            if root and root in self.revo:
                entry = self.revo[root]
                # Get first English definition
                if isinstance(entry, dict) and 'gloss' in entry:
                    return entry['gloss']
                elif isinstance(entry, str):
                    return entry

            return None

        except Exception as e:
            return None

    def translate(self, text: str) -> str:
        """Translate Esperanto text to English (word by word with dictionary)."""
        words = text.split()
        translated = []

        for word in words:
            # Remove punctuation for translation
            clean_word = word.strip('.,!?;:"\'')
            punct = word[len(clean_word):] if len(word) > len(clean_word) else ''

            trans = self.translate_word(clean_word)
            if trans:
                translated.append(trans + punct)
            else:
                translated.append(word)  # Keep original if no translation

        return ' '.join(translated)


class NeuralTranslator:
    """Translate using Helsinki-NLP/opus-mt-eo-en."""

    def __init__(self):
        """Initialize neural translator."""
        try:
            from transformers import MarianMTModel, MarianTokenizer

            model_name = "Helsinki-NLP/opus-mt-eo-en"
            print(f"Loading {model_name}...", file=sys.stderr)

            self.tokenizer = MarianTokenizer.from_pretrained(model_name)
            self.model = MarianMTModel.from_pretrained(model_name)

            print(f"Neural translator loaded", file=sys.stderr)

        except ImportError:
            print("Error: transformers not installed. Install with: pip install transformers sentencepiece", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error loading neural translator: {e}", file=sys.stderr)
            sys.exit(1)

    def translate(self, text: str) -> str:
        """Translate Esperanto text to English using neural model."""
        if not text.strip():
            return text

        try:
            # Tokenize
            tokens = self.tokenizer([text], return_tensors="pt", padding=True)

            # Translate
            translated = self.model.generate(**tokens)

            # Decode
            result = self.tokenizer.decode(translated[0], skip_special_tokens=True)

            return result

        except Exception as e:
            print(f"Translation error: {e}", file=sys.stderr)
            return text


def is_esperanto(text: str) -> bool:
    """
    Detect if text is Esperanto.

    Checks for Esperanto-specific characters and common words.
    """
    if not text.strip():
        return False

    # Check for Esperanto-specific characters
    esperanto_chars = 'ĉĝĥĵŝŭĈĜĤĴŜŬ'
    if any(c in text for c in esperanto_chars):
        return True

    # Check for common Esperanto words
    esperanto_words = {
        'estas', 'kaj', 'la', 'mi', 'vi', 'li', 'ŝi', 'ĝi', 'ni', 'ili',
        'amas', 'vidas', 'parolas', 'legas', 'skribas',
        'hundo', 'kato', 'domo', 'tago', 'nokto',
        'bona', 'bela', 'granda', 'malgranda',
        'ĉu', 'kie', 'kio', 'kiu', 'kial', 'kiam', 'kiel',
        'tio', 'tiu', 'tie', 'tiam', 'tiel',
    }
    words = set(text.lower().split())
    if words & esperanto_words:
        return True

    return False


def is_likely_english(text: str) -> bool:
    """Quick check if text is likely English (to avoid false positives)."""
    english_indicators = {
        'the', 'is', 'are', 'was', 'were', 'have', 'has', 'had',
        'will', 'would', 'should', 'could', 'can', 'may', 'might',
        'this', 'that', 'these', 'those', 'what', 'where', 'when', 'why', 'how',
        'and', 'or', 'but', 'if', 'then',
    }
    words = set(text.lower().split())
    return len(words & english_indicators) >= 1


def split_into_segments(text: str) -> List[Tuple[str, bool]]:
    """
    Split text into segments of (text, is_esperanto).

    Attempts to segment by sentences/phrases while preserving structure.
    """
    # Split on sentence boundaries while preserving delimiters
    sentence_pattern = r'([.!?;]+\s*|\n+)'
    parts = re.split(sentence_pattern, text)

    segments = []
    for part in parts:
        if part.strip():
            # Skip if clearly English
            if is_likely_english(part):
                segments.append((part, False))
            else:
                is_eo = is_esperanto(part)
                segments.append((part, is_eo))

    return segments


def process_text(text: str, translator, mode: str = 'append') -> str:
    """
    Process text, translating Esperanto segments.

    Args:
        text: Input text
        translator: Translator instance (NativeTranslator or NeuralTranslator)
        mode: 'append' (add translation in parentheses) or 'replace' (replace with translation)

    Returns:
        Processed text
    """
    segments = split_into_segments(text)
    result = []

    for segment_text, is_eo in segments:
        if is_eo:
            translation = translator.translate(segment_text.strip())

            if mode == 'replace':
                result.append(translation)
            else:  # append
                result.append(f"{segment_text.strip()} ({translation})")
        else:
            result.append(segment_text)

    return ''.join(result)


def main():
    parser = argparse.ArgumentParser(
        description='Inline Esperanto-to-English translator for text streams',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Backends:
  native  - Uses Klareco parser + ReVo dictionary (fast, offline, lexical)
  neural  - Uses Helsinki-NLP/opus-mt-eo-en (better quality, requires download)

Examples:
  cat file.txt | %(prog)s
  echo "Mi estas programisto" | %(prog)s --backend neural
  %(prog)s --replace --backend native < input.txt
  %(prog)s --mode append --backend neural < input.txt

Requirements:
  native: Klareco parser + ReVo dictionary (already installed)
  neural: pip install transformers sentencepiece
        """
    )
    parser.add_argument('--backend', choices=['native', 'neural'], default='native',
                       help='Translation backend (default: native)')
    parser.add_argument('--mode', choices=['append', 'replace'], default='append',
                       help='append: add translation in parentheses, replace: replace with translation')
    parser.add_argument('--replace', action='store_true',
                       help='Shortcut for --mode replace')
    parser.add_argument('-i', '--input', type=argparse.FileType('r'), default=sys.stdin,
                       help='Input file (default: stdin)')
    parser.add_argument('-o', '--output', type=argparse.FileType('w'), default=sys.stdout,
                       help='Output file (default: stdout)')

    args = parser.parse_args()

    # Handle --replace shortcut
    if args.replace:
        args.mode = 'replace'

    # Initialize translator based on backend
    if args.backend == 'native':
        translator = NativeTranslator()
    else:  # neural
        translator = NeuralTranslator()

    # Process input
    for line in args.input:
        processed = process_text(line, translator, args.mode)
        args.output.write(processed)
        args.output.flush()


if __name__ == '__main__':
    main()
