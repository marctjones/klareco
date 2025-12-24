"""
Proper noun dictionary for Esperanto parsing.

Maintains known proper nouns with metadata to improve parse rates.
Two sources:
1. Static: Hand-curated common names (optional)
2. Dynamic: Extracted from corpus (primary)
"""

import json
from pathlib import Path
from typing import Optional, Dict

# Default paths
DEFAULT_DYNAMIC_PATH = Path(__file__).parent.parent / "data" / "proper_nouns_dynamic.json"
DEFAULT_STATIC_PATH = Path(__file__).parent.parent / "data" / "proper_nouns_static.json"

# Singleton instance
_dictionary_instance: Optional["ProperNounDictionary"] = None


class ProperNounDictionary:
    """
    Maintains known proper nouns with metadata.

    Usage:
        from klareco.proper_nouns import get_proper_noun_dictionary

        pn_dict = get_proper_noun_dictionary()
        if pn_dict.is_proper_noun("Frodo"):
            print("Known proper noun!")
    """

    def __init__(
        self,
        dynamic_path: Optional[Path] = None,
        static_path: Optional[Path] = None,
    ):
        """
        Initialize proper noun dictionary.

        Args:
            dynamic_path: Path to corpus-extracted dictionary (JSON)
            static_path: Path to hand-curated dictionary (JSON, optional)
        """
        self.dynamic: Dict[str, dict] = {}
        self.static: Dict[str, dict] = {}
        self.session_cache: Dict[str, dict] = {}  # Temporary additions

        # Load dynamic dictionary (primary)
        if dynamic_path is None:
            dynamic_path = DEFAULT_DYNAMIC_PATH

        if dynamic_path.exists():
            with open(dynamic_path, 'r', encoding='utf-8') as f:
                self.dynamic = json.load(f)

        # Load static dictionary (optional, hand-curated)
        if static_path is None:
            static_path = DEFAULT_STATIC_PATH

        if static_path.exists():
            with open(static_path, 'r', encoding='utf-8') as f:
                self.static = json.load(f)

    def is_proper_noun(self, word: str) -> bool:
        """
        Check if word is a known proper noun.

        Args:
            word: Word to check (with or without Esperanto endings)

        Returns:
            True if word is a known proper noun
        """
        base = self._strip_esperanto_endings(word)

        # Check all sources (static takes priority, then dynamic, then session)
        return (
            base in self.static or
            base in self.dynamic or
            base in self.session_cache
        )

    def get_metadata(self, word: str) -> dict:
        """
        Get metadata for a proper noun.

        Args:
            word: Word to look up

        Returns:
            Metadata dict with category, frequency, source, etc.
            Empty dict if not found.
        """
        base = self._strip_esperanto_endings(word)

        # Priority: static > dynamic > session
        if base in self.static:
            return self.static[base]
        elif base in self.dynamic:
            return self.dynamic[base]
        elif base in self.session_cache:
            return self.session_cache[base]
        else:
            return {}

    def get_category(self, word: str) -> Optional[str]:
        """
        Get category of proper noun (person, place, organization, other).

        Args:
            word: Word to look up

        Returns:
            Category string or None if not found
        """
        metadata = self.get_metadata(word)
        return metadata.get('category')

    def add_to_session(self, word: str, category: str = "other"):
        """
        Add a proper noun discovered during current session.

        This is temporary and not persisted. Use for proper nouns
        discovered during conversation that should be recognized
        for the duration of the session.

        Args:
            word: Word to add
            category: Category (person, place, organization, other)
        """
        base = self._strip_esperanto_endings(word)
        self.session_cache[base] = {
            "category": category,
            "source": "session",
            "temporary": True,
        }

    def _strip_esperanto_endings(self, word: str) -> str:
        """
        Remove Esperanto noun endings to get the base form.

        Handles: -ojn, -oj, -on, -o, -an, -ajn, -aj (in order of length)
        """
        # Normalize case - keep original capitalization for first letter
        if not word:
            return word

        # Strip common endings (order matters - longest first)
        for ending in ('ojn', 'ajn', 'oj', 'aj', 'on', 'an', 'o', 'a'):
            if word.endswith(ending) and len(word) > len(ending) + 1:
                return word[:-len(ending)]

        return word

    def __len__(self) -> int:
        """Return total number of known proper nouns."""
        return len(self.static) + len(self.dynamic) + len(self.session_cache)

    def __contains__(self, word: str) -> bool:
        """Allow 'word in dictionary' syntax."""
        return self.is_proper_noun(word)


def get_proper_noun_dictionary() -> ProperNounDictionary:
    """
    Get singleton instance of proper noun dictionary.

    This ensures the dictionary is loaded only once and reused.

    Returns:
        ProperNounDictionary instance
    """
    global _dictionary_instance

    if _dictionary_instance is None:
        _dictionary_instance = ProperNounDictionary()

    return _dictionary_instance


def reset_dictionary():
    """Reset singleton (mainly for testing)."""
    global _dictionary_instance
    _dictionary_instance = None
