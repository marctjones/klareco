"""
Unknown Root Tracker for Vocabulary Expansion.

Tracks unknown roots encountered during inference, along with
their contexts, word forms, and frequency. This enables:
1. Identifying common unknown roots that should be added to vocabulary
2. Collecting contexts for fine-tuning new root embeddings
3. Reviewing and expanding the vocabulary incrementally
"""

import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class UnknownRootTracker:
    """
    Tracks unknown roots encountered during embedding inference.

    Stores:
    - Root string
    - Count (how many times seen)
    - Words (morphological forms seen)
    - Contexts (sentences where root appeared)
    - First/last seen timestamps
    - Status (pending/added/rejected)

    Example:
        tracker = UnknownRootTracker(Path("data/unknown_roots.json"))
        tracker.log("programar", sentence="Mi programaras ĉiutage.", word="programaras")
        tracker.save()

        # Later, review candidates
        candidates = tracker.get_candidates(min_count=10)
    """

    def __init__(self, storage_path: Path):
        """
        Initialize tracker with storage path.

        Args:
            storage_path: Path to JSON file for persistence
        """
        self.storage_path = Path(storage_path)
        self._data: Dict[str, Dict[str, Any]] = {}
        self._added_roots: Set[str] = set()  # Roots already added to vocabulary
        self._rejected_roots: Set[str] = set()  # Roots marked as invalid

        # Load existing data if present
        if self.storage_path.exists():
            self._load()

        logger.debug(f"UnknownRootTracker initialized with {len(self._data)} tracked roots")

    def _load(self) -> None:
        """Load data from storage file."""
        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                saved = json.load(f)

            self._data = saved.get('roots', {})
            self._added_roots = set(saved.get('added', []))
            self._rejected_roots = set(saved.get('rejected', []))

            logger.debug(f"Loaded {len(self._data)} roots, {len(self._added_roots)} added, {len(self._rejected_roots)} rejected")
        except Exception as e:
            logger.warning(f"Failed to load unknown roots from {self.storage_path}: {e}")
            self._data = {}
            self._added_roots = set()
            self._rejected_roots = set()

    def save(self) -> None:
        """Save data to storage file."""
        try:
            # Ensure directory exists
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)

            # Prepare data for serialization
            save_data = {
                'roots': self._data,
                'added': list(self._added_roots),
                'rejected': list(self._rejected_roots),
                'last_updated': datetime.now().isoformat(),
                'total_roots': len(self._data),
                'total_added': len(self._added_roots),
            }

            # Write atomically
            temp_path = self.storage_path.with_suffix('.tmp')
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            temp_path.rename(self.storage_path)

            logger.debug(f"Saved {len(self._data)} roots to {self.storage_path}")
        except Exception as e:
            logger.error(f"Failed to save unknown roots: {e}")

    def log(
        self,
        root: str,
        sentence: Optional[str] = None,
        word: Optional[str] = None,
    ) -> None:
        """
        Log an unknown root encounter.

        Args:
            root: The unknown root string
            sentence: Full sentence context (optional)
            word: Full word form (optional)
        """
        root = root.lower()

        # Skip if already added to vocabulary
        if root in self._added_roots:
            return

        # Skip if rejected
        if root in self._rejected_roots:
            return

        now = datetime.now().isoformat()

        if root not in self._data:
            self._data[root] = {
                'root': root,
                'count': 0,
                'words': [],
                'contexts': [],
                'first_seen': now,
                'last_seen': now,
                'status': 'pending',
            }

        entry = self._data[root]
        entry['count'] += 1
        entry['last_seen'] = now

        # Track unique word forms
        if word and word.lower() not in [w.lower() for w in entry['words']]:
            entry['words'].append(word)
            # Limit stored word forms
            if len(entry['words']) > 10:
                entry['words'] = entry['words'][:10]

        # Track contexts (limit to avoid huge files)
        if sentence and sentence not in entry['contexts']:
            entry['contexts'].append(sentence)
            # Limit stored contexts
            if len(entry['contexts']) > 20:
                entry['contexts'] = entry['contexts'][:20]

    def get_candidates(
        self,
        min_count: int = 5,
        limit: int = 100,
        exclude_added: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Get candidate roots for vocabulary expansion.

        Args:
            min_count: Minimum occurrence count
            limit: Maximum number of candidates to return
            exclude_added: Whether to exclude already-added roots

        Returns:
            List of candidate dicts sorted by count (descending)
        """
        candidates = []

        for root, entry in self._data.items():
            # Skip if already added
            if exclude_added and root in self._added_roots:
                continue

            # Skip if rejected
            if root in self._rejected_roots:
                continue

            # Skip if below threshold
            if entry['count'] < min_count:
                continue

            candidates.append({
                'root': root,
                'count': entry['count'],
                'words': entry['words'],
                'contexts': entry['contexts'],
                'first_seen': entry['first_seen'],
                'last_seen': entry['last_seen'],
            })

        # Sort by count descending
        candidates.sort(key=lambda x: x['count'], reverse=True)

        return candidates[:limit]

    def mark_added(self, roots: List[str]) -> None:
        """
        Mark roots as added to vocabulary.

        Args:
            roots: List of root strings that were added
        """
        for root in roots:
            root = root.lower()
            self._added_roots.add(root)
            if root in self._data:
                self._data[root]['status'] = 'added'

        logger.info(f"Marked {len(roots)} roots as added to vocabulary")

    def mark_rejected(self, roots: List[str]) -> None:
        """
        Mark roots as rejected (not valid Esperanto).

        Args:
            roots: List of root strings to reject
        """
        for root in roots:
            root = root.lower()
            self._rejected_roots.add(root)
            if root in self._data:
                self._data[root]['status'] = 'rejected'

        logger.info(f"Marked {len(roots)} roots as rejected")

    def get_stats(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            'total_tracked': len(self._data),
            'total_added': len(self._added_roots),
            'total_rejected': len(self._rejected_roots),
            'pending': len([r for r in self._data if r not in self._added_roots and r not in self._rejected_roots]),
            'total_occurrences': sum(e['count'] for e in self._data.values()),
        }

    def clear_pending(self) -> None:
        """Clear all pending (non-added, non-rejected) roots."""
        to_remove = [
            root for root in self._data
            if root not in self._added_roots and root not in self._rejected_roots
        ]
        for root in to_remove:
            del self._data[root]
        logger.info(f"Cleared {len(to_remove)} pending roots")


# Global tracker instance (lazy initialization)
_global_tracker: Optional[UnknownRootTracker] = None


def get_tracker(storage_path: Optional[Path] = None) -> UnknownRootTracker:
    """
    Get the global unknown root tracker instance.

    Args:
        storage_path: Path for storage (only used on first call)

    Returns:
        Global UnknownRootTracker instance
    """
    global _global_tracker

    if _global_tracker is None:
        path = storage_path or Path("data/unknown_roots.json")
        _global_tracker = UnknownRootTracker(path)

    return _global_tracker


def log_unknown_root(
    root: str,
    sentence: Optional[str] = None,
    word: Optional[str] = None,
) -> None:
    """
    Convenience function to log an unknown root to global tracker.

    Args:
        root: The unknown root string
        sentence: Full sentence context (optional)
        word: Full word form (optional)
    """
    tracker = get_tracker()
    tracker.log(root, sentence=sentence, word=word)
