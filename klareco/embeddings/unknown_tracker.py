"""
Unknown Root Tracker for Vocabulary Expansion.

Tracks roots that aren't in the vocabulary during parsing/inference,
allowing periodic review and vocabulary expansion.
"""

import json
import threading
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import logging

logger = logging.getLogger(__name__)


class UnknownRootTracker:
    """
    Tracks unknown roots encountered during parsing.

    Thread-safe tracker that logs:
    - Root string
    - Occurrence count
    - Example contexts (sentences where it appeared)
    - First/last seen timestamps

    Usage:
        tracker = UnknownRootTracker("data/unknown_roots.json")

        # During parsing
        if root not in vocabulary:
            tracker.log(root, sentence="La novvorto estas utila")

        # Periodic save
        tracker.save()

        # Review
        candidates = tracker.get_candidates(min_count=10)
    """

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        max_contexts_per_root: int = 5,
        auto_save_interval: int = 1000,
    ):
        """
        Initialize tracker.

        Args:
            storage_path: Path to JSON file for persistence
            max_contexts_per_root: Max example sentences to store per root
            auto_save_interval: Save after this many new logs (0 = no auto-save)
        """
        self.storage_path = Path(storage_path) if storage_path else None
        self.max_contexts = max_contexts_per_root
        self.auto_save_interval = auto_save_interval

        self._lock = threading.Lock()
        self._log_count = 0

        # Data structure: {root: {count, contexts, first_seen, last_seen}}
        self._data: Dict[str, dict] = {}

        # Roots that have been added to vocabulary (skip tracking)
        self._added_roots: Set[str] = set()

        # Load existing data if available
        if self.storage_path and self.storage_path.exists():
            self._load()

    def log(
        self,
        root: str,
        sentence: Optional[str] = None,
        word: Optional[str] = None,
    ) -> None:
        """
        Log an unknown root occurrence.

        Args:
            root: The unknown root string
            sentence: Optional context sentence
            word: Optional full word containing the root
        """
        if not root or root in self._added_roots:
            return

        with self._lock:
            now = datetime.now().isoformat()

            if root not in self._data:
                self._data[root] = {
                    'count': 0,
                    'contexts': [],
                    'words': [],
                    'first_seen': now,
                    'last_seen': now,
                }

            entry = self._data[root]
            entry['count'] += 1
            entry['last_seen'] = now

            # Store context if provided and under limit
            if sentence and len(entry['contexts']) < self.max_contexts:
                # Avoid duplicates
                if sentence not in entry['contexts']:
                    entry['contexts'].append(sentence[:200])  # Truncate long sentences

            # Store word forms
            if word and len(entry['words']) < 10:
                if word not in entry['words']:
                    entry['words'].append(word)

            self._log_count += 1

            # Auto-save periodically
            if self.auto_save_interval > 0 and self._log_count >= self.auto_save_interval:
                self._save_unlocked()
                self._log_count = 0

    def get_candidates(
        self,
        min_count: int = 10,
        limit: int = 100,
        sort_by: str = 'count',
    ) -> List[Dict]:
        """
        Get candidate roots for vocabulary expansion.

        Args:
            min_count: Minimum occurrence count
            limit: Maximum candidates to return
            sort_by: 'count' or 'recent'

        Returns:
            List of candidate dicts with root info
        """
        with self._lock:
            candidates = []

            for root, info in self._data.items():
                if info['count'] >= min_count:
                    candidates.append({
                        'root': root,
                        'count': info['count'],
                        'contexts': info['contexts'],
                        'words': info['words'],
                        'first_seen': info['first_seen'],
                        'last_seen': info['last_seen'],
                    })

            # Sort
            if sort_by == 'count':
                candidates.sort(key=lambda x: x['count'], reverse=True)
            elif sort_by == 'recent':
                candidates.sort(key=lambda x: x['last_seen'], reverse=True)

            return candidates[:limit]

    def get_stats(self) -> Dict:
        """Get summary statistics."""
        with self._lock:
            if not self._data:
                return {
                    'total_unknown_roots': 0,
                    'total_occurrences': 0,
                    'roots_seen_10plus': 0,
                    'roots_seen_100plus': 0,
                }

            counts = [info['count'] for info in self._data.values()]
            return {
                'total_unknown_roots': len(self._data),
                'total_occurrences': sum(counts),
                'roots_seen_10plus': sum(1 for c in counts if c >= 10),
                'roots_seen_100plus': sum(1 for c in counts if c >= 100),
                'top_5': sorted(
                    [(r, i['count']) for r, i in self._data.items()],
                    key=lambda x: x[1],
                    reverse=True
                )[:5],
            }

    def mark_added(self, roots: List[str]) -> None:
        """
        Mark roots as added to vocabulary.

        These will be removed from tracking and ignored in future logs.
        """
        with self._lock:
            for root in roots:
                self._added_roots.add(root)
                if root in self._data:
                    del self._data[root]

    def clear(self) -> None:
        """Clear all tracked data."""
        with self._lock:
            self._data.clear()
            self._log_count = 0

    def save(self) -> None:
        """Save tracked data to storage."""
        with self._lock:
            self._save_unlocked()

    def _save_unlocked(self) -> None:
        """Internal save (must hold lock)."""
        if not self.storage_path:
            return

        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        save_data = {
            'unknown_roots': self._data,
            'added_roots': list(self._added_roots),
            'last_saved': datetime.now().isoformat(),
        }

        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)

        logger.debug(f"Saved {len(self._data)} unknown roots to {self.storage_path}")

    def _load(self) -> None:
        """Load tracked data from storage."""
        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self._data = data.get('unknown_roots', {})
            self._added_roots = set(data.get('added_roots', []))

            logger.info(f"Loaded {len(self._data)} unknown roots from {self.storage_path}")
        except Exception as e:
            logger.warning(f"Could not load unknown roots: {e}")
            self._data = {}
            self._added_roots = set()


# Global tracker instance (optional convenience)
_global_tracker: Optional[UnknownRootTracker] = None


def get_tracker(storage_path: Optional[Path] = None) -> UnknownRootTracker:
    """Get or create the global tracker instance."""
    global _global_tracker

    if _global_tracker is None:
        default_path = Path("data/unknown_roots.json")
        _global_tracker = UnknownRootTracker(storage_path or default_path)

    return _global_tracker


def log_unknown_root(
    root: str,
    sentence: Optional[str] = None,
    word: Optional[str] = None,
) -> None:
    """Convenience function to log to global tracker."""
    get_tracker().log(root, sentence, word)
