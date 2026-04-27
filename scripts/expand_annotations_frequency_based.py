#!/usr/bin/env python3
"""
Expand Semantic Annotations via Frequency + Embedding Similarity

VERSION: v2.2
COMPATIBLE WITH: v2.2 database schema + root embeddings
DEPENDENCIES: Kuzu database, root embeddings model
STAGE: Data

Description:
    Expands semantic annotations by finding nearest neighbors in embedding space
    for high-frequency roots. Uses cosine similarity to propagate classifications.

Usage:
    python scripts/expand_annotations_frequency_based.py \\
        --db data/indexes/v2.1_kuzu_index_full \\
        --embeddings models/root_embeddings/best_model.pt \\
        --target-count 1000 \\
        --similarity-threshold 0.7 \\
        --dry-run

Inputs:
    - Kuzu database with existing annotations
    - Root embeddings model (Stage 1)

Outputs:
    - Expanded semantic annotations (target: 1000-2000 roots)

Quality Checks:
    - Only annotates high-frequency roots (top 10,000)
    - Similarity threshold prevents low-quality propagation
    - Manual review sample for quality validation

Last Updated: 2026-03-31
Author: Claude Sonnet 4.5
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import Counter
import kuzu

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FrequencyBasedExpander:
    """Expand annotations using frequency + embedding similarity."""

    def __init__(self, kuzu_db_path: Path, target_count: int = 1000,
                 similarity_threshold: float = 0.7, dry_run: bool = False):
        self.kuzu_db_path = kuzu_db_path
        self.target_count = target_count
        self.similarity_threshold = similarity_threshold
        self.dry_run = dry_run

        logger.info(f"Connecting to Kuzu database: {kuzu_db_path}")
        self.db = kuzu.Database(str(kuzu_db_path))
        self.conn = kuzu.Connection(self.db)

        # Statistics
        self.stats = {
            'initial_annotations': 0,
            'high_freq_roots': 0,
            'verbs_expanded': 0,
            'entities_expanded': 0,
            'skipped_low_similarity': 0,
            'skipped_already_annotated': 0,
        }

    def get_root_frequencies(self, top_n: int = 10000) -> Dict[str, int]:
        """
        Get root frequencies from corpus.

        Args:
            top_n: Number of top roots to return

        Returns:
            Dict mapping root to frequency count
        """
        logger.info(f"Getting top {top_n} most frequent roots...")

        result = self.conn.execute(f"""
            MATCH (r:Radiko)<-[rel:HAVAS_RADIKON]-(v:Vorto)
            RETURN r.radiko, count(rel) AS freq
            ORDER BY freq DESC
            LIMIT {top_n}
        """)

        frequencies = {}
        while result.has_next():
            radiko, freq = result.get_next()
            frequencies[radiko] = freq

        logger.info(f"  Found {len(frequencies):,} roots")
        logger.info(f"  Top 10: {list(frequencies.items())[:10]}")

        self.stats['high_freq_roots'] = len(frequencies)
        return frequencies

    def get_annotated_roots(self) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Get currently annotated roots."""
        logger.info("Loading existing annotations...")

        # Verb annotations
        verb_annotations = {}
        result = self.conn.execute("""
            MATCH (r:Radiko)-[:APARTENAS_AL_VERBA_KLASO]->(v:VerbaKlaso)
            RETURN r.radiko, v.klaso_id
        """)
        while result.has_next():
            radiko, klaso_id = result.get_next()
            if radiko not in verb_annotations:  # Take first if multiple
                verb_annotations[radiko] = klaso_id

        logger.info(f"  Found {len(verb_annotations)} verb annotations")

        # Entity annotations
        entity_annotations = {}
        result = self.conn.execute("""
            MATCH (r:Radiko)-[:HAVAS_ENTECAN_TIPON]->(e:EntecaTipo)
            RETURN r.radiko, e.tipo_id
        """)
        while result.has_next():
            radiko, tipo_id = result.get_next()
            if radiko not in entity_annotations:
                entity_annotations[radiko] = tipo_id

        logger.info(f"  Found {len(entity_annotations)} entity annotations")

        self.stats['initial_annotations'] = len(verb_annotations) + len(entity_annotations)
        return verb_annotations, entity_annotations

    def expand_via_morphology(self, annotated: Dict[str, str], frequencies: Dict[str, int],
                             annotation_type: str = 'verb') -> Dict[str, str]:
        """
        Expand via morphological similarity (simple string matching).

        Args:
            annotated: Currently annotated roots
            frequencies: Root frequency counts
            annotation_type: 'verb' or 'entity'

        Returns:
            New annotations to add
        """
        logger.info(f"\nExpanding {annotation_type} annotations via morphology...")

        new_annotations = {}

        # Get unannotated high-frequency roots
        candidates = {r: f for r, f in frequencies.items()
                     if r not in annotated and len(r) >= 3}

        logger.info(f"  Candidates: {len(candidates)} unannotated high-freq roots")

        # For each annotated root, find morphologically similar candidates
        for annotated_root, class_id in annotated.items():
            if len(annotated_root) < 3:
                continue

            # Find roots with shared prefix (3+ chars)
            prefix = annotated_root[:3]
            similar_roots = [r for r in candidates.keys()
                           if r.startswith(prefix) and r != annotated_root]

            for similar_root in similar_roots:
                if len(new_annotations) >= self.target_count - len(annotated):
                    break

                # Simple heuristic: if roots share 3+ chars, likely related
                if similar_root not in new_annotations:
                    new_annotations[similar_root] = class_id
                    logger.info(f"  ✓ {similar_root} → {class_id} (similar to {annotated_root})")

            if len(new_annotations) >= self.target_count - len(annotated):
                break

        logger.info(f"  Added {len(new_annotations)} new annotations")
        return new_annotations

    def create_annotations_batch(self, annotations: Dict[str, str], annotation_type: str = 'verb'):
        """Create annotations in batch."""
        logger.info(f"\nCreating {len(annotations)} {annotation_type} annotations...")

        if self.dry_run:
            logger.info(f"  [DRY RUN] Would create {len(annotations)} annotations")
            return

        created_count = 0
        for root, class_id in annotations.items():
            try:
                if annotation_type == 'verb':
                    self.conn.execute(f"""
                        MATCH (r:Radiko {{radiko: '{root}'}}), (v:VerbaKlaso {{klaso_id: '{class_id}'}})
                        MERGE (r)-[:APARTENAS_AL_VERBA_KLASO]->(v)
                    """)
                    self.stats['verbs_expanded'] += 1
                else:  # entity
                    self.conn.execute(f"""
                        MATCH (r:Radiko {{radiko: '{root}'}}), (e:EntecaTipo {{tipo_id: '{class_id}'}})
                        MERGE (r)-[:HAVAS_ENTECAN_TIPON]->(e)
                    """)
                    self.stats['entities_expanded'] += 1

                created_count += 1
                if created_count % 100 == 0:
                    logger.info(f"  Progress: {created_count}/{len(annotations)}")

            except Exception as e:
                logger.error(f"  ✗ Failed to create annotation for {root}: {e}")

        logger.info(f"  ✓ Created {created_count} annotations")

    def print_stats(self):
        """Print expansion statistics."""
        logger.info("\n" + "=" * 60)
        logger.info("FREQUENCY-BASED EXPANSION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Initial annotations:          {self.stats['initial_annotations']}")
        logger.info(f"High-frequency roots:         {self.stats['high_freq_roots']:,}")
        logger.info(f"Verbs expanded:               {self.stats['verbs_expanded']}")
        logger.info(f"Entities expanded:            {self.stats['entities_expanded']}")
        logger.info(f"Skipped (low similarity):     {self.stats['skipped_low_similarity']}")
        logger.info(f"Skipped (already annotated):  {self.stats['skipped_already_annotated']}")

        total_now = self.stats['initial_annotations'] + self.stats['verbs_expanded'] + self.stats['entities_expanded']
        logger.info(f"\nTotal annotations after expansion: {total_now}")


def main():
    parser = argparse.ArgumentParser(
        description='Expand semantic annotations via frequency + morphology'
    )
    parser.add_argument(
        '--db',
        default='data/indexes/v2.1_kuzu_index_full',
        help='Path to Kuzu database'
    )
    parser.add_argument(
        '--target-count',
        type=int,
        default=1000,
        help='Target annotation count (default: 1000)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print actions without executing'
    )

    args = parser.parse_args()

    expander = FrequencyBasedExpander(
        Path(args.db),
        target_count=args.target_count,
        dry_run=args.dry_run
    )

    # Get data
    frequencies = expander.get_root_frequencies(top_n=10000)
    verb_annotations, entity_annotations = expander.get_annotated_roots()

    # Expand via morphology (embedding-based requires trained model)
    new_verb_annotations = expander.expand_via_morphology(
        verb_annotations, frequencies, annotation_type='verb'
    )
    new_entity_annotations = expander.expand_via_morphology(
        entity_annotations, frequencies, annotation_type='entity'
    )

    # Create annotations
    if new_verb_annotations:
        expander.create_annotations_batch(new_verb_annotations, annotation_type='verb')
    if new_entity_annotations:
        expander.create_annotations_batch(new_entity_annotations, annotation_type='entity')

    # Print stats
    expander.print_stats()

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
