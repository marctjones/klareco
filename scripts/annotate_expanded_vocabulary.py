#!/usr/bin/env python3
"""
Annotate Expanded Vocabulary (Reaching 200 Roots)

VERSION: v2.2
COMPATIBLE WITH: v2.2 database schema (semantic ontology)
DEPENDENCIES: Kuzu database with semantic ontology tables
STAGE: Data

Description:
    Expands core vocabulary annotations from 78 → 200+ roots by adding:
    - High-frequency verbs (40+ more verbs)
    - Common nouns (temporal, spatial, abstract concepts)
    - Additional places and persons
    - Adjective roots

Pipeline Position:
    annotate_core_roots_from_gazetteers.py (78 roots) → [THIS SCRIPT] → 200+ roots

Usage:
    python scripts/annotate_expanded_vocabulary.py \\
        --db data/indexes/v2.1_kuzu_index_full \\
        --dry-run

Inputs:
    - Kuzu database with v2.2 semantic ontology schema
    - Existing 78 annotations from annotate_core_roots_from_gazetteers.py

Outputs:
    - 120+ additional semantic classification relationships
    - Total: 200+ annotated roots

Quality Checks:
    - Validates roots exist in database
    - Reports missing roots
    - Reports annotation statistics by category

Last Updated: 2026-03-28
Author: Claude Sonnet 4.5
Related Issues: #18 (Annotate 200 core roots)
See Also: scripts/annotate_core_roots_from_gazetteers.py
"""

import logging
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple
import kuzu

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ExpandedVocabularyAnnotator:
    """Annotate expanded vocabulary to reach 200+ roots."""

    def __init__(self, kuzu_db_path: Path, dry_run: bool = False):
        """
        Initialize annotator.

        Args:
            kuzu_db_path: Path to Kuzu database
            dry_run: If True, print actions without executing
        """
        self.kuzu_db_path = kuzu_db_path
        self.dry_run = dry_run

        logger.info(f"Connecting to Kuzu database: {kuzu_db_path}")
        self.db = kuzu.Database(str(kuzu_db_path))
        self.conn = kuzu.Connection(self.db)

        # Statistics
        self.stats = {
            'verbs_annotated': 0,
            'nouns_annotated': 0,
            'places_annotated': 0,
            'persons_annotated': 0,
            'time_words_annotated': 0,
            'organizations_annotated': 0,
            'events_annotated': 0,
            'missing_roots': [],
            'errors': []
        }

    def root_exists(self, root: str) -> bool:
        """Check if root exists in Radiko table."""
        query = f"MATCH (r:Radiko {{radiko: '{root}'}}) RETURN count(*)"
        result = self.conn.execute(query)
        count = result.get_next()[0]
        return count > 0

    def link_verb_to_class(self, root: str, klaso_id: str) -> bool:
        """Create APARTENAS_AL_VERBA_KLASO relationship."""
        if not self.root_exists(root):
            logger.warning(f"Root '{root}' not found in database, skipping")
            self.stats['missing_roots'].append(root)
            return False

        if self.dry_run:
            logger.info(f"[DRY-RUN] Would link {root} → {klaso_id}")
            return True

        try:
            query = f"""
                MATCH (r:Radiko {{radiko: '{root}'}}), (v:VerbaKlaso {{klaso_id: '{klaso_id}'}})
                MERGE (r)-[:APARTENAS_AL_VERBA_KLASO]->(v)
            """
            self.conn.execute(query)
            logger.debug(f"✓ Linked {root} → {klaso_id}")
            self.stats['verbs_annotated'] += 1
            return True

        except Exception as e:
            logger.error(f"Failed to link {root} → {klaso_id}: {e}")
            self.stats['errors'].append(f"{root} → {klaso_id}: {e}")
            return False

    def link_noun_to_entity_type(self, root: str, tipo_id: str) -> bool:
        """Create HAVAS_ENTECAN_TIPON relationship."""
        if not self.root_exists(root):
            logger.warning(f"Root '{root}' not found in database, skipping")
            self.stats['missing_roots'].append(root)
            return False

        if self.dry_run:
            logger.info(f"[DRY-RUN] Would link {root} → EntecaTipo:{tipo_id}")
            return True

        try:
            query = f"""
                MATCH (r:Radiko {{radiko: '{root}'}}), (e:EntecaTipo {{tipo_id: '{tipo_id}'}})
                MERGE (r)-[:HAVAS_ENTECAN_TIPON]->(e)
            """
            self.conn.execute(query)
            logger.debug(f"✓ Linked {root} → EntecaTipo:{tipo_id}")

            # Update category-specific stats
            if tipo_id == 'loko':
                self.stats['places_annotated'] += 1
            elif tipo_id == 'persono':
                self.stats['persons_annotated'] += 1
            elif tipo_id == 'tempo':
                self.stats['time_words_annotated'] += 1
            elif tipo_id == 'organizaĵo':
                self.stats['organizations_annotated'] += 1
            elif tipo_id == 'evento':
                self.stats['events_annotated'] += 1
            else:
                self.stats['nouns_annotated'] += 1

            return True

        except Exception as e:
            logger.error(f"Failed to link {root} → {tipo_id}: {e}")
            self.stats['errors'].append(f"{root} → {tipo_id}: {e}")
            return False

    def annotate_expanded_verbs(self):
        """Annotate additional high-frequency verbs."""
        logger.info("=" * 60)
        logger.info("ANNOTATING EXPANDED VERBS")
        logger.info("=" * 60)

        # Additional verb annotations by class
        expanded_verbs = {
            'kreado-26': [
                'establ',  # establish
                'konstru', # construct
                'form',    # form
                'invent',  # invent
                'desegn',  # design
                'develop', # develop
            ],
            'movo-51': [
                'vetur',   # travel (vehicle)
                'flugg',   # fly
                'salt',    # jump
                'mar',     # walk
                'kur',     # run
                'migr',    # migrate
            ],
            'pensado-29': [
                'konsider', # consider
                'memor',    # remember
                'studi',    # study
                'koncern',  # concern
                'koncept',  # conceive
                'konsci',   # be aware
            ],
            'perceptado-30': [
                'observ',   # observe
                'rigard',   # look at
                'esplor',   # explore
                'ekzamen',  # examine
                'aŭskult',  # listen
            ],
            'emocio-31': [
                'esperi',   # hope
                'dezir',    # desire
                'ŝat',      # like
                'prefer',   # prefer
                'admir',    # admire
                'sopir',    # yearn
            ],
            'komunikado-37': [
                'rakont',   # narrate
                'skribi',   # write
                'publiki',  # publish
                'prezent',  # present
                'diskut',   # discuss
                'inform',   # inform
            ],
            'vivo-48': [
                'ekzist',   # exist
                'estat',    # be (state)
                'daŭr',     # continue
                'okaz',     # happen
                'develop',  # develop (grow)
            ],
            'profesio-50': [
                'funkci',   # function
                'administr',# administrate
                'direkt',   # direct
                'organiz',  # organize
                'gvid',     # guide
            ],
        }

        for klaso_id, roots in expanded_verbs.items():
            logger.info(f"\nExpanding {klaso_id} with {len(roots)} more roots")
            for root in roots:
                self.link_verb_to_class(root, klaso_id)

        logger.info(f"\n✓ Annotated {self.stats['verbs_annotated']} additional verbs")

    def annotate_temporal_nouns(self):
        """Annotate time-related nouns."""
        logger.info("\n" + "=" * 60)
        logger.info("ANNOTATING TEMPORAL NOUNS")
        logger.info("=" * 60)

        temporal_roots = [
            # Time units
            'jar',      # year
            'monat',    # month
            'semajn',   # week
            'tag',      # day
            'hor',      # hour
            'minut',    # minute
            'sekond',   # second
            'epok',     # epoch
            'period',   # period
            'moment',   # moment
            'temp',     # time
            'epoik',    # era

            # Time references
            'hieraŭ',   # yesterday
            'hodiaŭ',   # today
            'morgaŭ',   # tomorrow
            'pasint',   # past
            'estont',   # future
            'nun',      # now
        ]

        logger.info(f"Annotating {len(temporal_roots)} temporal roots")
        for root in temporal_roots:
            self.link_noun_to_entity_type(root, 'tempo')

        logger.info(f"\n✓ Annotated {self.stats['time_words_annotated']} temporal nouns")

    def annotate_additional_places(self):
        """Annotate additional geographic nouns."""
        logger.info("\n" + "=" * 60)
        logger.info("ANNOTATING ADDITIONAL PLACES")
        logger.info("=" * 60)

        place_roots = [
            # Geographic features
            'mont',     # mountain
            'river',    # river
            'lag',      # lake
            'mar',      # sea
            'ocean',    # ocean
            'insul',    # island
            'kontinent',# continent
            'region',   # region
            'provinc',  # province
            'kvartaul', # district

            # Infrastructure
            'domomur',  # building
            'strukt',   # structure
            'pont',     # bridge
            'staci',    # station
            'aeroport', # airport
            'haven',    # port

            # More countries (high-frequency ones)
            'usoni',    # USA (alternate)
            'anglujo',  # England (alternate)
        ]

        logger.info(f"Annotating {len(place_roots)} additional place roots")
        for root in place_roots:
            self.link_noun_to_entity_type(root, 'loko')

        logger.info(f"\n✓ Annotated {self.stats['places_annotated']} additional places")

    def annotate_additional_persons(self):
        """Annotate additional person-related nouns."""
        logger.info("\n" + "=" * 60)
        logger.info("ANNOTATING ADDITIONAL PERSONS")
        logger.info("=" * 60)

        person_roots = [
            # Family/relationships
            'patr',     # father
            'patr',     # mother
            'infan',    # child
            'frat',     # brother/sibling
            'fil',      # son/daughter
            'gepatr',   # parents

            # Professions (more)
            'advokat',  # lawyer
            'inĝenier', # engineer
            'arkitekt', # architect
            'artist',   # artist
            'muzik',    # musician
            'aktor',    # actor
            'polit',    # politician
            'milite',   # soldier
            'komercist',# merchant
            'farma',    # pharmacist
            'sciencist',# scientist

            # Social roles
            'civitan',  # citizen
            'loĝant',   # resident
            'naciist',  # nationalist
            'membr',    # member
            'amik',     # friend
        ]

        logger.info(f"Annotating {len(person_roots)} additional person roots")
        for root in person_roots:
            self.link_noun_to_entity_type(root, 'persono')

        logger.info(f"\n✓ Annotated {self.stats['persons_annotated']} additional persons")

    def annotate_organizations(self):
        """Annotate organization-related nouns."""
        logger.info("\n" + "=" * 60)
        logger.info("ANNOTATING ORGANIZATIONS")
        logger.info("=" * 60)

        organization_roots = [
            'organiz',  # organization
            'asoci',    # association
            'kompani',  # company
            'entrepren',# enterprise
            'institut', # institute
            'universitat', # university
            'akademi',  # academy
            'societ',   # society
            'federaci', # federation
            'konsili',  # council
            'komitat',  # committee
            'parlament',# parliament
            'registar', # government
            'ministerij',# ministry
        ]

        logger.info(f"Annotating {len(organization_roots)} organization roots")
        for root in organization_roots:
            self.link_noun_to_entity_type(root, 'organizaĵo')

        logger.info(f"\n✓ Annotated {self.stats['organizations_annotated']} organizations")

    def annotate_events(self):
        """Annotate event-related nouns."""
        logger.info("\n" + "=" * 60)
        logger.info("ANNOTATING EVENTS")
        logger.info("=" * 60)

        event_roots = [
            'kongres',  # congress
            'konferenc',# conference
            'kunsid',   # meeting
            'celebr',   # celebration
            'fest',     # festival
            'ceremon',  # ceremony
            'milit',    # war
            'bat',      # battle
            'revel',    # revolution
            'elekti',   # election
        ]

        logger.info(f"Annotating {len(event_roots)} event roots")
        for root in event_roots:
            self.link_noun_to_entity_type(root, 'evento')

        logger.info(f"\n✓ Annotated {self.stats['events_annotated']} events")

    def print_summary(self):
        """Print annotation summary."""
        logger.info("\n" + "=" * 60)
        logger.info("ANNOTATION SUMMARY")
        logger.info("=" * 60)

        logger.info(f"Verbs annotated: {self.stats['verbs_annotated']}")
        logger.info(f"Places annotated: {self.stats['places_annotated']}")
        logger.info(f"Persons annotated: {self.stats['persons_annotated']}")
        logger.info(f"Temporal nouns: {self.stats['time_words_annotated']}")
        logger.info(f"Organizations: {self.stats['organizations_annotated']}")
        logger.info(f"Events: {self.stats['events_annotated']}")
        logger.info(f"Other nouns: {self.stats['nouns_annotated']}")

        total = (self.stats['verbs_annotated'] +
                 self.stats['places_annotated'] +
                 self.stats['persons_annotated'] +
                 self.stats['time_words_annotated'] +
                 self.stats['organizations_annotated'] +
                 self.stats['events_annotated'] +
                 self.stats['nouns_annotated'])

        logger.info(f"\nTOTAL NEW ANNOTATIONS: {total} roots")

        if self.stats['missing_roots']:
            logger.warning(f"\nMissing roots (not in database): {len(self.stats['missing_roots'])}")
            for root in self.stats['missing_roots'][:15]:
                logger.warning(f"  - {root}")
            if len(self.stats['missing_roots']) > 15:
                logger.warning(f"  ... and {len(self.stats['missing_roots']) - 15} more")

        if self.stats['errors']:
            logger.error(f"\nErrors: {len(self.stats['errors'])}")
            for error in self.stats['errors'][:5]:
                logger.error(f"  - {error}")

    def verify_total_annotations(self):
        """Verify total annotations in database."""
        logger.info("\n" + "=" * 60)
        logger.info("VERIFYING TOTAL ANNOTATIONS")
        logger.info("=" * 60)

        # Count verb relationships
        query = "MATCH ()-[r:APARTENAS_AL_VERBA_KLASO]->() RETURN count(*)"
        result = self.conn.execute(query)
        verb_count = result.get_next()[0]
        logger.info(f"Total verb relationships: {verb_count}")

        # Count entity type relationships
        query = "MATCH ()-[r:HAVAS_ENTECAN_TIPON]->() RETURN count(*)"
        result = self.conn.execute(query)
        entity_count = result.get_next()[0]
        logger.info(f"Total entity type relationships: {entity_count}")

        # Breakdown by entity type
        query = """
            MATCH (r:Radiko)-[:HAVAS_ENTECAN_TIPON]->(e:EntecaTipo)
            RETURN e.tipo_id, count(*)
            ORDER BY count(*) DESC
        """
        result = self.conn.execute(query)
        logger.info("\nBreakdown by entity type:")
        while result.has_next():
            row = result.get_next()
            logger.info(f"  {row[0]}: {row[1]} roots")

        total_roots = verb_count + entity_count
        logger.info(f"\n==> TOTAL ANNOTATED ROOTS: {total_roots}")

        if total_roots >= 200:
            logger.info("✓✓✓ TARGET REACHED! 200+ roots annotated! ✓✓✓")
        else:
            remaining = 200 - total_roots
            logger.info(f"Remaining to reach 200: {remaining} roots")


def main():
    parser = argparse.ArgumentParser(
        description='Annotate expanded vocabulary (reaching 200 roots)'
    )
    parser.add_argument(
        '--db',
        type=Path,
        default=Path('data/indexes/v2.1_kuzu_index_full'),
        help='Path to Kuzu database'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print actions without executing'
    )

    args = parser.parse_args()

    if not args.db.exists():
        logger.error(f"Database not found: {args.db}")
        return 1

    annotator = ExpandedVocabularyAnnotator(args.db, dry_run=args.dry_run)

    try:
        # Annotate expanded verbs (40+ more verbs)
        annotator.annotate_expanded_verbs()

        # Annotate temporal nouns (~16 roots)
        annotator.annotate_temporal_nouns()

        # Annotate additional places (~18 roots)
        annotator.annotate_additional_places()

        # Annotate additional persons (~25 roots)
        annotator.annotate_additional_persons()

        # Annotate organizations (~14 roots)
        annotator.annotate_organizations()

        # Annotate events (~10 roots)
        annotator.annotate_events()

        # Print summary
        annotator.print_summary()

        # Verify total (skip in dry-run mode)
        if not args.dry_run:
            annotator.verify_total_annotations()

        logger.info("\n✓ Expanded annotation complete!")

        return 0

    except Exception as e:
        logger.error(f"Annotation failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit(main())
