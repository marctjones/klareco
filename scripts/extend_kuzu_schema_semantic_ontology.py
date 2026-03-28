#!/usr/bin/env python3
"""
Extend Kuzu Schema with 4-Layer Semantic Ontology

VERSION: v2.2
COMPATIBLE WITH: v2.1 database (extends, does not replace)
STAGE: Data
RELATED ISSUES: #654, #655

This script extends the v2.1 Kuzu database with a comprehensive semantic ontology:
- Layer 1: Lexical semantics (verb/noun classes, thematic roles, aspectual classes)
- Layer 2: Frame semantics (semantic frames, event participants)
- Layer 3: Discourse semantics (RST relations, information structure)
- Layer 4: Schema semantics (biographical/definitional/event schemas)

The ontology is based on proven linguistic frameworks:
- VerbNet for verb classification
- WordNet for noun hierarchies
- FrameNet for semantic frames
- RST (Rhetorical Structure Theory) for discourse relations

All terminology is in Esperanto to maintain the "Pure Esperanto AI" principle.

Usage:
    python scripts/extend_kuzu_schema_semantic_ontology.py
    python scripts/extend_kuzu_schema_semantic_ontology.py --dry-run  # Show SQL only

Inputs:
    - Existing v2.1 Kuzu database at data/indexes/v2.1_kuzu_index_full

Outputs:
    - Extended schema with 4-layer ontology tables
    - Migration log at logs/schema_migration_v2.2.log

Last Updated: 2026-03-28
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List
import kuzu

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/schema_migration_v2.2.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SemanticOntologySchemaExtension:
    """Extend Kuzu database with 4-layer semantic ontology."""

    def __init__(self, db_path: Path, dry_run: bool = False):
        self.db_path = db_path
        self.dry_run = dry_run

        if not dry_run:
            self.db = kuzu.Database(str(db_path))
            self.conn = kuzu.Connection(self.db)

        # Track created tables
        self.created_tables = []

    def execute(self, query: str, description: str):
        """Execute query or log if dry run."""
        if self.dry_run:
            logger.info(f"[DRY RUN] {description}")
            logger.info(f"Query: {query}\n")
        else:
            logger.info(f"Executing: {description}")
            try:
                self.conn.execute(query)
                logger.info("✓ Success")
            except Exception as e:
                logger.error(f"✗ Failed: {e}")
                raise

    # ===================================================================
    # LAYER 1: LEXICAL SEMANTICS
    # ===================================================================

    def create_layer1_verb_classes(self):
        """Create verb classification tables (VerbNet-style)."""

        logger.info("=" * 60)
        logger.info("LAYER 1: LEXICAL SEMANTICS - Verb Classes")
        logger.info("=" * 60)

        # 1. VerbaKlaso (Verb Class) - Hierarchical verb taxonomy
        self.execute("""
            CREATE NODE TABLE VerbaKlaso (
                klaso_id STRING,
                klaso_nomo STRING,
                priskribo STRING,
                superklaso_id STRING,
                ekzemplaj_radikoj STRING[],
                graveco_biografia DOUBLE DEFAULT 0.5,
                graveco_difina DOUBLE DEFAULT 0.5,
                graveco_okazaĵa DOUBLE DEFAULT 0.5,
                PRIMARY KEY (klaso_id)
            )
        """, "Create VerbaKlaso (Verb Class) node table")

        # 2. AspektaKlaso (Aspectual Class) - Vendler classification
        self.execute("""
            CREATE NODE TABLE AspektaKlaso (
                klaso_id STRING,
                klaso_nomo STRING,
                priskribo STRING,
                ekzemploj STRING[],
                telikeco BOOLEAN,
                durativeco BOOLEAN,
                dinamikeco BOOLEAN,
                PRIMARY KEY (klaso_id)
            )
        """, "Create AspektaKlaso (Aspectual Class) node table")

        # 3. TemaRolo (Thematic Role) - Agent, Patient, Theme, etc.
        self.execute("""
            CREATE NODE TABLE TemaRolo (
                rolo_id STRING,
                rolo_nomo STRING,
                priskribo STRING,
                ekzemploj STRING[],
                kerneco BOOLEAN,
                PRIMARY KEY (rolo_id)
            )
        """, "Create TemaRolo (Thematic Role) node table")

        # Relationship: Root belongs to verb class
        self.execute("""
            CREATE REL TABLE APARTENAS_AL_VERBA_KLASO (
                FROM Radiko TO VerbaKlaso,
                fonto STRING,
                konfideco DOUBLE DEFAULT 1.0
            )
        """, "Create APARTENAS_AL_VERBA_KLASO relationship")

        # Relationship: Root has aspectual class
        self.execute("""
            CREATE REL TABLE HAVAS_ASPEKTAN_KLASON (
                FROM Radiko TO AspektaKlaso,
                fonto STRING
            )
        """, "Create HAVAS_ASPEKTAN_KLASON relationship")

    def create_layer1_noun_classes(self):
        """Create noun classification tables (WordNet-style)."""

        logger.info("\nLayer 1: Noun Classes")
        logger.info("-" * 60)

        # 1. SubstantivaKlaso (Noun Class) - Hierarchical noun taxonomy
        self.execute("""
            CREATE NODE TABLE SubstantivaKlaso (
                klaso_id STRING,
                klaso_nomo STRING,
                priskribo STRING,
                superklaso_id STRING,
                ekzemplaj_radikoj STRING[],
                enteca_tipo STRING,
                animeco BOOLEAN DEFAULT false,
                konkreteco BOOLEAN DEFAULT true,
                PRIMARY KEY (klaso_id)
            )
        """, "Create SubstantivaKlaso (Noun Class) node table")

        # 2. EntecaTipo (Entity Type) - Person, Place, Time, etc.
        self.execute("""
            CREATE NODE TABLE EntecaTipo (
                tipo_id STRING,
                tipo_nomo STRING,
                priskribo STRING,
                supergrupo STRING,
                ekzemploj STRING[],
                PRIMARY KEY (tipo_id)
            )
        """, "Create EntecaTipo (Entity Type) node table")

        # Relationship: Root belongs to noun class
        self.execute("""
            CREATE REL TABLE APARTENAS_AL_SUBSTANTIVA_KLASO (
                FROM Radiko TO SubstantivaKlaso,
                fonto STRING,
                konfideco DOUBLE DEFAULT 1.0
            )
        """, "Create APARTENAS_AL_SUBSTANTIVA_KLASO relationship")

        # Relationship: Root has entity type
        self.execute("""
            CREATE REL TABLE HAVAS_ENTECAN_TIPON (
                FROM Radiko TO EntecaTipo,
                fonto STRING
            )
        """, "Create HAVAS_ENTECAN_TIPON relationship")

    def create_layer1_adjective_classes(self):
        """Create adjective classification tables."""

        logger.info("\nLayer 1: Adjective Classes")
        logger.info("-" * 60)

        # AdjektivaKlaso (Adjective Class)
        self.execute("""
            CREATE NODE TABLE AdjektivaKlaso (
                klaso_id STRING,
                klaso_nomo STRING,
                priskribo STRING,
                ekzemploj STRING[],
                gradeblo BOOLEAN DEFAULT true,
                esenceco STRING,
                PRIMARY KEY (klaso_id)
            )
        """, "Create AdjektivaKlaso (Adjective Class) node table")

        # Relationship: Root belongs to adjective class
        self.execute("""
            CREATE REL TABLE APARTENAS_AL_ADJEKTIVA_KLASO (
                FROM Radiko TO AdjektivaKlaso,
                fonto STRING
            )
        """, "Create APARTENAS_AL_ADJEKTIVA_KLASO relationship")

    # ===================================================================
    # LAYER 2: FRAME SEMANTICS
    # ===================================================================

    def create_layer2_frames(self):
        """Create semantic frame tables (FrameNet-style)."""

        logger.info("\n" + "=" * 60)
        logger.info("LAYER 2: FRAME SEMANTICS")
        logger.info("=" * 60)

        # 1. SemantikKadro (Semantic Frame)
        self.execute("""
            CREATE NODE TABLE SemantikKadro (
                kadro_id STRING,
                kadro_nomo STRING,
                priskribo STRING,
                kernaj_roloj STRING[],
                periferaj_roloj STRING[],
                ekzemplaj_verboj STRING[],
                PRIMARY KEY (kadro_id)
            )
        """, "Create SemantikKadro (Semantic Frame) node table")

        # 2. KadraRolo (Frame Role) - Frame-specific roles
        self.execute("""
            CREATE NODE TABLE KadraRolo (
                rolo_id STRING,
                rolo_nomo STRING,
                priskribo STRING,
                kerneco BOOLEAN,
                ekzemploj STRING[],
                PRIMARY KEY (rolo_id)
            )
        """, "Create KadraRolo (Frame Role) node table")

        # Relationship: Verb evokes frame
        self.execute("""
            CREATE REL TABLE ELVOKIS_KADRON (
                FROM Radiko TO SemantikKadro,
                tipeco STRING
            )
        """, "Create ELVOKIS_KADRON relationship")

        # Relationship: Frame has role
        self.execute("""
            CREATE REL TABLE KADRO_HAVAS_ROLON (
                FROM SemantikKadro TO KadraRolo,
                kerneco BOOLEAN,
                ordigo INT64
            )
        """, "Create KADRO_HAVAS_ROLON relationship")

    # ===================================================================
    # LAYER 3: DISCOURSE SEMANTICS
    # ===================================================================

    def create_layer3_discourse(self):
        """Create discourse relation tables (RST)."""

        logger.info("\n" + "=" * 60)
        logger.info("LAYER 3: DISCOURSE SEMANTICS")
        logger.info("=" * 60)

        # 1. DiskursaRilato (RST Relation)
        self.execute("""
            CREATE NODE TABLE DiskursaRilato (
                rilato_id STRING,
                rilato_nomo STRING,
                priskribo STRING,
                markantoj STRING[],
                strukturaj_indikilo STRING,
                kerna_rolo STRING,
                satelita_rolo STRING,
                tipo STRING,
                PRIMARY KEY (rilato_id)
            )
        """, "Create DiskursaRilato (Discourse Relation) node table")

        # 2. InformStruktura (Information Structure)
        self.execute("""
            CREATE NODE TABLE InformStruktura (
                strukturo_id STRING,
                strukturo_nomo STRING,
                priskribo STRING,
                ekzemploj STRING[],
                PRIMARY KEY (strukturo_id)
            )
        """, "Create InformStruktura (Information Structure) node table")

    # ===================================================================
    # LAYER 4: SCHEMA SEMANTICS
    # ===================================================================

    def create_layer4_schemas(self):
        """Create schema tables for biographical/definitional/event schemas."""

        logger.info("\n" + "=" * 60)
        logger.info("LAYER 4: SCHEMA SEMANTICS")
        logger.info("=" * 60)

        # 1. EnhavaSkemo (Content Schema)
        self.execute("""
            CREATE NODE TABLE EnhavaSkemo (
                skemo_id STRING,
                skemo_nomo STRING,
                priskribo STRING,
                demanda_tipo STRING,
                slotoj STRING[],
                PRIMARY KEY (skemo_id)
            )
        """, "Create EnhavaSkemo (Content Schema) node table")

        # 2. SkemaSloto (Schema Slot)
        self.execute("""
            CREATE NODE TABLE SkemaSloto (
                sloto_id STRING,
                sloto_nomo STRING,
                priskribo STRING,
                graveco_pezo DOUBLE,
                semantikaj_limigoj STRING[],
                verba_klasoj STRING[],
                substantiva_klasoj STRING[],
                ekzemploj STRING[],
                PRIMARY KEY (sloto_id)
            )
        """, "Create SkemaSloto (Schema Slot) node table")

        # Relationship: Schema has slot
        self.execute("""
            CREATE REL TABLE SKEMO_HAVAS_SLOTON (
                FROM EnhavaSkemo TO SkemaSloto,
                ordigo INT64,
                devigeco BOOLEAN DEFAULT false
            )
        """, "Create SKEMO_HAVAS_SLOTON relationship")

    # ===================================================================
    # POPULATE WITH CORE TAXONOMY
    # ===================================================================

    def populate_core_taxonomy(self):
        """Populate core verb/noun/adjective classes."""

        logger.info("\n" + "=" * 60)
        logger.info("POPULATING CORE TAXONOMY")
        logger.info("=" * 60)

        # Core verb classes (top-level only, detailed annotation comes later)
        verb_classes = [
            ("kreado-26", "Kreado", "Ago de krei aŭ produkti ion novan", None, ["fond", "kre", "produk", "far"], 0.95, 0.70, 0.85),
            ("movo-51", "Movo", "Ŝanĝo de loko aŭ pozicio", None, ["ir", "ven", "fur", "voj"], 0.60, 0.40, 0.80),
            ("pensado-29", "Pensado", "Mensa aktiveco kaj kogna procezo", None, ["pens", "sci", "kred", "komprend"], 0.70, 0.85, 0.60),
            ("perceptado-30", "Perceptado", "Senca percepto", None, ["vid", "aŭd", "sent", "gust"], 0.50, 0.60, 0.70),
            ("emocio-31", "Emocio", "Emocia stato aŭ sento", None, ["am", "ĝoj", "tim", "trist"], 0.65, 0.55, 0.75),
            ("komunikado-37", "Komunikado", "Interŝanĝo de informoj", None, ["dir", "parol", "demand", "respond"], 0.75, 0.80, 0.85),
            ("vivo-48", "Vivo", "Vivaĵaj procezoj", None, ["viv", "mort", "nask", "kresk"], 0.90, 0.70, 0.85),
            ("profesio-50", "Profesio", "Profesia aktiveco", None, ["labor", "instrui", "kurac", "vend"], 0.80, 0.75, 0.70),
        ]

        for klaso_id, nomo, priskribo, super_id, ekzemploj, grav_bio, grav_def, grav_okaz in verb_classes:
            ekzemploj_str = "', '".join(ekzemploj)
            self.execute(f"""
                CREATE (v:VerbaKlaso {{
                    klaso_id: '{klaso_id}',
                    klaso_nomo: '{nomo}',
                    priskribo: '{priskribo}',
                    superklaso_id: {f"'{super_id}'" if super_id else 'NULL'},
                    ekzemplaj_radikoj: ['{ekzemploj_str}'],
                    graveco_biografia: {grav_bio},
                    graveco_difina: {grav_def},
                    graveco_okazaĵa: {grav_okaz}
                }})
            """, f"Create verb class: {nomo}")

        # Core aspectual classes
        aspectual_classes = [
            ("stato", "Stato", "Statika situacio sen interna dinamiko", ["est", "hav", "apert"], False, True, False),
            ("aktiveco", "Aktiveco", "Daŭra ago sen interna finpunkto", ["kur", "parol", "labor"], False, True, True),
            ("plenumigo", "Plenumigo", "Procedo kun interna finpunkto", ["konstrui", "mort", "skrib"], True, True, True),
            ("atingaĵo", "Atingaĵo", "Momenta ŝanĝo de stato", ["trov", "kompreni", "ven"], True, False, True),
        ]

        for klaso_id, nomo, priskribo, ekzemploj, telikeco, durativeco, dinamikeco in aspectual_classes:
            ekzemploj_str = "', '".join(ekzemploj)
            self.execute(f"""
                CREATE (a:AspektaKlaso {{
                    klaso_id: '{klaso_id}',
                    klaso_nomo: '{nomo}',
                    priskribo: '{priskribo}',
                    ekzemploj: ['{ekzemploj_str}'],
                    telikeco: {str(telikeco).lower()},
                    durativeco: {str(durativeco).lower()},
                    dinamikeco: {str(dinamikeco).lower()}
                }})
            """, f"Create aspectual class: {nomo}")

        # Core thematic roles
        thematic_roles = [
            ("aganto", "Aganto", "Intenca inicianto de ago", ["Li konstruis domon"], True),
            ("paciento", "Paciento", "Enteco submetata al ago", ["La domo estis konstruita"], True),
            ("temo", "Temo", "Enteco movita aŭ priskribita", ["Li sendis leteron"], True),
            ("spertanto", "Spertanto", "Enteco spertanta staton", ["Mi amas muzikon"], True),
            ("instrumento", "Instrumento", "Ilo uzata por ago", ["Li tranĉis per tranĉilo"], False),
            ("fonto", "Fonto", "Devenpunkto de movo", ["Li venis el Pollando"], False),
            ("celo", "Celo", "Celpunkto de movo", ["Li iris al Varsovio"], False),
            ("loko", "Loko", "Lokacio de ago", ["Li laboras en Parizo"], False),
            ("tempo", "Tempo", "Tempa kadro", ["Li naskiĝis en 1887"], False),
        ]

        for rolo_id, nomo, priskribo, ekzemploj, kerneco in thematic_roles:
            ekzemploj_str = "', '".join(ekzemploj)
            self.execute(f"""
                CREATE (t:TemaRolo {{
                    rolo_id: '{rolo_id}',
                    rolo_nomo: '{nomo}',
                    priskribo: '{priskribo}',
                    ekzemploj: ['{ekzemploj_str}'],
                    kerneco: {str(kerneco).lower()}
                }})
            """, f"Create thematic role: {nomo}")

        # Core noun entity types
        entity_types = [
            ("persono", "Persono", "Homa individuo", "vivanta", ["homo", "vir", "infan", "kuracist"]),
            ("loko", "Loko", "Geografia aŭ fizika lokacio", "konkreta", ["urb", "land", "dom", "strat"]),
            ("tempo", "Tempo", "Tempa punkto aŭ periodo", "abstrakta", ["jar", "monat", "tag", "hor"]),
            ("organizaĵo", "Organizaĵo", "Socia strukturo", "abstrakta", ["kompani", "registar", "klub"]),
            ("eventoInformation", "Evento", "Okazaĵo aŭ evento", "abstrakta", ["kongres", "fest", "milit", "renkontiĝ"]),
            ("profesio", "Profesio", "Profesia rolo", "abstrakta", ["kuracist", "instruist", "verkist", "inĝenier"]),
        ]

        for tipo_id, nomo, priskribo, supergrupo, ekzemploj in entity_types:
            ekzemploj_str = "', '".join(ekzemploj)
            self.execute(f"""
                CREATE (e:EntecaTipo {{
                    tipo_id: '{tipo_id}',
                    tipo_nomo: '{nomo}',
                    priskribo: '{priskribo}',
                    supergrupo: '{supergrupo}',
                    ekzemploj: ['{ekzemploj_str}']
                }})
            """, f"Create entity type: {nomo}")

        # Core content schemas
        schemas = [
            ("biografia", "Biografia Skemo", "Strukturo por biografiaj resumoj", "WHO"),
            ("difina", "Difina Skemo", "Strukturo por difinoj", "WHAT"),
            ("okazaĵa", "Okazaĵa Skemo", "Strukturo por eventoj", "WHEN"),
        ]

        for skemo_id, nomo, priskribo, demanda_tipo in schemas:
            self.execute(f"""
                CREATE (s:EnhavaSkemo {{
                    skemo_id: '{skemo_id}',
                    skemo_nomo: '{nomo}',
                    priskribo: '{priskribo}',
                    demanda_tipo: '{demanda_tipo}',
                    slotoj: []
                }})
            """, f"Create content schema: {nomo}")

        # Core schema slots (biographical)
        biographical_slots = [
            ("identigo", "Identigo", "Kiu estas la persono", 1.0, ["verbo:est", "objekto:persono|profesio"], ["est"], ["persono", "profesio"]),
            ("ĉefa_realigo", "Ĉefa Realigo", "Plej grava kontribuo", 0.95, ["verbo:kreado", "aspekto:plenumigo"], ["fond", "kre", "produk"], ["organizaĵo", "sistemo"]),
            ("naskiĝo_morto", "Naskiĝo kaj Morto", "Datoj de vivo", 0.85, ["verbo:vivo"], ["nask", "mort"], ["tempo", "loko"]),
            ("profesio", "Profesio", "Profesia rolo", 0.80, ["substantivo:profesio"], [], ["profesio"]),
            ("loko", "Loko", "Geografia kunteksto", 0.70, ["substantivo:loko"], [], ["loko", "urb", "land"]),
        ]

        for sloto_id, nomo, priskribo, graveco, limigoj, verba_klasoj, subst_klasoj in biographical_slots:
            limigoj_str = "', '".join(limigoj)
            verba_str = "', '".join(verba_klasoj)
            subst_str = "', '".join(subst_klasoj)

            self.execute(f"""
                CREATE (sl:SkemaSloto {{
                    sloto_id: '{sloto_id}',
                    sloto_nomo: '{nomo}',
                    priskribo: '{priskribo}',
                    graveco_pezo: {graveco},
                    semantikaj_limigoj: ['{limigoj_str}'],
                    verba_klasoj: ['{verba_str}'],
                    substantiva_klasoj: ['{subst_str}'],
                    ekzemploj: []
                }})
            """, f"Create schema slot: {nomo}")

            # Link to biographical schema
            self.execute(f"""
                MATCH (s:EnhavaSkemo {{skemo_id: 'biografia'}}), (sl:SkemaSloto {{sloto_id: '{sloto_id}'}})
                CREATE (s)-[:SKEMO_HAVAS_SLOTON {{ordigo: {len(biographical_slots)}, devigeco: false}}]->(sl)
            """, f"Link slot {nomo} to biographical schema")

    def run(self):
        """Execute full schema extension."""
        logger.info("Starting Kuzu schema extension: Semantic Ontology v2.2")
        logger.info(f"Database: {self.db_path}")
        logger.info(f"Dry run: {self.dry_run}")
        logger.info("=" * 60)

        try:
            # Create all layers
            self.create_layer1_verb_classes()
            self.create_layer1_noun_classes()
            self.create_layer1_adjective_classes()
            self.create_layer2_frames()
            self.create_layer3_discourse()
            self.create_layer4_schemas()

            # Populate core taxonomy
            if not self.dry_run:
                self.populate_core_taxonomy()

            logger.info("\n" + "=" * 60)
            logger.info("✓ Schema extension complete!")
            logger.info("=" * 60)
            logger.info("\nNext steps:")
            logger.info("1. Annotate 200 core roots (scripts/annotate_core_vocabulary.py)")
            logger.info("2. Update retrieval to use semantic classes")
            logger.info("3. Update QA to use schema slots")

        except Exception as e:
            logger.error(f"\n✗ Schema extension failed: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(
        description='Extend Kuzu schema with 4-layer semantic ontology'
    )
    parser.add_argument('--db-path', type=Path,
                       default=Path('data/indexes/v2.1_kuzu_index_full'),
                       help='Path to Kuzu database')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show queries without executing')

    args = parser.parse_args()

    # Create logs directory
    Path('logs').mkdir(exist_ok=True)

    # Run extension
    extender = SemanticOntologySchemaExtension(args.db_path, args.dry_run)
    extender.run()


if __name__ == '__main__':
    main()
