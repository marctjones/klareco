#!/usr/bin/env python3
"""
Kuzu AST-Native Schema Definition (v2.1 - Pure Esperanto)

This module defines the complete graph schema for storing parsed ASTs,
document hierarchy, and extensible annotations in Kuzu.

ALL NAMES IN PURE ESPERANTO (except technical acronyms: AST, CSV, JSON)

Design principles:
1. ASTs stored as native graph nodes (immutable, versioned)
2. Document hierarchy: Fontaro → Dokumento → Sekcio → Paragrafo → Frazoteksto
3. Extensible annotation system (zero schema changes for new types)
4. Circular vocabulary: Annotations link to Esperanto roots
5. Full reconstruction: Can rebuild exact original AST + annotations

Version 2.1 Changes:
- Pure Esperanto naming (was mixed English/Esperanto in v2.0)
- Node names: Fontaro, Dokumento, Sekcio, Paragrafo, Frazoteksto
- Relationship names: HAVAS_, ESTAS_, EN_, SEKVA_
- Property names: Esperanto where possible
- Property values: Esperanto (ORO/ARĜENTO/BRONZO, sukceso/malsukceso)

See: #592 (EPIC), docs/NAMING_CONSISTENCY_AUDIT.md
"""

from typing import List

# ============================================================================
# SCHEMA VERSION
# ============================================================================

SCHEMA_VERSION = "2.1.0-beta"


# ============================================================================
# DOCUMENT HIERARCHY SCHEMA (Pure Esperanto)
# ============================================================================

HIERARCHY_NODE_SCHEMA = """
CREATE NODE TABLE IF NOT EXISTS Fontaro (
    id INT64 PRIMARY KEY,
    nomo STRING,
    fontotipo STRING,
    lingvo STRING,
    metadatenoj STRING
);

CREATE NODE TABLE IF NOT EXISTS Dokumento (
    id INT64 PRIMARY KEY,
    fontaro_id INT64,
    titolo STRING,
    ekstera_id STRING,
    dokumentipo STRING,
    aŭtoro STRING,
    jaro INT64,
    kvalito STRING,
    metadatenoj STRING
);

CREATE NODE TABLE IF NOT EXISTS Sekcio (
    id INT64 PRIMARY KEY,
    dokumento_id INT64,
    sekcio_nomo STRING,
    sekcio_nivelo INT64,
    sekcio_ordo INT64,
    gepatra_sekcio_id INT64
);

CREATE NODE TABLE IF NOT EXISTS Paragrafo (
    id INT64 PRIMARY KEY,
    sekcio_id INT64,
    paragrafo_ordo INT64,
    paragrafo_tipo STRING,
    metadatenoj STRING
);

CREATE NODE TABLE IF NOT EXISTS Frazoteksto (
    id INT64 PRIMARY KEY,
    paragrafo_id INT64,
    teksto STRING,
    frazo_ordo INT64,
    tutmonda_ordo INT64
);
"""

HIERARCHY_REL_SCHEMA = """
CREATE REL TABLE IF NOT EXISTS EN_FONTARO (FROM Dokumento TO Fontaro);
CREATE REL TABLE IF NOT EXISTS EN_DOKUMENTO (FROM Sekcio TO Dokumento);
CREATE REL TABLE IF NOT EXISTS EN_SEKCIO (FROM Paragrafo TO Sekcio);
CREATE REL TABLE IF NOT EXISTS EN_PARAGRAFO (FROM Frazoteksto TO Paragrafo);
CREATE REL TABLE IF NOT EXISTS GEPATRA_SEKCIO (FROM Sekcio TO Sekcio);

CREATE REL TABLE IF NOT EXISTS SEKVA_SEKCIO (FROM Sekcio TO Sekcio);
CREATE REL TABLE IF NOT EXISTS SEKVA_PARAGRAFO (FROM Paragrafo TO Paragrafo);
CREATE REL TABLE IF NOT EXISTS SEKVA_FRAZOTEKSTO (FROM Frazoteksto TO Frazoteksto);
"""


# ============================================================================
# AST STORAGE SCHEMA (Immutable parser output - Pure Esperanto)
# ============================================================================

AST_NODE_SCHEMA = """
CREATE NODE TABLE IF NOT EXISTS AST (
    id INT64 PRIMARY KEY,
    frazoteksto_id INT64,
    versio INT64,
    kreita_je TIMESTAMP,
    kreita_de STRING,
    estas_nuna BOOLEAN,

    fraztipo STRING,
    demandotipo STRING,
    negita BOOLEAN,

    tutaj_vortoj INT64,
    esperantaj_vortoj INT64,
    neesperantaj_vortoj INT64,
    sukcesoprocento DOUBLE,
    analizkategorioj STRING
);

CREATE NODE TABLE IF NOT EXISTS Frazo (
    id INT64 PRIMARY KEY,
    ast_id INT64,
    tipo STRING DEFAULT 'frazo'
);

CREATE NODE TABLE IF NOT EXISTS Vortgrupo (
    id INT64 PRIMARY KEY,
    ast_id INT64,
    tipo STRING DEFAULT 'vortgrupo'
);

CREATE NODE TABLE IF NOT EXISTS Vorto (
    id INT64 PRIMARY KEY,
    ast_id INT64,

    plena_vorto STRING,
    radiko STRING,
    vortspeco STRING,

    nombro STRING,
    kazo STRING,
    tempo STRING,
    modo STRING,

    participo_voco STRING,
    participo_tempo STRING,

    prefiksoj STRING,
    sufiksoj STRING,

    analizstato STRING,
    analizeraro STRING,
    kategorio STRING,

    propranoma_kategorio STRING,
    propranoma_ofteco INT64,

    korelativo_prefikso STRING,
    korelativo_sufikso STRING,
    korelativo_signifo STRING,

    estas_kunmetita BOOLEAN,
    kunmetitaj_radikoj STRING
);
"""

AST_REL_SCHEMA = """
CREATE REL TABLE IF NOT EXISTS FRAZOTEKSTO_HAVAS_AST (
    FROM Frazoteksto TO AST,
    estas_nuna BOOLEAN
);

CREATE REL TABLE IF NOT EXISTS AST_HAVAS_FRAZON (FROM AST TO Frazo);

CREATE REL TABLE IF NOT EXISTS HAVAS_SUBJEKTON_VORTGRUPO (FROM Frazo TO Vortgrupo);
CREATE REL TABLE IF NOT EXISTS HAVAS_SUBJEKTON_VORTO (FROM Frazo TO Vorto);
CREATE REL TABLE IF NOT EXISTS HAVAS_VERBON (FROM Frazo TO Vorto);
CREATE REL TABLE IF NOT EXISTS HAVAS_OBJEKTON_VORTGRUPO (FROM Frazo TO Vortgrupo);
CREATE REL TABLE IF NOT EXISTS HAVAS_OBJEKTON_VORTO (FROM Frazo TO Vorto);
CREATE REL TABLE IF NOT EXISTS HAVAS_ALIAJN (
    FROM Frazo TO Vorto,
    pozicio INT64
);

CREATE REL TABLE IF NOT EXISTS HAVAS_KERNON (FROM Vortgrupo TO Vorto);
CREATE REL TABLE IF NOT EXISTS HAVAS_PRISKRIBON (
    FROM Vortgrupo TO Vorto,
    pozicio INT64
);

CREATE REL TABLE IF NOT EXISTS HAVAS_KUNMETAĴON (
    FROM Vorto TO Vorto,
    pozicio INT64
);
"""

ROOT_NODE_SCHEMA = """
CREATE NODE TABLE IF NOT EXISTS Radiko (
    radiko STRING PRIMARY KEY,
    dokumenta_ofteco INT64,
    tuta_ofteco INT64
);
"""

ROOT_REL_SCHEMA = """
CREATE REL TABLE IF NOT EXISTS HAVAS_RADIKON (
    FROM Vorto TO Radiko,
    estas_ĉefa BOOLEAN,
    pozicio INT64
);

CREATE REL TABLE IF NOT EXISTS ESTAS_SINONIMO (FROM Radiko TO Radiko);
CREATE REL TABLE IF NOT EXISTS ESTAS_HIPERONIMO (FROM Radiko TO Radiko);
CREATE REL TABLE IF NOT EXISTS ESTAS_ANTONIMO (FROM Radiko TO Radiko);
"""

ANNOTATION_NODE_SCHEMA = """
CREATE NODE TABLE IF NOT EXISTS AnotacioTipo (
    id INT64 PRIMARY KEY,
    tipo_nomo STRING,
    priskribo STRING,
    radiko STRING,
    valoro_tipo STRING,
    kreita_de STRING,
    skema_versio INT64
);

CREATE NODE TABLE IF NOT EXISTS AnotacioValoro (
    id INT64 PRIMARY KEY,
    valoro_nomo STRING,
    radiko STRING,
    valoro_tipo STRING,
    cifera_valoro DOUBLE,
    metadatenoj STRING
);

CREATE NODE TABLE IF NOT EXISTS Anotacio (
    id INT64 PRIMARY KEY,
    anotacioaro_id INT64,
    anotaciotipo_id INT64,
    anotaciovaloro_id INT64,
    konfido DOUBLE,
    metadatenoj STRING
);

CREATE NODE TABLE IF NOT EXISTS AnotacioAro (
    id INT64 PRIMARY KEY,
    nomo STRING,
    versio INT64,
    kreita_je TIMESTAMP,
    kreita_de STRING,
    bazita_sur_ast_versio INT64,
    estas_aktiva BOOLEAN
);
"""

ANNOTATION_REL_SCHEMA = """
CREATE REL TABLE IF NOT EXISTS VORTO_HAVAS_ANOTACION (FROM Vorto TO Anotacio);
CREATE REL TABLE IF NOT EXISTS VORTGRUPO_HAVAS_ANOTACION (FROM Vortgrupo TO Anotacio);
CREATE REL TABLE IF NOT EXISTS FRAZO_HAVAS_ANOTACION (FROM Frazo TO Anotacio);
CREATE REL TABLE IF NOT EXISTS AST_HAVAS_ANOTACION (FROM AST TO Anotacio);
CREATE REL TABLE IF NOT EXISTS FRAZOTEKSTO_HAVAS_ANOTACION (FROM Frazoteksto TO Anotacio);
CREATE REL TABLE IF NOT EXISTS PARAGRAFO_HAVAS_ANOTACION (FROM Paragrafo TO Anotacio);
CREATE REL TABLE IF NOT EXISTS SEKCIO_HAVAS_ANOTACION (FROM Sekcio TO Anotacio);
CREATE REL TABLE IF NOT EXISTS DOKUMENTO_HAVAS_ANOTACION (FROM Dokumento TO Anotacio);

CREATE REL TABLE IF NOT EXISTS EN_ANOTACIOARO (FROM Anotacio TO AnotacioAro);
CREATE REL TABLE IF NOT EXISTS ANOTACIO_HAVAS_TIPON (FROM Anotacio TO AnotacioTipo);
CREATE REL TABLE IF NOT EXISTS ANOTACIO_HAVAS_VALORON (FROM Anotacio TO AnotacioValoro);
CREATE REL TABLE IF NOT EXISTS ANOTACIAS_AST (FROM AnotacioAro TO AST);

CREATE REL TABLE IF NOT EXISTS ANOTACIOTIPO_ESTAS_RADIKO (FROM AnotacioTipo TO Radiko);
CREATE REL TABLE IF NOT EXISTS ANOTACIOVALORO_ESTAS_RADIKO (FROM AnotacioValoro TO Radiko);
"""

# CRITICAL: Kuzu requires all NODE tables created first, then REL tables
FULL_SCHEMA = [
    # Phase 1: Create all NODE tables
    HIERARCHY_NODE_SCHEMA,
    AST_NODE_SCHEMA,
    ROOT_NODE_SCHEMA,
    ANNOTATION_NODE_SCHEMA,
    # Phase 2: Create all REL tables
    HIERARCHY_REL_SCHEMA,
    AST_REL_SCHEMA,
    ROOT_REL_SCHEMA,
    ANNOTATION_REL_SCHEMA,
]


def get_create_statements() -> List[str]:
    """
    Get all CREATE statements as a list.

    Returns:
        List of SQL statements to create the schema.
    """
    statements = []
    for schema_block in FULL_SCHEMA:
        # Split by semicolon, clean up
        for stmt in schema_block.split(';'):
            stmt = stmt.strip()
            # Keep statements that contain CREATE
            if stmt and 'CREATE' in stmt.upper():
                # Remove ALL comments (both line comments and inline comments)
                lines = []
                for line in stmt.split('\n'):
                    # Remove inline comments (everything after --)
                    if '--' in line:
                        line = line[:line.index('--')]
                    # Skip pure comment lines (now empty after stripping)
                    if line.strip():
                        lines.append(line)
                cleaned_stmt = '\n'.join(lines).strip()
                if cleaned_stmt:
                    statements.append(cleaned_stmt + ';')
    return statements


def print_schema_summary():
    """Print a summary of the schema."""
    print(f"Schema Version: {SCHEMA_VERSION}")
    print("\nNode Tables:")
    print("  Hierarchy: Fontaro, Dokumento, Sekcio, Paragrafo, Frazoteksto")
    print("  AST: AST, Frazo, Vortgrupo, Vorto")
    print("  Index: Radiko")
    print("  Annotations: AnotacioTipo, AnotacioValoro, Anotacio, AnotacioAro")
    print("\nRelationship Tables:")
    print("  Hierarchy: EN_*, SEKVA_*, GEPATRA_*")
    print("  AST: HAVAS_*, AST_HAVAS_FRAZON, FRAZOTEKSTO_HAVAS_AST")
    print("  Index: HAVAS_RADIKON, ESTAS_SINONIMO/HIPERONIMO/ANTONIMO")
    print("  Annotations: *_HAVAS_ANOTACION, EN_ANOTACIOARO, ANOTACIO_HAVAS_*")


if __name__ == '__main__':
    print_schema_summary()
    print(f"\nTotal CREATE statements: {len(get_create_statements())}")
