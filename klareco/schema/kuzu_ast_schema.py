#!/usr/bin/env python3
"""
Kuzu AST-Native Schema Definition (v2.0)

This module defines the complete graph schema for storing parsed ASTs,
document hierarchy, and extensible annotations in Kuzu.

Design principles:
1. ASTs stored as native graph nodes (immutable, versioned)
2. Document hierarchy: Collection → Document → Section → Paragraph → Sentence
3. Extensible annotation system (zero schema changes for new types)
4. Circular vocabulary: Annotations link to Esperanto roots
5. Full reconstruction: Can rebuild exact original AST + annotations

See: #592 (EPIC), docs/EVALUATION_ANALYSIS_v1.0.md
"""

from typing import List

# ============================================================================
# SCHEMA VERSION
# ============================================================================

SCHEMA_VERSION = "2.0.0-alpha"


# ============================================================================
# DOCUMENT HIERARCHY SCHEMA
# ============================================================================

HIERARCHY_NODE_SCHEMA = """
-- Source Collection (Top level: Wikipedia, Gutenberg, ReVo)
CREATE NODE TABLE IF NOT EXISTS SourceCollection (
    id INT64 PRIMARY KEY,
    name STRING,              -- "Vikipedio", "Project Gutenberg"
    source_type STRING,       -- "encyclopedia", "book", "dictionary"
    language STRING,          -- "eo"
    metadata STRING           -- JSON: year, author, etc.
);

-- Document (Wikipedia article, book, dictionary entry)
CREATE NODE TABLE IF NOT EXISTS Document (
    id INT64 PRIMARY KEY,
    collection_id INT64,
    title STRING,             -- Article title, book title
    external_id STRING,       -- Wikipedia article_id, ISBN
    doc_type STRING,          -- "article", "book", "qa"
    author STRING,
    year INT64,
    quality STRING,           -- "GOLD", "SILVER", "BRONZE"
    metadata STRING           -- JSON: timestamp, revision, etc.
);

-- Section (Wikipedia section, book chapter)
CREATE NODE TABLE IF NOT EXISTS Section (
    id INT64 PRIMARY KEY,
    doc_id INT64,
    section_name STRING,      -- "Historio", "Chapter 3"
    section_level INT64,      -- 1 = top-level, 2 = subsection
    section_order INT64,      -- Order within document
    parent_section_id INT64   -- For nested sections (NULL for top-level)
);

-- Paragraph (paragraph within section)
CREATE NODE TABLE IF NOT EXISTS Paragraph (
    id INT64 PRIMARY KEY,
    section_id INT64,
    paragraph_order INT64,    -- Order within section
    paragraph_type STRING,    -- "text", "list_item", "quote"
    metadata STRING           -- JSON: list level, quote depth
);

-- Sentence (individual sentence)
CREATE NODE TABLE IF NOT EXISTS Sentence (
    id INT64 PRIMARY KEY,
    paragraph_id INT64,
    text STRING,              -- Original text (for display)
    sentence_order INT64,     -- Order within paragraph
    global_order INT64        -- Order within entire document (for pronoun resolution)
);
"""

HIERARCHY_REL_SCHEMA = """
-- Hierarchy edges (navigation up/down)
CREATE REL TABLE IF NOT EXISTS IN_COLLECTION (FROM Document TO SourceCollection);
CREATE REL TABLE IF NOT EXISTS IN_DOCUMENT (FROM Section TO Document);
CREATE REL TABLE IF NOT EXISTS IN_SECTION (FROM Paragraph TO Section);
CREATE REL TABLE IF NOT EXISTS IN_PARAGRAPH (FROM Sentence TO Paragraph);
CREATE REL TABLE IF NOT EXISTS PARENT_SECTION (FROM Section TO Section);

-- Ordering edges (navigation left/right)
CREATE REL TABLE IF NOT EXISTS NEXT_SECTION (FROM Section TO Section);
CREATE REL TABLE IF NOT EXISTS NEXT_PARAGRAPH (FROM Paragraph TO Paragraph);
CREATE REL TABLE IF NOT EXISTS NEXT_SENTENCE (FROM Sentence TO Sentence);
"""


# ============================================================================
# AST STORAGE SCHEMA (Immutable parser output)
# ============================================================================

AST_NODE_SCHEMA = """
-- AST: Complete parsed syntax tree (versioned, immutable)
CREATE NODE TABLE IF NOT EXISTS AST (
    id INT64 PRIMARY KEY,
    sentence_id INT64,
    version INT64,              -- AST version (parser v1, v2, etc.)
    created_at TIMESTAMP,
    created_by STRING,          -- "parser_v1.2.3"
    is_current BOOLEAN,         -- TRUE for latest version

    -- AST-level metadata from parser
    fraztipo STRING,            -- "demando", "deklaro", "ordono"
    demandotipo STRING,         -- "ki", "ĉu", NULL
    negita BOOLEAN,             -- Is sentence negated?

    -- Parse statistics
    total_words INT64,
    esperanto_words INT64,
    non_esperanto_words INT64,
    success_rate DOUBLE,
    parse_categories STRING     -- JSON: {"foreign_word": 1, ...}
);

-- Frazo: Sentence-level AST node
CREATE NODE TABLE IF NOT EXISTS Frazo (
    id INT64 PRIMARY KEY,
    ast_id INT64,               -- Which AST version
    tipo STRING DEFAULT 'frazo'
);

-- Vortgrupo: Word group (noun phrase, etc.)
CREATE NODE TABLE IF NOT EXISTS Vortgrupo (
    id INT64 PRIMARY KEY,
    ast_id INT64,
    tipo STRING DEFAULT 'vortgrupo'
);

-- Vorto: Individual word (all parser fields, immutable)
CREATE NODE TABLE IF NOT EXISTS Vorto (
    id INT64 PRIMARY KEY,
    ast_id INT64,               -- Which AST version

    -- Core fields
    plena_vorto STRING,
    radiko STRING,
    vortspeco STRING,

    -- Grammar
    nombro STRING,
    kazo STRING,
    tempo STRING,
    modo STRING,

    -- Participles
    participo_voco STRING,
    participo_tempo STRING,

    -- Affixes (JSON arrays)
    prefiksoj STRING,           -- JSON: ["re", "mal"]
    sufiksoj STRING,            -- JSON: ["ig", "il", "et"]

    -- Parse metadata (immutable from parser)
    parse_status STRING,        -- "success", "unknown_root", "failed"
    parse_error STRING,
    category STRING,            -- "foreign_word", "proper_name_known"

    -- Proper nouns
    proper_noun_category STRING,
    proper_noun_frequency INT64,

    -- Correlatives
    korelativo_prefikso STRING,
    korelativo_sufikso STRING,
    korelativo_signifo STRING,

    -- Compound words
    estas_kunmetita BOOLEAN,
    kunmetitaj_radikoj STRING  -- JSON: ["verk", "jar"]
);
"""




# ============================================================================
# FULL SCHEMA (Combined - NODES FIRST, THEN RELS)
# ============================================================================

# Split schemas into node and rel parts
AST_REL_SCHEMA = """
-- AST structure edges
CREATE REL TABLE IF NOT EXISTS SENTENCE_HAS_AST (
    FROM Sentence TO AST,
    is_current BOOLEAN
);

CREATE REL TABLE IF NOT EXISTS AST_HAS_FRAZO (FROM AST TO Frazo);

-- Frazo structure (union types: can link to Vortgrupo OR Vorto)
CREATE REL TABLE IF NOT EXISTS HAS_SUBJEKTO_VORTGRUPO (FROM Frazo TO Vortgrupo);
CREATE REL TABLE IF NOT EXISTS HAS_SUBJEKTO_VORTO (FROM Frazo TO Vorto);
CREATE REL TABLE IF NOT EXISTS HAS_VERBO (FROM Frazo TO Vorto);
CREATE REL TABLE IF NOT EXISTS HAS_OBJEKTO_VORTGRUPO (FROM Frazo TO Vortgrupo);
CREATE REL TABLE IF NOT EXISTS HAS_OBJEKTO_VORTO (FROM Frazo TO Vorto);
CREATE REL TABLE IF NOT EXISTS HAS_ALIAJ (
    FROM Frazo TO Vorto,
    position INT64
);

-- Vortgrupo structure
CREATE REL TABLE IF NOT EXISTS HAS_KERNO (FROM Vortgrupo TO Vorto);
CREATE REL TABLE IF NOT EXISTS HAS_PRISKRIBO (
    FROM Vortgrupo TO Vorto,
    position INT64
);

-- Recursive: Compound word parts
CREATE REL TABLE IF NOT EXISTS HAS_KUNMETAJHO (
    FROM Vorto TO Vorto,
    position INT64
);
"""

ROOT_NODE_SCHEMA = """
-- Root: Esperanto root index (for fast lookup)
CREATE NODE TABLE IF NOT EXISTS Root (
    root STRING PRIMARY KEY,
    doc_freq INT64,
    total_freq INT64
);
"""

ROOT_REL_SCHEMA = """
-- Link words to roots
CREATE REL TABLE IF NOT EXISTS HAS_ROOT (
    FROM Vorto TO Root,
    is_primary BOOLEAN,
    position INT64
);

-- Semantic relations
CREATE REL TABLE IF NOT EXISTS IS_SYNONYM (FROM Root TO Root);
CREATE REL TABLE IF NOT EXISTS IS_HYPERNYM (FROM Root TO Root);
CREATE REL TABLE IF NOT EXISTS IS_ANTONYM (FROM Root TO Root);
"""

ANNOTATION_NODE_SCHEMA = """
-- AnnotationType: Defines what kinds of annotations exist
CREATE NODE TABLE IF NOT EXISTS AnnotationType (
    id INT64 PRIMARY KEY,
    type_name STRING,
    description STRING,
    root STRING,
    value_type STRING,
    created_by STRING,
    schema_version INT64
);

-- AnnotationValue: The actual values annotations can have
CREATE NODE TABLE IF NOT EXISTS AnnotationValue (
    id INT64 PRIMARY KEY,
    value_name STRING,
    root STRING,
    value_type STRING,
    numeric_value DOUBLE,
    metadata STRING
);

-- Annotation: Links entities to annotation values
CREATE NODE TABLE IF NOT EXISTS Annotation (
    id INT64 PRIMARY KEY,
    annotation_set_id INT64,
    annotation_type_id INT64,
    annotation_value_id INT64,
    confidence DOUBLE,
    metadata STRING
);

-- AnnotationSet: Versioned collection of annotations
CREATE NODE TABLE IF NOT EXISTS AnnotationSet (
    id INT64 PRIMARY KEY,
    name STRING,
    version INT64,
    created_at TIMESTAMP,
    created_by STRING,
    based_on_ast_version INT64,
    is_active BOOLEAN
);
"""

ANNOTATION_REL_SCHEMA = """
-- Annotation edges (can attach to ANY level)
CREATE REL TABLE IF NOT EXISTS WORD_HAS_ANNOTATION (FROM Vorto TO Annotation);
CREATE REL TABLE IF NOT EXISTS VORTGRUPO_HAS_ANNOTATION (FROM Vortgrupo TO Annotation);
CREATE REL TABLE IF NOT EXISTS FRAZO_HAS_ANNOTATION (FROM Frazo TO Annotation);
CREATE REL TABLE IF NOT EXISTS AST_HAS_ANNOTATION (FROM AST TO Annotation);
CREATE REL TABLE IF NOT EXISTS SENTENCE_HAS_ANNOTATION (FROM Sentence TO Annotation);
CREATE REL TABLE IF NOT EXISTS PARAGRAPH_HAS_ANNOTATION (FROM Paragraph TO Annotation);
CREATE REL TABLE IF NOT EXISTS SECTION_HAS_ANNOTATION (FROM Section TO Annotation);
CREATE REL TABLE IF NOT EXISTS DOCUMENT_HAS_ANNOTATION (FROM Document TO Annotation);

-- Annotation system edges
CREATE REL TABLE IF NOT EXISTS IN_ANNOTATION_SET (FROM Annotation TO AnnotationSet);
CREATE REL TABLE IF NOT EXISTS ANNOTATION_HAS_TYPE (FROM Annotation TO AnnotationType);
CREATE REL TABLE IF NOT EXISTS ANNOTATION_HAS_VALUE (FROM Annotation TO AnnotationValue);
CREATE REL TABLE IF NOT EXISTS ANNOTATES_AST (FROM AnnotationSet TO AST);

-- Circular vocabulary: Annotations link to roots
CREATE REL TABLE IF NOT EXISTS ANNOTATION_TYPE_IS_ROOT (FROM AnnotationType TO Root);
CREATE REL TABLE IF NOT EXISTS ANNOTATION_VALUE_IS_ROOT (FROM AnnotationValue TO Root);
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
    """Print summary of schema."""
    print(f"Kuzu AST-Native Schema v{SCHEMA_VERSION}")
    print("=" * 80)
    print()
    print("Node Tables:")
    print("  Document Hierarchy: SourceCollection, Document, Section, Paragraph, Sentence")
    print("  AST Structure: AST, Frazo, Vortgrupo, Vorto")
    print("  Root Index: Root")
    print("  Annotations: AnnotationType, AnnotationValue, Annotation, AnnotationSet")
    print()
    print("Total Statements:", len(get_create_statements()))
    print()


if __name__ == '__main__':
    print_schema_summary()
    print("\nSchema Statements:")
    print("=" * 80)
    for i, stmt in enumerate(get_create_statements(), 1):
        print(f"\n-- Statement {i}")
        print(stmt)
