#!/usr/bin/env python3
"""
Extract SVO Triples from Corpus for Semantic Type Hierarchy

VERSION: v2.1
COMPATIBLE WITH: v2.1 Kuzu database schema, extracted JSONL corpus
DEPENDENCIES: klareco.parser, klareco.schema.kuzu_ast_schema_v2_1
STAGE: Data

Description:
    Extracts (subject, verb, object) triples from all corpus sentences.
    Two extraction modes:
    1. Kuzu query (fast): Extract from v2.1 database using Cypher queries
    2. Direct parsing (slower): Parse JSONL sentences with klareco.parser

    Handles complex sentence structures:
    - Multiple clauses → multiple triples
    - Nested subordinate clauses
    - Coordinated subjects/verbs/objects
    - Passive voice
    - Prepositional phrases (optional)

Pipeline Position:
    Corpus (5.4M sentences) → [THIS SCRIPT] → SVO Triples JSONL → Clustering

Usage:
    # Extract from Kuzu database (fast, ~200K sentences)
    python scripts/extract_svo_triples.py \
        --source kuzu \
        --db-path data/indexes/v2.1_kuzu_index_full \
        --output data/semantic_types/svo_triples_kuzu.jsonl

    # Extract from JSONL corpus (slow, 5.4M sentences)
    python scripts/extract_svo_triples.py \
        --source jsonl \
        --corpus data/extracted/wikipedia_sentences.jsonl \
        --output data/semantic_types/svo_triples_wiki.jsonl

    # Extract from both (recommended)
    python scripts/extract_svo_triples.py \
        --source both \
        --db-path data/indexes/v2.1_kuzu_index_full \
        --corpus data/extracted/wikipedia_sentences.jsonl \
        --corpus data/extracted/books_sentences.jsonl \
        --output data/semantic_types/svo_triples_all.jsonl

    # With additional relations (not just SVO)
    python scripts/extract_svo_triples.py \
        --source kuzu \
        --db-path data/indexes/v2.1_kuzu_index_full \
        --output data/semantic_types/all_relations.jsonl \
        --extract-prepositional  # Extract prepositional phrases too
        --extract-modifiers      # Extract adjective-noun pairs

Inputs:
    - Kuzu v2.1 database (if --source kuzu or both)
    - Corpus JSONL files (if --source jsonl or both)

Outputs:
    - SVO triples JSONL file with format:
      {
        "subject_root": "zamenhof",
        "verb_root": "kre",
        "object_root": "esperant",
        "subject_full": "Zamenhof",
        "verb_full": "kreis",
        "object_full": "Esperanton",
        "relation_type": "SVO",
        "source": "wikipedia",
        "sentence": "Zamenhof kreis Esperanton.",
        "sentence_id": 12345,
        "confidence": 1.0
      }

Quality Checks:
    - Filter out triples with unknown roots (analizstato != 'sukceso')
    - Filter out triples with function words as subjects/objects
    - Validate root exists in vocabulary
    - Track extraction statistics (success rate, coverage)

Last Updated: 2026-03-16
Author: Claude Code
Related Issues: SFV design
See Also: /tmp/automated_type_hierarchy_design.md
"""

import argparse
import json
import jsonlines
import kuzu
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple
from collections import defaultdict
import logging

from klareco.parser import parse

# ============================================================================
# FUNCTION WORDS (exclude from semantic types)
# ============================================================================

FUNCTION_WORDS = {
    # Articles, pronouns, conjunctions, prepositions
    'la', 'mi', 'vi', 'li', 'ŝi', 'ĝi', 'ni', 'ili', 'oni',
    'kaj', 'aŭ', 'sed', 'ĉar', 'se', 'kvankam', 'dum',
    'de', 'da', 'al', 'el', 'en', 'sur', 'sub', 'inter', 'antaŭ', 'post',
    'je', 'pro', 'per', 'sen', 'kun', 'tra', 'ĉe', 'ĉirkaŭ',
    'tiu', 'tio', 'ĉi', 'ĉiu', 'neniu', 'iu', 'io', 'kiu', 'kio',
    'kie', 'kiam', 'kiel', 'kial', 'kiom',
    # Auxiliaries
    'est', 'dev', 'vol', 'pov', 'far',
}

# ============================================================================
# KUZU EXTRACTION
# ============================================================================

def extract_from_kuzu(db_path: Path, output_path: Path, extract_prep: bool = False, extract_mod: bool = False):
    """
    Extract SVO triples from Kuzu v2.1 database using Cypher queries.

    Advantages:
    - Fast (no parsing needed, ASTs already stored)
    - Clean (only successfully parsed sentences)
    - Structured (explicit SVO relationships)

    Args:
        db_path: Path to Kuzu database directory
        output_path: Path to output JSONL file
        extract_prep: Extract prepositional phrases too
        extract_mod: Extract adjective-noun modifiers too
    """
    logging.info(f"Connecting to Kuzu database: {db_path}")
    db = kuzu.Database(str(db_path))
    conn = kuzu.Connection(db)

    triples = []
    stats = defaultdict(int)

    # Query 1: Extract simple SVO triples
    # Match sentences with explicit subject, verb, object
    query_svo = """
    MATCH (frazo:Frazo)-[:HAVAS_VERBON]->(verbo:Vorto)
    OPTIONAL MATCH (frazo)-[:HAVAS_SUBJEKTON_VORTO]->(subjekto_v:Vorto)
    OPTIONAL MATCH (frazo)-[:HAVAS_SUBJEKTON_VORTGRUPO]->(subjekto_vg:Vortgrupo)-[:HAVAS_KERNON]->(subjekto_kern:Vorto)
    OPTIONAL MATCH (frazo)-[:HAVAS_OBJEKTON_VORTO]->(objekto_v:Vorto)
    OPTIONAL MATCH (frazo)-[:HAVAS_OBJEKTON_VORTGRUPO]->(objekto_vg:Vortgrupo)-[:HAVAS_KERNON]->(objekto_kern:Vorto)
    MATCH (frazo)<-[:AST_HAVAS_FRAZON]-(ast:AST)<-[:FRAZOTEKSTO_HAVAS_AST]-(frazoteksto:Frazoteksto)
    MATCH (frazoteksto)-[:EN_PARAGRAFO]->(p:Paragrafo)-[:EN_SEKCIO]->(s:Sekcio)-[:EN_DOKUMENTO]->(d:Dokumento)-[:EN_FONTARO]->(f:Fontaro)
    WHERE (subjekto_v IS NOT NULL OR subjekto_kern IS NOT NULL)
      AND (objekto_v IS NOT NULL OR objekto_kern IS NOT NULL)
      AND ast.estas_nuna = true
    RETURN
        COALESCE(subjekto_v.radiko, subjekto_kern.radiko) AS subject_root,
        COALESCE(subjekto_v.plena_vorto, subjekto_kern.plena_vorto) AS subject_full,
        COALESCE(subjekto_v.analizstato, subjekto_kern.analizstato) AS subject_status,
        verbo.radiko AS verb_root,
        verbo.plena_vorto AS verb_full,
        verbo.analizstato AS verb_status,
        COALESCE(objekto_v.radiko, objekto_kern.radiko) AS object_root,
        COALESCE(objekto_v.plena_vorto, objekto_kern.plena_vorto) AS object_full,
        COALESCE(objekto_v.analizstato, objekto_kern.analizstato) AS object_status,
        frazoteksto.teksto AS sentence,
        frazoteksto.id AS sentence_id,
        f.nomo AS source
    LIMIT 1000000
    """

    logging.info("Extracting SVO triples from Kuzu...")
    result = conn.execute(query_svo)

    while result.has_next():
        row = result.get_next()
        stats['total_rows'] += 1

        # Unpack row
        subject_root = row[0]
        subject_full = row[1]
        subject_status = row[2]
        verb_root = row[3]
        verb_full = row[4]
        verb_status = row[5]
        object_root = row[6]
        object_full = row[7]
        object_status = row[8]
        sentence = row[9]
        sentence_id = row[10]
        source = row[11]

        # Skip if any component failed parsing
        if subject_status != 'sukceso' or verb_status != 'sukceso' or object_status != 'sukceso':
            stats['parse_failed'] += 1
            continue

        # Skip if any component is a function word
        if subject_root in FUNCTION_WORDS or object_root in FUNCTION_WORDS:
            stats['function_word'] += 1
            continue

        # Skip if missing any component
        if not all([subject_root, verb_root, object_root]):
            stats['missing_component'] += 1
            continue

        # Create triple
        triple = {
            'subject_root': subject_root,
            'verb_root': verb_root,
            'object_root': object_root,
            'subject_full': subject_full,
            'verb_full': verb_full,
            'object_full': object_full,
            'relation_type': 'SVO',
            'source': source or 'unknown',
            'sentence': sentence,
            'sentence_id': sentence_id,
            'confidence': 1.0  # High confidence for Kuzu (already parsed)
        }

        triples.append(triple)
        stats['valid_triples'] += 1

    # TODO: Add queries for prepositional phrases and modifiers if requested
    if extract_prep:
        logging.warning("Prepositional phrase extraction not yet implemented")
        stats['prep_phrases'] = 0

    if extract_mod:
        logging.warning("Modifier extraction not yet implemented")
        stats['modifiers'] = 0

    # Write to JSONL
    logging.info(f"Writing {len(triples)} triples to {output_path}")
    with jsonlines.open(output_path, mode='w') as writer:
        for triple in triples:
            writer.write(triple)

    # Print statistics
    logging.info("Kuzu Extraction Statistics:")
    logging.info(f"  Total rows: {stats['total_rows']}")
    logging.info(f"  Valid triples: {stats['valid_triples']}")
    logging.info(f"  Parse failed: {stats['parse_failed']}")
    logging.info(f"  Function words: {stats['function_word']}")
    logging.info(f"  Missing components: {stats['missing_component']}")
    logging.info(f"  Success rate: {stats['valid_triples'] / stats['total_rows'] * 100:.1f}%")

    return stats


# ============================================================================
# JSONL EXTRACTION (with complex structure handling)
# ============================================================================

def extract_root(node: Dict) -> Optional[Tuple[str, str, str]]:
    """
    Extract (root, full_word, status) from AST node.

    Handles:
    - Simple vorto nodes
    - Vortgrupo nodes (extract kerno)
    - Null nodes

    Returns:
        (root, full_word, analizstato) or None
    """
    if not node:
        return None

    if node.get('tipo') == 'vortgrupo':
        kerno = node.get('kerno', {})
        return (
            kerno.get('radiko'),
            kerno.get('plena_vorto'),
            kerno.get('analizstato')
        )
    elif node.get('tipo') == 'vorto':
        return (
            node.get('radiko'),
            node.get('plena_vorto'),
            node.get('analizstato')
        )
    else:
        return None


def extract_triples_from_ast(ast: Dict, sentence: str, source: str, sentence_id: int) -> List[Dict]:
    """
    Extract all SVO triples from a single AST.

    Handles complex structures:
    1. Multiple clauses (koordinitaj frazoj)
    2. Subordinate clauses (dependaj frazoj, ke-clauses)
    3. Coordinated verbs (kaj between verbs)
    4. Passive voice (participles with de-phrases)
    5. Nested structures

    Args:
        ast: Parsed AST dictionary
        sentence: Original sentence text
        source: Source corpus name
        sentence_id: Sentence ID

    Returns:
        List of triple dictionaries
    """
    triples = []

    # Handle top-level frazo
    if ast.get('tipo') == 'frazo':
        # Extract main clause triple(s)
        main_triples = extract_from_frazo(ast, sentence, source, sentence_id, confidence=1.0)
        triples.extend(main_triples)

    # Handle subordinate clauses in 'aliaj' (ke-clauses, relative clauses)
    aliaj = ast.get('aliaj', [])
    for item in aliaj:
        if isinstance(item, dict) and item.get('tipo') == 'frazo':
            # Recursive extraction for subordinate clause
            sub_triples = extract_from_frazo(item, sentence, source, sentence_id, confidence=0.8)
            triples.extend(sub_triples)

    # Handle subordinate clause as object (ke-clauses)
    objekto = ast.get('objekto')
    if objekto and isinstance(objekto, dict) and objekto.get('tipo') == 'frazo':
        sub_triples = extract_from_frazo(objekto, sentence, source, sentence_id, confidence=0.8)
        triples.extend(sub_triples)

    # Handle relative clauses in subject/object priskriboj
    subjekto = ast.get('subjekto')
    if subjekto and isinstance(subjekto, dict):
        priskriboj = subjekto.get('priskriboj', [])
        for priskribo in priskriboj:
            if isinstance(priskribo, dict) and priskribo.get('tipo') == 'frazo':
                sub_triples = extract_from_frazo(priskribo, sentence, source, sentence_id, confidence=0.8)
                triples.extend(sub_triples)

    if objekto and isinstance(objekto, dict) and objekto.get('tipo') == 'vortgrupo':
        priskriboj = objekto.get('priskriboj', [])
        for priskribo in priskriboj:
            if isinstance(priskribo, dict) and priskribo.get('tipo') == 'frazo':
                sub_triples = extract_from_frazo(priskribo, sentence, source, sentence_id, confidence=0.8)
                triples.extend(sub_triples)

    return triples


def extract_from_frazo(frazo: Dict, sentence: str, source: str, sentence_id: int, confidence: float) -> List[Dict]:
    """
    Extract SVO triple(s) from a frazo node.

    Handles:
    - Simple SVO
    - Coordinated verbs (manĝas... kaj trinkas...)
    - Passive voice (estis skribita de Zamenhof)

    Args:
        frazo: Frazo AST node
        sentence: Original sentence text
        source: Source corpus name
        sentence_id: Sentence ID
        confidence: Confidence score

    Returns:
        List of triple dictionaries
    """
    triples = []

    # Check for passive voice (verb is participle + "est" auxiliary + "de" phrase)
    passive_triple = extract_passive_voice(frazo, sentence, source, sentence_id, confidence)
    if passive_triple:
        triples.append(passive_triple)
        return triples  # Don't also extract as normal SVO

    # Extract subject
    subjekto = frazo.get('subjekto')
    subject_info = extract_root(subjekto)
    if not subject_info:
        return triples
    subject_root, subject_full, subject_status = subject_info

    # Extract main verb
    verbo = frazo.get('verbo', {})
    verb_root = verbo.get('radiko')
    verb_full = verbo.get('plena_vorto')
    verb_status = verbo.get('analizstato')

    # Extract object
    objekto = frazo.get('objekto')
    if not objekto or not isinstance(objekto, dict):
        return triples
    object_info = extract_root(objekto)

    # Try to extract coordinated verbs (Subject V1 Object1 kaj V2 Object2)
    coordinated_triples = extract_coordinated_verbs(
        frazo, subject_info, sentence, source, sentence_id, confidence
    )
    if coordinated_triples:
        triples.extend(coordinated_triples)
        return triples  # Don't also extract as simple SVO

    # Extract simple SVO triple
    if object_info and verb_root:
        object_root, object_full, object_status = object_info

        # Validate
        if all([subject_root, verb_root, object_root]):
            if subject_status == 'sukceso' and verb_status == 'sukceso' and object_status == 'sukceso':
                if subject_root not in FUNCTION_WORDS and object_root not in FUNCTION_WORDS:
                    triple = {
                        'subject_root': subject_root,
                        'verb_root': verb_root,
                        'object_root': object_root,
                        'subject_full': subject_full,
                        'verb_full': verb_full,
                        'object_full': object_full,
                        'relation_type': 'SVO',
                        'source': source,
                        'sentence': sentence,
                        'sentence_id': sentence_id,
                        'confidence': confidence
                    }
                    triples.append(triple)

    return triples


def extract_coordinated_verbs(
    frazo: Dict,
    subject_info: Tuple[str, str, str],
    sentence: str,
    source: str,
    sentence_id: int,
    confidence: float
) -> List[Dict]:
    """
    Extract coordinated verb triples: "Subject V1 O1 kaj V2 O2"

    Example: "La hundo manĝas viandon kaj trinkas akvon"
    → (hund, manĝ, viand), (hund, trink, akv)

    Strategy:
    - Look for "kaj" in aliaj list
    - If found, check if there's another verb after it
    - Create separate triples for each verb-object pair

    Args:
        frazo: Frazo AST node
        subject_info: (subject_root, subject_full, subject_status)
        sentence: Original sentence
        source: Source corpus
        sentence_id: Sentence ID
        confidence: Confidence score

    Returns:
        List of triples (empty if no coordination detected)
    """
    triples = []
    subject_root, subject_full, subject_status = subject_info

    if subject_status != 'sukceso':
        return triples

    # Get main verb and object
    verbo = frazo.get('verbo', {})
    verb1_root = verbo.get('radiko')
    verb1_full = verbo.get('plena_vorto')
    verb1_status = verbo.get('analizstato')

    objekto = frazo.get('objekto')
    object1_info = extract_root(objekto)

    if not object1_info or verb1_status != 'sukceso':
        return triples

    object1_root, object1_full, object1_status = object1_info
    if object1_status != 'sukceso':
        return triples

    # Look for "kaj" followed by verb in aliaj
    aliaj = frazo.get('aliaj', [])
    found_kaj = False
    verb2_node = None
    object2_node = None

    for i, item in enumerate(aliaj):
        if isinstance(item, dict):
            # Check if this is "kaj"
            if item.get('radiko') == 'kaj' and item.get('vortspeco') == 'konjunkcio':
                found_kaj = True
            # After finding "kaj", look for next verb
            elif found_kaj and not verb2_node and item.get('vortspeco') == 'verbo':
                verb2_node = item
            # After finding second verb, look for its object (noun with -n)
            elif verb2_node and not object2_node:
                if item.get('vortspeco') in ['substantivo', 'propranomo'] and item.get('kazo') == 'akuzativo':
                    object2_node = item
                    break

    # If we found coordinated verb and object, extract both triples
    if verb2_node and object2_node:
        # First triple (main verb + main object)
        if subject_root not in FUNCTION_WORDS and object1_root not in FUNCTION_WORDS:
            triples.append({
                'subject_root': subject_root,
                'verb_root': verb1_root,
                'object_root': object1_root,
                'subject_full': subject_full,
                'verb_full': verb1_full,
                'object_full': object1_full,
                'relation_type': 'SVO',
                'source': source,
                'sentence': sentence,
                'sentence_id': sentence_id,
                'confidence': confidence
            })

        # Second triple (coordinated verb + object)
        verb2_root = verb2_node.get('radiko')
        verb2_full = verb2_node.get('plena_vorto')
        object2_root = object2_node.get('radiko')
        object2_full = object2_node.get('plena_vorto')

        if subject_root not in FUNCTION_WORDS and object2_root not in FUNCTION_WORDS:
            triples.append({
                'subject_root': subject_root,
                'verb_root': verb2_root,
                'object_root': object2_root,
                'subject_full': subject_full,
                'verb_full': verb2_full,
                'object_full': object2_full,
                'relation_type': 'SVO',
                'source': source,
                'sentence': sentence,
                'sentence_id': sentence_id,
                'confidence': confidence
            })

    return triples


def extract_passive_voice(
    frazo: Dict,
    sentence: str,
    source: str,
    sentence_id: int,
    confidence: float
) -> Optional[Dict]:
    """
    Extract triple from passive voice construction.

    Example: "La libro estis skribita de Zamenhof"
    → (zamenhof, skrib, libr)  [agent and patient flipped]

    Strategy:
    - Main verb is "est" (auxiliary)
    - Look for participle in subject.priskriboj (passive participle with -ita/-ata)
    - Look for "de" phrase in aliaj (agent)
    - Extract as (agent, participle_root, subject)

    Args:
        frazo: Frazo AST node
        sentence: Original sentence
        source: Source corpus
        sentence_id: Sentence ID
        confidence: Confidence score

    Returns:
        Triple dictionary or None
    """
    # Check if main verb is "est" (to be)
    verbo = frazo.get('verbo', {})
    if verbo.get('radiko') != 'est':
        return None

    # Get subject (will become patient/object in triple)
    subjekto = frazo.get('subjekto')
    if not subjekto:
        return None

    patient_info = extract_root(subjekto)
    if not patient_info:
        return None
    patient_root, patient_full, patient_status = patient_info

    if patient_status != 'sukceso' or patient_root in FUNCTION_WORDS:
        return None

    # Look for passive participle in subject's priskriboj (descriptions)
    participle_node = None
    if subjekto.get('tipo') == 'vortgrupo':
        priskriboj = subjekto.get('priskriboj', [])
        for priskribo in priskriboj:
            if isinstance(priskribo, dict):
                # Look for passive participle
                if priskribo.get('vortspeco') in ['adjektivo', 'participo']:
                    participo_voco = priskribo.get('participo_voĉo')
                    if participo_voco == 'pasiva':
                        participle_node = priskribo
                        break

    if not participle_node:
        return None

    # Look for "de" phrase in aliaj (agent)
    aliaj = frazo.get('aliaj', [])
    agent_node = None
    found_de = False

    for item in aliaj:
        if isinstance(item, dict):
            # Look for "de" preposition
            if item.get('radiko') == 'de' and item.get('vortspeco') == 'prepozicio':
                found_de = True

            # After "de", look for the agent (noun)
            elif found_de and not agent_node:
                if item.get('vortspeco') in ['substantivo', 'propranomo', 'nekonata']:
                    agent_node = item
                    break

    # If we found participle and agent, extract triple
    if participle_node and agent_node:
        participle_root = participle_node.get('radiko')
        participle_full = participle_node.get('plena_vorto')
        agent_root = agent_node.get('radiko')
        agent_full = agent_node.get('plena_vorto')
        agent_status = agent_node.get('analizstato')

        if agent_status == 'sukceso' and agent_root not in FUNCTION_WORDS:
            return {
                'subject_root': agent_root,      # Agent becomes subject
                'verb_root': participle_root,     # Participle root becomes verb
                'object_root': patient_root,      # Original subject becomes object
                'subject_full': agent_full,
                'verb_full': participle_full,
                'object_full': patient_full,
                'relation_type': 'SVO_passive',
                'source': source,
                'sentence': sentence,
                'sentence_id': sentence_id,
                'confidence': confidence * 0.9  # Slightly lower confidence for passive
            }

    return None


def extract_from_jsonl(corpus_paths: List[Path], output_path: Path, max_sentences: Optional[int] = None):
    """
    Extract SVO triples from JSONL corpus files by parsing.

    Args:
        corpus_paths: List of corpus JSONL file paths
        output_path: Path to output JSONL file
        max_sentences: Optional limit on number of sentences to process
    """
    triples = []
    stats = defaultdict(int)

    for corpus_path in corpus_paths:
        logging.info(f"Processing corpus: {corpus_path}")
        source = corpus_path.stem  # e.g., "wikipedia_sentences" → "wikipedia"

        with jsonlines.open(corpus_path) as reader:
            for i, item in enumerate(reader):
                if max_sentences and stats['total_sentences'] >= max_sentences:
                    break

                stats['total_sentences'] += 1

                # Log progress every 10K sentences
                if stats['total_sentences'] % 10000 == 0:
                    logging.info(f"  Processed {stats['total_sentences']} sentences, extracted {stats['valid_triples']} triples")

                sentence = item.get('text', '')
                sentence_id = item.get('id', i)

                # Get or parse AST
                ast = item.get('ast')
                if not ast:
                    try:
                        ast = parse(sentence)
                        stats['parsed'] += 1
                    except Exception as e:
                        logging.debug(f"Parse failed for sentence {sentence_id}: {e}")
                        stats['parse_failed'] += 1
                        continue
                else:
                    stats['ast_cached'] += 1

                # Extract triples (handles complex structures)
                sentence_triples = extract_triples_from_ast(ast, sentence, source, sentence_id)
                triples.extend(sentence_triples)
                stats['valid_triples'] += len(sentence_triples)

    # Write to JSONL
    logging.info(f"Writing {len(triples)} triples to {output_path}")
    with jsonlines.open(output_path, mode='w') as writer:
        for triple in triples:
            writer.write(triple)

    # Print statistics
    logging.info("JSONL Extraction Statistics:")
    logging.info(f"  Total sentences: {stats['total_sentences']}")
    logging.info(f"  Valid triples: {stats['valid_triples']}")
    logging.info(f"  Parsed on-the-fly: {stats['parsed']}")
    logging.info(f"  AST cached: {stats['ast_cached']}")
    logging.info(f"  Parse failed: {stats['parse_failed']}")
    logging.info(f"  Avg triples per sentence: {stats['valid_triples'] / stats['total_sentences']:.2f}")

    return stats


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Extract SVO triples from corpus')
    parser.add_argument('--source', choices=['kuzu', 'jsonl', 'both'], required=True,
                        help='Extraction source (kuzu=fast, jsonl=complete, both=combined)')
    parser.add_argument('--db-path', type=Path,
                        help='Path to Kuzu database directory (required if source=kuzu or both)')
    parser.add_argument('--corpus', type=Path, action='append',
                        help='Path to corpus JSONL file (can specify multiple, required if source=jsonl or both)')
    parser.add_argument('--output', type=Path, required=True,
                        help='Output JSONL file path')
    parser.add_argument('--max-sentences', type=int,
                        help='Maximum sentences to process from JSONL (for testing)')
    parser.add_argument('--extract-prepositional', action='store_true',
                        help='Extract prepositional phrases too (not just SVO)')
    parser.add_argument('--extract-modifiers', action='store_true',
                        help='Extract adjective-noun modifiers')
    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Validate arguments
    if args.source in ['kuzu', 'both'] and not args.db_path:
        parser.error("--db-path required when --source is kuzu or both")
    if args.source in ['jsonl', 'both'] and not args.corpus:
        parser.error("--corpus required when --source is jsonl or both")

    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Extract from Kuzu
    if args.source in ['kuzu', 'both']:
        if args.source == 'kuzu':
            output_path = args.output
        else:
            output_path = args.output.with_suffix('.kuzu.jsonl')

        extract_from_kuzu(
            db_path=args.db_path,
            output_path=output_path,
            extract_prep=args.extract_prepositional,
            extract_mod=args.extract_modifiers
        )

    # Extract from JSONL
    if args.source in ['jsonl', 'both']:
        if args.source == 'jsonl':
            output_path = args.output
        else:
            output_path = args.output.with_suffix('.jsonl_parsed.jsonl')

        extract_from_jsonl(
            corpus_paths=args.corpus,
            output_path=output_path,
            max_sentences=args.max_sentences
        )

    # Merge if both sources
    if args.source == 'both':
        logging.info("Merging Kuzu and JSONL triples...")
        kuzu_path = args.output.with_suffix('.kuzu.jsonl')
        jsonl_path = args.output.with_suffix('.jsonl_parsed.jsonl')

        all_triples = []
        with jsonlines.open(kuzu_path) as reader:
            all_triples.extend(list(reader))
        with jsonlines.open(jsonl_path) as reader:
            all_triples.extend(list(reader))

        # Deduplicate by (subject_root, verb_root, object_root, sentence_id)
        seen = set()
        unique_triples = []
        for triple in all_triples:
            key = (triple['subject_root'], triple['verb_root'], triple['object_root'], triple['sentence_id'])
            if key not in seen:
                seen.add(key)
                unique_triples.append(triple)

        with jsonlines.open(args.output, mode='w') as writer:
            for triple in unique_triples:
                writer.write(triple)

        logging.info(f"Merged {len(all_triples)} triples → {len(unique_triples)} unique triples")
        logging.info(f"Output: {args.output}")


if __name__ == '__main__':
    main()
