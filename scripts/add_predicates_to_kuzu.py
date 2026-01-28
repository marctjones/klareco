#!/usr/bin/env python3
"""
Add Predicate table to existing Kuzu index.

This script adds the Predicate node table and HAS_PREDICATE relationship
to an existing Kuzu index, enabling fast O(1) predicate-based retrieval.

Usage:
    python scripts/add_predicates_to_kuzu.py --index data/indexes/kuzu_index
"""

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

try:
    import kuzu
except ImportError:
    print("Error: kuzu package not installed. Run: pip install kuzu")
    exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_predicate_from_ast(ast: Dict) -> Optional[Tuple[str, Optional[str], Optional[str]]]:
    """
    Extract predicate (subject-verb-object) from AST.

    Returns (verb, subject, object) tuple where verb is required, subject and object are optional.
    For compounds, preserves suffix structure to distinguish (e.g., "esperant" vs "esperant-rond").

    Returns None if no verb found.
    """
    if not ast or not isinstance(ast, dict):
        return None

    def get_root(node):
        """Extract root from a node (handling vortgrupo)."""
        if not node or not isinstance(node, dict):
            return None

        if node.get('tipo') == 'vortgrupo':
            kerno = node.get('kerno', {})
            return kerno.get('radiko', '').lower() if isinstance(kerno, dict) else None
        elif node.get('tipo') == 'vorto':
            return node.get('radiko', '').lower()

        return None

    def get_word_with_compounds(node):
        """
        Extract word preserving compound structure for semantic disambiguation.

        Examples:
        - "Esperanto" → "esperant"
        - "Esperanto-rondo" → "esperant-rond" (compound preserved)
        - "esperantisto" → "esperant-ist" (derivational suffix preserved)

        Issue #550: Preserve compounds to distinguish language from organizations.
        """
        if not node or not isinstance(node, dict):
            return None

        # Handle vortgrupo by extracting kerno
        if node.get('tipo') == 'vortgrupo':
            node = node.get('kerno', {})
            if not isinstance(node, dict):
                return None

        # Extract root
        root = node.get('radiko', '').lower()
        if not root:
            return None

        # Build compound structure
        parts = []

        # 1. Check for compound words (kunmetajhoj - using h not ĵ)
        if node.get('estas_kunmetita') and 'kunmetajhoj' in node:
            # Extract roots from compound components
            semantic_suffixes = ['ist', 'ul', 'ej', 'ar', 'aĵ', 'ism', 'an', 'estr', 'il', 'uj']
            for component in node.get('kunmetajhoj', []):
                if isinstance(component, dict):
                    comp_root = component.get('radiko', '').lower()
                    if comp_root:
                        parts.append(comp_root)
                    # Also extract semantic suffixes from components
                    comp_sufixes = component.get('sufiksoj', [])
                    if comp_sufixes:
                        preserved = [suf for suf in comp_sufixes if suf in semantic_suffixes]
                        parts.extend(preserved)

        # 2. Add main root
        parts.append(root)

        # 3. Check for derivational suffixes (semantic type changers)
        sufiksoj = node.get('sufiksoj', [])
        if sufiksoj:
            # Keep: -ist, -ul, -ej, -ar, -aĵ, -ism, -an (semantic type changers)
            # Skip: -in (gender), -et/-eg (size), -ĉj/-nj (affectionate)
            semantic_suffixes = ['ist', 'ul', 'ej', 'ar', 'aĵ', 'ism', 'an', 'estr', 'il', 'uj']
            preserved = [suf for suf in sufiksoj if suf in semantic_suffixes]
            parts.extend(preserved)

        # Join all parts with hyphen
        if len(parts) > 1:
            return '-'.join(parts)

        return root

    # Extract verb (required) - just root, no compounds
    verb = get_root(ast.get('verbo'))
    if not verb:
        return None

    # Extract subject - just root (names/nouns are simple)
    subj = get_root(ast.get('subjekto'))

    # Extract object - preserve compounds for disambiguation
    obj = get_word_with_compounds(ast.get('objekto'))

    return (verb, subj, obj)


def extract_virtual_predicates_from_noun_phrases(ast: Dict) -> List[Tuple[str, Optional[str], Optional[str]]]:
    """
    Extract virtual predicates from participial noun phrases.

    Issue #549: Most encyclopedia-style facts use participles, not verb predicates.
    """
    if not ast or not isinstance(ast, dict):
        return []

    virtual_predicates = []

    # Helper to extract word with participle info
    def get_word_info(node):
        """Get word root, suffixes, and participle metadata."""
        if not node or not isinstance(node, dict):
            return None

        # Handle vortgrupo
        if node.get('tipo') == 'vortgrupo':
            node = node.get('kerno', {})
            if not isinstance(node, dict):
                return None

        if node.get('tipo') != 'vorto':
            return None

        root = node.get('radiko', '').lower()
        sufiksoj = node.get('sufiksoj', [])

        return {
            'root': root,
            'sufiksoj': sufiksoj,
        }

    # Helper to find "de X" prepositional phrases in aliaj
    def find_de_phrase_object(aliaj_list):
        """Find object of 'de' prepositional phrase in aliaj."""
        if not aliaj_list:
            return None

        # Look for pattern: "de" followed by noun
        for i, word in enumerate(aliaj_list):
            if not isinstance(word, dict):
                continue

            # Check if this is "de" preposition
            if (word.get('tipo') == 'vorto' and
                word.get('radiko', '').lower() == 'de' and
                word.get('vortspeco') == 'prepozicio'):

                # Next word should be the object
                if i + 1 < len(aliaj_list):
                    next_word = aliaj_list[i + 1]
                    if isinstance(next_word, dict) and next_word.get('tipo') == 'vorto':
                        root = next_word.get('radiko', '').lower()
                        if root:
                            return root

        return None

    # Pattern 1: Subject with participle + "de Y"
    subjekto = ast.get('subjekto')
    if subjekto:
        subj_info = get_word_info(subjekto)
        if subj_info and 'int' in subj_info['sufiksoj']:
            verb_root = subj_info['root']
            de_object = find_de_phrase_object(ast.get('aliaj', []))
            if de_object:
                virtual_predicates.append((verb_root, None, de_object))

    # Pattern 2: "A estas B-into de C" → (B, A, C)
    verbo = ast.get('verbo')
    if verbo and isinstance(verbo, dict):
        verb_root = verbo.get('radiko', '').lower()

        if verb_root == 'est':
            aliaj = ast.get('aliaj', [])

            for word in aliaj:
                if not isinstance(word, dict):
                    continue

                word_info = get_word_info(word)
                if word_info and 'int' in word_info['sufiksoj']:
                    pred_verb = word_info['root']
                    subj_info = get_word_info(ast.get('subjekto'))
                    pred_subj = subj_info['root'] if subj_info else None
                    de_object = find_de_phrase_object(aliaj)

                    if pred_verb and pred_subj:
                        virtual_predicates.append((pred_verb, pred_subj, de_object))

    # Pattern 3: Participial adjective modifying noun
    if subjekto and isinstance(subjekto, dict) and subjekto.get('tipo') == 'vortgrupo':
        priskriboj = subjekto.get('priskriboj', [])
        kerno = subjekto.get('kerno', {})
        kerno_root = kerno.get('radiko', '').lower() if isinstance(kerno, dict) else None

        for priskribo in priskriboj:
            if not isinstance(priskribo, dict):
                continue

            pri_info = get_word_info(priskribo)
            if pri_info and 'int' in pri_info['sufiksoj']:
                verb_root = pri_info['root']
                if kerno_root:
                    virtual_predicates.append((verb_root, kerno_root, None))

    return virtual_predicates


def resolve_pronoun_subject(subj_root: Optional[str], document_context: List[str]) -> Optional[str]:
    """
    Resolve pronoun subject to nearest mentioned entity.

    Issue #551: Simple heuristic for pronoun resolution.
    """
    pronouns = {'li', 'ŝi', 'ĝi', 'ili', 'mi', 'vi', 'ni'}
    if not subj_root or subj_root.lower() not in pronouns:
        return subj_root

    if document_context:
        return document_context[-1]

    return subj_root


def add_predicates_to_index(index_path: Path, fresh: bool = False):
    """Add Predicate table and HAS_PREDICATE relationships to existing index."""

    logger.info("=" * 60)
    logger.info("Adding Predicate Table to Kuzu Index")
    logger.info("=" * 60)
    logger.info(f"Index: {index_path}")
    logger.info("")

    db_path = index_path / "kuzu.db"
    if not db_path.exists():
        logger.error(f"Kuzu database not found at {db_path}")
        return False

    # Open database
    logger.info("Opening database...")
    db = kuzu.Database(str(db_path))
    conn = kuzu.Connection(db)

    # Check if Predicate table already exists
    try:
        result = conn.execute("MATCH (p:Predicate) RETURN count(p) LIMIT 1")
        count = result.get_next()[0]
        if count > 0:
            if fresh:
                logger.info(f"Dropping existing Predicate table with {count:,} predicates...")
                conn.execute("DROP TABLE HAS_PREDICATE")
                conn.execute("DROP TABLE Predicate")
                logger.info("  ✓ Dropped existing tables")
            else:
                logger.info(f"Predicate table already exists with {count:,} predicates")
                logger.info("Use --fresh to rebuild")
                return True
    except Exception:
        # Table doesn't exist - that's expected
        pass

    # Create schema if needed
    logger.info("Creating Predicate schema...")
    try:
        conn.execute("""
            CREATE NODE TABLE IF NOT EXISTS Predicate (
                id STRING,
                verb STRING,
                subj STRING,
                obj STRING,
                PRIMARY KEY (id)
            )
        """)
        logger.info("  ✓ Predicate table created")
    except Exception as e:
        logger.info(f"  Predicate table already exists: {e}")

    try:
        conn.execute("""
            CREATE REL TABLE IF NOT EXISTS HAS_PREDICATE (
                FROM Sentence TO Predicate
            )
        """)
        logger.info("  ✓ HAS_PREDICATE relationship created")
    except Exception as e:
        logger.info(f"  HAS_PREDICATE relationship already exists: {e}")

    # Load corpus and extract predicates
    corpus_path = index_path.parent.parent / "corpus" / "unified_corpus.jsonl"
    if not corpus_path.exists():
        # Try alternative location
        corpus_path = Path("data/corpus/unified_corpus.jsonl")
        if not corpus_path.exists():
            logger.error(f"Corpus not found at {corpus_path}")
            return False

    logger.info(f"Loading corpus from: {corpus_path}")

    # Read sentences from Kuzu to map doc_id to sentence_id
    logger.info("Loading sentence mappings...")
    doc_to_sent: Dict[int, int] = {}  # doc_id -> sent_id (reversed for O(1) lookup)
    result = conn.execute("MATCH (s:Sentence) RETURN s.id, s.doc_id")
    while result.has_next():
        sent_id, doc_id = result.get_next()
        doc_to_sent[doc_id] = sent_id
    logger.info(f"  Loaded {len(doc_to_sent):,} sentence mappings")

    # Create temporary CSV files for bulk loading
    temp_dir = index_path / "temp_predicates"
    temp_dir.mkdir(exist_ok=True)

    predicates_csv = temp_dir / "predicates.csv"
    has_predicate_csv = temp_dir / "has_predicate.csv"

    logger.info("")
    logger.info("Extracting predicates from corpus...")

    predicates_seen: Set[str] = set()
    pred_count = 0
    edge_count = 0
    skipped = 0

    # Track document context for pronoun resolution (Issue #551)
    document_context: List[str] = []  # Recent proper nouns

    with open(predicates_csv, 'w', newline='', encoding='utf-8') as pred_f, \
         open(has_predicate_csv, 'w', newline='', encoding='utf-8') as edge_f:

        pred_writer = csv.writer(pred_f)
        edge_writer = csv.writer(edge_f)

        # Write headers
        pred_writer.writerow(['id', 'verb', 'subj', 'obj'])
        edge_writer.writerow(['sent_id', 'pred_id'])

        # Process corpus
        doc_id = 0
        with open(corpus_path) as f:
            for line in f:
                try:
                    doc = json.loads(line)
                except json.JSONDecodeError:
                    continue

                ast = doc.get('ast')
                if not ast:
                    skipped += 1
                    continue

                # Skip low-quality parses
                parse_rate = doc.get('parse_rate', 1.0)
                if parse_rate < 0.5:
                    skipped += 1
                    continue

                # Find corresponding sentence_id (O(1) lookup)
                sent_id = doc_to_sent.get(doc_id)

                if sent_id is None:
                    # Document not in index (maybe limit was used)
                    doc_id += 1
                    continue

                # Track proper nouns for pronoun resolution (Issue #551)
                subjekto = ast.get('subjekto')
                if subjekto and isinstance(subjekto, dict):
                    # Get kerno if vortgrupo
                    node = subjekto if subjekto.get('tipo') == 'vorto' else subjekto.get('kerno', {})
                    if isinstance(node, dict):
                        root = node.get('radiko', '').lower()
                        # Check if proper noun (not pronoun)
                        if root and root not in {'mi', 'vi', 'li', 'ŝi', 'ĝi', 'ni', 'ili'}:
                            # Add to context (keep last 10)
                            document_context.append(root)
                            if len(document_context) > 10:
                                document_context.pop(0)

                # Extract predicates (both verb predicates and virtual predicates from participles)
                all_predicates = []

                # 1. Extract standard verb predicate
                predicate = extract_predicate_from_ast(ast)
                if predicate:
                    all_predicates.append(predicate)

                # 2. Extract virtual predicates from participial noun phrases (Issue #549)
                virtual_predicates = extract_virtual_predicates_from_noun_phrases(ast)
                all_predicates.extend(virtual_predicates)

                # Create nodes and edges for all predicates (with pronoun resolution)
                for verb, subj, obj in all_predicates:
                    # Resolve pronoun subject to entity (Issue #551)
                    subj_resolved = resolve_pronoun_subject(subj, document_context)

                    # Create predicate ID
                    subj_str = subj_resolved or ''
                    obj_str = obj or ''
                    pred_id = f"{verb}|{subj_str}|{obj_str}"

                    # Add Predicate node if not seen
                    if pred_id not in predicates_seen:
                        pred_writer.writerow([pred_id, verb, subj_str, obj_str])
                        predicates_seen.add(pred_id)
                        pred_count += 1

                    # Add HAS_PREDICATE edge
                    edge_writer.writerow([sent_id, pred_id])
                    edge_count += 1

                doc_id += 1

                # Log progress every 10000 docs
                if doc_id % 10000 == 0:
                    logger.info(f"  Processed {doc_id:,} docs - {pred_count:,} predicates, {edge_count:,} edges")

    logger.info(f"  Extracted {pred_count:,} unique predicates, {edge_count:,} edges")
    logger.info(f"  Skipped {skipped:,} docs (no AST or low parse rate)")

    # Bulk load CSVs
    logger.info("")
    logger.info("Loading predicates into Kuzu...")
    try:
        conn.execute(f"COPY Predicate FROM '{predicates_csv}' (header=true)")
        logger.info(f"  ✓ Loaded {pred_count:,} predicates")
    except Exception as e:
        logger.error(f"  Error loading predicates: {e}")
        return False

    logger.info("Loading HAS_PREDICATE relationships...")
    try:
        conn.execute(f"COPY HAS_PREDICATE FROM '{has_predicate_csv}' (header=true)")
        logger.info(f"  ✓ Loaded {edge_count:,} HAS_PREDICATE edges")
    except Exception as e:
        logger.error(f"  Error loading edges: {e}")
        return False

    # Verify
    logger.info("")
    logger.info("Verifying...")
    result = conn.execute("MATCH (p:Predicate) RETURN count(p)")
    count = result.get_next()[0]
    logger.info(f"  Total predicates in database: {count:,}")

    result = conn.execute("MATCH ()-[r:HAS_PREDICATE]->() RETURN count(r)")
    count = result.get_next()[0]
    logger.info(f"  Total HAS_PREDICATE edges: {count:,}")

    # Clean up temp files
    logger.info("")
    logger.info("Cleaning up temporary files...")
    predicates_csv.unlink()
    has_predicate_csv.unlink()
    temp_dir.rmdir()

    logger.info("")
    logger.info("=" * 60)
    logger.info("✓ Predicate table added successfully!")
    logger.info("=" * 60)

    return True


def main():
    parser = argparse.ArgumentParser(description="Add Predicate table to existing Kuzu index")
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("data/indexes/kuzu_index"),
        help="Path to Kuzu index directory"
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Rebuild predicate table from scratch"
    )

    args = parser.parse_args()

    success = add_predicates_to_index(args.index, fresh=args.fresh)

    if not success:
        exit(1)


if __name__ == '__main__':
    main()
