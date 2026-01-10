#!/usr/bin/env python3
"""
Extract Predicate Triples from Unified Corpus.

This script extracts (verb, subject, object) triples from parsed ASTs
for predicate-based retrieval. Part of Issue #253.

Features:
- Handles incomplete predicates (null subjects/objects as wildcards)
- Handles passive voice (semantic role reversal)
- Extracts predicates from embedded clauses
- Skips correlatives and pronouns (handled by AST lookup)
- Checkpoint support for restartability
- Progress logging every 10 seconds

Output format:
    {"doc_id": 12345, "verb": "fond", "subj": "zamenhof", "obj": "esperant", "clause_depth": 0}

Usage:
    python scripts/extract_predicates.py
    python scripts/extract_predicates.py --limit 10000  # Test with subset
    python scripts/extract_predicates.py --resume       # Resume from checkpoint
    python scripts/extract_predicates.py --fresh        # Start fresh, ignore checkpoint

Input:
    data/corpus/unified_corpus.jsonl

Output:
    data/indexes/kuzu_index/predicates.jsonl
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Script version for tracking
VERSION = "1.0.0"

# Set up logging with both console and file output
def setup_logging(log_dir: Path) -> Path:
    """Set up logging to both console and file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"extract_predicates_{timestamp}.log"

    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    ))
    root_logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    root_logger.addHandler(file_handler)

    return log_file

logger = logging.getLogger(__name__)

# Skip these word types (function words - handled by AST, not predicates)
SKIP_VORTSPECO = {'korelativo', 'pronomo', 'artikolo', 'prepozicio', 'konjunkcio'}

# Content word types that should be included in predicates
CONTENT_VORTSPECO = {'substantivo', 'verbo', 'adjektivo', 'adverbo'}


def get_head_root(node: Optional[Dict]) -> Optional[str]:
    """
    Get the head root from a word or word group.

    For word groups, returns the root of the kerno (head).
    Skips function words (correlatives, pronouns, etc.).

    Args:
        node: AST node (vorto or vortgrupo)

    Returns:
        Root string (lowercase) or None if function word/empty
    """
    if not node or not isinstance(node, dict):
        return None

    tipo = node.get('tipo')

    if tipo == 'vorto':
        vortspeco = node.get('vortspeco', '')

        # Skip function words - they don't contribute to predicate semantics
        if vortspeco in SKIP_VORTSPECO:
            return None

        root = node.get('radiko', '')
        if root and len(root) >= 2:
            return root.lower()
        return None

    elif tipo == 'vortgrupo':
        # Get root from the head (kerno)
        return get_head_root(node.get('kerno'))

    return None


def extract_predicates_from_ast(
    ast: Dict,
    doc_id: int,
    clause_depth: int = 0,
) -> List[Dict]:
    """
    Extract all predicate triples from an AST.

    Handles:
    - Main clause predicates
    - Embedded clauses (ke, kiu, kio, kie, kiam)
    - Passive voice detection
    - Copula sentences (estas X)

    Args:
        ast: AST dict from parsed sentence
        doc_id: Document ID for output
        clause_depth: Current clause nesting depth

    Returns:
        List of predicate dicts: {doc_id, verb, subj, obj, clause_depth, is_passive, is_copula}
    """
    predicates = []

    if not ast or not isinstance(ast, dict):
        return predicates

    tipo = ast.get('tipo')

    if tipo == 'frazo':
        # Extract main predicate
        verb_node = ast.get('verbo')
        subj_node = ast.get('subjekto')
        obj_node = ast.get('objekto')

        verb = get_head_root(verb_node)
        subj = get_head_root(subj_node)
        obj = get_head_root(obj_node)

        # Only include if we have a verb
        if verb:
            # Detect passive voice (verb ends in -iĝ- or has passive participle)
            is_passive = False
            if verb_node and isinstance(verb_node, dict):
                sufiksoj = verb_node.get('sufiksoj', [])
                if 'iĝ' in sufiksoj:
                    is_passive = True
                # Also check for participle endings
                plena = verb_node.get('plena_vorto', '')
                if plena.endswith(('ita', 'ata', 'ota')):
                    is_passive = True

            # Detect copula (estas, estis, estos, etc.)
            is_copula = verb == 'est'

            # For passive voice, swap subject and object semantically
            # "La libro legiĝas" = "Someone reads the book"
            if is_passive and obj is None and subj:
                # Subject is actually the semantic object
                predicates.append({
                    'doc_id': doc_id,
                    'verb': verb,
                    'subj': None,  # Unknown agent
                    'obj': subj,   # Semantic object
                    'clause_depth': clause_depth,
                    'is_passive': True,
                    'is_copula': is_copula,
                })
            else:
                predicates.append({
                    'doc_id': doc_id,
                    'verb': verb,
                    'subj': subj,
                    'obj': obj,
                    'clause_depth': clause_depth,
                    'is_passive': is_passive,
                    'is_copula': is_copula,
                })

        # Recursively extract from aliaj (modifiers, embedded clauses)
        for aliaj_item in ast.get('aliaj', []):
            predicates.extend(
                extract_predicates_from_ast(aliaj_item, doc_id, clause_depth + 1)
            )

        # Also check priskriboj in subject/object groups for embedded clauses
        for role_node in [subj_node, obj_node]:
            if role_node and isinstance(role_node, dict):
                if role_node.get('tipo') == 'vortgrupo':
                    for priskribo in role_node.get('priskriboj', []):
                        predicates.extend(
                            extract_predicates_from_ast(priskribo, doc_id, clause_depth + 1)
                        )

    elif tipo == 'vortgrupo':
        # Check priskriboj for embedded clauses
        for priskribo in ast.get('priskriboj', []):
            predicates.extend(
                extract_predicates_from_ast(priskribo, doc_id, clause_depth)
            )

    return predicates


def save_checkpoint(checkpoint_path: Path, state: Dict) -> None:
    """Save checkpoint atomically (write to temp, then rename)."""
    temp_path = checkpoint_path.with_suffix('.tmp')
    try:
        with open(temp_path, 'w') as f:
            json.dump(state, f)
        temp_path.rename(checkpoint_path)
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")
        if temp_path.exists():
            temp_path.unlink()


def load_checkpoint(checkpoint_path: Path) -> Optional[Dict]:
    """Load checkpoint if it exists."""
    if not checkpoint_path.exists():
        return None
    try:
        with open(checkpoint_path) as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load checkpoint: {e}")
        return None


def process_corpus(
    corpus_path: Path,
    output_path: Path,
    limit: Optional[int] = None,
    resume: bool = False,
    fresh: bool = False,
):
    """
    Process corpus and extract all predicates.

    Args:
        corpus_path: Path to unified_corpus.jsonl
        output_path: Path for output predicates.jsonl
        limit: Optional limit on documents to process
        resume: Resume from checkpoint if available
        fresh: Start fresh, ignore any checkpoint
    """
    checkpoint_path = output_path.parent / "extract_predicates.checkpoint.json"

    logger.info("=" * 60)
    logger.info(f"Extracting Predicate Triples (v{VERSION})")
    logger.info("=" * 60)
    logger.info(f"Corpus: {corpus_path}")
    logger.info(f"Output: {output_path}")
    if limit:
        logger.info(f"Limit:  {limit:,}")

    start_time = datetime.now()

    # Count total documents
    logger.info("\nCounting documents...")
    with open(corpus_path) as f:
        total_docs = sum(1 for _ in f)
    logger.info(f"  Total: {total_docs:,}")

    if limit:
        total_docs = min(total_docs, limit)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check for checkpoint
    start_doc_id = 0
    stats = {
        'processed': 0,
        'skipped_no_ast': 0,
        'skipped_low_parse': 0,
        'total_predicates': 0,
        'complete_predicates': 0,  # Has verb, subj, obj
        'verb_only': 0,
        'verb_subj': 0,
        'verb_obj': 0,
        'passive': 0,
        'copula': 0,
        'embedded': 0,
    }

    if fresh:
        logger.info("\n--fresh specified, starting from scratch")
        if checkpoint_path.exists():
            checkpoint_path.unlink()
    elif resume or checkpoint_path.exists():
        checkpoint = load_checkpoint(checkpoint_path)
        if checkpoint:
            start_doc_id = checkpoint.get('last_doc_id', 0) + 1
            stats = checkpoint.get('stats', stats)
            logger.info(f"\nResuming from checkpoint:")
            logger.info(f"  Last doc_id: {checkpoint.get('last_doc_id', 0):,}")
            logger.info(f"  Processed:   {stats['processed']:,}")
            logger.info(f"  Predicates:  {stats['total_predicates']:,}")
        else:
            logger.info("\nNo valid checkpoint found, starting fresh")

    # Open output file in append mode if resuming, write mode otherwise
    output_mode = 'a' if start_doc_id > 0 else 'w'

    last_log_time = datetime.now()
    last_checkpoint_time = datetime.now()
    checkpoint_interval = 60  # Save checkpoint every 60 seconds

    with open(corpus_path) as f_in, open(output_path, output_mode) as f_out:
        for doc_id, line in enumerate(f_in):
            # Skip already processed documents
            if doc_id < start_doc_id:
                continue

            if limit and doc_id >= limit:
                break

            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Skip documents without AST
            ast = doc.get('ast')
            if not ast:
                stats['skipped_no_ast'] += 1
                continue

            # Skip documents with low parse rate
            parse_rate = doc.get('parse_rate', 1.0)
            if parse_rate < 0.5:
                stats['skipped_low_parse'] += 1
                continue

            stats['processed'] += 1

            # Extract predicates
            predicates = extract_predicates_from_ast(ast, doc_id)

            for pred in predicates:
                # Write to output
                f_out.write(json.dumps(pred, ensure_ascii=False) + '\n')

                # Update stats
                stats['total_predicates'] += 1

                if pred['subj'] and pred['obj']:
                    stats['complete_predicates'] += 1
                elif pred['subj']:
                    stats['verb_subj'] += 1
                elif pred['obj']:
                    stats['verb_obj'] += 1
                else:
                    stats['verb_only'] += 1

                if pred.get('is_passive'):
                    stats['passive'] += 1
                if pred.get('is_copula'):
                    stats['copula'] += 1
                if pred['clause_depth'] > 0:
                    stats['embedded'] += 1

            now = datetime.now()

            # Log progress every 10 seconds
            if (now - last_log_time).total_seconds() >= 10:
                elapsed = (now - start_time).total_seconds()
                rate = stats['processed'] / elapsed if elapsed > 0 else 0
                pct = (doc_id + 1) / total_docs * 100
                eta_seconds = (total_docs - doc_id - 1) / rate if rate > 0 else 0
                eta_minutes = eta_seconds / 60
                logger.info(
                    f"  {doc_id + 1:,}/{total_docs:,} ({pct:.1f}%) - "
                    f"{stats['total_predicates']:,} predicates - "
                    f"{rate:.0f}/sec - ETA: {eta_minutes:.1f}m"
                )
                last_log_time = now

            # Save checkpoint every minute
            if (now - last_checkpoint_time).total_seconds() >= checkpoint_interval:
                save_checkpoint(checkpoint_path, {
                    'last_doc_id': doc_id,
                    'stats': stats,
                    'version': VERSION,
                    'timestamp': now.isoformat(),
                })
                last_checkpoint_time = now

    # Final stats
    elapsed = (datetime.now() - start_time).total_seconds()

    logger.info("")
    logger.info("=" * 60)
    logger.info("Extraction Complete")
    logger.info("=" * 60)
    logger.info(f"Version: {VERSION}")
    logger.info(f"Time: {elapsed/60:.1f} minutes")
    logger.info("")
    logger.info("Documents:")
    logger.info(f"  Processed:        {stats['processed']:,}")
    logger.info(f"  Skipped (no AST): {stats['skipped_no_ast']:,}")
    logger.info(f"  Skipped (low parse): {stats['skipped_low_parse']:,}")
    logger.info("")
    logger.info("Predicates:")
    logger.info(f"  Total:            {stats['total_predicates']:,}")
    logger.info(f"  Complete (V,S,O): {stats['complete_predicates']:,}")
    logger.info(f"  Verb+Subj only:   {stats['verb_subj']:,}")
    logger.info(f"  Verb+Obj only:    {stats['verb_obj']:,}")
    logger.info(f"  Verb only:        {stats['verb_only']:,}")
    logger.info("")
    logger.info("Special cases:")
    logger.info(f"  Passive:          {stats['passive']:,}")
    logger.info(f"  Copula (estas):   {stats['copula']:,}")
    logger.info(f"  Embedded clause:  {stats['embedded']:,}")
    logger.info("")
    logger.info(f"Output: {output_path}")

    # Remove checkpoint on successful completion
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        logger.info("Checkpoint removed (extraction complete)")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Extract predicate triples from parsed corpus"
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/corpus/unified_corpus.jsonl"),
        help="Path to unified corpus",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/indexes/kuzu_index/predicates.jsonl"),
        help="Output path for predicates",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of documents (for testing)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if available",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Start fresh, ignore any checkpoint",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"extract_predicates.py v{VERSION}",
    )

    args = parser.parse_args()

    # Set up logging (console + file)
    log_file = setup_logging(Path("logs"))

    logger.info(f"extract_predicates.py v{VERSION}")
    logger.info(f"Log file: {log_file}")

    if not args.corpus.exists():
        logger.error(f"Corpus not found: {args.corpus}")
        sys.exit(1)

    process_corpus(
        args.corpus,
        args.output,
        args.limit,
        resume=args.resume,
        fresh=args.fresh,
    )


if __name__ == "__main__":
    main()
