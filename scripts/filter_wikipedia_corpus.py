#!/usr/bin/env python3
"""
Filter Wikipedia content from corpus and Kuzu index.

Implements filters from issues #538, #539, #540, #541:
- Non-article content (templates, test pages, modules)
- List-heavy articles and sections
- Disambiguation pages
- Table-heavy articles with minimal prose

Usage:
    python scripts/filter_wikipedia_corpus.py \\
        --corpus data/enhanced_corpus/corpus_with_metadata.jsonl \\
        --index data/indexes/kuzu_index \\
        --output data/enhanced_corpus/corpus_filtered.jsonl \\
        --dry-run  # Show what would be filtered without making changes
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Set, Tuple, Dict
import re

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import kuzu
except ImportError:
    print("Warning: kuzu not installed. Run: pip install kuzu")
    kuzu = None

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# Filter 1: Non-article content (#538)
EXCLUDED_NAMESPACES = {
    'Template:', 'Module:', 'Wikipedia:', 'Help:', 'User:',
    'MediaWiki:', 'Category:', 'File:', 'Portal:',
    'Draft:', 'Special:', 'Talk:'
}

def is_non_article_content(doc: dict) -> Tuple[bool, str]:
    """Check if document is non-article content (templates, modules, etc)."""
    source = doc.get('source', {})
    if source.get('name') != 'wikipedia':
        return False, ""

    # Get article title from metadata
    metadata = doc.get('metadata', {})
    article = metadata.get('article', '')

    # Check namespace
    for namespace in EXCLUDED_NAMESPACES:
        if article.startswith(namespace):
            return True, f"namespace:{namespace}"

    # Check for test pages in title
    if '/provejo' in article or '/sandbox' in article.lower():
        return True, "test_page_title"

    text = doc.get('text', '')
    if len(text) < 50:
        return False, ""

    # Check for test pages in content
    test_phrases = [
        'Ĉi tie aperu nur la testoj',  # "Here should appear only the tests"
        '/provejo',  # Test page path
        'por konservi ilin',  # "to preserve them" - common in test pages
    ]
    for phrase in test_phrases:
        if phrase in text:
            return True, f"test_page_content:{phrase}"

    # Check for template/module documentation ({{#invoke: patterns)
    if '{{#invoke:' in text:
        invoke_count = text.count('{{#invoke:')
        # More than 3 invocations likely template documentation
        if invoke_count >= 3:
            return True, f"template_doc:{invoke_count}_invokes"

    # Check for repeated "redonas" (returns) - common in template docs
    if text.count('redonas') > 5 or text.count('Dies gibt zurück') > 3 or text.count('This will output') > 3:
        return True, "template_doc:returns_pattern"

    # Check template code density (lowered to >15% from 30%)
    brace_count = text.count('{') + text.count('}')
    density = brace_count / len(text)
    if density > 0.15:
        return True, f"template_heavy:{density:.2f}"

    # Check for pipe-heavy content (wiki table markup)
    pipe_count = text.count('|')
    pipe_density = pipe_count / len(text)
    if pipe_density > 0.08:  # >8% pipes = heavy wiki markup
        return True, f"markup_heavy:{pipe_density:.2f}"

    return False, ""


# Filter 2: List-heavy content (#539)
def is_list_heavy(doc: dict) -> Tuple[bool, str]:
    """Check if document is list-heavy."""
    source = doc.get('source', {})
    if source.get('name') != 'wikipedia':
        return False, ""

    metadata = doc.get('metadata', {})
    article = metadata.get('article', '')

    # Check for list article titles
    if article.startswith(('Listo de ', 'Liste de ', 'List of ')):
        return True, "list_title"

    text = doc.get('text', '')
    if len(text) < 50:
        return False, ""

    # Check list markup density
    lines = text.split('\n')
    if not lines:
        return False, ""

    list_lines = sum(1 for line in lines if line.strip().startswith(('* ', '- ', '# ')))
    list_ratio = list_lines / len(lines)

    if list_ratio > 0.6:
        return True, f"list_heavy:{list_ratio:.2f}"

    # Check for name catalogs (high capitalized word ratio)
    words = text.split()
    if len(words) > 20:
        cap_words = sum(1 for w in words if w and w[0].isupper())
        cap_ratio = cap_words / len(words)
        if cap_ratio > 0.7:  # >70% capitalized = likely name list
            return True, f"name_catalog:{cap_ratio:.2f}"

    # Check for taxonomy lists (biological classification)
    # Pattern: :genro:, ::specio:, :::subspecio:
    taxonomy_markers = text.count(':genro:') + text.count('::specio:') + text.count(':::subspecio:')
    if taxonomy_markers > 5:
        return True, f"taxonomy_list:{taxonomy_markers}_markers"

    # Check for colon-heavy lists (often taxonomies or structured data)
    # Only applies if we have multiple lines
    if len(lines) > 3:
        lines_with_colons = sum(1 for line in lines if line.count(':') >= 2)
        if lines_with_colons > len(lines) * 0.4 and lines_with_colons > 5:  # >40% AND >5 lines
            return True, f"structured_list:{lines_with_colons}/{len(lines)}"

    return False, ""


# Filter 3: Disambiguation pages (#540)
def is_disambiguation(doc: dict) -> Tuple[bool, str]:
    """Check if document is a disambiguation page."""
    source = doc.get('source', {})
    if source.get('name') != 'wikipedia':
        return False, ""

    metadata = doc.get('metadata', {})
    article = metadata.get('article', '')
    text = doc.get('text', '')

    # Check title
    if '(apartigilo)' in article.lower() or '(disambiguation)' in article.lower():
        return True, "title_marker"

    # Check for disambiguation phrases
    disambig_phrases = [
        'povas rilati al:',
        'povas signifi:',
        'can refer to:',
        'may refer to:',
        'rilatas al:'
    ]

    text_lower = text.lower()
    for phrase in disambig_phrases:
        if phrase in text_lower:
            return True, f"phrase:{phrase}"

    return False, ""


# Filter 4: Table-heavy with minimal prose (#541)
def is_table_heavy(doc: dict) -> Tuple[bool, str]:
    """Check if document is table-heavy with minimal prose."""
    source = doc.get('source', {})
    if source.get('name') != 'wikipedia':
        return False, ""

    text = doc.get('text', '')
    if len(text) < 100:
        return False, ""

    # Count table markup: {|, |-, |}, ||
    table_markers = text.count('{|') + text.count('|-') + text.count('|}')

    # Also count pipe-heavy lines (table rows)
    lines = text.split('\n')
    pipe_heavy_lines = sum(1 for line in lines if line.count('|') > 3)

    if table_markers > 5 or pipe_heavy_lines > len(lines) * 0.3:
        # Has significant table markup - check prose content
        # Remove table-like lines
        prose_lines = [line for line in lines if line.count('|') <= 2]
        prose = '\n'.join(prose_lines)

        # Count sentences in prose
        sentence_markers = prose.count('.') + prose.count('!') + prose.count('?')

        if sentence_markers < 10:
            table_ratio = (len(text) - len(prose)) / len(text)
            return True, f"table_heavy:{table_ratio:.2f},prose_sentences:{sentence_markers}"

    return False, ""


def apply_filters(doc: dict) -> Tuple[bool, str]:
    """Apply all filters to a document.

    Returns:
        (should_filter, reason)
    """
    # Try each filter
    filters = [
        ("non_article", is_non_article_content),
        ("list_heavy", is_list_heavy),
        ("disambiguation", is_disambiguation),
        ("table_heavy", is_table_heavy),
    ]

    for filter_name, filter_func in filters:
        should_filter, detail = filter_func(doc)
        if should_filter:
            return True, f"{filter_name}:{detail}"

    return False, ""


def filter_corpus(corpus_path: Path, output_path: Path, dry_run: bool = False) -> Set[str]:
    """Filter corpus and write clean version.

    Returns:
        Set of sentence IDs to remove
    """
    logger.info(f"Reading corpus from {corpus_path}...")

    filtered_ids = set()
    filter_stats = {}
    kept_count = 0
    filtered_count = 0

    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out_file = open(output_path, 'w', encoding='utf-8')

    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line_num % 100000 == 0:
                logger.info(f"  Processed {line_num:,} sentences...")

            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(f"  Skipping invalid JSON at line {line_num}")
                continue

            # Apply filters
            should_filter, reason = apply_filters(doc)

            if should_filter:
                # Track for deletion
                sentence_id = doc.get('id')
                if sentence_id:
                    filtered_ids.add(sentence_id)

                # Update stats
                filter_type = reason.split(':')[0]
                filter_stats[filter_type] = filter_stats.get(filter_type, 0) + 1
                filtered_count += 1

                # Log first few examples
                if filtered_count <= 10:
                    article = doc.get('metadata', {}).get('article', 'unknown')
                    logger.info(f"  FILTER: {article} - {reason}")
            else:
                # Keep this sentence
                if not dry_run:
                    out_file.write(line)
                kept_count += 1

    if not dry_run:
        out_file.close()

    # Report statistics
    logger.info("")
    logger.info("=" * 60)
    logger.info("Filtering Summary")
    logger.info("=" * 60)
    logger.info(f"Total processed: {kept_count + filtered_count:,}")
    logger.info(f"Kept: {kept_count:,} ({kept_count/(kept_count+filtered_count)*100:.1f}%)")
    logger.info(f"Filtered: {filtered_count:,} ({filtered_count/(kept_count+filtered_count)*100:.1f}%)")
    logger.info("")
    logger.info("Filter breakdown:")
    for filter_type, count in sorted(filter_stats.items(), key=lambda x: -x[1]):
        logger.info(f"  {filter_type}: {count:,}")
    logger.info("")

    if not dry_run:
        logger.info(f"Filtered corpus written to: {output_path}")

    return filtered_ids


def delete_from_index(index_path: Path, filtered_ids: Set[str], dry_run: bool = False):
    """Delete filtered sentences from Kuzu index."""
    if kuzu is None:
        logger.error("Kuzu not installed - cannot update index")
        return

    if not filtered_ids:
        logger.info("No sentences to delete from index")
        return

    logger.info("")
    logger.info("=" * 60)
    logger.info("Updating Kuzu Index")
    logger.info("=" * 60)
    logger.info(f"Sentences to delete: {len(filtered_ids):,}")

    if dry_run:
        logger.info("DRY RUN - no changes will be made")
        return

    # Connect to database
    db_path = index_path / "kuzu.db"
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        return

    logger.info(f"Connecting to {db_path}...")
    db = kuzu.Database(str(index_path))
    conn = kuzu.Connection(db)

    # Delete in batches to avoid query size limits
    batch_size = 1000
    id_list = list(filtered_ids)
    total_batches = (len(id_list) + batch_size - 1) // batch_size

    logger.info(f"Deleting in {total_batches} batches...")

    for i in range(0, len(id_list), batch_size):
        batch = id_list[i:i+batch_size]
        batch_num = i // batch_size + 1

        # Format IDs for Cypher query
        id_list_str = ', '.join(f'"{id}"' for id in batch)

        # Delete sentences and their edges
        query = f"""
        MATCH (s:Sentence)
        WHERE s.id IN [{id_list_str}]
        DETACH DELETE s
        """

        try:
            conn.execute(query)
            logger.info(f"  Batch {batch_num}/{total_batches}: Deleted {len(batch)} sentences")
        except Exception as e:
            logger.error(f"  Batch {batch_num} failed: {e}")

    # Verify deletion
    result = conn.execute("MATCH (s:Sentence) RETURN count(s) as count")
    remaining = result.get_next()[0]
    logger.info(f"")
    logger.info(f"Remaining sentences in index: {remaining:,}")

    # Note: We don't delete Root nodes even if orphaned - they might be referenced elsewhere
    # and are small enough not to matter

    logger.info("Index update complete")


def main():
    parser = argparse.ArgumentParser(description='Filter Wikipedia content from corpus and index')
    parser.add_argument('--corpus', type=Path, required=True,
                       help='Input corpus file (corpus_with_metadata.jsonl)')
    parser.add_argument('--index', type=Path,
                       help='Kuzu index directory (optional - if provided, will update index)')
    parser.add_argument('--output', type=Path,
                       help='Output filtered corpus file (default: corpus_with_metadata.jsonl.filtered)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be filtered without making changes')

    args = parser.parse_args()

    if not args.output:
        args.output = args.corpus.with_suffix('.jsonl.filtered')

    # Filter corpus
    filtered_ids = filter_corpus(args.corpus, args.output, dry_run=args.dry_run)

    # Update index if provided
    if args.index and args.index.exists():
        delete_from_index(args.index, filtered_ids, dry_run=args.dry_run)
    elif args.index:
        logger.warning(f"Index directory not found: {args.index}")

    logger.info("")
    logger.info("Done!")


if __name__ == '__main__':
    main()
