#!/usr/bin/env python3
"""
Extract Wikipedia articles with full metadata (title, ID, sections).

Features:
- Extracts article title and ID from MediaWiki XML
- Preserves section structure within articles
- Progress indicators every 100 articles
- Error logging with context
- Resumable with checkpoints
- Memory-efficient streaming
"""

import bz2
import json
import logging
import sys
import time
from pathlib import Path
from typing import Iterator, Optional
import xml.etree.ElementTree as ET

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('logs/wikipedia_extraction.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def parse_mediawiki_dump(xml_file: Path) -> Iterator[dict]:
    """Stream parse MediaWiki XML dump."""
    logger.info(f"Opening Wikipedia dump: {xml_file}")

    with bz2.open(xml_file, 'rt', encoding='utf-8') as f:
        context = ET.iterparse(f, events=['start', 'end'])
        context = iter(context)
        event, root = next(context)

        article_count = 0
        current_page = {}
        in_page = False
        in_revision = False

        for event, elem in context:
            # Extract tag name without namespace
            if '}' in elem.tag:
                tag = elem.tag.split('}')[1]
            else:
                tag = elem.tag

            if event == 'start':
                if tag == 'page':
                    in_page = True
                    current_page = {}
                elif tag == 'revision':
                    in_revision = True

            elif event == 'end' and in_page:
                if tag == 'title':
                    current_page['article_title'] = elem.text or ''

                elif tag == 'id':
                    # First <id> is article ID (before <revision>)
                    if 'article_id' not in current_page and not in_revision:
                        try:
                            current_page['article_id'] = int(elem.text)
                        except (ValueError, TypeError):
                            pass

                elif tag == 'timestamp' and in_revision:
                    current_page['timestamp'] = elem.text

                elif tag == 'text' and in_revision:
                    current_page['text'] = elem.text or ''

                elif tag == 'revision':
                    in_revision = False

                elif tag == 'page':
                    in_page = False

                    # Skip redirects and empty pages
                    if current_page.get('text') and current_page.get('article_id'):
                        article_count += 1

                        if article_count % 100 == 0:
                            logger.info(f"✓ Processed {article_count} articles (current: '{current_page.get('article_title', 'N/A')}')")

                        yield current_page

                    current_page = {}
                    elem.clear()
                    root.clear()

        logger.info(f"Finished: {article_count} articles total")


def extract_sections(text: str) -> list[dict]:
    """
    Extract sections from MediaWiki text.

    MediaWiki sections are marked by == Section Name ==

    Returns:
        list of dicts with keys: section_name, section_level, content
    """
    sections = []
    current_section = {'section_name': None, 'section_level': 0, 'content': []}

    for line in text.split('\n'):
        # Check for section headers (== Header ==, === Subheader ===, etc.)
        if line.strip().startswith('==') and line.strip().endswith('=='):
            # Save previous section if it has content
            if current_section['content']:
                sections.append(current_section.copy())

            # Parse new section
            stripped = line.strip()
            level = 0
            while stripped.startswith('='):
                level += 1
                stripped = stripped[1:]
            while stripped.endswith('='):
                stripped = stripped[:-1]

            section_name = stripped.strip()
            current_section = {
                'section_name': section_name,
                'section_level': level,
                'content': []
            }
        else:
            # Add to current section content
            if line.strip():  # Skip empty lines
                current_section['content'].append(line.strip())

    # Add final section
    if current_section['content']:
        sections.append(current_section)

    return sections


def clean_mediawiki_markup(text: str) -> str:
    """
    Remove MediaWiki markup from text.

    Removes:
    - Wiki links: [[Link|Display]] -> Display
    - Bold/italic: '''bold''' -> bold
    - Templates: {{template}} -> (removed)
    - HTML comments: <!-- comment --> -> (removed)
    """
    import re

    # Remove HTML comments
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)

    # Remove templates ({{...}})
    # This is a simplified approach; real MediaWiki parsing is complex
    text = re.sub(r'\{\{[^}]*\}\}', '', text)

    # Convert wiki links [[Link|Display]] -> Display, [[Link]] -> Link
    text = re.sub(r'\[\[([^\]|]+)\|([^\]]+)\]\]', r'\2', text)
    text = re.sub(r'\[\[([^\]]+)\]\]', r'\1', text)

    # Remove bold/italic markup
    text = re.sub(r"'''([^']+)'''", r'\1', text)
    text = re.sub(r"''([^']+)''", r'\1', text)

    # Remove external links [http://... text] -> text
    text = re.sub(r'\[http[^\s]+ ([^\]]+)\]', r'\1', text)
    text = re.sub(r'\[http[^\s]+\]', '', text)

    return text.strip()


def extract_sentences(text: str) -> list[str]:
    """
    Extract sentences from text.

    Simple sentence splitting on . ! ?
    """
    import re

    # Split on sentence boundaries
    sentences = re.split(r'[.!?]+', text)

    # Clean and filter
    sentences = [s.strip() for s in sentences if s.strip()]

    # Filter out very short sentences (likely fragments)
    sentences = [s for s in sentences if len(s.split()) >= 3]

    return sentences


def process_wikipedia_dump(
    xml_file: Path,
    output_file: Path,
    checkpoint_file: Optional[Path] = None,
    checkpoint_interval: int = 1000
):
    """
    Process Wikipedia dump and extract sentences with metadata.

    Args:
        xml_file: Path to MediaWiki XML dump (.xml.bz2)
        output_file: Output JSONL file
        checkpoint_file: Path to checkpoint file for resuming
        checkpoint_interval: Save checkpoint every N articles
    """
    # Load checkpoint if exists
    processed_articles = set()
    if checkpoint_file and checkpoint_file.exists():
        logger.info(f"Loading checkpoint from {checkpoint_file}")
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
            processed_articles = set(checkpoint.get('processed_articles', []))
            logger.info(f"Resuming: {len(processed_articles)} articles already processed")

    # Open output file in append mode
    mode = 'a' if processed_articles else 'w'

    total_sentences = 0
    total_articles = 0
    error_count = 0

    logger.info("=" * 60)
    logger.info("Starting Wikipedia extraction")
    logger.info(f"Input: {xml_file}")
    logger.info(f"Output: {output_file}")
    logger.info("=" * 60)

    start_time = time.time()

    with open(output_file, mode, encoding='utf-8') as out:
        for article in parse_mediawiki_dump(xml_file):
            article_id = article.get('article_id')
            article_title = article.get('article_title', 'N/A')

            # Skip if already processed
            if article_id in processed_articles:
                continue

            try:
                # Clean text
                text = clean_mediawiki_markup(article.get('text', ''))

                # Extract sections
                sections = extract_sections(text)

                # If no sections, treat whole text as one section
                if not sections:
                    sections = [{
                        'section_name': None,
                        'section_level': 0,
                        'content': [text]
                    }]

                # Extract sentences from each section
                article_sentence_count = 0
                for section in sections:
                    section_text = ' '.join(section['content'])
                    sentences = extract_sentences(section_text)

                    for sentence in sentences:
                        entry = {
                            'text': sentence,
                            'source': 'wikipedia',
                            'source_name': 'Vikipedio Esperanto',
                            'article_title': article_title,
                            'article_id': article_id,
                            'section': section['section_name'],
                            'section_level': section['section_level'],
                            'timestamp': article.get('timestamp')
                        }

                        out.write(json.dumps(entry, ensure_ascii=False) + '\n')
                        article_sentence_count += 1
                        total_sentences += 1

                total_articles += 1
                processed_articles.add(article_id)

                # Periodic progress log
                if total_articles % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = total_articles / elapsed
                    logger.info(f"Progress: {total_articles} articles, {total_sentences} sentences ({rate:.1f} articles/sec)")

                # Save checkpoint
                if checkpoint_file and total_articles % checkpoint_interval == 0:
                    with open(checkpoint_file, 'w') as f:
                        json.dump({
                            'processed_articles': list(processed_articles),
                            'total_articles': total_articles,
                            'total_sentences': total_sentences
                        }, f)
                    logger.info(f"✓ Checkpoint saved: {total_articles} articles processed")

            except Exception as e:
                error_count += 1
                logger.error(f"Error processing article '{article_title}' (ID: {article_id}): {e}")

                # Log every 10 errors
                if error_count % 10 == 0:
                    logger.warning(f"⚠ {error_count} errors encountered so far")

    # Final summary
    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("Wikipedia extraction complete!")
    logger.info(f"Total articles processed: {total_articles:,}")
    logger.info(f"Total sentences extracted: {total_sentences:,}")
    logger.info(f"Errors encountered: {error_count}")
    logger.info(f"Time elapsed: {elapsed/60:.1f} minutes")
    logger.info(f"Rate: {total_articles/(elapsed/60):.1f} articles/min")
    logger.info("=" * 60)

    # Save final checkpoint
    if checkpoint_file:
        with open(checkpoint_file, 'w') as f:
            json.dump({
                'processed_articles': list(processed_articles),
                'total_articles': total_articles,
                'total_sentences': total_sentences,
                'completed': True
            }, f)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Extract Wikipedia with metadata')
    parser.add_argument('--xml', type=Path, default=Path('data/corpora/eo_wikipedia.xml.bz2'),
                        help='Path to Wikipedia XML dump')
    parser.add_argument('--output', type=Path, default=Path('data/extracted/wikipedia_sentences.jsonl'),
                        help='Output JSONL file')
    parser.add_argument('--checkpoint', type=Path, default=Path('data/extracted/wikipedia_checkpoint.json'),
                        help='Checkpoint file for resuming')
    parser.add_argument('--checkpoint-interval', type=int, default=1000,
                        help='Save checkpoint every N articles')

    args = parser.parse_args()

    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Create logs directory
    Path('logs').mkdir(exist_ok=True)

    process_wikipedia_dump(
        xml_file=args.xml,
        output_file=args.output,
        checkpoint_file=args.checkpoint,
        checkpoint_interval=args.checkpoint_interval
    )
