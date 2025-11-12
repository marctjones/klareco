#!/usr/bin/env python3
"""
Comprehensive cleaning script for all Esperanto texts.

Handles three levels of cleaning:
1. Light: Project Gutenberg (remove English headers/footers)
2. Moderate: Wikipedia (remove HTML tags, URLs, wiki markup)
3. Heavy: Literary works (aggressive HTML/JavaScript removal)
"""

import re
from pathlib import Path
from typing import Tuple


def clean_gutenberg_text(text: str) -> Tuple[str, dict]:
    """Clean Project Gutenberg text - remove English headers/footers and boilerplate."""

    stats = {
        'original_size': len(text),
        'removed_header': 0,
        'removed_footer': 0,
        'license_blocks': 0,
        'english_sections': 0
    }

    # Find START marker
    start_match = re.search(r'\*\*\* START OF [^\*]+\*\*\*', text)
    if start_match:
        stats['removed_header'] = start_match.end()
        text = text[start_match.end():]

    # Find END marker
    end_match = re.search(r'\*\*\* END OF [^\*]+\*\*\*', text)
    if end_match:
        stats['removed_footer'] = len(text) - end_match.start()
        text = text[:end_match.start()]

    # Remove Project Gutenberg license/copyright blocks (often in English)
    license_patterns = [
        r'Project Gutenberg.*?License.*?END OF .*?LICENSE',
        r'This eBook is for the use of anyone.*?restrictions whatsoever',
        r'\*\*\*.*?EBOOK.*?\*\*\*',
        r'START:.*?FULL LICENSE',
        r'END:.*?FULL LICENSE',
        r'The Full Project Gutenberg License.*?works',
        r'Section \d+\..*?Foundation',
        r'Information about.*?Project Gutenberg.*?\n\n',
        r'by.*?http://www\.gutenberg\.(org|net)',
        r'Updated editions.*?replacements.*?\n\n',
    ]
    for pattern in license_patterns:
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        stats['license_blocks'] += len(matches)
        text = re.sub(pattern, '\n', text, flags=re.DOTALL | re.IGNORECASE)

    # Remove common English boilerplate sections
    english_patterns = [
        r'Most people start.*?gutenberg\.org',
        r'This and all associated files.*?contact.*?\n\n',
        r'Updated editions will replace.*?defect.*?\n\n',
        r'We do not necessarily keep.*?readable.*?\n\n',
        r'Most people start at.*?located at.*?\n\n',
        r'by.*?E-text prepared by.*?\n\n',
        r'Transcribed from.*?by.*?\n\n',
        r'\[Illustration:.*?\]',
        r'\[NOTE:.*?\]',
    ]
    for pattern in english_patterns:
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        stats['english_sections'] += len(matches)
        text = re.sub(pattern, '\n', text, flags=re.DOTALL | re.IGNORECASE)

    # Normalize multiple newlines to double newlines
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Strip leading/trailing whitespace
    text = text.strip()

    stats['final_size'] = len(text)
    stats['reduction'] = stats['original_size'] - stats['final_size']

    return text, stats


def clean_wikipedia_text(text: str) -> Tuple[str, dict]:
    """Clean Wikipedia text - aggressive HTML, URLs, and MediaWiki markup removal."""

    stats = {
        'original_size': len(text),
        'html_tags': 0,
        'urls': 0,
        'wiki_markup': 0,
        'style_attrs': 0,
        'html_entities': 0
    }

    # Remove script and style blocks first
    text = re.sub(r'<script[^>]*>.*?</script>', ' ', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<style[^>]*>.*?</style>', ' ', text, flags=re.DOTALL | re.IGNORECASE)

    # Remove HTML comments
    text = re.sub(r'<!--.*?-->', ' ', text, flags=re.DOTALL)

    # Count and remove all HTML tags (including attributes)
    html_tags = re.findall(r'<[^>]+>', text)
    stats['html_tags'] = len(html_tags)
    text = re.sub(r'<[^>]+>', ' ', text)

    # Remove common HTML/CSS artifacts that slip through
    style_patterns = [
        r'style\s*=\s*["\'][^"\']*["\']',
        r'class\s*=\s*["\'][^"\']*["\']',
        r'id\s*=\s*["\'][^"\']*["\']',
        r'width\s*=\s*["\']?[\d]+["\']?',
        r'height\s*=\s*["\']?[\d]+["\']?',
        r'background\s*[:\-]\s*[^;]+;?',
        r'color\s*[:\-]\s*[^;]+;?',
        r'border\s*[:\-]\s*[^;]+;?',
        r'margin\s*[:\-]\s*[^;]+;?',
        r'padding\s*[:\-]\s*[^;]+;?',
        r'font\s*[:\-]\s*[^;]+;?',
        r'text-align\s*[:\-]\s*[^;]+;?',
    ]
    for pattern in style_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        stats['style_attrs'] += len(matches)
        text = re.sub(pattern, ' ', text, flags=re.IGNORECASE)

    # Remove HTML entities (&nbsp;, &lt;, etc.)
    entities = re.findall(r'&[a-z]+;', text, re.IGNORECASE)
    stats['html_entities'] = len(entities)
    text = re.sub(r'&[a-z]+;', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'&#\d+;', ' ', text)  # Numeric entities

    # Count and remove URLs
    urls = re.findall(r'https?://\S+', text)
    stats['urls'] = len(urls)
    text = re.sub(r'https?://\S+', ' ', text)

    # Remove MediaWiki templates and markup (more aggressive)
    wiki_patterns = [
        r'\{\{[^\}]*\}\}',              # {{templates}} - nested-safe
        r'\[\[Dosiero:[^\]]+\]\]',      # [[File:...]]
        r'\[\[Bildo:[^\]]+\]\]',        # [[Image:...]]
        r'\[\[Kategorio:[^\]]+\]\]',    # [[Category:...]]
        r'\|[^\|]*\|[^\|]*\|',          # table/template params
        r'cellspacing\s*=\s*["\']?\d+["\']?',
        r'cellpadding\s*=\s*["\']?\d+["\']?',
        r'rowspan\s*=\s*["\']?\d+["\']?',
        r'colspan\s*=\s*["\']?\d+["\']?',
        r'bgcolor\s*=\s*["\'][^"\']*["\']',
        r'valign\s*=\s*["\'][^"\']*["\']',
        r'align\s*=\s*["\'][^"\']*["\']',
    ]
    for pattern in wiki_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        stats['wiki_markup'] += len(matches)
        text = re.sub(pattern, ' ', text, flags=re.IGNORECASE)

    # Clean up [[wikilinks]] - keep only the displayed text
    text = re.sub(r'\[\[([^\|\]]+)\|([^\]]+)\]\]', r'\2', text)  # [[page|text]] -> text
    text = re.sub(r'\[\[([^\]]+)\]\]', r'\1', text)              # [[text]] -> text

    # Remove external links [http://...]
    text = re.sub(r'\[http[^\]]+\]', ' ', text)

    # Remove wiki table syntax remnants
    text = re.sub(r'\|-', ' ', text)
    text = re.sub(r'\|+', ' ', text)
    text = re.sub(r'!+', ' ', text)

    # Remove image file references
    text = re.sub(r'\b\w+\.(jpg|jpeg|png|gif|svg)\b', ' ', text, flags=re.IGNORECASE)

    # Remove common wiki artifacts
    text = re.sub(r"'''", '', text)  # Bold markup
    text = re.sub(r"''", '', text)   # Italic markup

    # Normalize whitespace (collapse multiple spaces/newlines)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    stats['final_size'] = len(text)
    stats['reduction'] = stats['original_size'] - stats['final_size']

    return text, stats


def clean_literary_text(text: str) -> Tuple[str, dict]:
    """Clean literary works - aggressive HTML/JavaScript and Gutenberg boilerplate removal."""

    stats = {
        'original_size': len(text),
        'script_blocks': 0,
        'style_blocks': 0,
        'html_tags': 0,
        'urls': 0,
        'gutenberg_blocks': 0
    }

    # Remove script blocks
    script_blocks = re.findall(r'<script[^>]*>.*?</script>', text, re.DOTALL | re.IGNORECASE)
    stats['script_blocks'] = len(script_blocks)
    text = re.sub(r'<script[^>]*>.*?</script>', ' ', text, flags=re.DOTALL | re.IGNORECASE)

    # Remove style blocks
    style_blocks = re.findall(r'<style[^>]*>.*?</style>', text, re.DOTALL | re.IGNORECASE)
    stats['style_blocks'] = len(style_blocks)
    text = re.sub(r'<style[^>]*>.*?</style>', ' ', text, flags=re.DOTALL | re.IGNORECASE)

    # Remove comments
    text = re.sub(r'<!--.*?-->', ' ', text, flags=re.DOTALL)

    # Count and remove all HTML tags
    html_tags = re.findall(r'<[^>]+>', text)
    stats['html_tags'] = len(html_tags)
    text = re.sub(r'<[^>]+>', ' ', text)

    # Remove URLs
    urls = re.findall(r'https?://\S+', text)
    stats['urls'] = len(urls)
    text = re.sub(r'https?://\S+', ' ', text)

    # Remove email addresses
    text = re.sub(r'\S+@\S+', ' ', text)

    # Remove Project Gutenberg boilerplate (common in literary files)
    gutenberg_patterns = [
        r'Project Gutenberg.*?License.*?\n\n',
        r'\*\*\*.*?START.*?\*\*\*',
        r'\*\*\*.*?END.*?\*\*\*',
        r'The Full Project Gutenberg License.*?\n\n',
        r'This eBook is for the use of.*?\n\n',
        r'Updated editions will replace.*?\n\n',
        r'by.*?www\.gutenberg\.(org|net).*?\n',
        r'E-text prepared by.*?\n',
        r'Transcribed from.*?\n',
        r'Literary Archive Foundation.*?\n',
        r'Section \d+\..*?Agreement.*?\n\n',
        r'donations.*?tax deductible.*?\n\n',
        r'Information about.*?how to help.*?\n\n',
        r'\[Illustration.*?\]',
        r'\[NOTE:.*?\]',
    ]
    for pattern in gutenberg_patterns:
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        stats['gutenberg_blocks'] += len(matches)
        text = re.sub(pattern, '\n', text, flags=re.DOTALL | re.IGNORECASE)

    # Remove common web artifacts
    text = re.sub(r'&\w+;', ' ', text)  # HTML entities like &nbsp;
    text = re.sub(r'\{[^\}]+\}', ' ', text)  # JSON-like structures

    # Normalize whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 newlines
    text = re.sub(r' +', ' ', text)  # Collapse spaces
    text = text.strip()

    stats['final_size'] = len(text)
    stats['reduction'] = stats['original_size'] - stats['final_size']

    return text, stats


def main():
    data_dir = Path(__file__).parent.parent / 'data'
    output_dir = data_dir / 'clean_corpus'
    output_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("CLEANING ALL ESPERANTO TEXTS")
    print("=" * 80)
    print()

    all_stats = []

    # 1. Project Gutenberg files (light cleaning)
    print("1. PROJECT GUTENBERG FILES (Light Cleaning)")
    print("-" * 80)

    gutenberg_dir = data_dir / 'gutenberg_esperanto'
    for filepath in sorted(gutenberg_dir.glob('*.txt')):
        text = filepath.read_text(encoding='utf-8')
        cleaned_text, stats = clean_gutenberg_text(text)

        output_file = output_dir / filepath.name
        output_file.write_text(cleaned_text, encoding='utf-8')

        reduction_pct = (stats['reduction'] / stats['original_size'] * 100) if stats['original_size'] > 0 else 0
        print(f"  ✓ {filepath.name:45s} {stats['original_size']:>8d} → {stats['final_size']:>8d} ({reduction_pct:>5.1f}% removed)")

        all_stats.append({'file': filepath.name, 'type': 'gutenberg', **stats})

    print()

    # 2. Wikipedia (moderate cleaning)
    print("2. WIKIPEDIA (Moderate Cleaning)")
    print("-" * 80)

    wiki_file = data_dir / 'cleaned' / 'cleaned_wikipedia.txt'
    if wiki_file.exists():
        text = wiki_file.read_text(encoding='utf-8')
        cleaned_text, stats = clean_wikipedia_text(text)

        output_file = output_dir / 'wikipedia.txt'
        output_file.write_text(cleaned_text, encoding='utf-8')

        reduction_pct = (stats['reduction'] / stats['original_size'] * 100) if stats['original_size'] > 0 else 0
        print(f"  ✓ wikipedia.txt {stats['original_size']:>10d} → {stats['final_size']:>10d} ({reduction_pct:>5.1f}% removed)")
        print(f"    Removed: {stats['html_tags']} HTML tags, {stats['urls']} URLs, {stats['wiki_markup']} wiki markup")

        all_stats.append({'file': 'wikipedia.txt', 'type': 'wikipedia', **stats})

    print()

    # 3. Literary works from Gutenberg (light cleaning - same as Gutenberg)
    print("3. LITERARY WORKS FROM GUTENBERG (Light Cleaning)")
    print("-" * 80)

    corpora_dir = data_dir / 'corpora'
    gutenberg_literary_files = [
        'kadavrejo_strato.txt',
        'la_korvo.txt',
        'puto_kaj_pendolo.txt',
        'ses_noveloj.txt',
        'usxero_domo.txt',
        'alicio.txt',
        'frankenstejno.txt',
        'jekyll_hyde.txt',
        'milito_de_la_mondoj.txt',
        'sorcxisto_de_oz.txt',
    ]

    for filename in gutenberg_literary_files:
        filepath = corpora_dir / filename
        if not filepath.exists():
            print(f"  ✗ {filename:45s} NOT FOUND")
            continue

        text = filepath.read_text(encoding='utf-8', errors='ignore')
        cleaned_text, stats = clean_gutenberg_text(text)  # Use Gutenberg cleaning

        output_file = output_dir / filename
        output_file.write_text(cleaned_text, encoding='utf-8')

        reduction_pct = (stats['reduction'] / stats['original_size'] * 100) if stats['original_size'] > 0 else 0
        print(f"  ✓ {filename:45s} {stats['original_size']:>8d} → {stats['final_size']:>8d} ({reduction_pct:>5.1f}% removed)")

        all_stats.append({'file': filename, 'type': 'gutenberg_literary', **stats})

    print()

    # 4. Tolkien works (heavy HTML cleaning)
    print("4. TOLKIEN WORKS (Heavy HTML Cleaning)")
    print("-" * 80)

    tolkien_files = [
        'la_hobito.txt',
        'la_mastro_de_l_ringoj.txt',
    ]

    for filename in tolkien_files:
        filepath = corpora_dir / filename
        if not filepath.exists():
            print(f"  ✗ {filename:45s} NOT FOUND")
            continue

        text = filepath.read_text(encoding='utf-8', errors='ignore')
        cleaned_text, stats = clean_literary_text(text)  # Use heavy cleaning for HTML

        output_file = output_dir / filename
        output_file.write_text(cleaned_text, encoding='utf-8')

        reduction_pct = (stats['reduction'] / stats['original_size'] * 100) if stats['original_size'] > 0 else 0
        print(f"  ✓ {filename:45s} {stats['original_size']:>8d} → {stats['final_size']:>8d} ({reduction_pct:>5.1f}% removed)")
        if stats.get('script_blocks', 0) > 0 or stats.get('style_blocks', 0) > 0:
            print(f"    Removed: {stats.get('script_blocks', 0)} scripts, {stats.get('style_blocks', 0)} styles, {stats.get('html_tags', 0)} tags")

        all_stats.append({'file': filename, 'type': 'tolkien', **stats})

    # Summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    total_original = sum(s['original_size'] for s in all_stats)
    total_final = sum(s['final_size'] for s in all_stats)
    total_reduction = total_original - total_final

    print(f"Files processed: {len(all_stats)}")
    print(f"Original size:   {total_original:,} bytes ({total_original/1024/1024:.1f} MB)")
    print(f"Final size:      {total_final:,} bytes ({total_final/1024/1024:.1f} MB)")
    print(f"Removed:         {total_reduction:,} bytes ({total_reduction/total_original*100:.1f}%)")
    print(f"\nAll clean files saved to: {output_dir}")

    # Create manifest
    manifest_file = output_dir / 'MANIFEST.md'
    with open(manifest_file, 'w', encoding='utf-8') as f:
        f.write("# Clean Esperanto Corpus\n\n")
        f.write(f"**Created:** {Path(__file__).name}\n")
        f.write(f"**Total files:** {len(all_stats)}\n")
        f.write(f"**Total size:** {total_final/1024/1024:.1f} MB\n\n")

        for text_type in ['gutenberg', 'wikipedia', 'literary']:
            f.write(f"## {text_type.upper()}\n\n")
            for s in all_stats:
                if s['type'] == text_type:
                    f.write(f"- {s['file']} ({s['final_size']:,} bytes)\n")
            f.write("\n")

    print(f"Manifest created: {manifest_file}")


if __name__ == '__main__':
    main()
