#!/usr/bin/env python3
"""
Inventory all Esperanto text files and assess their quality.

This script:
1. Finds all Esperanto text files
2. Checks if they contain actual Esperanto (vs HTML/metadata)
3. Reports on cleanliness and usability
4. Provides recommendations for cleaning
"""

import re
from pathlib import Path
from collections import defaultdict


def assess_file_quality(filepath: Path) -> dict:
    """Assess the quality of an Esperanto text file."""

    try:
        text = filepath.read_text(encoding='utf-8', errors='ignore')
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'size': 0
        }

    size = len(text)
    lines = text.split('\n')

    # Detect Esperanto content
    esperanto_chars = set('ĉĝĥĵŝŭĈĜĤĴŜŬ')
    eo_char_count = sum(1 for c in text if c in esperanto_chars)

    # Detect contamination
    html_tags = len(re.findall(r'<[^>]+>', text))
    urls = len(re.findall(r'https?://\S+', text))
    script_blocks = len(re.findall(r'<script[^>]*>.*?</script>', text, re.DOTALL))

    # Check for common Esperanto words
    common_words = ['la', 'kaj', 'estas', 'de', 'en', 'al', 'kaj']
    word_count = 0
    for word in common_words:
        word_count += len(re.findall(r'\b' + word + r'\b', text, re.IGNORECASE))

    # Detect garbage/corruption
    # Look for lines with mostly non-printable or weird characters
    corrupted_lines = 0
    for line in lines:
        if len(line) > 20:
            printable = sum(1 for c in line if c.isprintable() or c in '\n\t')
            if printable / len(line) < 0.8:
                corrupted_lines += 1

    # Determine quality
    if html_tags > 100 or script_blocks > 0:
        quality = 'needs_cleaning'
        issue = 'HTML/XML embedded'
    elif corrupted_lines > 10:
        quality = 'corrupted'
        issue = f'{corrupted_lines} corrupted lines'
    elif eo_char_count < 10 and word_count < 50:
        quality = 'not_esperanto'
        issue = 'Little/no Esperanto detected'
    elif html_tags > 0:
        quality = 'minor_cleanup'
        issue = f'{html_tags} HTML tags'
    else:
        quality = 'clean'
        issue = 'None'

    return {
        'status': 'ok',
        'size': size,
        'lines': len(lines),
        'esperanto_chars': eo_char_count,
        'common_words': word_count,
        'html_tags': html_tags,
        'urls': urls,
        'script_blocks': script_blocks,
        'corrupted_lines': corrupted_lines,
        'quality': quality,
        'issue': issue
    }


def main():
    data_dir = Path('/home/marc/klareco/data')

    # Find all text files
    text_files = []
    for pattern in ['**/*.txt', '**/*.text']:
        text_files.extend(data_dir.glob(pattern))

    # Exclude __pycache__ and similar
    text_files = [f for f in text_files if '__pycache__' not in str(f)]

    # Organize by directory
    by_dir = defaultdict(list)
    for f in text_files:
        rel_path = f.relative_to(data_dir)
        dir_name = rel_path.parts[0] if len(rel_path.parts) > 1 else 'root'
        by_dir[dir_name].append(f)

    print("=" * 80)
    print("ESPERANTO TEXT INVENTORY")
    print("=" * 80)
    print()

    all_results = []

    for dir_name in sorted(by_dir.keys()):
        files = sorted(by_dir[dir_name])
        print(f"\n{dir_name.upper()}/ ({len(files)} files)")
        print("-" * 80)

        for filepath in files:
            result = assess_file_quality(filepath)
            result['file'] = filepath.name
            result['path'] = str(filepath.relative_to(data_dir))
            all_results.append(result)

            if result['status'] == 'error':
                print(f"  ✗ {filepath.name:40s} ERROR: {result['error']}")
                continue

            size_mb = result['size'] / 1024 / 1024
            quality = result['quality']

            # Status symbol
            if quality == 'clean':
                symbol = '✓'
            elif quality == 'minor_cleanup':
                symbol = '~'
            elif quality == 'needs_cleaning':
                symbol = '⚠'
            elif quality == 'corrupted':
                symbol = '✗'
            else:
                symbol = '?'

            print(f"  {symbol} {filepath.name:40s} {size_mb:6.2f}MB  {quality:15s} {result['issue']}")

    # Summary by quality
    print("\n" + "=" * 80)
    print("SUMMARY BY QUALITY")
    print("=" * 80)

    quality_counts = defaultdict(lambda: {'count': 0, 'size': 0})
    for r in all_results:
        if r['status'] == 'ok':
            q = r['quality']
            quality_counts[q]['count'] += 1
            quality_counts[q]['size'] += r['size']

    for quality in ['clean', 'minor_cleanup', 'needs_cleaning', 'corrupted', 'not_esperanto']:
        if quality in quality_counts:
            c = quality_counts[quality]
            size_mb = c['size'] / 1024 / 1024
            print(f"  {quality:20s}: {c['count']:3d} files ({size_mb:8.2f} MB)")

    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    needs_cleaning = [r for r in all_results if r.get('quality') == 'needs_cleaning']
    corrupted = [r for r in all_results if r.get('quality') == 'corrupted']
    minor = [r for r in all_results if r.get('quality') == 'minor_cleanup']

    if needs_cleaning:
        print(f"\n✓ Files needing HTML cleanup ({len(needs_cleaning)}):")
        for r in needs_cleaning[:10]:
            print(f"  - {r['path']}")
        if len(needs_cleaning) > 10:
            print(f"  ... and {len(needs_cleaning) - 10} more")

    if corrupted:
        print(f"\n✗ Corrupted files to investigate ({len(corrupted)}):")
        for r in corrupted[:5]:
            print(f"  - {r['path']}")

    if minor:
        print(f"\n~ Files needing minor cleanup ({len(minor)}):")
        for r in minor[:5]:
            print(f"  - {r['path']}")

    print("\nNext steps:")
    print("1. Create cleaning script to remove HTML/XML from 'needs_cleaning' files")
    print("2. Investigate corrupted files (may need re-downloading)")
    print("3. Apply minor cleanup to remaining files")
    print("4. Create master clean corpus directory")


if __name__ == '__main__':
    main()
