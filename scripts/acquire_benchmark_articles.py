#!/usr/bin/env python3
"""
Fetch benchmark articles from Wikipedia for data quality testing.

Downloads the 50 most popular Esperanto Wikipedia articles and the
Esperanto equivalents of the 50 most popular English Wikipedia articles.

These are used to verify that our extracted Wikipedia data is complete.
"""

import json
import requests
import time
from pathlib import Path
from typing import Optional

# Output directory
OUTPUT_DIR = Path('data/benchmarks/wikipedia_articles')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# User agent as required by Wikipedia API
HEADERS = {
    'User-Agent': 'KlarecoBot/1.0 (https://github.com/marctjones/klareco; Esperanto NLP research)'
}

# Top 50 English Wikipedia articles (2024) - from Wikipedia:2024 Top 50 Report
# These are well-known articles that should have Esperanto equivalents
TOP_ENGLISH_ARTICLES = [
    "Deaths in 2024",
    "Kamala Harris",
    "2024 United States presidential election",
    "Menendez brothers",
    "Donald Trump",
    "Taylor Swift",
    "JD Vance",
    "Deadpool & Wolverine",
    "Project 2025",
    "YouTube",
    "Joe Biden",
    "ChatGPT",
    "Elon Musk",
    "Tim Walz",
    "Cristiano Ronaldo",
    "2024 Summer Olympics",
    "Lionel Messi",
    "India",
    "United States",
    "Cleopatra",
    "Napoleon",
    "World War II",
    "Albert Einstein",
    "Leonardo da Vinci",
    "William Shakespeare",
    "Adolf Hitler",
    "Abraham Lincoln",
    "Queen Elizabeth II",
    "Michael Jackson",
    "Elvis Presley",
    "The Beatles",
    "Barack Obama",
    "Vladimir Putin",
    "China",
    "Japan",
    "Germany",
    "France",
    "United Kingdom",
    "Russia",
    "Brazil",
    "Australia",
    "Canada",
    "Italy",
    "Spain",
    "Mexico",
    "Argentina",
    "South Korea",
    "Indonesia",
    "Netherlands",
    "Switzerland",
]

# Well-known Esperanto Wikipedia articles (popular topics in Esperanto community)
TOP_ESPERANTO_ARTICLES = [
    "Esperanto",
    "Ludoviko Lazaro Zamenhof",
    "Vikipedio",
    "Universala Kongreso de Esperanto",
    "Pasporta Servo",
    "Akademio de Esperanto",
    "Esperantio",
    "Esperantujo",
    "Fundamento de Esperanto",
    "La Espero",
    "Fina Venko",
    "Internacia Esperanto-Muzeo",
    "Propaedeutica valoro de Esperanto",
    "Esperanto-kulturo",
    "Esperanto-muziko",
    "Esperanto-literaturo",
    "Originala Esperanta Literaturo",
    "Historio de Esperanto",
    "Gramatiko de Esperanto",
    "Vortaro de Esperanto",
    "Unua Libro",
    "Dua Libro de l' Lingvo Internacia",
    "Fundamenta Krestomatio",
    "La Ondo de Esperanto",
    "Monato",
    "Kontakto",
    "Sennacieca Asocio Tutmonda",
    "Universala Esperanto-Asocio",
    "Junulara Esperanto-Organizo",
    "Internacia Kongresa Universitato",
    # General topics popular in any language
    "Eŭropo",
    "Azio",
    "Afriko",
    "Ameriko",
    "Oceanio",
    "Parizo",
    "Londono",
    "Berlino",
    "Moskvo",
    "Tokio",
    "Novjorko",
    "Sciencoj",
    "Matematiko",
    "Fiziko",
    "Kemio",
    "Biologio",
    "Medicino",
    "Astronomio",
    "Filozofio",
    "Historio",
]


def get_eo_equivalent(en_title: str) -> Optional[str]:
    """Get Esperanto Wikipedia title for an English article."""
    url = f"https://en.wikipedia.org/w/api.php"
    params = {
        'action': 'query',
        'titles': en_title,
        'prop': 'langlinks',
        'lllang': 'eo',
        'format': 'json',
    }

    try:
        resp = requests.get(url, params=params, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        pages = data.get('query', {}).get('pages', {})
        for page_id, page in pages.items():
            if page_id == '-1':
                return None
            langlinks = page.get('langlinks', [])
            for ll in langlinks:
                if ll.get('lang') == 'eo':
                    return ll.get('*')
        return None
    except Exception as e:
        print(f"  Error getting EO equivalent for {en_title}: {e}")
        return None


def get_article_content(title: str, lang: str = 'eo') -> Optional[dict]:
    """Get article content from Wikipedia."""
    url = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        'action': 'query',
        'titles': title,
        'prop': 'extracts|info',
        'explaintext': True,  # Plain text, no HTML
        'exsectionformat': 'plain',
        'format': 'json',
    }

    try:
        resp = requests.get(url, params=params, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        pages = data.get('query', {}).get('pages', {})
        for page_id, page in pages.items():
            if page_id == '-1':
                return None
            return {
                'title': page.get('title'),
                'pageid': int(page_id),
                'extract': page.get('extract', ''),
                'length': page.get('length', 0),
            }
        return None
    except Exception as e:
        print(f"  Error fetching {title}: {e}")
        return None


def main():
    print("=" * 60)
    print("Fetching benchmark Wikipedia articles")
    print("=" * 60)

    all_articles = []
    seen_titles = set()

    # 1. Get Esperanto equivalents of top English articles
    print("\n[1/3] Finding Esperanto equivalents of top English articles...")
    en_to_eo = {}

    for i, en_title in enumerate(TOP_ENGLISH_ARTICLES, 1):
        print(f"  [{i:2d}/50] {en_title}...", end=" ", flush=True)
        eo_title = get_eo_equivalent(en_title)
        if eo_title:
            en_to_eo[en_title] = eo_title
            print(f"→ {eo_title}")
        else:
            print("(no EO equivalent)")
        time.sleep(0.2)  # Rate limiting

    print(f"\n  Found {len(en_to_eo)} Esperanto equivalents")

    # 2. Download Esperanto articles
    print("\n[2/3] Downloading Esperanto articles from English equivalents...")

    for en_title, eo_title in en_to_eo.items():
        if eo_title in seen_titles:
            continue

        print(f"  Downloading: {eo_title}...", end=" ", flush=True)
        article = get_article_content(eo_title, 'eo')
        if article and article.get('extract'):
            article['source'] = 'english_top50'
            article['english_title'] = en_title
            all_articles.append(article)
            seen_titles.add(eo_title)
            print(f"OK ({len(article['extract'])} chars)")
        else:
            print("(empty or not found)")
        time.sleep(0.2)

    # 3. Download native Esperanto popular articles
    print("\n[3/3] Downloading popular Esperanto-native articles...")

    for eo_title in TOP_ESPERANTO_ARTICLES:
        if eo_title in seen_titles:
            print(f"  Skipping (duplicate): {eo_title}")
            continue

        print(f"  Downloading: {eo_title}...", end=" ", flush=True)
        article = get_article_content(eo_title, 'eo')
        if article and article.get('extract'):
            article['source'] = 'esperanto_popular'
            all_articles.append(article)
            seen_titles.add(eo_title)
            print(f"OK ({len(article['extract'])} chars)")
        else:
            print("(empty or not found)")
        time.sleep(0.2)

    # Save results
    print("\n" + "=" * 60)
    print(f"Downloaded {len(all_articles)} articles")

    # Save as JSON
    output_file = OUTPUT_DIR / 'benchmark_articles.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'metadata': {
                'total_articles': len(all_articles),
                'from_english_top50': sum(1 for a in all_articles if a.get('source') == 'english_top50'),
                'esperanto_popular': sum(1 for a in all_articles if a.get('source') == 'esperanto_popular'),
            },
            'articles': all_articles
        }, f, ensure_ascii=False, indent=2)

    print(f"Saved to: {output_file}")

    # Save titles list for quick reference
    titles_file = OUTPUT_DIR / 'benchmark_titles.txt'
    with open(titles_file, 'w', encoding='utf-8') as f:
        for article in all_articles:
            f.write(f"{article['title']}\n")
    print(f"Titles saved to: {titles_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("Summary of downloaded articles:")
    print("=" * 60)
    for i, article in enumerate(all_articles[:20], 1):
        title = article['title'][:40]
        chars = len(article['extract'])
        print(f"  {i:2d}. {title:42s} ({chars:,} chars)")
    if len(all_articles) > 20:
        print(f"  ... and {len(all_articles) - 20} more")


if __name__ == '__main__':
    main()
