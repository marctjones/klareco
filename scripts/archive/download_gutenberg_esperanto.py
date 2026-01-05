#!/usr/bin/env python3
"""
Download standard Esperanto texts from Project Gutenberg.

Focuses on:
- Works by Zamenhof (the founder)
- Grammar and reference works written IN Esperanto
- Historical documents and periodicals
- Original Esperanto works (not translations)

Avoids translated literature with foreign language borrowings.
"""

import os
import requests
from pathlib import Path
import time
from typing import Dict, List

# Project Gutenberg base URL for plain text UTF-8 files
GUTENBERG_BASE = "https://www.gutenberg.org/cache/epub/{}/pg{}.txt"

# High-priority texts for standard Esperanto
PRIORITY_TEXTS = {
    "zamenhof": {
        "description": "Works by L.L. Zamenhof (founder - most authoritative)",
        "texts": {
            8224: "Fundamenta Krestomatio",
            20006: "Dua Libro de l' Lingvo Internacia",
            11307: "El la Biblio",
        }
    },
    "grammar": {
        "description": "Grammar and reference works written IN Esperanto",
        "texts": {
            47855: "Esperanta sintakso - Paul Fruictier",
            52556: "Esperanto-Germana frazlibro - Anton & Borel",
            24525: "Karlo: Facila Legolibro - Edmond Privat",
        }
    },
    "historical": {
        "description": "Historical documents and periodicals",
        "texts": {
            57184: "Dokumentoj de Esperanto - A. Möbusz",
            26359: "Vivo de Zamenhof - Edmond Privat",
            38240: "The Esperantist, Complete",
        }
    },
    "original": {
        "description": "Original works by Esperantists (not translations)",
        "texts": {
            42028: "En Rusujo per Esperanto - A. Rivier",
            25311: "El la vivo de esperantistoj - V. Stankiević",
            42774: "Mondo kaj koro - K. Kalocsay (poetry)",
            48896: "Verdaj fajreroj - Roman Frenkel (poetry)",
            76273: "Por kaj kontraŭ Esperanto - Henri Vallienne",
            23670: "Nuntempaj Rakontoj - G.P. Stamatov",
        }
    }
}


def download_text(text_id: int, title: str, output_dir: Path) -> bool:
    """Download a single text from Project Gutenberg."""
    url = GUTENBERG_BASE.format(text_id, text_id)
    output_file = output_dir / f"{text_id:05d}_{sanitize_filename(title)}.txt"

    if output_file.exists():
        print(f"  ✓ Already downloaded: {title}")
        return True

    try:
        print(f"  Downloading: {title} (ID: {text_id})")
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # Write the content
        output_file.write_text(response.text, encoding='utf-8')
        print(f"  ✓ Saved to: {output_file.name}")

        # Be nice to Project Gutenberg servers
        time.sleep(2)
        return True

    except requests.exceptions.RequestException as e:
        print(f"  ✗ Failed to download {title}: {e}")
        return False


def sanitize_filename(title: str) -> str:
    """Convert title to safe filename."""
    # Remove author names and clean up
    title = title.split(" - ")[0]  # Remove author
    title = title.replace(":", "_")
    title = title.replace("/", "_")
    title = title.replace(" ", "_")
    # Keep only alphanumeric, underscore, and Esperanto letters
    safe_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_ĉĝĥĵŝŭĈĜĤĴŜŬ"
    return "".join(c for c in title if c in safe_chars)


def main():
    """Download all priority Esperanto texts."""
    # Create output directory
    output_dir = Path(__file__).parent.parent / "data" / "gutenberg_esperanto"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("DOWNLOADING STANDARD ESPERANTO TEXTS FROM PROJECT GUTENBERG")
    print("=" * 70)
    print()

    total_texts = sum(len(cat["texts"]) for cat in PRIORITY_TEXTS.values())
    downloaded = 0
    failed = 0
    skipped = 0

    # Download by category
    for category, info in PRIORITY_TEXTS.items():
        print(f"\n{info['description'].upper()}")
        print("-" * 70)

        for text_id, title in info["texts"].items():
            output_file = output_dir / f"{text_id:05d}_{sanitize_filename(title)}.txt"

            if output_file.exists():
                skipped += 1

            success = download_text(text_id, title, output_dir)
            if success:
                downloaded += 1
            else:
                failed += 1

    # Summary
    print("\n" + "=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)
    print(f"Total texts: {total_texts}")
    print(f"Downloaded: {downloaded}")
    print(f"Already had: {skipped}")
    print(f"Failed: {failed}")
    print(f"\nAll texts saved to: {output_dir}")
    print()

    # Create manifest
    manifest_file = output_dir / "MANIFEST.md"
    with open(manifest_file, "w", encoding="utf-8") as f:
        f.write("# Project Gutenberg Esperanto Texts\n\n")
        f.write("Downloaded: " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n\n")

        for category, info in PRIORITY_TEXTS.items():
            f.write(f"## {info['description']}\n\n")
            for text_id, title in info["texts"].items():
                f.write(f"- [{text_id:05d}] {title}\n")
            f.write("\n")

        f.write(f"\n**Total texts: {total_texts}**\n")

    print(f"Created manifest: {manifest_file}")


if __name__ == "__main__":
    main()
