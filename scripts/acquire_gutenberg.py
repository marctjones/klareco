#!/usr/bin/env python3
"""
Download all Esperanto texts from Project Gutenberg.

Only downloads texts that are IN Esperanto (not English texts about Esperanto).
Skips texts that already exist in the target directory.
Supports resuming from checkpoint.

Usage:
    python scripts/download_gutenberg_eo.py [--fresh]
"""

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path

# Esperanto texts to download (ID: title)
# Filtered to only include texts WRITTEN IN Esperanto
ESPERANTO_TEXTS = {
    # Literature - Translations
    17482: "La Aventuroj de Alicio en Mirlando",
    24145: "Doktoro Jekyll kaj Sinjoro Hyde",
    31348: "La Mirinda Sorcxisto de Oz",
    11511: "Robinsono Kruso",
    37279: "Hamleto Regxido de Danujo",
    47913: "Makbeto",
    61103: "Kiel placxas al vi",
    71836: "La ventego de Shakespeare",

    # Drama - Ibsen
    19183: "Hedda Gabler",
    20178: "John Gabriel Borkman",
    19030: "Puphejmo",
    19858: "Konstruestro Solness",
    19803: "Popolmalamiko",
    24355: "Rosmersholm",
    19182: "La Sinjorino el la Maro",
    20162: "La kolonoj de la socio",
    23837: "Eta Eyolf",
    26480: "La Tronpretendantoj",

    # Drama - Goethe, Schiller
    60429: "Hermano kaj Doroteo",
    22592: "Ifigenio en Taurido",
    65958: "La Rabistoj",
    45713: "Aspazio",

    # Fairy Tales
    27915: "Fabeloj de Andersen",
    76310: "Cindrulino",
    74405: "Orientaj fabeloj",
    48900: "Tiel do Rakontoj por malgrandaj infanoj",

    # Noveloj / Short Stories
    18178: "Rakontoj",
    63105: "Tri Noveloj",
    20943: "Mark Twain Tri Ceteraj Noveloj",
    17945: "Mark Twain Tri Noveloj",
    21195: "Tri Noveloj (Hawthorne)",
    21194: "Tri Noveloj (Irving)",
    20931: "Tri Noveloj de Bret Harte",
    66037: "Elektitaj noveloj",
    62511: "Ses noveloj el Rakontoj de mistero kaj imago",

    # Poe
    17425: "La Falo de Usxero-Domo",
    18326: "La Murdoj de Kadavrejo-Strato",
    61190: "La Puto kaj la Pendolo",

    # Irving
    19293: "La Legendo de Dorm-Valeto",

    # Adventure / Fiction
    20802: "Cxe la koro de la tero",
    55954: "Stranga heredajxo",
    61860: "Marta",
    62118: "Legendoj",
    55302: "Princo Serebrjanij",
    32480: "La Alaska stafeto kaj Kaptitoj de la glacirokoj",
    26099: "Aventuroj de Antonio",
    51069: "La Granda Admiralo",
    65835: "La Regxo de la Montoj",
    17665: "Mia Kontrabandulo",
    61581: "La kialo de la vivo",
    68879: "La firmao de la kato kiu pilkludas",
    23093: "Princo Vanc",
    74344: "Perdita kaj retrovita",
    35981: "La kolomba premio",
    23586: "La liturgio de l foiro",
    27593: "La Majstro kaj Martinelli",
    63926: "La mirinda historio de Petro Schlemihl",
    35917: "Rikke-tikke-tak",
    24501: "La Batalo de l Vivo",
    47259: "La Vendreda Klubo",
    22070: "La Karavano",
    24763: "La Kantistino",
    47249: "La jeso de knabinoj",
    69123: "Saltego trans jarmiloj",
    25539: "Vojagxo interne de mia cxambro",
    25386: "La lasta Usonano",
    22901: "Taglibro de Vilagx-pedelo",
    45612: "Sub la Meznokta Suno",

    # Poetry
    30536: "Rolandkanto",
    52111: "Ama Stelaro",
    24292: "La Montarino",
    64579: "Idoj de Orfeo",
    32035: "Lauroj",

    # Drama - Other
    63064: "Salome",
    27170: "Jeppe sur la Monto",
    23774: "La Asocio de la Junuloj",
    28971: "Botistoj",
    35743: "Tri unuaktaj komedioj",
    52876: "Advokato Patelin",
    75983: "La gefratoj",

    # Fables
    51690: "Elektitaj fabloj de La Fontaine",

    # Historical / Biography
    21951: "Jan Amos Komensky",
    26959: "La Lastaj Tagoj de Zamenhof",
    62394: "Ascendo al Monto-Blanka en 1787",
    48033: "Pagxoj el la Flandra Literaturo",

    # Religion
    24057: "La Libro Ruth",

    # Other
    25964: "Batalo pri la Domo Heikkila",
    56351: "Parizina",
    68874: "El la Camera obscura",
    68878: "Deklaracio",
    61579: "Al mia fratineto",
    28971: "Inaugurxa parolado de Barack Obama",

    # Already have but verify
    # 8224: Fundamenta Krestomatio - HAVE
    # 11307: El la Biblio - HAVE
    # 20006: Dua Libro - HAVE
    # 23670: Nuntempaj Rakontoj - HAVE
    # 24525: Karlo - HAVE
    # 25311: El la vivo de esperantistoj - HAVE
    # 26359: Vivo de Zamenhof - HAVE
    # 38240: The Esperantist Complete - HAVE
    # 42028: En Rusujo per Esperanto - HAVE
    # 42774: Mondo kaj koro - HAVE
    # 47855: Esperanta sintakso - HAVE
    # 48896: Verdaj fajreroj - HAVE
    # 52556: Esperanto-Germana frazlibro - HAVE
    # 57184: Dokumentoj de Esperanto - HAVE
    # 76273: Por kaj kontraux Esperanto - HAVE
}

# Skip these - English texts ABOUT Esperanto, not IN Esperanto
SKIP_IDS = {
    7787,   # A Complete Grammar of Esperanto (English)
    8177,   # The Esperanto Teacher (English)
    16967,  # English-Esperanto Dictionary (English)
    24575,  # Czech Esperanto textbook
}


def sanitize_filename(title: str) -> str:
    """Convert title to safe filename."""
    # Replace Esperanto special chars with ASCII equivalents
    replacements = {
        'ĉ': 'cx', 'Ĉ': 'Cx',
        'ĝ': 'gx', 'Ĝ': 'Gx',
        'ĥ': 'hx', 'Ĥ': 'Hx',
        'ĵ': 'jx', 'Ĵ': 'Jx',
        'ŝ': 'sx', 'Ŝ': 'Sx',
        'ŭ': 'ux', 'Ŭ': 'Ux',
        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
    }
    for old, new in replacements.items():
        title = title.replace(old, new)

    # Remove non-alphanumeric chars except spaces and underscores
    title = re.sub(r'[^\w\s-]', '', title)
    # Replace spaces with underscores
    title = re.sub(r'\s+', '_', title)
    return title


def download_text(ebook_id: int, output_dir: Path) -> tuple[bool, str]:
    """
    Download a single ebook from Project Gutenberg.
    Returns (success, message).
    """
    # Try different URL patterns
    urls = [
        f"https://www.gutenberg.org/cache/epub/{ebook_id}/pg{ebook_id}.txt",
        f"https://www.gutenberg.org/files/{ebook_id}/{ebook_id}-0.txt",
        f"https://www.gutenberg.org/files/{ebook_id}/{ebook_id}.txt",
    ]

    for url in urls:
        try:
            req = urllib.request.Request(
                url,
                headers={'User-Agent': 'Klareco/1.0 (Esperanto AI Research)'}
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                content = response.read()

                # Try to decode as UTF-8, fall back to latin-1
                try:
                    text = content.decode('utf-8')
                except UnicodeDecodeError:
                    text = content.decode('latin-1')

                # Get title for filename
                title = ESPERANTO_TEXTS.get(ebook_id, f"ebook_{ebook_id}")
                filename = f"{ebook_id}_{sanitize_filename(title)}.txt"

                output_path = output_dir / filename
                output_path.write_text(text, encoding='utf-8')

                return True, f"Downloaded: {filename}"

        except urllib.error.HTTPError as e:
            if e.code == 404:
                continue  # Try next URL
            return False, f"HTTP error {e.code}: {url}"
        except Exception as e:
            continue  # Try next URL

    return False, f"All URLs failed for ebook {ebook_id}"


def load_checkpoint(checkpoint_path: Path) -> set:
    """Load set of already downloaded ebook IDs."""
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            data = json.load(f)
            return set(data.get('downloaded', []))
    return set()


def save_checkpoint(checkpoint_path: Path, downloaded: set):
    """Save checkpoint with downloaded IDs."""
    temp_path = checkpoint_path.with_suffix('.tmp')
    with open(temp_path, 'w') as f:
        json.dump({'downloaded': list(downloaded)}, f)
    temp_path.rename(checkpoint_path)


def main():
    parser = argparse.ArgumentParser(description='Download Esperanto texts from Project Gutenberg')
    parser.add_argument('--fresh', action='store_true', help='Start fresh, ignore checkpoint')
    parser.add_argument('--output', type=Path, default=Path('data/raw/eo/gutenberg'),
                        help='Output directory')
    args = parser.parse_args()

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / '_download_checkpoint.json'

    # Load checkpoint
    if args.fresh:
        downloaded = set()
        print("Starting fresh download...")
    else:
        downloaded = load_checkpoint(checkpoint_path)
        if downloaded:
            print(f"Resuming from checkpoint: {len(downloaded)} already downloaded")

    # Check what we already have by filename
    existing_ids = set()
    for f in output_dir.glob('*.txt'):
        match = re.match(r'^(\d+)_', f.name)
        if match:
            existing_ids.add(int(match.group(1)))

    # Combine downloaded and existing
    already_have = downloaded | existing_ids

    # Filter to what we need to download
    to_download = {
        eid: title
        for eid, title in ESPERANTO_TEXTS.items()
        if eid not in already_have and eid not in SKIP_IDS
    }

    print(f"\nTexts to download: {len(to_download)}")
    print(f"Already have: {len(already_have)}")
    print(f"Skipping (English): {len(SKIP_IDS)}")
    print()

    if not to_download:
        print("Nothing to download!")
        return

    success_count = 0
    fail_count = 0

    for i, (ebook_id, title) in enumerate(to_download.items(), 1):
        print(f"[{i}/{len(to_download)}] {ebook_id}: {title}...", end=" ", flush=True)

        success, message = download_text(ebook_id, output_dir)

        if success:
            print("OK")
            downloaded.add(ebook_id)
            success_count += 1
        else:
            print(f"FAILED - {message}")
            fail_count += 1

        # Save checkpoint every 5 downloads
        if i % 5 == 0:
            save_checkpoint(checkpoint_path, downloaded)

        # Be nice to Gutenberg servers
        time.sleep(1)

    # Final checkpoint save
    save_checkpoint(checkpoint_path, downloaded)

    print(f"\n{'='*50}")
    print(f"Download complete!")
    print(f"  Success: {success_count}")
    print(f"  Failed:  {fail_count}")
    print(f"  Total in {output_dir}: {len(list(output_dir.glob('*.txt')))}")


if __name__ == '__main__':
    main()
