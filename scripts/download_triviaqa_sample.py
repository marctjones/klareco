#!/usr/bin/env python3
"""
Download and Prepare TriviaQA Sample

Downloads TriviaQA dataset and extracts sample questions for translation.

Usage:
    python scripts/download_triviaqa_sample.py --output triviaqa_sample_1000.jsonl --limit 1000
"""

import json
import argparse
import logging
import tarfile
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

TRIVIAQA_URL = "https://nlp.cs.washington.edu/triviaqa/data/triviaqa-unfiltered.tar.gz"


def download_triviaqa(download_dir: Path):
    """Download TriviaQA dataset."""
    download_dir.mkdir(parents=True, exist_ok=True)

    tar_path = download_dir / "triviaqa-unfiltered.tar.gz"

    if tar_path.exists():
        logger.info(f"TriviaQA already downloaded at {tar_path}")
        return tar_path

    logger.info(f"Downloading TriviaQA from {TRIVIAQA_URL}...")
    logger.info("This may take a few minutes (~600MB)...")

    urllib.request.urlretrieve(TRIVIAQA_URL, tar_path)

    logger.info(f"Downloaded to {tar_path}")
    return tar_path


def extract_triviaqa(tar_path: Path, extract_dir: Path):
    """Extract TriviaQA tarball."""
    logger.info(f"Extracting {tar_path}...")

    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(extract_dir)

    logger.info(f"Extracted to {extract_dir}")


def prepare_sample(triviaqa_dir: Path, output_file: Path, limit: int = 1000):
    """Prepare sample of Q&A pairs from TriviaQA."""

    # TriviaQA structure: triviaqa-unfiltered/unfiltered-web-train.json
    train_file = triviaqa_dir / "unfiltered-web-train.json"

    if not train_file.exists():
        raise FileNotFoundError(f"TriviaQA train file not found at {train_file}")

    logger.info(f"Loading TriviaQA from {train_file}...")

    with open(train_file) as f:
        data = json.load(f)

    # Extract Q&A pairs
    qa_pairs = []
    for item in data['Data'][:limit]:
        qa_pairs.append({
            'question': item['Question'],
            'answer': item['Answer']['Aliases']  # List of acceptable answers
        })

    # Save as JSONL
    logger.info(f"Saving {len(qa_pairs)} Q&A pairs to {output_file}...")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        for qa in qa_pairs:
            f.write(json.dumps(qa, ensure_ascii=False) + '\n')

    logger.info(f"Saved {len(qa_pairs)} Q&A pairs")

    # Print sample
    logger.info("\nSample questions:")
    for i, qa in enumerate(qa_pairs[:5]):
        logger.info(f"  {i+1}. Q: {qa['question']}")
        logger.info(f"     A: {qa['answer'][0]}")


def main():
    parser = argparse.ArgumentParser(description='Download and prepare TriviaQA sample')
    parser.add_argument('--output', type=Path,
                       default=Path('data/external/triviaqa_sample_1000.jsonl'),
                       help='Output JSONL file')
    parser.add_argument('--limit', type=int, default=1000,
                       help='Number of questions to extract')
    parser.add_argument('--download-dir', type=Path,
                       default=Path('data/external'),
                       help='Directory to download TriviaQA')
    parser.add_argument('--skip-download', action='store_true',
                       help='Skip download if already exists')

    args = parser.parse_args()

    # Download
    if not args.skip_download:
        tar_path = download_triviaqa(args.download_dir)

        # Extract
        extract_triviaqa(tar_path, args.download_dir)

    # Prepare sample
    triviaqa_dir = args.download_dir / "triviaqa-unfiltered"
    prepare_sample(triviaqa_dir, args.output, args.limit)


if __name__ == '__main__':
    main()
