#!/usr/bin/env python3
"""
Generate minimal pairs for Stage 2 grammatical model training.

Creates training data for Stage 2 by generating sentence pairs that differ only
in a specific grammatical feature. Uses deterministic grammar rules.

Features generated:
- Negation pairs: Add/remove "ne" (polarity flip)
- Tense pairs: Change verb tense (as/is/os)
- Mood pairs: Indicative vs conditional (as/us)
- Sentence type pairs: Statement vs question (add "Ĉu")

Usage:
    python scripts/training/generate_stage2_pairs.py \
        --input data/corpus/authoritative_corpus.jsonl \
        --output data/training/stage2_pairs.jsonl \
        --limit 5000
"""

import argparse
import json
import logging
import re
import random
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Iterator, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class MinimalPair:
    """A minimal pair for grammatical training."""
    sentence1: str
    sentence2: str
    feature_type: str  # negation, tense, mood, sentence_type
    feature_detail: str  # e.g., "present_to_past", "add_negation"
    similarity: float  # Target similarity (used for training)
    source_id: Optional[str] = None


# Verb tense patterns
TENSE_ENDINGS = {
    'present': 'as',
    'past': 'is',
    'future': 'os',
    'conditional': 'us',
    'imperative': 'u',
    'infinitive': 'i',
}

# Verb tense replacement patterns
VERB_PATTERN = re.compile(r'\b([a-zĉĝĥĵŝŭ]+)(as|is|os|us|u|i)\b', re.IGNORECASE)


def find_verbs(text: str) -> List[Tuple[str, str, int, int]]:
    """Find verbs in text. Returns list of (full_word, root, start, end)."""
    verbs = []
    for match in VERB_PATTERN.finditer(text):
        root = match.group(1)
        ending = match.group(2)
        # Skip if too short (likely not a verb root)
        if len(root) < 2:
            continue
        verbs.append((match.group(0), root, match.start(), match.end()))
    return verbs


def change_verb_tense(text: str, from_tense: str, to_tense: str) -> Optional[str]:
    """Change verb tense in text. Returns None if no change possible."""
    from_ending = TENSE_ENDINGS.get(from_tense)
    to_ending = TENSE_ENDINGS.get(to_tense)
    if not from_ending or not to_ending:
        return None

    # Find verbs with the from_tense
    verbs = find_verbs(text)
    matching_verbs = [(v, r, s, e) for v, r, s, e in verbs if v.endswith(from_ending)]

    if not matching_verbs:
        return None

    # Change all matching verbs
    result = text
    offset = 0
    for verb, root, start, end in matching_verbs:
        new_verb = root + to_ending
        # Preserve case
        if verb[0].isupper():
            new_verb = new_verb[0].upper() + new_verb[1:]
        result = result[:start + offset] + new_verb + result[end + offset:]
        offset += len(new_verb) - len(verb)

    return result if result != text else None


def add_negation(text: str) -> Optional[str]:
    """Add negation to sentence. Returns None if already negated."""
    # Check if already negated
    if re.search(r'\bne\b', text, re.IGNORECASE):
        return None

    # Find the first verb
    verbs = find_verbs(text)
    if not verbs:
        return None

    # Add "ne" before the first verb
    first_verb, _, start, _ = verbs[0]

    # Handle case of verb at start of sentence
    if start == 0:
        return "Ne " + text[0].lower() + text[1:]

    # Insert "ne" before the verb
    return text[:start] + "ne " + text[start:]


def remove_negation(text: str) -> Optional[str]:
    """Remove negation from sentence. Returns None if not negated."""
    # Find "ne" before a verb
    match = re.search(r'\bne\s+', text, re.IGNORECASE)
    if not match:
        return None

    # Remove "ne "
    result = text[:match.start()] + text[match.end():]

    # Capitalize if needed
    if match.start() == 0 and result:
        result = result[0].upper() + result[1:]

    return result if result != text else None


def make_question(text: str) -> Optional[str]:
    """Convert statement to yes/no question. Returns None if already a question."""
    text = text.strip()

    # Check if already a question
    if text.endswith('?') or text.lower().startswith('ĉu ') or text.lower().startswith('kiu '):
        return None

    # Remove trailing punctuation
    if text.endswith('.'):
        text = text[:-1]

    # Add "Ĉu" at the beginning
    if text[0].isupper():
        text = text[0].lower() + text[1:]

    return "Ĉu " + text + "?"


def make_statement(text: str) -> Optional[str]:
    """Convert question to statement. Returns None if not a simple yes/no question."""
    text = text.strip()

    # Check if it's a ĉu-question
    match = re.match(r'^Ĉu\s+', text, re.IGNORECASE)
    if not match:
        return None

    # Remove "Ĉu " and question mark
    result = text[match.end():]
    if result.endswith('?'):
        result = result[:-1] + '.'

    # Capitalize
    if result:
        result = result[0].upper() + result[1:]

    return result


def has_verb(text: str) -> bool:
    """Check if text contains a verb."""
    return bool(find_verbs(text))


def get_verb_tense(text: str) -> Optional[str]:
    """Get the tense of the first verb in text."""
    verbs = find_verbs(text)
    if not verbs:
        return None

    first_verb = verbs[0][0].lower()
    for tense, ending in TENSE_ENDINGS.items():
        if first_verb.endswith(ending):
            return tense
    return None


def is_valid_sentence(text: str) -> bool:
    """Check if sentence is suitable for pair generation."""
    # Skip very short sentences
    if len(text) < 10:
        return False

    # Skip sentences with too many unknown characters
    eo_chars = set('abcdefghijklmnoprstuvzĉĝĥĵŝŭABCDEFGHIJKLMNOPRSTUVZĈĜĤĴŜŬ')
    valid_chars = sum(1 for c in text if c.isalpha() and c.lower() in eo_chars)
    alpha_chars = sum(1 for c in text if c.isalpha())
    if alpha_chars > 0 and valid_chars / alpha_chars < 0.9:
        return False

    return True


def generate_pairs_from_sentence(
    text: str,
    source_id: Optional[str] = None
) -> Iterator[MinimalPair]:
    """Generate all possible minimal pairs from a sentence."""

    if not is_valid_sentence(text):
        return

    # Negation pairs
    negated = add_negation(text)
    if negated:
        yield MinimalPair(
            sentence1=text,
            sentence2=negated,
            feature_type="negation",
            feature_detail="add_negation",
            similarity=-0.8,  # Opposite meaning
            source_id=source_id
        )

    unnegated = remove_negation(text)
    if unnegated:
        yield MinimalPair(
            sentence1=text,
            sentence2=unnegated,
            feature_type="negation",
            feature_detail="remove_negation",
            similarity=-0.8,  # Opposite meaning
            source_id=source_id
        )

    # Tense pairs
    tense = get_verb_tense(text)
    if tense in ('present', 'past', 'future'):
        # Present/Past pairs
        if tense == 'present':
            past = change_verb_tense(text, 'present', 'past')
            if past:
                yield MinimalPair(
                    sentence1=text,
                    sentence2=past,
                    feature_type="tense",
                    feature_detail="present_to_past",
                    similarity=0.7,  # Same action, different time
                    source_id=source_id
                )
            future = change_verb_tense(text, 'present', 'future')
            if future:
                yield MinimalPair(
                    sentence1=text,
                    sentence2=future,
                    feature_type="tense",
                    feature_detail="present_to_future",
                    similarity=0.5,  # Same action, different time
                    source_id=source_id
                )
        elif tense == 'past':
            present = change_verb_tense(text, 'past', 'present')
            if present:
                yield MinimalPair(
                    sentence1=text,
                    sentence2=present,
                    feature_type="tense",
                    feature_detail="past_to_present",
                    similarity=0.7,
                    source_id=source_id
                )
            future = change_verb_tense(text, 'past', 'future')
            if future:
                yield MinimalPair(
                    sentence1=text,
                    sentence2=future,
                    feature_type="tense",
                    feature_detail="past_to_future",
                    similarity=0.4,  # Larger temporal gap
                    source_id=source_id
                )
        elif tense == 'future':
            present = change_verb_tense(text, 'future', 'present')
            if present:
                yield MinimalPair(
                    sentence1=text,
                    sentence2=present,
                    feature_type="tense",
                    feature_detail="future_to_present",
                    similarity=0.5,
                    source_id=source_id
                )

    # Mood pairs (indicative vs conditional)
    if tense in ('present', 'past', 'future'):
        conditional = change_verb_tense(text, tense, 'conditional')
        if conditional:
            yield MinimalPair(
                sentence1=text,
                sentence2=conditional,
                feature_type="mood",
                feature_detail=f"{tense}_to_conditional",
                similarity=0.3,  # Factual vs hypothetical
                source_id=source_id
            )

    # Sentence type pairs
    question = make_question(text)
    if question:
        yield MinimalPair(
            sentence1=text,
            sentence2=question,
            feature_type="sentence_type",
            feature_detail="statement_to_question",
            similarity=0.5,  # Same content, different illocution
            source_id=source_id
        )

    statement = make_statement(text)
    if statement:
        yield MinimalPair(
            sentence1=text,
            sentence2=statement,
            feature_type="sentence_type",
            feature_detail="question_to_statement",
            similarity=0.5,
            source_id=source_id
        )


def stream_corpus(input_path: Path) -> Iterator[Dict[str, Any]]:
    """Stream entries from corpus file."""
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def generate_all_pairs(
    input_path: Path,
    output_path: Path,
    limit_per_type: int = 5000,
    random_seed: int = 42
) -> Dict[str, int]:
    """Generate minimal pairs from corpus and write to output."""

    random.seed(random_seed)

    # Collect pairs by type
    pairs_by_type: Dict[str, List[MinimalPair]] = {
        'negation': [],
        'tense': [],
        'mood': [],
        'sentence_type': [],
    }

    logger.info(f"Reading from {input_path}")
    count = 0

    for entry in stream_corpus(input_path):
        text = entry.get('text', '')
        source = entry.get('source', {})
        source_id = source.get('citation') if isinstance(source, dict) else None

        for pair in generate_pairs_from_sentence(text, source_id):
            pairs_by_type[pair.feature_type].append(pair)

        count += 1
        if count % 10000 == 0:
            logger.info(f"  Processed {count} sentences...")
            # Check if we have enough
            all_full = all(len(pairs) >= limit_per_type for pairs in pairs_by_type.values())
            if all_full:
                break

    # Sample and shuffle
    logger.info("Sampling and shuffling pairs...")
    all_pairs = []
    for feature_type, pairs in pairs_by_type.items():
        logger.info(f"  {feature_type}: {len(pairs)} pairs generated")
        if len(pairs) > limit_per_type:
            sampled = random.sample(pairs, limit_per_type)
        else:
            sampled = pairs
        all_pairs.extend(sampled)

    random.shuffle(all_pairs)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing {len(all_pairs)} pairs to {output_path}")

    with open(output_path, 'w', encoding='utf-8') as f:
        for pair in all_pairs:
            f.write(json.dumps(asdict(pair), ensure_ascii=False) + '\n')

    # Return statistics
    stats = {ft: len([p for p in all_pairs if p.feature_type == ft]) for ft in pairs_by_type}
    stats['total'] = len(all_pairs)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Generate minimal pairs for Stage 2 grammatical model'
    )
    parser.add_argument(
        '--input', type=Path,
        default=Path('data/corpus/authoritative_corpus.jsonl'),
        help='Input corpus file'
    )
    parser.add_argument(
        '--output', type=Path,
        default=Path('data/training/stage2_pairs.jsonl'),
        help='Output pairs file'
    )
    parser.add_argument(
        '--limit', type=int, default=5000,
        help='Maximum pairs per feature type'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        return 1

    stats = generate_all_pairs(
        input_path=args.input,
        output_path=args.output,
        limit_per_type=args.limit,
        random_seed=args.seed
    )

    logger.info("=" * 60)
    logger.info("GENERATION COMPLETE")
    logger.info("=" * 60)
    for feature_type, count in sorted(stats.items()):
        logger.info(f"  {feature_type}: {count}")

    return 0


if __name__ == '__main__':
    exit(main())
