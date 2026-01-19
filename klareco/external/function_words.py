"""Function word filtering for external semantic category import.

Function words (pronouns, correlatives, particles) should NOT receive semantic
categorizations from external sources. They are handled deterministically by
the parser's grammar layer.

This module provides filtering to prevent wasting API calls on function words
during ConceptNet/Wikidata imports.
"""

from typing import Set


# Esperanto function words that should be excluded from semantic categorization
FUNCTION_WORDS: Set[str] = {
    # Personal pronouns
    'mi',      # I
    'vi',      # you
    'li',      # he
    'ŝi',      # she
    'ĝi',      # it
    'ni',      # we
    'ili',     # they
    'si',      # oneself
    'oni',     # one (impersonal)

    # Correlatives: demonstrative/interrogative
    'tio',     # that (thing)
    'tiu',     # that (one)
    'kio',     # what (thing)
    'kiu',     # who/which (one)
    'ĉio',     # everything
    'io',      # something
    'nenio',   # nothing
    'iu',      # someone
    'neniu',   # no one

    # Correlatives: place
    'tie',     # there
    'kie',     # where
    'ĉie',     # everywhere
    'ie',      # somewhere
    'nenie',   # nowhere
    'tien',    # to there
    'kien',    # to where
    'ĉien',    # to everywhere
    'ien',     # to somewhere
    'nenien',  # to nowhere

    # Correlatives: time
    'tiam',    # then
    'kiam',    # when
    'ĉiam',    # always
    'iam',     # sometime
    'neniam',  # never

    # Correlatives: manner
    'tiel',    # thus/so
    'kiel',    # how
    'ĉiel',    # in every way
    'iel',     # somehow
    'neniel',  # in no way

    # Correlatives: quantity
    'tiom',    # that much
    'kiom',    # how much
    'ĉiom',    # all of it
    'iom',     # some amount
    'neniom',  # none

    # Correlatives: reason
    'tial',    # therefore
    'kial',    # why
    'ĉial',    # for every reason
    'ial',     # for some reason
    'nenial',  # for no reason

    # Correlatives: possession
    'ties',    # that one's
    'kies',    # whose
    'ĉies',    # everyone's
    'ies',     # someone's
    'nenies',  # no one's

    # Articles and particles
    'la',      # the
    'je',      # at/for (general preposition)

    # Conjunctions (handled grammatically)
    'kaj',     # and
    'aŭ',      # or
    'sed',     # but
    'ĉar',     # because
    'ke',      # that (conjunction)
    'se',      # if
    'ĉu',      # whether/question particle

    # Common grammatical words
    'ne',      # not/no
    'jes',     # yes
    'ja',      # indeed
    'do',      # so/therefore
    'nu',      # well
    'eĉ',      # even
    'ankaŭ',   # also
    'nur',     # only
    'des',     # the (in "ju...des")
    'ju',      # the (in "ju...des")
}


def is_function_word(word: str) -> bool:
    """Check if a word is a function word that should be excluded.

    Args:
        word: Esperanto root word

    Returns:
        True if the word is a function word (should be excluded)
    """
    return word.lower() in FUNCTION_WORDS


def filter_function_words(words: list) -> list:
    """Filter out function words from a list of words.

    Args:
        words: List of word dictionaries (must have 'root' key)

    Returns:
        Filtered list with function words removed
    """
    return [w for w in words if not is_function_word(w.get('root', ''))]


def get_function_word_count(words: list) -> int:
    """Count how many function words are in a list.

    Args:
        words: List of word dictionaries (must have 'root' key)

    Returns:
        Number of function words found
    """
    return sum(1 for w in words if is_function_word(w.get('root', '')))
