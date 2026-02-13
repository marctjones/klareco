"""
Rigorous Entity Type Taxonomy for Klareco.

Based on academic standards (ACE, OntoNotes, Li & Roth, Aristotle) aligned with
Esperanto linguistic structure (correlatives, affixes, PMEG).

Three-tier hierarchy:
- Tier 1: Aristotelian (6 categories) - 100% deterministic from grammar
- Tier 2: NER-compatible (18 categories) - ~70% deterministic from affixes
- Tier 3: Fine-grained (30+ categories) - ~30% deterministic, rest learned

References:
- ACE (Automatic Content Extraction): PER, ORG, LOC, FAC, GPE, VEH, WEA
- OntoNotes 5.0: 18 entity types including DATE, TIME, QUANTITY
- Li & Roth (2002): Question answering taxonomy with 50 fine-grained classes
- Aristotle's Categories: Substance, Quality, Quantity, Relation, Place, Time
- Esperanto correlatives: 9 semantic suffixes (-u, -o, -a, -e, -am, -el, -al, -om, -es)
"""

from enum import Enum
from typing import Dict, Set


class TopLevelCategory(Enum):
    """
    Tier 1: Aristotelian top-level categories (6 categories).

    100% deterministic from vortspeco alone.
    Based on Aristotle's Categories (2000-year philosophical foundation).
    """
    ENTITY = "entity"          # Substance (what exists independently)
    ATTRIBUTE = "attribute"    # Quality (characteristics)
    QUANTITY = "quantity"      # Quantity/Number
    RELATION = "relation"      # Relation (comparative, possessive)
    SPACETIME = "spacetime"    # Place + Time (Aristotle's where/when)
    ACTION = "action"          # Doing/Being-affected (events, processes)


class EntityType(Enum):
    """
    Tier 2: NER-compatible mid-level categories (18 categories).

    ~70% deterministic from correlatives + affixes.
    Aligned with ACE, OntoNotes, and Esperanto correlative table.
    """
    # ENTITY branch
    PERSON = "person"                # ACE: PER | Correlative: -u
    ORGANIZATION = "organization"    # ACE: ORG
    LOCATION = "location"            # ACE: LOC | Correlative: -e
    FACILITY = "facility"            # ACE: FAC (buildings, infrastructure)
    GPE = "geo_political_entity"     # ACE: GPE (countries, cities as political entities)
    THING = "thing"                  # Correlative: -o (physical objects)
    CONCEPT = "concept"              # Abstract entities (ideas, theories)
    EVENT = "event"                  # OntoNotes: EVENT (conferences, wars)

    # ATTRIBUTE branch
    QUALITY = "quality"              # Correlative: -a (properties)
    MANNER = "manner"                # Correlative: -el (ways of doing)
    REASON = "reason"                # Correlative: -al (causes)

    # QUANTITY branch
    NUMBER = "number"                # OntoNotes: CARDINAL
    QUANTITY = "quantity"            # Correlative: -om, OntoNotes: QUANTITY
    PERCENT = "percent"              # OntoNotes: PERCENT
    MONEY = "money"                  # OntoNotes: MONEY
    ORDINAL = "ordinal"              # OntoNotes: ORDINAL

    # SPACETIME branch
    TIME_POINT = "time_point"        # Correlative: -am, OntoNotes: DATE, TIME
    TIME_DURATION = "time_duration"  # Periods, spans
    LOCATION_POINT = "location_point"    # Specific places
    LOCATION_REGION = "location_region"  # Areas, zones

    # RELATION branch
    POSSESSIVE = "possessive"        # Correlative: -es (ownership)


class PersonType(Enum):
    """
    Tier 3: Fine-grained person types.

    Based on Li & Roth HUMAN subtypes + Esperanto affixes.
    ~50% deterministic from affixes (marked DETERMINISTIC).
    """
    PERSON_NAME = "person_name"           # Proper names: Zamenhof, Schmidt
    PERSON_TITLE = "person_title"         # Doktoro, Profesoro
    PERSON_PROFESSION = "person_profession"  # Has -ist: instruisto (DETERMINISTIC)
    PERSON_ROLE = "person_role"           # Has -ul, -ant, -int: kreinto (DETERMINISTIC)
    PERSON_GENERIC = "person_generic"     # Generic: homo, persono
    PERSON_GROUP = "person_group"         # Has -ar: homaro (DETERMINISTIC)
    PERSON_PRONOUN = "person_pronoun"     # li, ŝi, ili (referential)


class LocationType(Enum):
    """
    Tier 3: Fine-grained location types.

    Based on Li & Roth LOCATION subtypes + Esperanto affixes.
    ~40% deterministic from affixes (marked DETERMINISTIC).
    """
    PLACE_NAME = "place_name"           # Proper: Berlino, Parizo
    PLACE_COUNTRY = "place_country"     # Has -land/-io: Anglaland, Germanio
    PLACE_CITY = "place_city"           # urbo
    PLACE_INSTITUTION = "place_institution"  # Has -ej: lernejo (DETERMINISTIC)
    PLACE_GEOGRAPHIC = "place_geographic"    # monto, rivero, lago
    PLACE_GENERIC = "place_generic"     # loko, ejo


class TimeType(Enum):
    """
    Tier 3: Fine-grained time types.

    Based on OntoNotes + Correlative -am.
    ~20% deterministic (mostly from correlatives).
    """
    TIME_ABSOLUTE = "time_absolute"     # 1887, januaro 15-a
    TIME_RELATIVE = "time_relative"     # hieraŭ, morgaŭ, antaŭe
    TIME_DURATION = "time_duration"     # jaro, monato, horo
    TIME_FREQUENCY = "time_frequency"   # ĉiutage, ofte, foje


class ThingType(Enum):
    """
    Tier 3: Thing subtypes.

    Aligned with Correlative -o + Esperanto affixes.
    ~60% deterministic from affixes (marked DETERMINISTIC).
    """
    THING_CONCRETE = "thing_concrete"   # Has -aĵ: objekto (DETERMINISTIC)
    THING_TOOL = "thing_tool"           # Has -il: tranĉilo (DETERMINISTIC)
    THING_COLLECTION = "thing_collection"  # Has -ar: libraro (DETERMINISTIC)
    THING_ABSTRACT = "thing_abstract"   # ideo, teorio


# Mapping tables for tier relationships

TIER2_TO_TIER1: Dict[EntityType, TopLevelCategory] = {
    # ENTITY branch
    EntityType.PERSON: TopLevelCategory.ENTITY,
    EntityType.ORGANIZATION: TopLevelCategory.ENTITY,
    EntityType.LOCATION: TopLevelCategory.SPACETIME,
    EntityType.FACILITY: TopLevelCategory.ENTITY,
    EntityType.GPE: TopLevelCategory.ENTITY,
    EntityType.THING: TopLevelCategory.ENTITY,
    EntityType.CONCEPT: TopLevelCategory.ENTITY,
    EntityType.EVENT: TopLevelCategory.ACTION,

    # ATTRIBUTE branch
    EntityType.QUALITY: TopLevelCategory.ATTRIBUTE,
    EntityType.MANNER: TopLevelCategory.ATTRIBUTE,
    EntityType.REASON: TopLevelCategory.RELATION,

    # QUANTITY branch
    EntityType.NUMBER: TopLevelCategory.QUANTITY,
    EntityType.QUANTITY: TopLevelCategory.QUANTITY,
    EntityType.PERCENT: TopLevelCategory.QUANTITY,
    EntityType.MONEY: TopLevelCategory.QUANTITY,
    EntityType.ORDINAL: TopLevelCategory.QUANTITY,

    # SPACETIME branch
    EntityType.TIME_POINT: TopLevelCategory.SPACETIME,
    EntityType.TIME_DURATION: TopLevelCategory.SPACETIME,
    EntityType.LOCATION_POINT: TopLevelCategory.SPACETIME,
    EntityType.LOCATION_REGION: TopLevelCategory.SPACETIME,

    # RELATION branch
    EntityType.POSSESSIVE: TopLevelCategory.RELATION,
}


TIER3_TO_TIER2: Dict[Enum, EntityType] = {
    # Person types → PERSON
    PersonType.PERSON_NAME: EntityType.PERSON,
    PersonType.PERSON_TITLE: EntityType.PERSON,
    PersonType.PERSON_PROFESSION: EntityType.PERSON,
    PersonType.PERSON_ROLE: EntityType.PERSON,
    PersonType.PERSON_GENERIC: EntityType.PERSON,
    PersonType.PERSON_GROUP: EntityType.PERSON,
    PersonType.PERSON_PRONOUN: EntityType.PERSON,

    # Location types → LOCATION
    LocationType.PLACE_NAME: EntityType.LOCATION,
    LocationType.PLACE_COUNTRY: EntityType.GPE,
    LocationType.PLACE_CITY: EntityType.GPE,
    LocationType.PLACE_INSTITUTION: EntityType.FACILITY,
    LocationType.PLACE_GEOGRAPHIC: EntityType.LOCATION,
    LocationType.PLACE_GENERIC: EntityType.LOCATION,

    # Time types → TIME
    TimeType.TIME_ABSOLUTE: EntityType.TIME_POINT,
    TimeType.TIME_RELATIVE: EntityType.TIME_POINT,
    TimeType.TIME_DURATION: EntityType.TIME_DURATION,
    TimeType.TIME_FREQUENCY: EntityType.TIME_POINT,

    # Thing types → THING
    ThingType.THING_CONCRETE: EntityType.THING,
    ThingType.THING_TOOL: EntityType.THING,
    ThingType.THING_COLLECTION: EntityType.THING,
    ThingType.THING_ABSTRACT: EntityType.CONCEPT,
}


def get_tier1_category(tier2_type: EntityType) -> TopLevelCategory:
    """Get Tier 1 category from Tier 2 type."""
    return TIER2_TO_TIER1.get(tier2_type, TopLevelCategory.ENTITY)


def get_tier2_type(tier3_type: Enum) -> EntityType:
    """Get Tier 2 type from Tier 3 type."""
    return TIER3_TO_TIER2.get(tier3_type, EntityType.THING)
