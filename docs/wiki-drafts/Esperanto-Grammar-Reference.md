# Esperanto Grammar Reference

**For**: Klareco Wiki
**Audience**: Developers, linguists, anyone learning about Esperanto's regular grammar
**Purpose**: Complete reference for Esperanto's 16 rules and how they enable deterministic parsing

---

## Table of Contents

1. [Why Esperanto for AI](#why-esperanto-for-ai)
2. [The 16 Fundamental Rules](#the-16-fundamental-rules)
3. [Morphology: Root + Affixes](#morphology-root--affixes)
4. [Word Formation](#word-formation)
5. [Grammatical Features](#grammatical-features)
6. [How This Enables Deterministic Parsing](#how-this-enables-deterministic-parsing)

---

## Why Esperanto for AI

Esperanto was designed (by L. L. Zamenhof, 1887) to be **perfectly regular** - no exceptions, no irregular verbs, no grammatical gender complications. This regularity makes it ideal for AST-based AI:

### Key Advantages

| Feature | Benefit for AI |
|---------|---------------|
| **100% regular morphology** | Programmatic root+affix decomposition (no learned POS tagging) |
| **Fixed grammatical endings** | Deterministic case/tense/number detection (no attention needed) |
| **Compositional lexicon** | Root embeddings only (5K roots vs 50K+ words for English) |
| **No exceptions** | Every construction follows a rule (100% coverage possible) |
| **Explicit roles** | Subject (nominative -o) vs object (accusative -on) - no ambiguity |

### Comparison to English

| Task | English (Traditional LLM) | Esperanto (Klareco) |
|------|--------------------------|---------------------|
| **Tokenization** | Subword (BPE): 50K+ tokens, semantic drift | Morpheme: root+prefix+suffix+ending, stable |
| **POS Tagging** | Learned (attention-based) | Deterministic (ending reveals POS) |
| **Subject/Object** | Learned (word order + context) | Deterministic (nominative vs accusative) |
| **Tense Detection** | Learned (verb form lookup) | Deterministic (-as=present, -is=past, -os=future) |
| **Parameters Needed** | 110M+ for grammar understanding | 0 (grammar is programmatic) |

---

## The 16 Fundamental Rules

Esperanto has exactly **16 grammar rules** with **no exceptions**. Here they are:

### Rule 1: The Article
- **Single article**: "la" (the)
- **No indefinite article** (no "a/an")
- Example: "la hundo" = the dog, "hundo" = a dog

### Rule 2: Nouns
- **Ending**: -o (singular), -oj (plural)
- **Accusative** (direct object): -n (singular), -jn (plural)
- Examples:
  - "hundo" = dog (subject)
  - "hundon" = dog (object)
  - "hundoj" = dogs
  - "hundojn" = dogs (object)

### Rule 3: Adjectives
- **Ending**: -a (singular), -aj (plural)
- **Accusative**: -an, -ajn
- **Agreement**: Must match noun in number and case
- Examples:
  - "bela hundo" = beautiful dog
  - "belan hundon" = beautiful dog (object)
  - "belaj hundoj" = beautiful dogs

### Rule 4: Cardinal Numbers
- **Fixed forms**: unu (1), du (2), tri (3), kvar (4), kvin (5), etc.
- **No inflection** (numbers don't change)

### Rule 5: Personal Pronouns
- **Forms**: mi (I), vi (you), li (he), ŝi (she), ĝi (it), ni (we), ili (they)
- **Possessive**: add -a (mia = my, via = your)
- **Accusative**: add -n (min = me, vin = you-object)

### Rule 6: Verbs - No Conjugation by Person!
Verbs do NOT change based on who does the action - only tense:
- **Present**: -as (mi manĝas, vi manĝas, li manĝas - all same!)
- **Past**: -is (mi manĝis, vi manĝis, li manĝis)
- **Future**: -os (mi manĝos, vi manĝos, li manĝos)
- **Conditional**: -us (mi manĝus)
- **Volitive** (command/wish): -u (manĝu!)
- **Infinitive**: -i (manĝi = to eat)

This is HUGE - English has dozens of verb forms (eat/eats/ate/eaten/eating), Esperanto has 6.

### Rule 7: Adverbs
- **Ending**: -e
- **Formed from**: adjective root + -e
- Examples:
  - "bela" (beautiful) → "bele" (beautifully)
  - "rapida" (fast) → "rapide" (quickly)

### Rule 8: Prepositions
- **Govern nominative** (no case change)
- **Exception**: Direction can use accusative
- Examples:
  - "en la domo" (in the house - location)
  - "en la domon" (into the house - direction)

### Rule 9: Word Formation
- **Every word is spelled as pronounced**
- **Each letter = one sound** (phonetic!)
- No silent letters, no ambiguous spellings

### Rule 10: Stress
- **Always on second-to-last syllable**
- "ho-BIT-o" (hobbit), "Es-per-AN-to"

### Rule 11: Compound Words
- **Words can be freely combined**
- **Last root determines POS**
- Examples:
  - "hundo" (dog) + "domo" (house) = "hundodomo" (doghouse)
  - "tago" (day) + "manĝo" (meal) = "tagmanĝo" (lunch - day-meal)

### Rule 12: Negation
- **Single word**: "ne" (no/not)
- **Position**: Before verb or word being negated
- Examples:
  - "Mi ne manĝas" (I don't eat)
  - "Ne mi" (Not me)

### Rule 13: Object Direction
- **Accusative (-n)** shows direction/goal
- "Mi iras Londonon" (I go TO London)
- "Mi estas en Londono" (I am IN London - no accusative, static location)

### Rule 14: Each Preposition Has Definite Meaning
- **Fixed semantics** (no overloading like English "in/on/at")
- Examples:
  - "en" = in, inside
  - "sur" = on (top of)
  - "ĉe" = at (location)
  - "al" = to (toward)
  - "de" = from, of

### Rule 15: Foreign Words
- **Take Esperanto endings**
- "Shakespeare" → "Ŝekspiro" (with -o noun ending)
- "McDonald's" → "Mekdonaldo"

### Rule 16: Omission of Article and Final -o
- **Article can be omitted** in proverbs, titles
- **-o can be dropped** in poetry (for meter)
- **Rarely used** in modern Esperanto

---

## Morphology: Root + Affixes

Every Esperanto word (except grammatical particles) follows this structure:

```
[PREFIX] + ROOT + [SUFFIX(ES)] + ENDING
```

### Prefixes (7 standard)

| Prefix | Meaning | Example |
|--------|---------|---------|
| **mal-** | Opposite | bona (good) → malbona (bad) |
| **re-** | Again, back | fari (do) → refari (redo) |
| **ge-** | Both sexes | patro (father) → gepatroj (parents) |
| **eks-** | Former, ex- | reĝo (king) → eksreĝo (ex-king) |
| **ek-** | Sudden action | krii (cry) → ekkrii (cry out suddenly) |
| **pra-** | Great-, remote | avo (grandfather) → praavo (great-grandfather) |
| **for-** | Away, off | iri (go) → foriri (go away) |

### Suffixes (31 standard, here are the most common)

| Suffix | Meaning | Example |
|--------|---------|---------|
| **-ul** | Person characterized by | bona (good) → bonulo (good person) |
| **-in** | Feminine | kato (cat) → katino (female cat) |
| **-ej** | Place | lerni (learn) → lernejo (school) |
| **-et** | Diminutive (small) | domo (house) → dometo (cottage) |
| **-eg** | Augmentative (large) | domo → domego (mansion) |
| **-ar** | Collection | arbo (tree) → arbaro (forest) |
| **-il** | Tool/instrument | tranĉi (cut) → tranĉilo (knife) |
| **-ĉj** | Male diminutive (affection) | patro (father) → paĉjo (daddy) |
| **-nj** | Female diminutive | patrino (mother) → panjo (mommy) |
| **-ad** | Continuous action | kanti (sing) → kantado (singing - activity) |
| **-aĵ** | Concrete thing | manĝi (eat) → manĝaĵo (food) |
| **-abl** | Capable of | manĝi → manĝebla (edible) |
| **-ind** | Worthy of | laŭdi (praise) → laŭdinda (praiseworthy) |
| **-ebl** | Possible | kredi (believe) → kredebla (believable) |
| **-em** | Tendency to | paroli (speak) → parolema (talkative) |
| **-ig** | Cause to (transitive) | pura (clean) → purigi (to clean something) |
| **-iĝ** | Become (intransitive) | pura → puriĝi (to become clean) |
| **-end** | Must be done | pagi (pay) → pagenda (must be paid) |

### Endings (10 total)

**Verbs** (6):
- -as (present)
- -is (past)
- -os (future)
- -us (conditional)
- -u (volitive/imperative)
- -i (infinitive)

**Nouns**: -o

**Adjectives**: -a

**Adverbs**: -e

**Accusative marker**: -n (added to nouns/adjectives)

**Plural marker**: -j (added before -n if present)

---

## Word Formation

### Simple Words
```
hund + o = hundo (dog)
bel + a = bela (beautiful)
rapid + e = rapide (quickly)
manĝ + as = manĝas (eats/eat/eating-present)
```

### With Prefixes
```
mal + bona = malbona (bad - opposite of good)
re + veni + i = reveni (to come back)
ek + vidi + is = ekvidis (suddenly saw)
```

### With Suffixes
```
hund + ej + o = hundejo (kennel - dog-place)
lern + ej + o = lernejo (school - learning-place)
bel + ul + o = belulo (beautiful person)
```

### Multiple Suffixes
```
lern + ej + an + o = lernejano (school student - school-place-person-noun)
hund + in + et + o = hundineto (puppy - dog-female-small)
```

### Compounds
```
tag + manĝ + o = tagmanĝo (lunch - day-meal)
nod + ĉambr + o = noktĉambro (bedroom - night-room)
fer + voj + o = fervojo (railway - iron-road)
```

### Complex Example
```
mal + bon + ul + ej + o = malbonulejo
  mal- = opposite
  bon = good
  -ul = person
  -ej = place
  -o = noun
  = "place for bad people" = prison!
```

---

## Grammatical Features

### Case (2 total - simplest in any language!)
- **Nominative** (subject): -o, -oj
- **Accusative** (object): -on, -ojn

Example:
```
La hundo vidas la katon.
   ↓     ↓      ↓    ↓
  the   dog   sees  the cat
subject(nom) verb  object(acc)
```

### Number (2 total)
- **Singular**: (no marker)
- **Plural**: -j

Example:
```
hundo = dog (singular)
hundoj = dogs (plural)
```

### Tense (6 total)
- **Present**: -as
- **Past**: -is
- **Future**: -os
- **Conditional**: -us
- **Volitive**: -u
- **Infinitive**: -i

Example:
```
Mi manĝas.   (I eat/am eating)
Mi manĝis.   (I ate)
Mi manĝos.   (I will eat)
Mi manĝus.   (I would eat)
Manĝu!       (Eat! - command)
Mi volas manĝi. (I want to eat)
```

### Mood (expressed through tense endings)
- **Indicative**: -as, -is, -os
- **Conditional**: -us
- **Imperative**: -u

### Voice (expressed through participles)
- **Active**: -ant (doing), -int (done), -ont (will do)
- **Passive**: -at (being done to), -it (was done to), -ot (will be done to)

Example:
```
La hundo estas vidata. (The dog is being seen - passive present)
  vidata = vid (see) + -at (passive present participle) + -a (adjective)

La hundo estas vidinta. (The dog has seen - active past)
  vidinta = vid + -int (active past participle) + -a (adjective)
```

### Aspect (expressed through -ad suffix)
- **Continuous**: -ado
- Example: "kanti" (to sing) → "kantado" (singing - the activity)

---

## How This Enables Deterministic Parsing

### 1. Role Detection is Deterministic

**English** (ambiguous):
```
"The dog sees the cat" - which is subject?
→ Must learn from word order and context
→ Requires attention mechanism
```

**Esperanto** (explicit):
```
"La hundo vidas la katon"
     -o (nom)         -on (acc)
   = SUBJECT        = OBJECT
→ Zero ambiguity, pure rule-based detection
```

### 2. POS Tagging is Deterministic

**English**:
```
"I love reading" - is "reading" a verb or noun?
→ Must learn from context
```

**Esperanto**:
```
Verb: "Mi amas legi" (I love to read - legi ends in -i = infinitive verb)
Noun: "Mi amas legadon" (I love reading - legado ends in -o = noun)
→ Ending reveals POS with 100% accuracy
```

### 3. Tense Detection is Deterministic

**English**:
```
"I eat" vs "I ate" vs "I will eat" vs "I have eaten" vs "I am eating"
→ Multiple irregular forms, must memorize all
```

**Esperanto**:
```
"Mi manĝas" (-as = present)
"Mi manĝis" (-is = past)
"Mi manĝos" (-os = future)
→ One root, three endings, 100% regular
```

### 4. Morpheme Decomposition is Deterministic

**English**:
```
"unhappiness" = un + happy + ness
BUT: "understand" ≠ un + der + stand (not compositional!)
→ Need to memorize which words are compositional
```

**Esperanto**:
```
"malbonulo" = mal + bon + ul + o
  EVERY word is compositional, always!
  = opposite-good-person-noun = bad person
→ 100% of words can be decomposed programmatically
```

### 5. No Exceptions = 100% Coverage

**English irregular verbs**:
```
go → went (completely different word!)
be → am/is/are/was/were/been (8 forms!)
→ Must memorize ~200 irregular verbs
```

**Esperanto**:
```
ZERO irregular verbs
ZERO irregular nouns
ZERO irregular anything
→ One set of rules covers 100% of language
```

---

## Klareco Implementation

### Parser Strategy

Given these rules, Klareco's parser is deterministic:

1. **Tokenize by morphemes**: Split "malbonulejo" → ["mal", "bon", "ul", "ej", "o"]
2. **Identify POS from ending**: "-o" = noun
3. **Extract case**: "malbonulejon" has -n = accusative = OBJECT
4. **Extract number**: "malbonulejoj" has -j = plural
5. **Build AST**: Structured tree with explicit roles

```python
# Pseudocode
def parse_word(word):
    # Extract ending (last 1-2 chars)
    ending = extract_ending(word)  # -o, -a, -e, -as, -is, etc.

    # POS is deterministic from ending
    pos = POS_MAP[ending]  # {"-o": "noun", "-a": "adj", ...}

    # Extract case/number
    is_accusative = ending.endswith("n")
    is_plural = "j" in ending

    # Extract root + affixes
    root, prefixes, suffixes = decompose_morphemes(word)

    # Build AST node
    return {
        "vortspeco": pos,
        "kazo": "akuzativo" if is_accusative else "nominativo",
        "nombro": "pluralo" if is_plural else "singularo",
        "radiko": root,
        "prefiksoj": prefixes,
        "sufiksoj": suffixes
    }
```

### AST Structure

For sentence "La hundo vidas la katon":

```json
{
  "tipo": "frazo",
  "subjekto": {
    "radiko": "hund",
    "vortspeco": "substantivo",
    "kazo": "nominativo",
    "nombro": "singularo"
  },
  "verbo": {
    "radiko": "vid",
    "vortspeco": "verbo",
    "tempo": "prezenco"
  },
  "objekto": {
    "radiko": "kat",
    "vortspeco": "substantivo",
    "kazo": "akuzativo",
    "nombro": "singularo"
  }
}
```

**Every field is deterministically extracted - zero learned parameters!**

---

## References

### Official Esperanto Resources
- **PMEG** (Plena Manlibro de Esperanta Gramatiko): https://bertilow.com/pmeg/
- **Fundamento de Esperanto**: The official foundation document
- **ReVo** (Reta Vortaro): https://www.reta-vortaro.de/revo/

### Klareco Documentation
- `klareco/parser.py` - Implementation of these rules
- `DESIGN.md` - Technical design using this grammar
- `VISION.md` - Why Esperanto enables AST-first architecture
- `docs/CORPUS_INVENTORY.md` - Available Esperanto texts

### Learning Esperanto
- **lernu.net**: Free interactive course
- **Duolingo**: Esperanto course
- **Kurso de Esperanto**: Free comprehensive course
