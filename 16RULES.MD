That is an excellent historical and linguistic question!

The **"16 Rules of Grammar"** (or *Dek Ses Reguloj*) are the core, foundational set of grammatical principles published by L. L. Zamenhof in 1887 in *Unua Libro* (First Book), which introduced Esperanto to the world. They were designed to demonstrate the language's perfect regularity and ease of learning.

This set of rules is the **sole basis for your Symbolic Esperanto Grammar Expert**—they are all the parser needs to know.

Here is the list of the 16 rules, grouped by topic for clarity:

---

## 1. Syntax, Phonetics, and Articles

These rules govern the most fundamental aspects of how words are structured, pronounced, and sequenced.

| Rule | Description | Impact on Parser (Symbolic AI) |
| :--- | :--- | :--- |
| **Rule 1: Pronunciation** | There are 28 letters; each letter has only one sound. | The parser doesn't need complex phonetic rules; spelling is perfectly phonetic. |
| **Rule 2: Stress** | The accent (stress) always falls on the **penultimate (second-to-last) syllable**. | Essential for metrical analysis (poetry) and the simplest rule to code. |
| **Rule 3: Article** | There is only one definite article, **`la`** (the), used for all genders, numbers, and cases. There is **no indefinite article** ("a" or "an"). | The parser can immediately categorize `la` and ignores the complexity of indefinite articles entirely. |
| **Rule 4: Nouns** | The noun is always characterized by the ending **`-o`**. | The parser uses this to identify the noun's function and strip the `-o` to find the semantic root. |

---

## 2. Morphology (Inflection and Word Endings)

These rules define how words are formed, changed, and their grammatical function within a sentence. This section is the most crucial for your AST annotations.

| Rule | Description | Impact on Parser (Symbolic AI) |
| :--- | :--- | :--- |
| **Rule 5: Plural** | The plural is formed by adding the letter **`-j`** to the noun. | The parser checks for the `-j` suffix immediately following the `-o` or `-a` to tag Plural number. |
| **Rule 6: Case** | The noun has two cases: **Nominative** (subject) and **Accusative** (direct object). The Accusative is formed by adding **`-n`** to the Nominative form. | This is the basis for your parser's high accuracy: any noun ending in **`-n`** is the object. This allows the parser to determine the subject and object regardless of word order. |
| **Rule 7: Adjectives** | The adjective is always characterized by the ending **`-a`** and agrees with the noun in number and case. | The parser looks for the **`-a`** suffix and then confirms the adjective also carries the `-j` and/or `-n` suffixes if the noun does. |
| **Rule 8: Adverbs** | The primary adverb is characterized by the ending **`-e`**. | A simple rule to identify adverbs used to modify verbs. |
| **Rule 9: Prepositions** | Prepositions govern the **Nominative** case (they do not take the Accusative). | This simplifies the parser logic by limiting where the Accusative suffix can appear (only on direct objects of verbs). |

---

## 3. Verb Conjugation (Tense and Mood)

Verbs are entirely regular and are one of the most simple aspects of the grammar.

| Rule | Description | Impact on Parser (Symbolic AI) |
| :--- | :--- | :--- |
| **Rule 10: Verb Tense** | Verbs are completely regular and do not change form based on person or number (no conjugation for I, you, he, they, etc.). | The parser ignores the complexity of personal pronouns entirely, focusing only on the tense/mood suffix. |
| **Rule 11: Verb Suffixes** | The tenses are formed using fixed suffixes: | This is a simple lookup table for the parser: |
| | **Present Tense:** `-as` |
| | **Past Tense:** `-is` |
| | **Future Tense:** `-os` |
| **Rule 12: Verb Moods** | The moods are also formed using fixed suffixes: | This is a simple lookup table for the parser: |
| | **Infinitive:** `-i` (To run) |
| | **Conditional:** `-us` (Would run) |
| | **Jussive/Imperative:** `-u` (Run! or May it run) |
| **Rule 13: Participles** | The six possible participles (active/passive, present/past/future) are formed with a regular suffix: e.g., active present is **`-ant-`**. | The parser uses these regular suffixes to identify and tag complex verb phrases (e.g., `mi estas kuranta` - I am running). |

---

## 4. Derived Words and Numbers

These rules govern how the vocabulary is expanded and how non-word elements are handled.

| Rule | Description | Impact on Parser (Symbolic AI) |
| :--- | :--- | :--- |
| **Rule 14: Affixes** | Words are formed by attaching prefixes and suffixes to root words. Affixes are treated as independent words. | This is the foundation for your **Morphological Decomposition Expert**—the rule guarantees that all affixes have a fixed, codifiable meaning. |
| **Rule 15: Compound Words** | Compound words are formed by joining two root words, with the main meaning coming last (e.g., *vaporŝipo* - steam-ship). | The parser must be coded to identify compound roots and analyze them sequentially. |
| **Rule 16: Elision** | The final vowel of a noun's singular nominative form (`-o`) may be omitted, usually only in poetry. The accent remains on the second-to-last syllable of the original word. | This is the only exception to Rule 4/9 the parser must handle, primarily for text like "The Raven" translation. |

---

## Summary for the Symbolic Parser

Your Symbolic Esperanto Expert is robust because it only needs to implement these 16 rules (and their derived affixes) deterministically. The total lack of irregularity means that every grammatical feature required for the AST (PoS, Tense, Case, Number) can be tagged perfectly without the need for probabilistic deep learning.

