# Proper Nouns: infer them, don't look them up

> **The challenge that produced this document:** *"Do we need parser
> dictionaries? Building a huge dictionary seems like giving up."*
>
> It is. This is what to do instead — and the method comes from Esperanto's own
> normative tradition, not from us.

## The reframe

The two artifacts we lost in the June migration were **not the same kind of
thing**, and conflating them is what made "restore the dictionaries" sound
reasonable:

| Artifact | What it really is | Verdict |
|---|---|---|
| `proper_nouns_dynamic_v3.json` | An **open-world list of every name that exists** | ❌ **Giving up.** Unbounded, stale by construction, pure world knowledge. |
| `protected_roots.json` | A **closed-world fact about the language**: which forms have *lexicalized* | ✅ Legitimate — but **derive** it, don't hand-maintain it |

**We do not need a list of names. We need a list of Esperanto roots** — finite,
closed, and derivable from our own corpus. Then proper-nounhood is *inferred*:

> **capitalized** (in a position where capitalization carries signal)
> **AND stem is NOT a known Esperanto root** → **proper noun**

An open-world *lookup* becomes a closed-world *inference*. That is the whole
idea.

---

## What Esperanto itself says

These rules are not heuristics we invented. Each is grounded in the language's
own tradition — and two of the sources are **already in our corpus**.

### 1. The alphabet is closed (16RULES, Rule 1)

> *"There are 28 letters; each letter has only one sound."*

Esperanto's orthography is **phonetic and closed**. `q`, `w`, `x`, `y` do not
exist. Neither do clusters like `sch`, `th`, `ph`, `ck`, or doubled consonants.
**Their presence is proof the token is not an Esperanto word.**

### 2. Zamenhof licensed foreign spelling for names (*Lingvaj Respondoj* 63)

*La Esperantisto*, 1891 — and this text is **in our own store**:

> *"Propran nomon oni povas nun skribi tiel, kiel ĝi estas skribata en la
> gepatra lingvo de ĝia posedanto, ĉar en la nuna tempo la fonetika skribado de
> multaj nomoj kaŭzus tro grandan kriplaĵon de tiuj nomoj…"*
>
> *(A proper name may now be written as it is written in the native language of
> its owner, because phonetic spelling would mutilate many names…)*

So **non-Esperanto orthography positively licenses proper-nounhood.** This is not
a hack — it is the founder's ruling, and it is why the signal is reliable.

### 3. Unassimilated names resist the accusative (PMEG / Akademio)

Foreign, formally unassimilated names are treated **as quotations**. They resist
`-n`, and a **head noun carries the case instead**:

```
la urbo New York          <- "urbo" is the head; the name stays invariant
la verkon «Faŭsto»        <- "verkon" carries -n; the title does not
Mi konas sinjoron Glazunovski
```

This is a **syntactic** signal, not a lexical one — and it is the correct
Esperanto construction, which is why our test-set generator producing
`Kiu venkis Rorke's Driftn?` was not merely ugly but *ungrammatical* (#791).

### 4. Assimilation is a spectrum, not a binary

- Continents, oceans, countries → translated (`Britujo`, `Francio`)
- Large well-known cities → often Esperantized (`Parizo`, `Londono`, `Nov-Jorko`)
- Local names, personal names → usually kept native (`Shakespeare`, `Makita`)
- Personal names → **the owner decides** (`Johano` or `John`)

So a name may look *fully* Esperanto (`Prago`, `Esperanto`, `Homaranismo` — all
end in `-o` and inflect normally). **The "invalid ending" test alone is
therefore weak**: measured on UD-Prago, **85% of gold proper nouns have a valid
Esperanto ending.** Assimilated names are morphologically indistinguishable from
common nouns — which is exactly why the *root lexicon* is the load-bearing
signal and orthography is only a supplement.

---

## The rule stack, and what it buys

Measured against **UD_Esperanto-Prago** (linguist-curated, external, touches none
of our code — the one ruler that cannot lie to us):

| Rule | P | R | **F1** |
|---|---|---|---|
| **Current parser** (dictionary missing) | 18.2% | 57.1% | **27.6%** |
| capitalized + stem not in the **root lexicon** | 29.1% | 85.2% | **43.4%** |
| + ignore **ALL-CAPS** (a heading carries no capitalization signal) | 32.8% | 81.5% | **46.8%** |
| + **position reset** after `.` `!` `?` `«` `(` `:` | 38.0% | 70.4% | **49.4%** |
| + **foreign orthography** licenses a name (Zamenhof LR63) | 38.5% | 74.1% | **50.6%** |

**27.6% → 50.6%, with no name list at all.**

And note: with only the Fundamento's **2,481** roots, the rule already reaches
**F1 42.2% at 100% recall** — it misses *nothing*. The bottleneck was never the
concept. It was that our lexicon held 2,481 roots when Esperanto has ~20,000.

`scripts/index/build_root_lexicon.py` now harvests **12,377** roots from the
corpus (lowercase-attested only ∪ the Fundamento).

### Why lowercase attestation is the discriminator

A root is an Esperanto root if the corpus uses it as a **common word** — i.e.
lowercase. **Names are capitalized; common words are not.** The corpus separates
them for us, for free. We never have to know anything about the world.

It works:

```
nord, brit, hund, urb    -> IN the lexicon   (Esperanto roots)
zamenhof, shakespear     -> NOT in           -> correctly INFERRED as names
```

---

## `protected_roots` is a *lexicalization* fact — and it is derivable

The parser splits `Esperanton` → `esper` + `ant`. **Etymologically the parser is
right**: Zamenhof's pseudonym was *Doktoro Esperanto* — "Doctor One-Who-Hopes".
The word genuinely *is* `esper-ant-o`.

The phenomenon is **lexicalization**: a compositionally-derived form has become a
fixed lexeme with its own meaning. That is a fact about *usage*, not *grammar*,
so no grammar rule recovers it — **but it is visible in the corpus.** A
lexicalized form takes further derivation *as if it were a root*:

```
esperant-ist-o, esperant-uj-o, esperant-ig-i, esperant-ec-o
    -> 102 distinct derivational tails attested in the corpus
```

So **compute it** from derivational productivity, human-review it once, and
regenerate it when the corpus changes. It is a *derived artifact*, and it is
small — lexicalization is rare.

---

## The residue that actually survives

After all of the above, what genuinely remains is:

> A **capitalized** token, in a **signal-bearing position**, with **valid
> Esperanto morphology**, whose stem **IS a known root**.

`Nordo` (the North / north). `Brita` (British / a surname). `Maria` (Mary / an
adjective). Here morphology, position, and the lexicon *all* say "ordinary
Esperanto word", and only world knowledge says otherwise.

**That is a far smaller residue than "every name in the world"** — and it is the
honest one. Sizing it is exactly what **#819** does, and until it is sized,
VISION.md's claim that this residue is "confirmed" remains unearned.

## Sources

- [The Sixteen Rules of Esperanto Grammar, commented by Don Harlow](https://babel.ucsc.edu/~hank/105/Esperanto16.pdf)
- [Rekomendo de la Akademio pri la uzo de propraj nomoj](https://sezonoj.ru/2013/10/akademio-2/)
- [Respondoj de la Konsultejo — Akademio de Esperanto](https://www.akademio-de-esperanto.org/akademio/index.php?title=Respondoj_de_la_Konsultejo)
- [Propraj nomoj, la esperantaj kaj la fremdaj](https://kovro.heliohost.org/eo/artikoloj/fremdaj-nomoj.html)
- [Esperanto Proper Names — David G. Simpson](http://esperanto.davidgsimpson.com/eo-proper.html)
- Zamenhof, *Lingvaj Respondoj* 61 & 63 — **in our own corpus** (`data/raw/eo/lingvaj_respondoj/`)

---

## Improving it: what actually moved the number (measured 2026-07-13)

### 1. Morphology beat "more roots" — decisively

The false positives were **not** hard cases. They were `Homaranismo`, `Homaranoj`,
`Presejo`, `Oficejo`, `Britio` — ordinary **derived** Esperanto nouns:
`homar+an+ism+o`, `pres+ej+o`, `ofic+ej+o`, `brit+i+o`. **Every one of those roots
was already in the lexicon.** The naive stemmer only stripped the *final ending*,
so it never reached them.

Replacing it with **full affix decomposition** against the closed affix inventory
(the 16 rules give us a *finite* list of prefixes and suffixes — this is the
language's own morphology, not a heuristic):

| | P | R | **F1** |
|---|---|---|---|
| naive final-ending strip | 38.5% | 74.1% | **50.6%** |
| **full affix decomposition** | **53.6%** | 55.6% | **54.5%** |
| **…scheme-adjusted** | **83.3%** | 55.6% | **66.7%** |

**Precision 38.5% → 83.3%.** Of the remaining false positives: **6** are
abbreviations/initials (`D-ro`, `L.` — a separate fix DESIGN.md already lists),
**4** are UD scheme choices (UD annotates `Esperanto` and `-ismo` doctrines as
NOUN — the same class as `PRON→adjektivo` that we already credit), and only **3**
are genuine errors.

### 2. ⚠️ Do NOT feed the parser's own `radiko` back in — it is circular

Using the parser's morphological decomposition instead of an independent one
**collapsed F1 to 12.8%**. The root lexicon is *harvested from the parser's
output*, so feeding that output back in is failure mode **F13** — the parser
grading its own homework. Independence is what makes the signal worth anything.

### 3. So — should we expand the root lexicon? **Yes, but the reason is subtle**

With the positional heuristics in place, lexicon size looks **flat**:

| roots | 4,502 | 12,118 | 31,855 |
|---|---|---|---|
| F1 | 47.4% | 47.9% | 47.3% |

That flatness is an **artifact of the position veto**, which was masking the
lexicon's real contribution. Positional evidence is *absent* at sentence-start —
and that is exactly where we were missing names (`Varsovio`, `Zamenhof`). Turn
the veto off, so morphology must carry the decision alone, and lexicon size
matters enormously:

| lexicon | size | P | R | **F1** (scheme-adj) |
|---|---|---|---|---|
| Fundamento only | 2,481 | 22.1% | 77.8% | **34.4%** |
| + corpus ≥50 | 5,229 | 42.0% | 77.8% | **54.5%** |
| + corpus ≥10 *(current)* | 12,377 | 45.7% | 77.8% | **57.5%** |
| **+ corpus ≥3** | **31,928** | **50.0%** | 74.1% | **59.7%** |
| + corpus ≥1 (everything) | 115,832 | 50.0% | 66.7% | **57.1%** ⬇ |

**22.1% → 50.0% precision.** But note the last row: at ≥1 occurrence it gets
*worse*. **Names leak into the lexicon** (harvested from a degraded parser that
over-tags `propra_nomo`), and the recall drops as real names start "decomposing"
to a contaminated root.

> **The limit is lexicon PURITY, not lexicon SIZE.** Beyond ~30K roots, adding
> more corpus-attested strings adds more contamination than coverage.

**That is the case for ReVo (#806)**: not "more roots" but a *curated* ~20K-root
lexicon with no name contamination. Quality is the axis that is still open; raw
size is not.

### 4. The real limiter on further progress is the ruler, not the method

UD-Prago contains **27 PROPN tokens**. Every number on this page has enormous
error bars, and the difference between 55% and 65% is not measurable on it.
**Further tuning against this treebank would be fitting noise.**

Before optimising further we need a bigger proper-noun evaluation set — which is
a *measurement* problem, not a *modelling* one, and it is the honest next step.

---

## The AST helps — and improving morphology *hurt*. Both are real. (2026-07-13)

> **The question that produced this section:** *"Does the AST of the rest of the
> sentence help identify if something is a proper noun, or a common noun at the
> start of a sentence?"*

**Yes — and chasing the answer overturned the residue claim written above it.**

### 1. Syntax resolves cases morphology provably cannot

Two rules, both **deductive** — they come from the 16 rules, not from fitting:

**Rules 2–7 — a content word must carry a grammatical ending.** `sam` is a
*root*; `sama` is a *word*. `pet`+`er` is a root plus a suffix with no ending —
not a word form at all. `decomposes_to_root` matched a **bare root** and so
accepted `Sam` and `Peter` as ordinary Esperanto words. **That was a bug**, and
it is why they could never be recognised. Fixed.

**Rule 3 — an adjective must agree with its head noun in number and case.** This
is the one that needs the *rest of the sentence*:

```
Maria gajnis bronzon      mar-i-a is an ADJECTIVE form. The next token is a VERB.
                          There is no noun to agree with, so the adjective reading
                          is UNGRAMMATICAL -> the token is not that word -> a name.

Centra Oficejo            agrees with Oficejo -> an ordinary adjective. Not a name.
```

`Maria` = `mar-i-a` ("of the sea") is a genuine ambiguity **in isolation**. The
sentence destroys it. This is exactly the information token-internal analysis
lacks, and it produced **zero false positives** on gold.

Position must therefore **stop vetoing grammar**. Capitalisation is *evidential*
and is worthless at sentence-start; ending-validity and agreement are
*deductive* and hold everywhere. `Varsovio` and `ZAMENHOF` were forced misses
purely because a position/ALL-CAPS veto ran before the grammar. Reordered.

### 2. But a *more correct* decomposer made name-detection *worse*

Diagnosing the false positives showed they were not rule errors — they were
**gaps in our own morphology**:

| missing | examples | rule |
|---|---|---|
| the six **participles** | `Konsciante`, `Planita`, `Lanĉita` | **Rule 6** |
| **root+root compounding** | `Hispanlando`, `Plurlingveco`, `Multokaze` | the most productive process in the language |
| **prepositional prefixes** | `Subskribo`, `Transnacia`, `Antaŭparolo` | prepositions prefix freely |
| inflected **correlatives** | `Kion`, `Ĉian` | closed class |

Fixing all of them removed **17 genuine false positives** — and *dropped recall*,
because the rule is *"fails to decompose → name"*, so a decomposer that
decomposes more things detects fewer names. `Esperanto` (`esper-ant-o`) and
`Ruslando` (`rus-land-o`) became undetectable **precisely by getting the grammar
right.**

> **This tension is itself a boundary finding.** Morphological completeness and
> proper-noun-detection-by-morphological-failure are in **direct opposition**. You
> cannot maximise both. Esperanto's productivity means almost any name can be
> *given* a derivation — so the better the morphology, the weaker the signal.
> What survives is not a grammar fact but a **usage** fact: **lexicalization**.
> `Esperanto` *is* `esper-ant-o` and is *also* a name. That is `protected_roots`
> (#804) — the artifact lost in the June migration — and this is independent
> evidence that it is load-bearing, not legacy cruft.

### 3. ⚠️ The ruler contradicts itself — no F1 claim is admissible

**Of 26 remaining misses, 16 are on token types UD annotates BOTH ways:**

| token | gold tags in the same treebank |
|---|---|
| `Esperanto` | **PROPN ×10, NOUN ×4** |
| `Homaranismo` | **NOUN ×4, PROPN ×4** |
| `L` | **NOUN ×6, PROPN ×2** |

With 41 gold PROPN and the ruler self-contradicting on the majority of our
disagreements, **neither a gain nor a regression is measurable here.** The rules
above are landed on **deductive** grounds — they are grammar, not fitted
parameters — and are explicitly **NOT** claimed as a merge-gate number. Under the
merge gate this is a research-track characterization finding. **#820 (a real
proper-noun eval set) is the unblock, and nothing here should be tuned further
until it lands.**

### 4. So what is the residue *now*?

Smaller than "every name in the world", and **differently shaped** than this
document claimed a day ago:

- ❌ **not** foreign words — orthography solves those (Zamenhof LR63), 100% recall
- ❌ **not** non-decomposing names — ending-validity solves those, at any position
- ❌ **not** adjectival collisions like `Maria` — **syntax** solves those
- ✅ **lexicalized** forms: `Esperanto`, `Ruslando` — derivable from the corpus (#804)
- ✅ **the hard core**: a token that is a *valid noun form*, *correctly slotted*,
  and *article-less* — `Petro` (`petr-o` = "rock"). Morphology says word, syntax
  says word, and only usage says name.

`Petro` is the honest residue. It is **not** dissolved by syntax, and `de Petro`
without an article is odd but not ungrammatical (`libro de papero`), so the
article signal is soft — its strongest form was confounded anyway (`has_la` was
3/11, and all three were the self-contradicting `Homaranismo`).

**Next signal to try, and it is still deterministic:** the corpus's own
**capitalisation ratio** per token type. A name is capitalised mid-sentence
almost always; a common noun almost never. That is a closed-world statistic over
our own corpus — the same move as lowercase-attestation for the root lexicon —
and it targets exactly the lexicalized + hard-core cases. It is *not* a gazetteer
of the world's names.

## Ranked next moves

1. **A larger proper-noun eval set.** Everything else is unmeasurable without it.
2. **ReVo (#806)** — a *curated* root lexicon. Purity, not size.
3. **Abbreviations and initials** (`D-ro`, `L.`) — 6 of 13 remaining FPs; DESIGN.md
   already lists it as deterministically fixable.
4. **Hybrid position handling** — use positional evidence where it exists, fall
   back to morphology where it does not, rather than letting position *veto*
   morphology outright.
5. **Scheme-adjusted reporting**, as we already do for POS — UD's treatment of
   `Esperanto` as NOUN is a scheme difference, not our error.
