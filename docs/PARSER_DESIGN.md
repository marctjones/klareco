# Parser redesign: enumerate deterministically, rank only what the grammar leaves open

> **The question that produced this document:** *"Shouldn't the AST be a tree? After
> all it stands for abstract syntax TREE."*
>
> It should. It isn't. That single observation explains more of our measured
> failure than everything else in this repo combined.

---

## 0. The two architectural errors

**(a) The AST is not a tree. It is a fixed-arity record.**

```python
{'tipo': 'frazo', 'subjekto': {...}, 'verbo': {...}, 'objekto': {...}, 'aliaj': [...]}
```

One subject slot. One verb slot. One object slot. **Nothing recurses.** Measured
consequence, on gold:

| | value |
|---|---|
| gold subjects per sentence | **1.64** |
| sentences with 2+ subjects (multi-clause) | **35.8%** |
| **hard recall ceiling for a single-slot AST** | **42.5%** |
| our measured subject recall | 33.7% |

**Rules cannot break 42.5%. Only a shape change can.** We spent months tuning
rules against a wall.

And the boundary is visible in the split:

| treebank | subjects/sentence | subject F1 | % of *shape* ceiling reached |
|---|---|---|---|
| **Cairo** (simple) | 1.10 | **85.7%** | **95%** |
| **Prago** (multi-clause prose) | 1.42 | 35.1% | 61% |

> The deterministic parser is **excellent** on sentences its AST can represent, and
> collapses **exactly** where the AST cannot represent the sentence. The rules were
> never the problem. The container was.

**(b) The parser returns ONE analysis where the grammar licenses MANY — silently.**

> **A parser that returns one parse where the grammar licenses two is not
> deterministic. It is arbitrary.**

Measured on our own corpus and lexicon (`scripts/eval/measure_morpheme_ambiguity.py`):

| | ours | Guinard 2016 (published) |
|---|---|---|
| mean analyses per word type | **1.86** | 2.15 |
| types with ≥2 analyses | **48.1%** | 53.5% |
| **tokens with ≥2 analyses, in running text** | **32.0%** | — |

```
diskriminanta   18 licensed analyses:  diskrimin-ant|a · diskrim-in-ant|a · dis-krim-in-ant|a
parlamentanino  14 licensed analyses:  parlamentan-in|o · parlament-an-in|o · par+lament-an-in|o
```

Every one is legal under the 16 rules. The parser keeps one and discards the rest
without recording that a choice was ever made. **That is how `Esperanton` →
`esper+ant` happened.** It did not *fail*. It *committed*.

---

## 1. What the field already knows (and we didn't)

### The deterministic ceiling for Esperanto is **published**

**EspGram** (Eckhard Bick, Constraint Grammar, ~2,600 rules, 32k-entry valency
lexicon), measured on an 18.5M-word corpus and the 52k-token **Arbobanko** treebank:

| | EspGram | klareco today |
|---|---|---|
| POS | **99.5%** | 80.9% strict / 94.5% scheme-adj |
| Syntactic function / attachment | **96.5%** | subject F1 41.4% |

**The gap to the rule-based ceiling is ~19 points of POS and ~55 of syntax, and
none of it requires machine learning.** Do not reach for a model to fix a lexicon.

### Bick published our boundary map — measured, on Esperanto

He hand-corrected EspGram's output and used the change log as an error analysis:

| class | share of errors |
|---|---|
| **postnominal PP attachment** | **~1/4 – 1/3 of ALL errors** |
| **coordination scope** | **4× over-represented** |
| relative-clause attachment | 4× over-represented |
| direct object (`-n`) | *easy* — "due to the morphological accusative marker" |

> *"Especially the problems with pp attachment and coordination indicate that at the
> syntactic level, **Esperanto is not so different from other languages**, and that
> **ambiguity in this area arises from semantics rather than morphology.**"*
> — Bick, LREC 2020

That is the author of the *rule* parser, about his *own* rule parser. He names the
same two residues we found independently: **proper nouns** ("the only systematic
POS ambiguity … because of upper-casing") and **morpheme-boundary ambiguity**
(`altiri` = `al+tir` "attract" *or* `alt+ir` "go high" — both legal).

### The FST answer to `organo` vs `org+an` — and I had it half wrong

I assumed a finite-state analyzer resolves this by lexicon-authority / longest-match.
**It does not.** An analyzer FST emits **every accepting path, unranked**; there is
no preference operator in `lexc`/`twolc`/`lttoolbox`. Listing `organ` does **not**
suppress `org+an`, because `org` and `an` are *themselves* legitimate entries.

- **Hana (1998, PC-KIMMO, Esperanto)**: 13.6% lexical homonymy over 460k words.
  `doktoro` → `dok|tor|o` ("dock+torus"). `papero` → `pap|er|o` ("element of a pope").
- **Longest-match is a heuristic, and a measurably bad one**: 94.4%, versus 98.9% for
  a trivial n-gram Markov model over morpheme classes (Guinard 2016).

**Apertium sidesteps the problem by refusing to derive at all** — 93,100 hand-listed
stems, *no productive affix rules*, ambiguity ≈ 0. It pays for that: Bick reports
**25.1% of noun lemmas in real text are not in the lexicon** and must be guessed.

> **You can have generativity or you can have unambiguity. Not both.**
> We chose generativity. The 32% is therefore ours *by construction* — not a bug.

---

## 2. The redesign

### 2.1 The AST becomes an actual tree

One recursive node type. The predicate-argument frame moves **from the sentence to
the clause**, and clauses nest. This is the minimal change that breaks the 42.5%
ceiling, and it preserves everything already built.

```
Dokumento
└── Paragrafo
    └── Frazo
        ├── Propozicio  (main)                 ← the frame lives HERE, once per finite verb
        │   ├── predikato : Vorto(fond)
        │   ├── subjekto  : Vortgrupo(kerno=Zamenhof)
        │   └── objekto   : Vortgrupo(kerno=esperant)
        └── Propozicio  (subordinate, rolo=rilativa)
            ├── predikato : Vorto(parol)
            └── subjekto  : Vortgrupo(kerno=li)
```

`Vorto` is *already* a tree (root + prefixes + suffixes + ending), so the morpheme
level nests inside the phrase level inside the clause level. **One type, all the way
down** — word, phrase, clause, sentence, paragraph, document.

Every node carries:
- `rolo` — its role in its parent
- `fonto` — **rule** or **model** (VISION.md: attribution is built in)
- `alternativoj` — see below

### 2.2 Emit the licensed SET, not one arbitrary member

The analyzer's contract changes from *"return the analysis"* to *"return the
analyses"*. This is a **rules-only change, zero learned parameters**, and it:

1. makes the residue **visible and countable** (32.0% of tokens — now measured);
2. makes the disambiguator a **separate, swappable, attributable stage**;
3. lets us honestly say *"the grammar licensed 18 readings; the ranker chose #3"* —
   which is the decomposability the whole project is for.

This is not novel. It is **the production standard** for morphologically rich
languages — Apertium, GiellaLT, South Sámi, Turkish, Icelandic — always factored as:

> **symbolic analyzer enumerates the licensed readings → disambiguator ranks them.**

### 2.3 Emit CoNLL-U

Oya (2025), on the Esperanto UD treebanks:

> *"automatic parsing was not conducted because **the parsers for Esperanto available
> at present do not yield parse output in the format of CoNLL-U**."*

He then explicitly calls for someone to build and evaluate an Esperanto UD parser.
**No parsing result on the Esperanto UD treebanks has ever been published.** Emitting
CoNLL-U costs us little, gives us LAS/UAS as a native metric, lets us benchmark
directly against Stanza/UDPipe/Trankit — and makes us first.

---

## 3. The auxiliary systems (what actually generates a high-quality AST)

The parser is ~20% of the quality. The **lexical resources are the other 80%**, and
this is where our 19-point gap to EspGram lives.

| resource | what it gives | size | license | status |
|---|---|---|---|---|
| **ReVo** (`revuloj/revo-fonto`) | curated root inventory, independent of our parser | **12,131 roots** | GPL-2.0 | ✅ **integrated** (#806) |
| **ESPSOF** (espsof.com) | **PRE-SEGMENTED words with GOLD morpheme boundaries** | **50,000+** | free | ⬜ **the morphology gold set we lack** |
| **apertium-epo** `.dix` | proper-noun gazetteer (`__np` paradigms) | **~36,750 names** | GPL-3 | ⬜ drop-in for the lost `proper_nouns_dynamic_v3.json` |
| **apertium-epo** `.dix` | lexicalized derived stems | 93,100 entries | GPL-3 | ⬜ a *lexicalization* oracle — every entry is a form that did NOT need deriving |
| **Hunspell `eo.dic`** | flagged roots + compound flags | **29,571** | GPL | ⬜ cross-check on ReVo |
| **ESPDIC** (Denisowski) | Esperanto–English lexicon | **63,380 entries** | free | ⬜ sense inventory |
| **Arbobanko** (Bick) | Esperanto treebank + semantic ontology | **52,000 tokens** | proprietary — must ask | ⬜ **16× UD-Prago; the #820 unblock if obtainable** |
| **Guinard's segmenter** | trained morpheme segmenter + model | 98.9% | open, GitHub | ⬜ a *baseline to beat*, and a labelled corpus |
| **Tekstaro** | reference Esperanto corpus | — | free | ⬜ |
| **The ontology** | **root → semantic class** | **0 rows** | ours | ❌ **EMPTY — and it is the load-bearing one. See §4.** |

**ESPSOF is the most valuable thing on this list that we do not have.** 50,000
words with gold morpheme boundaries is *exactly* the test set for the 32% ambiguity
— it turns morphological disambiguation from unmeasurable into a benchmark, today.

---

## 4. Where deep learning belongs — and precisely why rules cannot get there

### The theorem

**Church & Patil (1982)** proved a broad-coverage grammar licenses a number of
PP-attachment parses growing as the **Catalan numbers** — 1, 2, 5, 14, 42, …
*"Put the block in the box on the table"* is ambiguous **by grammar**. Both parses are
**well-formed**. You cannot write a rule that removes one without also removing the
sentences that *mean* it.

> **Disambiguation is not a grammatical operation. It is a *ranking* operation over a
> set the grammar has declared equally legal.** The grammar already did its job — and
> returned every answer.

### Where the ranking information actually lives — measured

- Human accuracy on PP-attachment given only `(verb, noun₁, prep, noun₂)`: **88%**.
  Given the full sentence: **93%**.
- **Hindle & Rooth (1993): 80% from pure lexical co-occurrence mutual information —
  no grammar at all.**

*"Eat pasta **with a fork**"* vs *"eat pasta **with meatballs**"* differ in **nothing
syntactic whatsoever**. Only in what forks and meatballs *are*.

That information is what a pretrained encoder holds — a compressed model of which
nouns co-occur with which verbs under which prepositions, across billions of tokens.
A grammar **cannot** contain it, because **it is not grammatical information**. A
lexicon could — but only by becoming a hand-written table of selectional preferences
over every verb × preposition × noun triple, at which point you have hand-written a
co-occurrence matrix, badly.

**And Esperanto does not escape this.** Its regularity buys POS, case, number, tense,
agreement — and Bick confirms those become near-perfect (`-n` makes object
identification *easy*). It buys **nothing** on PP-attachment or coordination scope,
because those were never morphological. `Mi vidis la viron kun teleskopo` is exactly
as ambiguous as its English source.

> **A designed language cannot design away an ambiguity that lives in the world
> rather than in the code.**

### And the same argument, in miniature, inside a *word*

Hana diagnosed his own failure in **1998**, and it is the whole thesis:

> `papero` → "element of a pope" *"could be prevented by prohibiting assigning the
> affix `-er-` to countable nouns. However, the classification of roots is very time
> consuming."*

`pap`+`er` is **legal morphology and nonsense semantics**. The rule that kills it is
not a grammar rule — it is *"`paper` is a substance, and substances do not take
`-er-`."* That is the **semantic class of the root**. It is not in the grammar and
**cannot be**.

**It is exactly what an ontology holds. And ours is empty** — `ontology_nodes` and
`ontology_edges` are 0 rows, `verb_klaso` 0% populated. Guinard's Markov model over
morpheme **semantic classes** scores 98.9% vs 94.4% for longest-match: it is the
*statistical stand-in for the same missing information*.

> **The ontology is not a deferred nice-to-have. It is the artifact that collapses
> 32% of the corpus from ambiguous to determined — and the 1998 literature says so.**

### So the model's job, stated exactly

> **A ranker over deterministically-enumerated candidate structures.**

It never proposes a structure the grammar forbids. It only chooses among structures
the grammar permits. That keeps the system decomposable: the rules still run, and we
can always state precisely how many points the model added, on exactly which
ambiguity class.

And the surface is **small and named**:

| residue | why deterministic methods provably stop | size |
|---|---|---|
| **PP attachment** | grammar licenses both; only selectional preference decides | ~1/4–1/3 of Bick's errors |
| **Coordination scope** | ditto; conjunct *similarity* is the signal, and it is semantic | 4× over-represented |
| **Morpheme segmentation** | needs the root's semantic class | **32.0% of tokens** |
| **Lexicalized proper nouns** | `Petro` = `petr-o` = "rock"; only usage decides | ~small, and the capitalization ratio already handles the *seen* ones |

Everything else — morphology, case, number, tense, agreement, object identification —
is deterministic, and near ceiling.

---

## 5. The order of work

1. **Close the 19-point POS gap to EspGram.** Zero ML. Lexical resources (§3) +
   the role-assignment rule bugs (verbless headings, adverbs-as-subjects,
   determiners stealing the head — ~100% of our precision errors).
2. **Make the AST a tree** (§2.1). Breaks the 42.5% ceiling. Zero ML.
3. **Emit the licensed set + CoNLL-U** (§2.2, §2.3). Zero ML. Makes the residue
   countable and the disambiguator swappable.
4. **Populate the ontology** (root → semantic class). Zero ML. Collapses a large
   share of the 32%.
5. **Only then**: a ranker over the residual candidates. Measure what it adds, per
   ambiguity class, against a deterministic system that still runs.

**We are at step 1.** Everything through step 4 is classical computation, and the
literature says it takes us to ~99.5% POS / ~96.5% syntax.

## Sources

- Bick, *Syntax and Semantics in a Treebank for Esperanto*, LREC 2020 — https://aclanthology.org/2020.lrec-1.630.pdf
- Bick, *Tagging and Parsing an Artificial Language* (EspGram), CL 2007
- Oya, *UD Treebanks for Esperanto as a Natural Language*, UDW/SyntaxFest 2025 — https://aclanthology.org/2025.udw-1.3/
- Hana, *Two-level Morphology of Esperanto*, 1998 — https://ufal.mff.cuni.cz/~hana/esr/thesis.pdf
- Guinard, *Esperanto Word Segmentation*, PBML 105, 2016 — https://ufal.mff.cuni.cz/pbml/105/art-guinard.pdf
- Church & Patil, *Coping with Syntactic Ambiguity*, AJCL 1982 — https://aclanthology.org/J82-3004/
- Hindle & Rooth, *Structural Ambiguity and Lexical Relations*, CL 1993 — https://aclanthology.org/J93-1005/
- Ficler & Goldberg, *Coordination Boundary Prediction*, EMNLP 2016 — https://aclanthology.org/D16-1003/
- Beesley & Karttunen, *Finite State Morphology*, CSLI 2003
- apertium-epo — https://github.com/apertium/apertium-epo
