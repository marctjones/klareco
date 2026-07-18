# Klareco Vision: Mapping the Boundary

## The Thesis

Klareco is not primarily an attempt to build a small language model. It is an
attempt to **find, empirically, where deterministic computation stops.**

The question is:

> Given a language whose grammar is genuinely regular, how much of "understanding"
> can be done with ordinary, classical, deterministic programming — rules, tables,
> graph queries, unification, constraint solving, search — and what is the
> *irreducible residue* that resists it?

The residue is the interesting part. Whatever is left over after we have pushed
deterministic methods as far as they honestly go is, by construction, the part
that actually requires learning. That residue is where machine learning belongs
— and nowhere else.

## What changed (and why this document was rewritten)

An earlier version of this document said the goal was to "get grammar for free so
learned parameters can be spent entirely on reasoning." That framing pre-assigned
the boundary: grammar → rules, reasoning → neural network.

**That pre-assignment was an assumption, not a finding, and we are dropping it.**

We do not know in advance that reasoning requires learning. A great deal of what
gets called "reasoning" — transitive inference, type hierarchies, constraint
propagation, quantifier scope, arithmetic, planning, path-finding through a fact
graph — is classical computer science with decades of theory behind it. We should
attempt all of it deterministically first, and find out where it breaks.

Likewise, we do not know that all of grammar is deterministic. Esperanto has no
morphological marker distinguishing a proper noun from a common noun; a novel
capitalized surname that is also an ordinary word is genuinely ambiguous, and no
rule can settle it. That is a *grammatical* problem that provably needs
distributional knowledge.

So the boundary does not run neatly between "grammar" and "reasoning." It runs
somewhere else, and its actual shape is the research output of this project.

## The method

1. **Attempt it deterministically.** Every capability starts as a rule, a table,
   a graph query, a unification, or a search. Implement it. Measure it.
2. **Find the failure mode.** Where does the deterministic version break, and
   *why*? Not "it scores 71%" — but "it fails precisely on cases where two roots
   are synonymous and no rule can know that."
3. **Characterize the residue.** State the failure as a property of the problem,
   not of the implementation. A residue is only real if you can say what
   information the deterministic method provably lacks.
4. **Only then, learn it.** Introduce a learned component targeted at that
   specific, characterized residue — as small as it can be — and measure how much
   of the residue it actually recovers.
5. **Keep the contribution decomposable.** Because the deterministic version still
   exists and still runs, we can always say how much the learned component added.

The output of step 3, accumulated across many capabilities, *is the thesis*. A
map of the boundary is a more durable contribution than a benchmark number.

## Why Esperanto

Esperanto is the ideal testbed because it maximizes the deterministic side of the
experiment. Its grammar has 16 rules and no exceptions. Its morphology is fully
compositional: every word decomposes into prefix + root + suffixes + ending, and
the ending states the part of speech, case, number, and tense. Its correlative
system is a closed, regular table. Its accusative case marks grammatical role
explicitly, so the role of a constituent need not be inferred from word order.

This means that when a deterministic method fails on Esperanto, we learn
something real. The failure cannot be blamed on morphological irregularity or on
syntactic ambiguity that a better parser would have resolved — because in
Esperanto those ambiguities largely do not exist. **Esperanto lets us isolate the
residue.** A failure here is evidence that the problem is genuinely not
rule-shaped.

If a capability *can* be done deterministically in Esperanto but not in English,
that is also a finding — it tells us the obstacle was linguistic irregularity,
not the nature of the task.

## What the residue looks like so far

These are the places where deterministic methods have actually broken, with the
reason stated as a property of the problem. This list is the real deliverable and
should grow.

**Proper-noun disambiguation (⚠️ CLAIMED residue — under test, see #819).**
Esperanto has no morphological proper-noun marker. A capitalized token whose
morphology, position, and function-word context give no signal — a novel surname
that is also a common word, sentence-initial position, an all-caps title, a
quoted phrase — cannot be resolved by any rule, so disambiguation requires
distributional or world knowledge.

**That argument is plausible. It is not yet earned** — and this document
previously called it *"confirmed"* without ever measuring it. Two things are
wrong with that:

- It is currently being invoked to excuse an F1 of **27.6%** (UD-Prago, measured
  2026-07-13: precision 18.2%, recall 57.1%, 36 false positives against 8 true
  positives). **You do not get to call a problem unsolvable while your solution
  is that bad** — and ours is bad for a mundane reason: much of the deterministic
  scaffolding was degraded or unmeasured. (Update 2026-07-18: a live-store audit
  found some of these already repaired — `protected_roots.json` is back
  (`Esperanton`→`esperant`), the ontology is **loaded and consumed**, not empty
  (12,798 nodes / 13,212 edges), and `entity_facts` is present. Still open: the
  `proper_nouns_dynamic_v*` dictionary (#804) and whether all Wikipedia article
  titles are in the store (#803). The F1 above has **not been re-measured** since
  those repairs.) A 27.6% F1 measured against a degraded scaffold is a property of
  the **implementation**, not the **problem** — re-measure before concluding
  anything.
- As written, the claim is **unfalsifiable**: "a token where morphology, position
  and context give *no* signal" defines the residue as whatever is left over. The
  honest question is not *"does an irreducible core exist?"* — trivially it does,
  for some tokens — but **"how big is it?"** A residue of 5% and a residue of 60%
  are entirely different findings, and only one of them justifies a model.

**#819 tests it properly**: push the deterministic method to its ceiling
(dictionary → article-title gazetteer → positional rules → morphological
decomposition → function-word context), measuring each step against UD-Prago —
the one ruler that is external and cannot lie to us — and report the ceiling **as
a number**. If it lands at 0.98, this entry gets **deleted**, and that is a *win*:
it is the thesis working. If it stalls at 0.6, we will have *earned* the word
irreducible, and be able to name exactly which token classes defeat every rule.

**This is the method this document itself demands, applied to this document's own
favourite example.** A hypothesis must not become a load-bearing assumption
merely because it is written down in the vision.

**Lexical synonymy (⚠️ CLAIMED residue — NOT yet earned, see #873).** Deciding
that `fond-`, `kre-`, `starig-`, and `establ-` denote the same relation *in a
given context* is plausibly not derivable from morphology. We approximate it with
a hand-seeded verb-class ontology — a lookup table pretending to be knowledge:
thin coverage, no generalization to an unenumerated root.

**But this document was calling it *"confirmed"* while never running the one
deterministic method that most directly targets it.** The store already holds
**2,864 curated ReVo `SINONIMO` edges**, and they are **not wired into first-stage
retrieval** — the live query expander (#855) is morphology-only. So the
deterministic ceiling for synonymy has never been measured. #873 runs that test:
OR-expand the 147-question synonym residue with ReVo synonyms and measure. Until
it does, this is a *claim*, not a residue — and note that the residue's own
headline example, "posedis" ⇄ "vendas" (*own* vs *sell*), **is not synonymy at
all** but a converse/world-knowledge relation, which no synonym table would bridge
(#874). This is the exact error the proper-noun entry above warns about, committed
here: a hypothesis made load-bearing because it was written down.

**Word-sense disambiguation (suspected residue).** Which sense of a root is in
play depends on context in ways the AST records but does not resolve.

**Cross-sentence coreference (suspected residue).** Resolving `li` / `tiu` / `ĝi`
to an antecedent in a *different* sentence requires discourse modeling. A large
share of the corpus has pronoun subjects, and for those sentences the answer to
"who" is simply not present in the sentence at all.

**Ranking among structurally valid candidates (open question).** When many
passages satisfy every structural constraint, something must break the tie. It is
genuinely unclear how much of this is deterministic — information-theoretic
specificity, which we have not yet exploited — versus learned calibration. This
is unresolved, and it is the most interesting open question in the project.

**Not residue — deterministic, and being built:** transitive inference over facts,
type hierarchies, constraint propagation, aggregation and quantifiers, arithmetic,
path-finding, task decomposition. These have working symbolic implementations.
Whether they hold up under real questions is a measurement problem, not a theory
problem.

## What we still believe

The architectural commitments below survive the reframing, because they serve
boundary-discovery rather than assuming its answer.

**The AST is the universal contract.** Every component consumes and produces
role-annotated ASTs. This is what makes the boundary *visible*: you can see
exactly what structure was derived by rule and what was left underdetermined.

**The pipeline state is a *thought*, and it is dual-layer.** The orchestrator
passes an immutable context between stages: a **SymbolicLayer** — everything
expressible as Esperanto AST or AST-derived structure (the question AST,
retrieved passage ASTs, extracted fact triples, answer segments, citations) —
and a **LatentLayer** — dense representations with no clean AST encoding.
Everything a module contributes must land in one of those two layers. Side
channels — private tables, ad-hoc flags, regex over raw question text — are
contract violations, and we have measured what they cost: on 2026-07-18 three
subsystems were found dead in production from one privately-drifted schema,
silently (#881).

**Every capability is a dual-track slot.** A slot has a deterministic
implementation (required — it is the floor being measured) and optionally a
learned one, composed in one of three modes: *shadow* (runs on the same
thoughts, output recorded but unused — measurement without shipping), *enrich*
(fills only what the deterministic pass left underdetermined, tagged per node),
or *replace* (earned through the merge gate). This is how "attempt it
deterministically first" becomes an architecture instead of a slogan.

**Any thought is decodable at any stage.** Because the grammar is regular and
the core root vocabulary is small, every symbolic enrichment can be rendered
back into readable Esperanto deterministically — the deparser for sentence
ASTs, glossers for facts, candidates, and plans, each item tagged rule-vs-model.
The universal thought decoder is both the observability instrument ("watch the
AST evolve through each layer", below) and a constraint on learned components:
they must speak thought-language — decodable symbols with attribution — never
opaque state.

**The contract is enforced, not merely documented.** Conformance tests run over
every stage (immutability, delta discipline, decodability, attribution, loud
failure). Optional modules run default-off until they pass the suite and carry
a number. A contract that is not tested is a naming convention.

**Attribution is built in, not post-hoc.** Each AST node tracks whether it came
from a rule or a model. Explainability does not require zero learned parameters —
it requires *decomposable contributions*. A prediction that is "77% deterministic
rule, 23% learned adjustment" is explainable in a way a monolithic model is not.

**Output is grammatically correct by construction.** The linearizer converts AST
back to text by rule, not by generation. Correct grammar is not something the
system learns to usually do; it is something it cannot fail to do.

**Answers are grounded and cited.** Retrieval operates over structure, and answers
are extracted from retrieved evidence with citation trails. The system cannot
assert a fact it did not retrieve.

**Small enough to train on a laptop.** If the residue is small, the models that
cover it are small. This is a *prediction* of the thesis, not an axiom of it — and
if the residue turns out to be large, that is a finding too, and an honest one.

## Success looks like

Not a leaderboard score. Success is being able to say, with evidence:

- Here is the set of capabilities we implemented deterministically, and how well
  each works.
- Here is precisely where each one breaks, stated as a property of the problem.
- Here is the learned component we added to cover that specific gap, how large it
  is, and how much of the gap it actually recovered.
- Here is the resulting map of the deterministic/learned boundary for a language
  with a perfectly regular grammar.

And, as a working artifact: you can ask a question in Esperanto, watch the AST
evolve through each layer, see which evidence was retrieved and why, see what came
from rules versus models, and get a grammatically perfect, grounded, cited answer.

## What this is not

- **Not eliminating machine learning.** ML is the tool for the residue. The point
  is to know what the residue *is* before reaching for it.
- **Not pure rule-based AI.** It is a hybrid whose boundary is measured rather
  than assumed.
- **Not a translation system.** Klareco works natively in Esperanto.
- **Not a grammar checker.** Grammar is infrastructure, not the goal.
- **Not competing with frontier models on English.** It is answering a different
  question.

## See also

- `DESIGN.md` — the architecture as it actually is today, including what is broken
  and what is unmeasured.
- `CLAUDE.md` — working conventions.
