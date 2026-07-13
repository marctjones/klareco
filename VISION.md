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

**Proper-noun disambiguation (confirmed residue).** Esperanto has no
morphological proper-noun marker. A capitalized token whose morphology, position,
and function-word context give no signal — a novel surname that is also a common
word, sentence-initial position, an all-caps title, a quoted phrase — cannot be
resolved by any rule. Disambiguation requires distributional or world knowledge.
This is the cleanest confirmed residue we have.

**Lexical synonymy (confirmed residue, currently faked).** Deciding that `fond-`,
`kre-`, `starig-`, and `establ-` denote the same relation *in a given context* is
not derivable from morphology. We currently approximate this with a hand-seeded
verb-class ontology, which is a lookup table pretending to be knowledge: its
coverage is thin, and it cannot generalize to a root nobody enumerated. The
honest description is that this is a learned problem we are currently solving
with a list.

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
