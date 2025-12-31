#!/usr/bin/env python3
"""
Stage 2 Grammar Demo: Deterministic Grammatical Similarity Adjustment

This demo showcases how the GrammaticalAdjuster uses AST annotations
to adjust semantic similarity based on grammatical features.

Key insight: The parser already extracts grammatical features into the AST.
Why learn what we already know? Stage 2 uses ZERO learned parameters.

Grammatical features from AST:
  - negita: True/False (negation)
  - tempo: 'pasinteco', 'prezenco', 'futuro' (tense)
  - fraztipo: 'deklaro', 'demando', 'ordono' (sentence type)
  - modo: 'indikativo', 'kondicxa', 'vola', 'infinitivo' (mood)

Usage:
    python scripts/demo_stage2_grammar.py
    python scripts/demo_stage2_grammar.py --interactive
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F


def print_header(title: str):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_subheader(title: str):
    """Print a subsection header."""
    print(f"\n--- {title} ---")


def cosine_sim(emb1, emb2) -> float:
    """Calculate cosine similarity between two embeddings."""
    if emb1 is None or emb2 is None:
        return 0.0
    return F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()


def demo_concept():
    """Explain the Stage 2 concept."""
    print_header("Stage 2: Deterministic Grammatical Adjustment")

    print("""
THE PROBLEM:
  Stage 1 embeddings capture semantic content (root meanings).
  "La kato dormas" and "La kato ne dormas" share the same roots
  (kato, dorm-), so Stage 1 gives them similarity ~1.0.

  But they have OPPOSITE meanings!

THE INSIGHT:
  The parser ALREADY extracts grammatical features into the AST:
  - negita: True/False
  - tempo: past/present/future
  - fraztipo: statement/question/command
  - modo: indicative/conditional/etc.

THE SOLUTION:
  GrammaticalAdjuster uses these AST annotations directly.
  No learning needed - pure deterministic rules.

  Stage 2 has ZERO learned parameters.

THE PIPELINE:
  Text -> Parser (Stage 0) -> AST with grammar annotations
       -> SemanticModel (Stage 1) -> semantic similarity
       -> GrammaticalAdjuster (Stage 2) -> adjusted similarity
""")


def demo_negation():
    """Demonstrate negation detection and adjustment."""
    from klareco import SemanticPipeline, GrammaticalAdjuster

    print_header("Negation: Polarity Flip")

    pipeline = SemanticPipeline.load()
    adjuster = GrammaticalAdjuster()

    pairs = [
        ("La kato dormas.", "La kato ne dormas."),
        ("Mi amas vin.", "Mi ne amas vin."),
        ("Li venis.", "Li ne venis."),
        ("Estas bela tago.", "Ne estas bela tago."),
    ]

    print("\nNegation flips meaning. Adjustment factor: -0.8")
    print("(Multiplies similarity by -0.8, reversing the sign)\n")

    print(f"{'Sentence 1':<30} {'Sentence 2':<30} {'Stage 1':>8} {'Stage 2':>8}")
    print("-" * 80)

    for s1, s2 in pairs:
        e1 = pipeline.for_retrieval(s1)
        e2 = pipeline.for_retrieval(s2)

        semantic_sim = cosine_sim(e1.sentence_embedding, e2.sentence_embedding)
        result = adjuster.adjust_with_explanation(e1, e2, semantic_sim)

        s1_short = s1[:28] + ".." if len(s1) > 30 else s1
        s2_short = s2[:28] + ".." if len(s2) > 30 else s2

        print(f"{s1_short:<30} {s2_short:<30} {semantic_sim:>8.3f} {result.adjusted_similarity:>8.3f}")

    print("\nNote: Stage 1 sees same roots -> high similarity")
    print("      Stage 2 detects negation -> flips to negative similarity")


def demo_tense():
    """Demonstrate tense comparison and adjustment."""
    from klareco import SemanticPipeline, GrammaticalAdjuster

    print_header("Tense: Temporal Distance")

    pipeline = SemanticPipeline.load()
    adjuster = GrammaticalAdjuster()

    # Present as baseline
    present = "Mi manĝas."
    past = "Mi manĝis."
    future = "Mi manĝos."

    print("\nTense affects meaning but doesn't flip it.")
    print("Adjacent tenses (present<->past, present<->future): x0.8")
    print("Distant tenses (past<->future): x0.6\n")

    e_present = pipeline.for_retrieval(present)
    e_past = pipeline.for_retrieval(past)
    e_future = pipeline.for_retrieval(future)

    comparisons = [
        ("Present vs Past", e_present, e_past, present, past),
        ("Present vs Future", e_present, e_future, present, future),
        ("Past vs Future", e_past, e_future, past, future),
    ]

    print(f"{'Comparison':<20} {'Sent 1':<15} {'Sent 2':<15} {'Stage 1':>8} {'Stage 2':>8} {'Factor':>8}")
    print("-" * 84)

    for name, e1, e2, s1, s2 in comparisons:
        semantic_sim = cosine_sim(e1.sentence_embedding, e2.sentence_embedding)
        result = adjuster.adjust_with_explanation(e1, e2, semantic_sim)

        factor = result.adjustments.get('tense', 1.0)
        print(f"{name:<20} {s1:<15} {s2:<15} {semantic_sim:>8.3f} {result.adjusted_similarity:>8.3f} {factor:>8.2f}")

    # Show AST tempo values
    print("\nAST tempo values:")
    print(f"  '{present}' -> tempo: {e_present.tempo}")
    print(f"  '{past}' -> tempo: {e_past.tempo}")
    print(f"  '{future}' -> tempo: {e_future.tempo}")


def demo_sentence_type():
    """Demonstrate sentence type comparison."""
    from klareco import SemanticPipeline, GrammaticalAdjuster

    print_header("Sentence Type: Illocutionary Force")

    pipeline = SemanticPipeline.load()
    adjuster = GrammaticalAdjuster()

    statement = "Vi parolas Esperanton."
    question = "Cxu vi parolas Esperanton?"
    command = "Parolu Esperanton!"

    print("\nSentence type changes the speech act:")
    print("  - deklaro (statement): asserting a fact")
    print("  - demando (question): requesting information")
    print("  - ordono (command): directing action")
    print("\nMismatch factor: x0.7\n")

    e_statement = pipeline.for_retrieval(statement)
    e_question = pipeline.for_retrieval(question)
    e_command = pipeline.for_retrieval(command)

    comparisons = [
        ("Statement vs Question", e_statement, e_question, statement, question),
        ("Statement vs Command", e_statement, e_command, statement, command),
        ("Question vs Command", e_question, e_command, question, command),
    ]

    print(f"{'Comparison':<22} {'Stage 1':>8} {'Stage 2':>8} {'Type 1':<12} {'Type 2':<12}")
    print("-" * 70)

    for name, e1, e2, s1, s2 in comparisons:
        semantic_sim = cosine_sim(e1.sentence_embedding, e2.sentence_embedding)
        result = adjuster.adjust_with_explanation(e1, e2, semantic_sim)

        print(f"{name:<22} {semantic_sim:>8.3f} {result.adjusted_similarity:>8.3f} {e1.fraztipo:<12} {e2.fraztipo:<12}")

    print("\nAST fraztipo values:")
    print(f"  '{statement}' -> fraztipo: {e_statement.fraztipo}")
    print(f"  '{question}' -> fraztipo: {e_question.fraztipo}")
    print(f"  '{command}' -> fraztipo: {e_command.fraztipo}")


def demo_combined():
    """Demonstrate multiple adjustments combining."""
    from klareco import SemanticPipeline, GrammaticalAdjuster

    print_header("Combined Adjustments")

    pipeline = SemanticPipeline.load()
    adjuster = GrammaticalAdjuster()

    print("\nMultiple grammatical differences stack multiplicatively.\n")

    pairs = [
        # Negation only
        ("Li venas.", "Li ne venas.", "negation only"),
        # Tense only
        ("Li venas.", "Li venis.", "tense only"),
        # Negation + tense
        ("Li venas.", "Li ne venis.", "negation + tense"),
        # Question + tense
        ("Li venas.", "Cxu li venis?", "question + tense"),
        # Negation + question + tense
        ("Li venas.", "Cxu li ne venis?", "neg + question + tense"),
    ]

    print(f"{'Sent 1':<15} {'Sent 2':<20} {'Description':<22} {'S1':>6} {'S2':>6} {'Adjustments'}")
    print("-" * 95)

    for s1, s2, desc in pairs:
        e1 = pipeline.for_retrieval(s1)
        e2 = pipeline.for_retrieval(s2)

        semantic_sim = cosine_sim(e1.sentence_embedding, e2.sentence_embedding)
        result = adjuster.adjust_with_explanation(e1, e2, semantic_sim)

        adj_str = ", ".join(f"{k}:{v}" for k, v in result.adjustments.items()) or "(none)"

        print(f"{s1:<15} {s2:<20} {desc:<22} {semantic_sim:>6.3f} {result.adjusted_similarity:>6.3f} {adj_str}")


def demo_custom_factors():
    """Demonstrate custom adjustment factors."""
    from klareco import SemanticPipeline, GrammaticalAdjuster

    print_header("Custom Adjustment Factors")

    pipeline = SemanticPipeline.load()

    print("\nDefault factors:")
    default = GrammaticalAdjuster()
    for name, value in default.adjustments.items():
        print(f"  {name}: {value}")

    print("\nCustom factors (stronger negation, weaker tense):")
    custom = GrammaticalAdjuster(adjustments={
        'negation_mismatch': -1.0,  # Full polarity flip
        'tense_adjacent': 0.95,     # Almost no tense penalty
        'tense_distant': 0.9,
    })
    for name, value in custom.adjustments.items():
        print(f"  {name}: {value}")

    # Compare results
    s1 = "Mi laboras."
    s2_neg = "Mi ne laboras."
    s2_tense = "Mi laboris."

    e1 = pipeline.for_retrieval(s1)
    e2_neg = pipeline.for_retrieval(s2_neg)
    e2_tense = pipeline.for_retrieval(s2_tense)

    sem_neg = cosine_sim(e1.sentence_embedding, e2_neg.sentence_embedding)
    sem_tense = cosine_sim(e1.sentence_embedding, e2_tense.sentence_embedding)

    print(f"\n{'Comparison':<25} {'Semantic':>10} {'Default':>10} {'Custom':>10}")
    print("-" * 60)

    # Negation
    def_neg = default.adjust(e1, e2_neg, sem_neg)
    cust_neg = custom.adjust(e1, e2_neg, sem_neg)
    print(f"{'Negation (laboras/ne laboras)':<25} {sem_neg:>10.3f} {def_neg:>10.3f} {cust_neg:>10.3f}")

    # Tense
    def_tense = default.adjust(e1, e2_tense, sem_tense)
    cust_tense = custom.adjust(e1, e2_tense, sem_tense)
    print(f"{'Tense (laboras/laboris)':<25} {sem_tense:>10.3f} {def_tense:>10.3f} {cust_tense:>10.3f}")


def demo_comparison_api():
    """Demonstrate the compare() API for feature inspection."""
    from klareco import SemanticPipeline, GrammaticalAdjuster

    print_header("Feature Comparison API")

    pipeline = SemanticPipeline.load()
    adjuster = GrammaticalAdjuster()

    s1 = "La hundo kuris rapide."
    s2 = "Cxu la kato ne kuras?"

    print(f"\nComparing:")
    print(f"  1: {s1}")
    print(f"  2: {s2}")

    e1 = pipeline.for_retrieval(s1)
    e2 = pipeline.for_retrieval(s2)

    comparison = adjuster.compare(e1, e2)

    print("\nFeature comparison (adjuster.compare()):")
    for feature, data in comparison.items():
        print(f"\n  {feature}:")
        for key, value in data.items():
            print(f"    {key}: {value}")

    # Now show adjustment
    semantic_sim = cosine_sim(e1.sentence_embedding, e2.sentence_embedding)
    result = adjuster.adjust_with_explanation(e1, e2, semantic_sim)

    print(f"\nSimilarity adjustment:")
    print(f"  Semantic (Stage 1): {result.original_similarity:.3f}")
    print(f"  Adjusted (Stage 2): {result.adjusted_similarity:.3f}")
    print(f"  Adjustments applied: {result.adjustments}")


def interactive_mode():
    """Interactive demo mode."""
    from klareco import SemanticPipeline, GrammaticalAdjuster

    print_header("Interactive Mode")

    pipeline = SemanticPipeline.load()
    adjuster = GrammaticalAdjuster()

    print("\nEnter two sentences to compare their grammatical similarity.")
    print("Commands:")
    print("  <sentence 1> | <sentence 2>  - Compare two sentences")
    print("  factors                       - Show adjustment factors")
    print("  quit                          - Exit")
    print()

    while True:
        try:
            user_input = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGxis revido!")
            break

        if not user_input:
            continue

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Gxis revido!")
            break

        if user_input.lower() == 'factors':
            print("\nAdjustment factors:")
            for name, value in adjuster.adjustments.items():
                print(f"  {name}: {value}")
            print()
            continue

        if '|' not in user_input:
            print("  Use: <sentence 1> | <sentence 2>")
            print("  Example: La kato dormas. | La kato ne dormas.")
            continue

        parts = user_input.split('|')
        if len(parts) != 2:
            print("  Please provide exactly two sentences separated by |")
            continue

        s1, s2 = parts[0].strip(), parts[1].strip()

        if not s1 or not s2:
            print("  Both sentences must be non-empty")
            continue

        try:
            e1 = pipeline.for_retrieval(s1)
            e2 = pipeline.for_retrieval(s2)

            semantic_sim = cosine_sim(e1.sentence_embedding, e2.sentence_embedding)
            result = adjuster.adjust_with_explanation(e1, e2, semantic_sim)
            comparison = adjuster.compare(e1, e2)

            print(f"\n  Sentence 1: {s1}")
            print(f"    negita: {e1.negita}, tempo: {e1.tempo}, fraztipo: {e1.fraztipo}")

            print(f"\n  Sentence 2: {s2}")
            print(f"    negita: {e2.negita}, tempo: {e2.tempo}, fraztipo: {e2.fraztipo}")

            print(f"\n  Stage 1 (semantic): {semantic_sim:.3f}")
            print(f"  Stage 2 (adjusted): {result.adjusted_similarity:.3f}")

            if result.adjustments:
                print(f"  Adjustments: {result.adjustments}")
            else:
                print("  Adjustments: (none - grammatically identical)")

            print()

        except Exception as e:
            print(f"  Error: {e}")
            continue


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Stage 2 Grammar Demo")
    parser.add_argument('--interactive', '-i', action='store_true',
                        help="Run in interactive mode")
    parser.add_argument('--section', '-s', type=str,
                        choices=['concept', 'negation', 'tense', 'type',
                                'combined', 'custom', 'api', 'all'],
                        default='all',
                        help="Which section to run")
    args = parser.parse_args()

    print("=" * 70)
    print(" KLARECO STAGE 2: GRAMMATICAL ADJUSTMENT DEMO")
    print(" Zero-parameter deterministic grammar rules")
    print("=" * 70)

    if args.interactive:
        interactive_mode()
        return

    try:
        from klareco import SemanticPipeline
        pipeline = SemanticPipeline.load()
        print("\nModels loaded successfully.")
    except Exception as e:
        print(f"\nError loading models: {e}")
        print("Make sure Stage 1 models are trained.")
        return

    sections = {
        'concept': demo_concept,
        'negation': demo_negation,
        'tense': demo_tense,
        'type': demo_sentence_type,
        'combined': demo_combined,
        'custom': demo_custom_factors,
        'api': demo_comparison_api,
    }

    if args.section == 'all':
        for name, func in sections.items():
            try:
                func()
            except Exception as e:
                print(f"\nError in {name}: {e}")
    else:
        sections[args.section]()

    print_header("Summary")
    print("""
Stage 2 Key Points:

1. ZERO LEARNED PARAMETERS
   - All grammatical features come from the parser (Stage 0)
   - GrammaticalAdjuster applies deterministic rules
   - No training needed, no overfitting possible

2. MULTIPLICATIVE ADJUSTMENTS
   - Negation: x-0.8 (flips polarity)
   - Tense adjacent: x0.8
   - Tense distant: x0.6
   - Mood mismatch: x0.5
   - Type mismatch: x0.7

3. PHILOSOPHY
   "Don't learn what you already know"
   - Parser extracts grammar -> use it directly
   - Focus learned capacity on what's actually unknown (reasoning)

Usage:
  from klareco import SemanticPipeline, GrammaticalAdjuster

  pipeline = SemanticPipeline.load()
  adjuster = GrammaticalAdjuster()

  e1 = pipeline.for_retrieval("La kato dormas.")
  e2 = pipeline.for_retrieval("La kato ne dormas.")

  semantic_sim = cosine_similarity(e1.embedding, e2.embedding)
  adjusted_sim = adjuster.adjust(e1, e2, semantic_sim)

  # semantic_sim: ~1.0 (same roots)
  # adjusted_sim: ~-0.8 (negation detected!)
""")

    print("\nRun with --interactive for live testing!")


if __name__ == '__main__':
    main()
