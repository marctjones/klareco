#!/usr/bin/env python3
"""
Multi-Stage AST Enrichment Pipeline Demo

This script demonstrates how a single sentence flows through the complete
enrichment pipeline from parsing to multi-model orchestration.

Stages demonstrated:
- Stage 0: Parser (deterministic, 0 params)
- Stage 1: Semantic Embeddings (learned, ~320K params)
- M1: Selectional Preference (mock, will be 10M params when trained)
- M2: Taxonomic Relations (mock, will be 10M params when trained)
- M3: Discourse Coherence (mock, will be 30-50M params when trained)
- M4: Multi-Model Orchestration (0 params, coordination only)

At each stage, we display:
- What annotations were added
- Memory footprint
- Human-readable explanation
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from klareco.parser import parse
from klareco.enriched_ast import (
    EnrichedAST,
    SelectionalAnnotation,
    TaxonomicAnnotation,
    DiscourseAnnotation,
    MultiModelAnnotation,
    AnnotationMetadata
)
from klareco.models import MockSelectionalModel
from klareco.thought_decoder import ThoughtDecoder
import time


def print_separator(title: str, char: str = "="):
    """Print a section separator."""
    width = 70
    print(f"\n{char * width}")
    print(f"{title:^{width}}")
    print(f"{char * width}\n")


def print_memory_footprint(enriched: EnrichedAST):
    """Display memory footprint breakdown."""
    footprint = enriched.memory_footprint()
    print("Memory Footprint:")
    for component, bytes_used in footprint.items():
        if component == 'total':
            print(f"  {'TOTAL':20s}: {bytes_used:6d} bytes ({bytes_used / 1024:.2f} KB)")
        else:
            print(f"  {component:20s}: {bytes_used:6d} bytes")


def stage_0_parsing():
    """Stage 0: Deterministic parsing."""
    print_separator("STAGE 0: PARSER (Deterministic, 0 params)")

    text = "La hundo manĝas viandon."
    print(f"Input: {text}\n")

    # Parse to AST
    parser_ast = parse(text)
    enriched = EnrichedAST.from_parser_output(parser_ast, text)

    # Display results
    print("Parsed Structure:")
    print(f"  tipo:             {enriched.tipo}")
    print(f"  fraztipo:         {enriched.fraztipo}")
    print(f"  negita:           {enriched.negita}")
    if enriched.subjekto:
        print(f"  subjekto.radiko:  {enriched.subjekto.get('kerno', {}).get('radiko')}")
    if enriched.verbo:
        print(f"  verbo.radiko:     {enriched.verbo.get('radiko')}")
    if enriched.objekto:
        print(f"  objekto.radiko:   {enriched.objekto.get('kerno', {}).get('radiko')}")
    print(f"  stages_applied:   {enriched.stages_applied}")

    # Memory
    print()
    print_memory_footprint(enriched)

    # Explanation
    print("\nExplanation:")
    decoder = ThoughtDecoder()
    explanation = decoder.decode(enriched)
    print(explanation)

    return enriched


def stage_1_semantic_embeddings(enriched: EnrichedAST) -> EnrichedAST:
    """Stage 1: Semantic embeddings (would use trained model in production)."""
    print_separator("STAGE 1: SEMANTIC EMBEDDINGS (~320K params)")

    print("NOTE: This stage would normally load a trained CompositionalEmbedding model:")
    print("  semantic_model = CompositionalEmbedding.load('models/root_embeddings/best_model.pt')")
    print("  enriched = semantic_model.enrich(enriched)")
    print("\nFor this demo, we'll simulate the enrichment:\n")

    # Simulate Stage 1 enrichment
    import numpy as np
    enriched_copy = enriched.clone()

    # Add mock sentence embedding (would be computed by model)
    enriched_copy.sentence_embedding = np.random.randn(128).astype(np.float32)
    enriched_copy.known_roots = ['hund', 'manĝ', 'viand']
    enriched_copy.unknown_roots = []
    enriched_copy.stages_applied.add('stage1_semantic')

    print("Semantic Embeddings Added:")
    print(f"  sentence_embedding.shape: {enriched_copy.sentence_embedding.shape}")
    print(f"  sentence_embedding.dtype: {enriched_copy.sentence_embedding.dtype}")
    print(f"  known_roots:              {enriched_copy.known_roots}")
    print(f"  unknown_roots:            {enriched_copy.unknown_roots}")
    print(f"  embedding_coverage:       100%")
    print(f"  stages_applied:           {enriched_copy.stages_applied}")

    print()
    print_memory_footprint(enriched_copy)

    print("\nExplanation:")
    print("  All roots in this sentence are known (hund, manĝ, viand).")
    print("  Sentence embedding computed compositionally from root + affix embeddings.")
    print("  This embedding will be used by downstream models for semantic reasoning.")

    return enriched_copy


def m1_selectional_preference(enriched: EnrichedAST) -> EnrichedAST:
    """M1: Selectional Preference (mock implementation)."""
    print_separator("M1: SELECTIONAL PREFERENCE (Mock, will be 10M params)")

    print("NOTE: Using MockSelectionalModel until M1 is trained.\n")

    # Use mock model
    model = MockSelectionalModel()
    enriched_m1 = model.enrich(enriched)

    print("Selectional Preference Scores:")
    print(f"  subject_verb_score:   {enriched_m1.selectional.subject_verb_score:.3f}")
    print(f"  verb_object_score:    {enriched_m1.selectional.verb_object_score:.3f}")
    print(f"  triple_plausibility:  {enriched_m1.selectional.triple_plausibility:.3f}")
    print(f"  stages_applied:       {enriched_m1.stages_applied}")

    print("\nMetadata:")
    meta = enriched_m1.selectional.meta
    print(f"  model_name:     {meta.model_name}")
    print(f"  model_version:  {meta.model_version}")
    print(f"  confidence:     {meta.confidence:.3f}")
    print(f"  compute_time:   {meta.compute_time_ms:.1f} ms")

    print()
    print_memory_footprint(enriched_m1)

    print("\nExplanation:")
    print(model.explain(enriched_m1))

    return enriched_m1


def m2_taxonomic_relations(enriched: EnrichedAST) -> EnrichedAST:
    """M2: Taxonomic Relations (mock implementation)."""
    print_separator("M2: TAXONOMIC RELATIONS (Mock, will be 10M params)")

    print("NOTE: Using mock implementation until M2 is trained.\n")

    # Mock M2 annotations
    enriched_m2 = enriched.clone()

    enriched_m2.taxonomic = TaxonomicAnnotation(
        hypernyms={
            'hund': [('mamul', 0.95), ('best', 0.92), ('animalo', 0.88)],
            'viand': [('manĝaĵ', 0.90), ('produkto', 0.85)]
        },
        hyponyms={
            'hund': [('ĉashund', 0.85), ('gardohund', 0.80)]
        },
        concept_clusters={
            'hund': 'animals.mammals',
            'viand': 'food.meat'
        },
        taxonomic_similarity=0.75,
        meta=AnnotationMetadata(
            model_name="mock_taxonomic",
            model_version="0.1.0",
            confidence=0.5,
            compute_time_ms=2.0,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            parameters={"mode": "mock"}
        )
    )

    enriched_m2.stages_applied.add('taxonomic')

    print("Taxonomic Relations:")
    print(f"  hypernyms['hund']:  {enriched_m2.taxonomic.hypernyms.get('hund', [])}")
    print(f"  hypernyms['viand']: {enriched_m2.taxonomic.hypernyms.get('viand', [])}")
    print(f"  hyponyms['hund']:   {enriched_m2.taxonomic.hyponyms.get('hund', [])}")
    print(f"  concept_clusters:")
    for root, cluster in enriched_m2.taxonomic.concept_clusters.items():
        print(f"    {root:10s} → {cluster}")
    print(f"  taxonomic_similarity: {enriched_m2.taxonomic.taxonomic_similarity:.3f}")
    print(f"  stages_applied:       {enriched_m2.stages_applied}")

    print()
    print_memory_footprint(enriched_m2)

    print("\nExplanation:")
    print("  Mock Taxonomic Relations:")
    print("    hund IS-A: mamul (0.950), best (0.920), animalo (0.880)")
    print("    viand IS-A: manĝaĵ (0.900), produkto (0.850)")
    print("    Semantic clusters: animals.mammals (hund), food.meat (viand)")
    print("  (These are placeholder relations until M2 is trained)")

    return enriched_m2


def m3_discourse_coherence(enriched: EnrichedAST, previous_sentence: str = None) -> EnrichedAST:
    """M3: Discourse Coherence (mock implementation)."""
    print_separator("M3: DISCOURSE COHERENCE (Mock, will be 30-50M params)")

    if previous_sentence:
        print(f"Previous context: \"{previous_sentence}\"\n")
    else:
        print("No previous context (first sentence in document).\n")

    print("NOTE: Using mock implementation until M3 is trained.\n")

    # Mock M3 annotations
    enriched_m3 = enriched.clone()

    # Mock discourse annotation (would be computed by model)
    enriched_m3.discourse = DiscourseAnnotation(
        coherence_with_previous=0.35 if previous_sentence else None,
        discourse_relation=None if not previous_sentence else "topic_shift",
        relation_confidence=0.60 if previous_sentence else None,
        coreferences={},  # No pronouns to resolve
        discourse_embedding=None,  # Optional 256d embedding
        meta=AnnotationMetadata(
            model_name="mock_discourse",
            model_version="0.1.0",
            confidence=0.5,
            compute_time_ms=3.0,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            parameters={"mode": "mock", "context_window": 3}
        )
    )

    enriched_m3.stages_applied.add('discourse')

    print("Discourse Coherence:")
    if enriched_m3.discourse.coherence_with_previous is not None:
        print(f"  coherence_with_previous: {enriched_m3.discourse.coherence_with_previous:.3f} (LOW - topic shift)")
        print(f"  discourse_relation:      {enriched_m3.discourse.discourse_relation}")
        print(f"  relation_confidence:     {enriched_m3.discourse.relation_confidence:.3f}")
    else:
        print(f"  coherence_with_previous: None (first sentence)")
        print(f"  discourse_relation:      None")
    print(f"  coreferences:            {enriched_m3.discourse.coreferences}")
    print(f"  stages_applied:          {enriched_m3.stages_applied}")

    print()
    print_memory_footprint(enriched_m3)

    print("\nExplanation:")
    if previous_sentence:
        print(f"  Previous: \"{previous_sentence}\" (cat sleeps)")
        print(f"  Current:  \"La hundo manĝas viandon.\" (dog eats meat)")
        print(f"  → LOW coherence (0.350) - different topics (cat vs dog)")
        print(f"  → Discourse relation: topic_shift")
    else:
        print(f"  No previous context - this is the first sentence.")
    print("  (These are placeholder scores until M3 is trained)")

    return enriched_m3


def m4_multi_model_orchestration(enriched: EnrichedAST) -> EnrichedAST:
    """M4: Multi-Model Orchestration (mock implementation)."""
    print_separator("M4: MULTI-MODEL ORCHESTRATION (0 params, coordination)")

    print("NOTE: Using mock implementation until M4 is trained.\n")

    # Mock M4 orchestration
    enriched_m4 = enriched.clone()

    # Combine scores from M1, M2, M3
    selectional_score = enriched.selectional.triple_plausibility if enriched.selectional else 0.0
    taxonomic_score = enriched.taxonomic.taxonomic_similarity if enriched.taxonomic else 0.0
    discourse_score = enriched.discourse.coherence_with_previous if enriched.discourse and enriched.discourse.coherence_with_previous else 0.5

    # Weighted combination
    weights = {
        'selectional': 0.4,
        'taxonomic': 0.3,
        'discourse': 0.3
    }

    model_scores = {
        'selectional': selectional_score,
        'taxonomic': taxonomic_score,
        'discourse': discourse_score
    }

    combined_score = sum(model_scores[k] * weights[k] for k in weights.keys())

    enriched_m4.multi_model = MultiModelAnnotation(
        model_scores=model_scores,
        combined_score=combined_score,
        score_breakdown={
            'selectional': {'weight': weights['selectional'], 'contribution': model_scores['selectional'] * weights['selectional']},
            'taxonomic': {'weight': weights['taxonomic'], 'contribution': model_scores['taxonomic'] * weights['taxonomic']},
            'discourse': {'weight': weights['discourse'], 'contribution': model_scores['discourse'] * weights['discourse']}
        },
        active_models={'selectional', 'taxonomic', 'discourse'},
        meta=AnnotationMetadata(
            model_name="mock_orchestrator",
            model_version="0.1.0",
            confidence=0.8,
            compute_time_ms=0.5,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            parameters={"weights": weights}
        )
    )

    enriched_m4.stages_applied.add('multi_model')

    print("Multi-Model Orchestration:")
    print(f"  model_scores:")
    for model_name, score in model_scores.items():
        contribution = score * weights[model_name]
        print(f"    {model_name:12s}: {score:.3f} (weight {weights[model_name]:.1f}, contribution {contribution:.3f})")
    print(f"  combined_score:  {combined_score:.3f}")
    print(f"  active_models:   {enriched_m4.multi_model.active_models}")
    print(f"  stages_applied:  {enriched_m4.stages_applied}")

    print()
    print_memory_footprint(enriched_m4)

    print("\nExplanation:")
    print(f"  Combined Score Calculation:")
    print(f"    (selectional: {selectional_score:.3f} × 0.4) + (taxonomic: {taxonomic_score:.3f} × 0.3) + (discourse: {discourse_score:.3f} × 0.3)")
    print(f"    = {combined_score:.3f}")
    print(f"  Top contributor: selectional (plausibility)")

    return enriched_m4


def demonstrate_serialization(enriched: EnrichedAST):
    """Demonstrate serialization capabilities."""
    print_separator("SERIALIZATION DEMONSTRATION")

    # Full JSON serialization
    json_str = enriched.to_json(indent=2)
    print(f"Full JSON serialization:")
    print(f"  Size: {len(json_str)} characters")
    print(f"  First 500 chars:")
    print(f"  {json_str[:500]}...")

    # Selective serialization (for training data)
    print(f"\nSelective serialization (M1 training needs):")
    training_dict = enriched.to_dict(include=['parser_ast', 'sentence_embedding', 'selectional'])
    print(f"  Keys included: {list(training_dict.keys())}")
    print(f"  Excluded: taxonomic, discourse, multi_model (not needed for M1)")

    # Round-trip test
    print(f"\nRound-trip test:")
    loaded = EnrichedAST.from_json(json_str)
    print(f"  ✓ Original text:        {enriched.original_text}")
    print(f"  ✓ Loaded text:          {loaded.original_text}")
    print(f"  ✓ Selectional preserved: {loaded.selectional.triple_plausibility:.3f}")
    print(f"  ✓ Taxonomic preserved:   {len(loaded.taxonomic.hypernyms)} hypernym entries")
    print(f"  ✓ Discourse preserved:   {loaded.discourse.coherence_with_previous}")
    print(f"  ✓ Multi-model preserved: {loaded.multi_model.combined_score:.3f}")


def main():
    """Run the complete enrichment pipeline demonstration."""
    print_separator("MULTI-STAGE AST ENRICHMENT PIPELINE DEMO", "=")
    print("This demo shows one sentence flowing through the complete")
    print("Klareco enrichment pipeline from parsing to multi-model orchestration.")

    # Stage 0: Parsing
    enriched = stage_0_parsing()

    # Stage 1: Semantic Embeddings
    enriched = stage_1_semantic_embeddings(enriched)

    # M1: Selectional Preference
    enriched = m1_selectional_preference(enriched)

    # M2: Taxonomic Relations
    enriched = m2_taxonomic_relations(enriched)

    # M3: Discourse Coherence (with mock previous sentence)
    previous_sentence = "La kato dormas."
    enriched = m3_discourse_coherence(enriched, previous_sentence)

    # M4: Multi-Model Orchestration
    enriched = m4_multi_model_orchestration(enriched)

    # Demonstrate serialization
    demonstrate_serialization(enriched)

    # Final summary
    print_separator("SUMMARY: WHAT WE DEMONSTRATED")
    print("""
1. ✓ EnrichedAST is the universal data container
   - Preserves deterministic parser output
   - Accumulates learned annotations from each model
   - Immutable progression (clone() before modifying)

2. ✓ Each model adds specific annotations:
   - M1 (Selectional): Plausibility scores (~200 bytes)
   - M2 (Taxonomic): IS-A relations (~500 bytes)
   - M3 (Discourse): Coherence + coreference (~300 bytes)
   - M4 (Orchestration): Combined scores (~300 bytes)

3. ✓ Memory-efficient design:
   - Total: ~2.2 KB per sentence (minimal configuration)
   - Can scale to ~6.4 KB with optional embeddings
   - For 4.2M corpus: 9-27 GB total

4. ✓ Fully explainable:
   - Every annotation has metadata (model version, confidence)
   - ThoughtDecoder generates human-readable explanations
   - Can trace which model contributed to final decision

5. ✓ Serialization ready:
   - Full JSON serialization for storage
   - Selective serialization for training efficiency
   - Round-trip preserves all information

6. ✓ Type-safe and composable:
   - Models build on each other's outputs
   - SemanticModel interface ensures consistency
   - Optional fields prevent errors when stages are skipped

NEXT STEPS:
- Train M1 (Selectional Preference) using this infrastructure
- Train M2 (Taxonomic Relations)
- Train M3 (Discourse Coherence)
- Implement M4 (Multi-Model Orchestration)

The infrastructure is ready! 🚀
""")


if __name__ == '__main__':
    main()
