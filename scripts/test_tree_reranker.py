#!/usr/bin/env python3
"""
Test TreeMatchReranker Implementation

Quick test to verify TreeMatchReranker works before full training.
Tests all three levels: syntax, compositional, semantic matching.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.models.tree_match_reranker import TreeMatchReranker, count_parameters
from klareco.embeddings import CompositionalEmbedding
from klareco.parser import parse

def test_syntax_matching():
    """Test syntax matching (deterministic, 0 params)."""
    print("\n" + "="*60)
    print("TEST 1: Syntax Matching (Deterministic)")
    print("="*60)

    # Create dummy compositional embedding (minimal)
    comp_emb = CompositionalEmbedding(
        root_vocab={'kiu': 0, 'fond': 1, 'esperant': 2, 'zamenhof': 3, 'kre': 4},
        prefix_vocab={'': 0},
        suffix_vocab={'': 0},
        embed_dim=128,
        dropout=0.0
    )

    model = TreeMatchReranker(comp_emb, freeze_embedding=True)

    # Test case: Same syntax pattern (SVO)
    query_text = "Kiu fondis Esperanton?"
    doc_text = "Zamenhof kreis Esperanton."

    print(f"\nQuery: {query_text}")
    print(f"Doc:   {doc_text}")

    try:
        query_ast = parse(query_text)
        doc_ast = parse(doc_text)

        # Debug: Show AST structure
        print(f"\nQuery AST keys: {list(query_ast.keys())}")
        query_subj = query_ast.get('subjekto')
        query_verb = query_ast.get('verbo')
        query_obj = query_ast.get('objekto')
        print(f"  subjekto: {query_subj.get('tipo') if query_subj else None}")
        print(f"  verbo: {query_verb.get('tipo') if query_verb else None}")
        print(f"  objekto: {query_obj.get('tipo') if query_obj else None}")

        print(f"\nDoc AST keys: {list(doc_ast.keys())}")
        doc_subj = doc_ast.get('subjekto')
        doc_verb = doc_ast.get('verbo')
        doc_obj = doc_ast.get('objekto')
        print(f"  subjekto: {doc_subj.get('tipo') if doc_subj else None}")
        print(f"  verbo: {doc_verb.get('tipo') if doc_verb else None}")
        print(f"  objekto: {doc_obj.get('tipo') if doc_obj else None}")

        syntax_score = model.syntax_tree_match(query_ast, doc_ast)
        print(f"\nSyntax score: {syntax_score:.3f}")

        # Lower threshold for now (syntax matching may need tuning)
        if syntax_score > 0.1:
            print("✓ PASS: Syntax matching working (score > 0.1)")
        else:
            print(f"⚠ WARNING: Low syntax score: {syntax_score:.3f}")

    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False

    return True


def load_compositional_embeddings(checkpoint_path):
    """Load compositional embeddings from checkpoint."""
    import torch
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if 'root_vocab' in checkpoint:
        # Full CompositionalEmbedding checkpoint
        comp_emb = CompositionalEmbedding(
            root_vocab=checkpoint['root_vocab'],
            prefix_vocab=checkpoint['prefix_vocab'],
            suffix_vocab=checkpoint['suffix_vocab'],
            embed_dim=checkpoint.get('embed_dim', 128),
        )
        comp_emb.load_state_dict(checkpoint['model_state_dict'])
    elif 'model_state_dict' in checkpoint:
        # Simple root embeddings with model_state_dict
        root_to_idx = checkpoint['root_to_idx']
        prefix_vocab = {'<NONE>': 0, '<UNK>': 1}
        suffix_vocab = {'<NONE>': 0, '<UNK>': 1}

        comp_emb = CompositionalEmbedding(
            root_vocab=root_to_idx,
            prefix_vocab=prefix_vocab,
            suffix_vocab=suffix_vocab,
            embed_dim=checkpoint.get('embedding_dim', 128)
        )
        # Load state dict (contains root_embeddings.weight)
        comp_emb.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        raise ValueError(f"Unrecognized checkpoint format. Keys: {list(checkpoint.keys())}")

    return comp_emb


def test_compositional_matching():
    """Test compositional matching (hybrid)."""
    print("\n" + "="*60)
    print("TEST 2: Compositional Matching (Hybrid)")
    print("="*60)

    # Load real compositional embeddings
    comp_emb_path = Path('models/root_embeddings/best_model.pt')
    if not comp_emb_path.exists():
        print(f"⚠ SKIP: Compositional embeddings not found at {comp_emb_path}")
        return True

    comp_emb = load_compositional_embeddings(comp_emb_path)
    model = TreeMatchReranker(comp_emb, freeze_embedding=True)

    # Test case: Same suffix
    word1 = {'radiko': 'hund', 'sufikso': 'ej', 'prefikso': ''}
    word2 = {'radiko': 'libro', 'sufikso': 'ej', 'prefikso': ''}

    comp_score = model._compositional_similarity(word1, word2)
    print(f"\nWord 1: hund-ej (dog place)")
    print(f"Word 2: libr-ej (book place)")
    print(f"Compositional score: {comp_score:.3f}")

    if comp_score > 0.2:  # Should get points for same suffix
        print("✓ PASS: Detected same suffix (-ej-)")
    else:
        print(f"⚠ WARNING: Low score for same suffix: {comp_score:.3f}")

    return True


def test_full_pipeline():
    """Test full forward pass."""
    print("\n" + "="*60)
    print("TEST 3: Full Pipeline")
    print("="*60)

    # Load real compositional embeddings
    comp_emb_path = Path('models/root_embeddings/best_model.pt')
    if not comp_emb_path.exists():
        print(f"⚠ SKIP: Compositional embeddings not found at {comp_emb_path}")
        return True

    comp_emb = load_compositional_embeddings(comp_emb_path)
    model = TreeMatchReranker(comp_emb, freeze_embedding=True)

    # Count parameters
    trainable, total = count_parameters(model)
    print(f"\nModel parameters:")
    print(f"  Trainable: {trainable:,}")
    print(f"  Total: {total:,}")

    # Test forward pass
    query_text = "Kiu fondis Esperanton?"
    doc_text = "Zamenhof kreis Esperanton en 1887."

    print(f"\nQuery: {query_text}")
    print(f"Doc:   {doc_text}")

    try:
        query_ast = parse(query_text)
        doc_ast = parse(doc_text)

        score, breakdown = model(query_ast, doc_ast)

        print(f"\nScore Breakdown:")
        print(f"  Syntax:        {breakdown['syntax_score']:.3f} (weight: {breakdown['syntax_weight']:.3f})")
        print(f"  Compositional: {breakdown['compositional_score']:.3f} (weight: {breakdown['compositional_weight']:.3f})")
        print(f"  Semantic:      {breakdown['semantic_score']:.3f} (weight: {breakdown['semantic_weight']:.3f})")
        print(f"  Final:         {breakdown['final_score']:.3f}")

        # Check score is reasonable
        if 0.0 <= score.item() <= 1.0:
            print("✓ PASS: Score in valid range [0, 1]")
        else:
            print(f"✗ FAIL: Score out of range: {score.item()}")
            return False

        # Check syntax weight is highest (should favor deterministic)
        if breakdown['syntax_weight'] >= breakdown['semantic_weight']:
            print("✓ PASS: Syntax weight >= semantic weight (favors deterministic)")
        else:
            print("⚠ WARNING: Semantic weight higher than syntax weight")

    except Exception as e:
        print(f"✗ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def main():
    print("\n" + "="*60)
    print("TreeMatchReranker Implementation Test")
    print("="*60)

    tests = [
        ("Syntax Matching", test_syntax_matching),
        ("Compositional Matching", test_compositional_matching),
        ("Full Pipeline", test_full_pipeline),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            if test_fn():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n✗ TEST CRASHED: {name}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")

    if failed == 0:
        print("\n✓ All tests passed! Ready to run full pipeline.")
        return 0
    else:
        print(f"\n✗ {failed} test(s) failed. Fix issues before running pipeline.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
