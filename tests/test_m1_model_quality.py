"""
M1 Selectional Preference Model Quality Tests

Tests for trained M1 model performance and quality metrics.

Requires:
- Trained M1 model: models/m1_selectional/best_model.pt
- Stage 1 embeddings: models/root_embeddings/best_model.pt
- Test data: data/training/m1_selectional_hard_only/test.jsonl

Run:
    pytest tests/test_m1_model_quality.py -v
    pytest tests/test_m1_model_quality.py -v -m stage2
"""

import json
from pathlib import Path
from typing import Dict, List

import pytest
import torch

from klareco.models.m1_selectional import M1SelectionalPreference


@pytest.fixture(scope="module")
def m1_model():
    """Load trained M1 model and Stage 1 embeddings."""
    model_path = Path('models/m1_selectional/best_model.pt')
    stage1_path = Path('models/root_embeddings/best_model.pt')

    if not model_path.exists():
        pytest.skip(f"M1 model not found: {model_path}")

    if not stage1_path.exists():
        pytest.skip(f"Stage 1 model not found: {stage1_path}")

    # Load Stage 1 embeddings
    stage1_checkpoint = torch.load(stage1_path, map_location='cpu', weights_only=False)
    root_embeddings = stage1_checkpoint['model_state_dict']['embeddings.weight']
    root_to_idx = stage1_checkpoint['root_to_idx']

    # Load M1 model
    m1_checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model = M1SelectionalPreference(
        embedding_dim=m1_checkpoint['embedding_dim'],
        hidden_dim=m1_checkpoint['hidden_dim']
    )
    model.load_state_dict(m1_checkpoint['model_state_dict'])
    model.eval()

    return {
        'model': model,
        'root_embeddings': root_embeddings,
        'root_to_idx': root_to_idx,
        'best_accuracy': m1_checkpoint['best_accuracy'],
        'embedding_dim': m1_checkpoint['embedding_dim'],
        'hidden_dim': m1_checkpoint['hidden_dim']
    }


@pytest.fixture(scope="module")
def test_data():
    """Load test data."""
    test_path = Path('data/training/m1_selectional_hard_only/test.jsonl')

    if not test_path.exists():
        pytest.skip(f"Test data not found: {test_path}")

    examples = []
    with open(test_path) as f:
        for line in f:
            examples.append(json.loads(line))

    return examples


def get_embedding(root: str, root_embeddings: torch.Tensor,
                  root_to_idx: Dict[str, int]) -> torch.Tensor:
    """Get embedding for a root."""
    idx = root_to_idx.get(root.lower(), 0)
    return root_embeddings[idx]


def score_example(example: Dict, model: M1SelectionalPreference,
                  root_embeddings: torch.Tensor, root_to_idx: Dict[str, int]) -> Dict:
    """Score an example with M1 model."""
    # Get embeddings
    subj_emb = get_embedding(example['subject_root'], root_embeddings, root_to_idx).unsqueeze(0)
    verb_emb = get_embedding(example['verb_root'], root_embeddings, root_to_idx).unsqueeze(0)
    obj_emb = get_embedding(example['object_root'], root_embeddings, root_to_idx).unsqueeze(0)

    # Score
    with torch.no_grad():
        outputs = model(subj_emb, verb_emb, obj_emb)

    return {
        'triple_score': outputs['triple_score'].item(),
        'subj_verb_score': outputs['subj_verb_score'].item(),
        'verb_obj_score': outputs['verb_obj_score'].item(),
        'label': example['label']
    }


@pytest.mark.model_quality
@pytest.mark.stage2
@pytest.mark.requires_model
def test_model_architecture(m1_model):
    """Test that M1 model has expected architecture."""
    assert m1_model['embedding_dim'] == 64, "Expected 64-dim input embeddings"
    assert m1_model['hidden_dim'] == 128, "Expected 128-dim hidden layer"
    assert m1_model['model'].count_parameters() > 200000, "Expected ~222K parameters"


@pytest.mark.model_quality
@pytest.mark.stage2
@pytest.mark.requires_model
def test_overall_accuracy(m1_model, test_data):
    """Test that overall accuracy meets threshold (≥80%)."""
    model = m1_model['model']
    root_embeddings = m1_model['root_embeddings']
    root_to_idx = m1_model['root_to_idx']

    correct = 0
    for example in test_data:
        result = score_example(example, model, root_embeddings, root_to_idx)
        prediction = 1.0 if result['triple_score'] > 0.5 else 0.0
        if prediction == result['label']:
            correct += 1

    accuracy = correct / len(test_data)

    assert accuracy >= 0.80, f"Overall accuracy {accuracy:.1%} below 80% threshold"


@pytest.mark.model_quality
@pytest.mark.stage2
@pytest.mark.requires_model
def test_plausible_detection(m1_model, test_data):
    """Test that plausible detection rate meets threshold (≥85%)."""
    model = m1_model['model']
    root_embeddings = m1_model['root_embeddings']
    root_to_idx = m1_model['root_to_idx']

    plausible_correct = 0
    plausible_total = 0

    for example in test_data:
        if example['label'] == 1.0:
            plausible_total += 1
            result = score_example(example, model, root_embeddings, root_to_idx)
            prediction = 1.0 if result['triple_score'] > 0.5 else 0.0
            if prediction == 1.0:
                plausible_correct += 1

    plausible_detection = plausible_correct / plausible_total if plausible_total > 0 else 0

    assert plausible_detection >= 0.85, \
        f"Plausible detection {plausible_detection:.1%} below 85% threshold"


@pytest.mark.model_quality
@pytest.mark.stage2
@pytest.mark.requires_model
def test_implausible_detection(m1_model, test_data):
    """Test that implausible detection rate meets threshold (≥70%)."""
    model = m1_model['model']
    root_embeddings = m1_model['root_embeddings']
    root_to_idx = m1_model['root_to_idx']

    implausible_correct = 0
    implausible_total = 0

    for example in test_data:
        if example['label'] == 0.0:
            implausible_total += 1
            result = score_example(example, model, root_embeddings, root_to_idx)
            prediction = 1.0 if result['triple_score'] > 0.5 else 0.0
            if prediction == 0.0:
                implausible_correct += 1

    implausible_detection = implausible_correct / implausible_total if implausible_total > 0 else 0

    assert implausible_detection >= 0.70, \
        f"Implausible detection {implausible_detection:.1%} below 70% threshold"


@pytest.mark.model_quality
@pytest.mark.stage2
@pytest.mark.requires_model
def test_plausible_examples():
    """Test specific plausible triples that should score high."""
    model_path = Path('models/m1_selectional/best_model.pt')
    stage1_path = Path('models/root_embeddings/best_model.pt')

    if not model_path.exists() or not stage1_path.exists():
        pytest.skip("Models not found")

    # Load models
    stage1_checkpoint = torch.load(stage1_path, map_location='cpu', weights_only=False)
    root_embeddings = stage1_checkpoint['model_state_dict']['embeddings.weight']
    root_to_idx = stage1_checkpoint['root_to_idx']

    m1_checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model = M1SelectionalPreference(
        embedding_dim=m1_checkpoint['embedding_dim'],
        hidden_dim=m1_checkpoint['hidden_dim']
    )
    model.load_state_dict(m1_checkpoint['model_state_dict'])
    model.eval()

    # Test plausible triples
    plausible_triples = [
        ('mi', 'uz', 'ĝi'),      # I use it
        ('li', 'hav', 'barb'),   # he has beard
        ('mi', 'rigard', 'hund'), # I look-at dog
    ]

    for subj, verb, obj in plausible_triples:
        # Skip if roots not in vocabulary
        if (subj not in root_to_idx or verb not in root_to_idx or obj not in root_to_idx):
            continue

        example = {'subject_root': subj, 'verb_root': verb, 'object_root': obj, 'label': 1.0}
        result = score_example(example, model, root_embeddings, root_to_idx)

        assert result['triple_score'] > 0.5, \
            f"Plausible triple ({subj}, {verb}, {obj}) scored {result['triple_score']:.3f} ≤ 0.5"


@pytest.mark.model_quality
@pytest.mark.stage2
@pytest.mark.requires_model
def test_implausible_examples():
    """Test specific implausible triples that should score low."""
    model_path = Path('models/m1_selectional/best_model.pt')
    stage1_path = Path('models/root_embeddings/best_model.pt')

    if not model_path.exists() or not stage1_path.exists():
        pytest.skip("Models not found")

    # Load models
    stage1_checkpoint = torch.load(stage1_path, map_location='cpu', weights_only=False)
    root_embeddings = stage1_checkpoint['model_state_dict']['embeddings.weight']
    root_to_idx = stage1_checkpoint['root_to_idx']

    m1_checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model = M1SelectionalPreference(
        embedding_dim=m1_checkpoint['embedding_dim'],
        hidden_dim=m1_checkpoint['hidden_dim']
    )
    model.load_state_dict(m1_checkpoint['model_state_dict'])
    model.eval()

    # Test implausible triples (from hard negatives)
    implausible_triples = [
        ('ŝtup', 'renkont', 'humid'),  # step meets humidity
        ('mi', 'hav', 'mult'),         # I have much (grammatical issue)
    ]

    for subj, verb, obj in implausible_triples:
        # Skip if roots not in vocabulary
        if (subj not in root_to_idx or verb not in root_to_idx or obj not in root_to_idx):
            continue

        example = {'subject_root': subj, 'verb_root': verb, 'object_root': obj, 'label': 0.0}
        result = score_example(example, model, root_embeddings, root_to_idx)

        assert result['triple_score'] <= 0.5, \
            f"Implausible triple ({subj}, {verb}, {obj}) scored {result['triple_score']:.3f} > 0.5"


@pytest.mark.model_quality
@pytest.mark.stage2
@pytest.mark.requires_model
def test_hard_negative_discrimination(m1_model, test_data):
    """Test that model can discriminate hard negatives (≥70% accuracy on hard negatives only)."""
    model = m1_model['model']
    root_embeddings = m1_model['root_embeddings']
    root_to_idx = m1_model['root_to_idx']

    # Filter to hard negatives only
    hard_negatives = [ex for ex in test_data if ex['label'] == 0.0 and 'hard' in ex.get('corruption', '')]

    if len(hard_negatives) < 10:
        pytest.skip("Not enough hard negatives in test data")

    correct = 0
    for example in hard_negatives:
        result = score_example(example, model, root_embeddings, root_to_idx)
        prediction = 1.0 if result['triple_score'] > 0.5 else 0.0
        if prediction == 0.0:  # Correctly identified as implausible
            correct += 1

    hard_negative_accuracy = correct / len(hard_negatives)

    assert hard_negative_accuracy >= 0.70, \
        f"Hard negative discrimination {hard_negative_accuracy:.1%} below 70% threshold"


@pytest.mark.model_quality
@pytest.mark.stage2
@pytest.mark.requires_model
def test_score_components_correlation(m1_model, test_data):
    """Test that score components (subj-verb, verb-obj, triple) are positively correlated."""
    model = m1_model['model']
    root_embeddings = m1_model['root_embeddings']
    root_to_idx = m1_model['root_to_idx']

    # Sample 100 examples
    import random
    sample = random.sample(test_data, min(100, len(test_data)))

    subj_verb_scores = []
    verb_obj_scores = []
    triple_scores = []

    for example in sample:
        result = score_example(example, model, root_embeddings, root_to_idx)
        subj_verb_scores.append(result['subj_verb_score'])
        verb_obj_scores.append(result['verb_obj_score'])
        triple_scores.append(result['triple_score'])

    # Compute correlation (simple Pearson)
    import statistics

    def correlation(x, y):
        mean_x = statistics.mean(x)
        mean_y = statistics.mean(y)
        numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        denominator = (sum((xi - mean_x) ** 2 for xi in x) * sum((yi - mean_y) ** 2 for yi in y)) ** 0.5
        return numerator / denominator if denominator > 0 else 0

    corr_subj_triple = correlation(subj_verb_scores, triple_scores)
    corr_obj_triple = correlation(verb_obj_scores, triple_scores)

    assert corr_subj_triple > 0.3, \
        f"Subject-verb and triple scores poorly correlated: {corr_subj_triple:.3f}"
    assert corr_obj_triple > 0.3, \
        f"Verb-object and triple scores poorly correlated: {corr_obj_triple:.3f}"
