"""The diagnostic inventory must account for the same tokens as the score."""
from collections import Counter

import pytest

from scripts.eval import eval_conllu
from scripts.eval.build_parser_pilot import document_split, normalized, select


GOLD = ('# text = Mi venas.\n'
        '1\tMi\tmi\tPRON\t_\t_\t2\tnsubj\t_\t_\n'
        '2\tvenas\tveni\tVERB\t_\t_\t0\troot\t_\t_\n'
        '3\t.\t.\tPUNCT\t_\t_\t2\tpunct\t_\t_\n\n')


@pytest.mark.parametrize('prediction, expected_coverage', [
    (GOLD, 1.0),
    ('1\tvenas\tveni\tVERB\t_\t_\t0\troot\t_\t_\n', 0.5),
])
def test_diagnostics_agree_with_fixed_denominator(tmp_path, monkeypatch,
                                                 prediction, expected_coverage):
    path = tmp_path / 'gold.conllu'
    path.write_text(GOLD)
    monkeypatch.setattr(eval_conllu, 'to_conllu', lambda *a, **kw: prediction)
    tokens = []
    metrics = eval_conllu.evaluate(str(path), diagnostics=tokens)
    assert len(tokens) == metrics['gold_tokens'] == 2
    assert metrics['coverage'] == expected_coverage
    correct = sum(not (set(t['errors']) - {'pos'}) for t in tokens)
    assert metrics['las_all'] == correct / len(tokens)
    if prediction == GOLD:
        assert metrics['lemma'] == 1.0
        assert metrics['morphology'] == 1.0
        assert metrics['lemma_all'] == 1.0
        assert metrics['morphology_all'] == 1.0
        assert metrics['root_sentence_accuracy'] == 1.0
        assert metrics['complete_tree_sentence_accuracy'] == 1.0


def test_crash_stays_in_diagnostic_denominator(tmp_path, monkeypatch):
    path = tmp_path / 'gold.conllu'
    path.write_text(GOLD)
    def crash(*args, **kwargs):
        raise ValueError('bad parse')
    monkeypatch.setattr(eval_conllu, 'to_conllu', crash)
    tokens = []
    metrics = eval_conllu.evaluate(str(path), diagnostics=tokens)
    assert metrics['gold_tokens'] == len(tokens) == 2
    assert metrics['las_all'] == 0
    assert all(t['errors'] == ['crash'] for t in tokens)
    assert metrics['root_sentence_accuracy'] == 0.0
    assert metrics['complete_tree_sentence_accuracy'] == 0.0


def test_annotation_queue_is_deterministic_and_document_disjoint():
    rows = [(i, f'Unika teksto {i}.', 'source', f'doc-{i // 2}') for i in range(100)]
    excluded = {normalized(rows[0][1])}
    selected = select(rows, excluded, 20)
    assert selected == select(list(reversed(rows)), excluded, 20)
    assert Counter(r['split'] for r in selected) == {'development': 16, 'heldout': 4}
    assert all(r['gold_conllu'] is None and r['annotation_status'] == 'unreviewed'
               for r in selected)
    assert all(normalized(r['text']) not in excluded for r in selected)
    for row in selected:
        assert row['split'] == document_split(row['source_name'], row['document_title'])


def test_annotation_queue_fails_if_it_cannot_fill_split():
    with pytest.raises(ValueError, match='eligible rows'):
        select([(1, 'Unu frazo.', 'source', 'one-document')], set(), 20)
