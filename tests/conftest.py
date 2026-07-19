"""Test taxonomy — auto-marking, so the tiers are real instead of declared.

`pytest.ini` used to declare eight markers (code, data_quality, model_quality,
integration, stage0, stage1, stage2, requires_model, requires_data). **Every one
of them was used by exactly zero tests.** CLAUDE.md described a four-category
strategy that did not exist in practice. A taxonomy nobody applies is not a
taxonomy — it is a comment.

So the tier is derived from the FILENAME here, in one place, rather than being
sprinkled by hand across 21 files where it would rot again.

The tiers answer five different questions, and they fail for different reasons:

  unit         Is the CODE correct?                   fast, offline, no data
  environment  Is the RUNTIME set up?                 artifacts exist & cohere
  data         Did the data LOAD correctly?           no garbage, no gaps
  pipeline     Do the PIECES work, end to end?        stages + orchestrator
  perf         Is it FAST ENOUGH vs. the baseline?    latency regression
  accuracy     Is it GOOD ENOUGH vs. the baseline?    quality regression

Run one tier:

    pytest -m unit                 # the fast inner loop — no data needed
    pytest -m environment          # "can I trust anything on this machine?"
    pytest -m "data or pipeline"   # "is the system actually working?"
    pytest -m "perf or accuracy"   # "did I make it worse?"  <- the merge gate
    pytest -m "not slow"           # everything cheap
"""

from pathlib import Path

import pytest

# filename stem -> tier. Anything unlisted defaults to `unit`, which is the safe
# default: a test that needs no data and asserts on code.
_TIERS: dict[str, str] = {
    # --- environment: is the runtime set up and coherent? ---------------------
    'test_preflight':            'environment',
    'test_environment_contract': 'environment',

    # --- data: did the data load / prepare correctly, and is it clean? --------
    'test_corpus_integrity':     'data',
    'test_data_quality':         'data',
    'test_cleaned_data':         'data',
    'test_extracted_data':       'data',
    'test_fundamento_completeness': 'data',
    'test_wikipedia_benchmark':  'data',

    # --- contract: does the ORCHESTRATOR hold its stages to the contract? -----
    # The primary suite. Runs against a tiny in-memory store (tests/contract/),
    # so it needs no production indexes — fast and CI-safe. See DESIGN.md →
    # "The orchestration contract".
    'test_stage_conformance':    'contract',
    'test_golden_trace':         'contract',
    'test_decoder':              'contract',
    'test_dependencies':         'contract',
    'test_loud_failure_lint':    'contract',

    # --- pipeline: do the pieces work, end to end? ----------------------------
    'test_orchestrator':         'pipeline',
    'test_question_classifier':  'pipeline',
    'test_entity_recognizer':    'pipeline',

    # --- perf / accuracy: regression against a recorded baseline --------------
    'test_perf_baseline':        'perf',
    'test_accuracy_baseline':    'accuracy',
    # parser quality vs. the UD gold treebanks — needs NO store (fixtures under
    # tests/fixtures/ud/), so it also runs in fast CI. See the file's docstring.
    'test_parser_ud_accuracy':   'accuracy',

    # --- unit: code correctness, offline ------------------------------------
    'test_parser':               'unit',
    'test_parser_fundamento':    'unit',
    'test_deparser':             'unit',
    'test_answer_scoring':       'unit',
    'test_testset_gates':        'unit',
    'test_language_quality_audit': 'unit',
    'test_llm_triage':           'unit',
    'test_logging_config':       'unit',
}


def pytest_collection_modifyitems(config, items):
    for item in items:
        stem = Path(str(item.fspath)).stem
        tier = _TIERS.get(stem, 'unit')
        item.add_marker(getattr(pytest.mark, tier))
