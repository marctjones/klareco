# Klareco Rework TODO List (AST-Driven Pivot)

This list tracks the modules that must be rewritten to align with the new strategy of using a deep, morpheme-based Abstract Syntax Tree (AST) instead of a shallow, word-based one.

### Development Guidelines (Ongoing)

- **Code Comments:** Add comments during development, especially for complex logic, to explain the *why* behind the code.
- **User Documentation:** Write user documentation whenever significant new functionality is successfully added to a script or the tool suite.

### High-Priority Testing & Infrastructure Enhancements

- [x] **0.1: Enhance Integration Test Runner**
  - **Reason:** Our current end-to-end test is hardcoded in `pipeline.py`. We need a more flexible way to run integration tests.
  - **Required Fix:** Create a new script, `scripts/run_integration_test.py`, that can:
    - Accept a command-line argument `--stop-after <step_name>` to run the pipeline only up to a specified step (e.g., "Parser", "IntentClassifier").
    - Accept a command-line argument `--num-sentences <N>` to specify how many sentences from the test corpus to run.
    - Default to running the full pipeline on all test sentences if no arguments are given.

- [x] **0.2: Create a Test Corpus**
  - **Reason:** We need a standardized, diverse set of sentences to test against, rather than just a few hardcoded examples.
  - **Required Fix:** Create a new script, `scripts/create_test_corpus.py`, that:
    - Samples 50 sentences or paragraphs from the `data/cleaned/` directory.
    - Ensures the samples are taken from multiple different texts (e.g., 5 from Wikipedia, 5 from 'alicio.txt', etc.).
    - Saves the sampled sentences to a new file, `data/test_corpus.json`.

- [ ] **0.3: Implement Standard Python Logging**
  - **Reason:** Improve debugging, monitoring, and overall application professionalism.
  - **Required Fix:** Refactor all scripts to use Python's `logging` library. Configure logging to output to a `run.log` file, ensuring only the most recent log is kept (e.g., by overwriting or simple file handling).

- [ ] **0.4: Integrate Test Coverage**
  - **Reason:** Understand the effectiveness of our tests and identify untested code paths.
  - **Required Fix:** Integrate `coverage.py` into our test execution workflow to generate coverage reports.

- [ ] **0.5: Perfect the Esperanto Parser**
  - **Reason:** A robust and accurate parser is fundamental to the entire neuro-symbolic AI model.
  - **Required Fix:** Continuously refine `parser.py` to handle complex Esperanto grammar, edge cases, and ensure 100% accuracy in AST generation for the test corpus.

- [ ] **0.6: Create `watch.sh` Script**
  - **Reason:** Provide a convenient way to monitor real-time log output during development and debugging.
  - **Required Fix:** Create a `watch.sh` script that uses `tail -f` to display the contents of the `run.log` file as it is updated.

### Rework Tasks

- [x] **1. Rewrite the Parser (`parser.py`)**
- [x] **2. Rewrite the De-parser (`deparser.py`)**
- [x] **3. Rewrite the Intent Classifier (`intent_classifier.py`)**
- [x] **4. Rewrite the Responder (`responder.py`)**
- [x] **5. Adjust the Safety Monitor (`safety.py`)**
- [x] **6. Update the Pipeline and End-to-End Tests (`pipeline.py`)**
