import unittest

from klareco.experts.extractive import ExtractiveResponder


class DummyRetriever:
    def __init__(self, results):
        self._results = results

    def retrieve(self, query, k=3, return_scores=True):
        return self._results[:k]


class TestExtractiveResponder(unittest.TestCase):
    def test_returns_top_answer_with_sources(self):
        results = [
            {"text": "La hundo kuras en la parko.", "score": 1.2, "source": "test_corpus"},
            {"text": "La kato dormas.", "score": 0.8, "source": "test_corpus"},
        ]
        responder = ExtractiveResponder(DummyRetriever(results), top_k=2)

        out = responder.execute({"tipo": "frazo"}, original_text="La hundo kuras?")
        self.assertEqual(out["answer"], results[0]["text"])
        self.assertAlmostEqual(out["confidence"], 1.2)
        self.assertEqual(len(out["sources"]), 2)

    def test_handles_empty_results(self):
        responder = ExtractiveResponder(DummyRetriever([]))
        out = responder.execute({"tipo": "frazo"}, original_text="Demando")
        self.assertEqual(out["confidence"], 0.0)
        self.assertTrue("sources" in out)
        self.assertIn("ne trovis", out["answer"])


if __name__ == "__main__":
    unittest.main()
