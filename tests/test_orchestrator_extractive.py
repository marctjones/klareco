import unittest

from klareco.orchestrator import Orchestrator
from klareco.parser import parse


class DummyRetriever:
    def __init__(self, text):
        self.text = text

    def retrieve(self, query, k=3, return_scores=True):
        return [{"text": self.text, "score": 1.0}]


class TestOrchestratorExtractive(unittest.TestCase):
    def test_factoid_uses_extractive_when_retriever_available(self):
        orch = Orchestrator(retriever=DummyRetriever("La hundo kuras."))
        ast = parse("Kio estas tio?")
        result = orch.route(ast)
        self.assertEqual(result["answer"], "La hundo kuras.")
        self.assertEqual(result["intent"], "factoid_question")

    def test_summarization_uses_summarizer(self):
        orch = Orchestrator(retriever=DummyRetriever("La hundo kuras."))
        ast = parse("Bonvolu resumi tion.")
        orch.gating_network.classify = lambda a: {"intent": "summarization_request", "confidence": 1.0}
        result = orch.route(ast)
        self.assertIn("hundo", result["answer"])
        self.assertEqual(result["intent"], "summarization_request")


if __name__ == "__main__":
    unittest.main()
