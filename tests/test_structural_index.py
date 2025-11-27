import unittest

from klareco.parser import parse
from klareco.structural_index import build_structural_metadata, rank_candidates_by_slot_overlap


class TestStructuralIndex(unittest.TestCase):
    def test_build_structural_metadata(self):
        ast = parse("La hundo vidas la grandan katon.")
        meta = build_structural_metadata(ast)
        self.assertIn("signature", meta)
        self.assertIn("grammar_tokens", meta)
        self.assertIn("slot_roots", meta)

        # Slot roots should capture main roles
        self.assertEqual(meta["slot_roots"].get("subjekto"), "hund")
        self.assertEqual(meta["slot_roots"].get("objekto"), "kat")
        self.assertEqual(meta["slot_roots"].get("verbo"), "vid")

        # Signature includes roles and case
        self.assertIn("SUBJ:role=subj", meta["signature"])
        self.assertIn("OBJ:role=obj", meta["signature"])
        self.assertTrue(any(tok.startswith("root:kat") for tok in meta["grammar_tokens"]))

    def test_rank_candidates_by_slot_overlap(self):
        meta = [
            {"slot_roots": {"subjekto": "hund", "objekto": "kat", "verbo": "vid"}},
            {"slot_roots": {"subjekto": "knab", "objekto": "libro", "verbo": "leg"}},
            {"slot_roots": {"subjekto": "kat", "verbo": "dorm"}},
        ]
        query = parse("La hundo vidas la katon.")
        query_meta = build_structural_metadata(query)
        indices = rank_candidates_by_slot_overlap(query_meta["slot_roots"], meta, limit=5)
        # Expect first item (exact match) before others
        self.assertEqual(indices[0], 0)
        self.assertIn(2, indices)  # shares 'kat' root


if __name__ == "__main__":
    unittest.main()
