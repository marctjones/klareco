import unittest

from klareco.parser import parse, parse_word
from klareco.canonicalizer import (
    canonicalize_sentence,
    signature_string,
    tokens_for_sentence,
    _word_tokens,
)


class TestCanonicalizer(unittest.TestCase):
    def test_signature_simple_sentence(self):
        ast = parse("La hundo vidas la grandan katon.")
        sig = signature_string(ast)
        self.assertIn("SUBJ:role=subj|root=hund|pos=substantivo|num=singularo|case=nominativo|mods=", sig)
        self.assertIn("VERB:role=verb|root=vid|pos=verbo|tense=prezenco|mods=", sig)
        self.assertIn("OBJ:role=obj|root=kat|pos=substantivo|num=singularo|case=akuzativo|mods=grand", sig)

    def test_signature_plural_subject(self):
        ast = parse("Malgrandaj hundoj vidas la grandan katon.")
        slots = canonicalize_sentence(ast)
        subj = slots["subjekto"]
        self.assertEqual(subj.root, "hund")
        self.assertEqual(subj.number, "pluralo")
        self.assertIn("malgrand", subj.modifiers)

    def test_word_tokens_verb_with_prefix_suffix(self):
        word = parse_word("resanigos")
        tokens = _word_tokens(word)
        self.assertIn("pref:re", tokens)
        self.assertIn("root:san", tokens)
        self.assertIn("suf:ig", tokens)
        self.assertIn("ending:os", tokens)
        self.assertIn("pos:verbo", tokens)
        self.assertIn("tense:futuro", tokens)

    def test_sentence_tokens_include_modifiers(self):
        ast = parse("La hundo vidas la grandan katon.")
        tokens = tokens_for_sentence(ast)
        self.assertTrue(any(tok == "root:hund" for tok in tokens))
        self.assertTrue(any(tok.startswith("root:kat") for tok in tokens))
        self.assertTrue(any(tok.startswith("root:grand") for tok in tokens))


if __name__ == "__main__":
    unittest.main()
