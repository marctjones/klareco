"""
Tests for the from-scratch, pure Python Esperanto De-parser.
"""
import unittest
from klareco.parser import parse
from klareco.deparser import deparse, _reconstruct_word

class TestScratchDeparser(unittest.TestCase):

    def test_reconstruct_simple_word(self):
        """Tests reconstructing a simple word from the new AST format."""
        ast = {
            "tipo": "vorto",
            "radiko": "hund",
            "vortspeco": "substantivo",
            "nombro": "singularo",
            "kazo": "nominativo"
        }
        self.assertEqual(_reconstruct_word(ast), "hundo")

    def test_reconstruct_complex_adjective(self):
        """Tests reconstructing a plural, accusative adjective."""
        ast = {
            "tipo": "vorto",
            "radiko": "grand",
            "vortspeco": "adjektivo",
            "nombro": "pluralo",
            "kazo": "akuzativo"
        }
        self.assertEqual(_reconstruct_word(ast), "grandajn")

    def test_reconstruct_word_with_affixes(self):
        """Tests reconstructing a word with both prefix and suffix."""
        ast = {
            "tipo": "vorto",
            "radiko": "san",
            "prefikso": "re",
            "sufiksoj": ["ig"],
            "vortspeco": "verbo",
            "tempo": "futuro"
        }
        self.assertEqual(_reconstruct_word(ast), "resanigos")

    def test_round_trip_simple_sentence(self):
        """
        Tests that parsing and then deparsing a sentence returns the original.
        This is the ultimate integration test for the parser/deparser pair.
        """
        # Use lowercase 'mi' so parser recognizes it as pronoun subject
        original_sentence = "mi amas la grandan katon."

        # 1. Parse the sentence into our new AST format
        ast = parse(original_sentence)

        # 2. Deparse the AST back into a string
        reconstructed_sentence = deparse(ast)

        # 3. Compare the result (case-insensitively, ignoring final punctuation)
        self.assertEqual(
            reconstructed_sentence.lower().strip('.'),
            original_sentence.lower().strip('.')
        )

    def test_round_trip_complex_sentence(self):
        """Tests a more complex round-trip operation."""
        original_sentence = "Malgrandaj hundoj vidas la grandan katon."
        ast = parse(original_sentence)
        reconstructed_sentence = deparse(ast)
        self.assertEqual(
            reconstructed_sentence.lower().strip('.'),
            original_sentence.lower().strip('.')
        )


class TestDeparsePunctuation(unittest.TestCase):

    def test_terminal_punctuation_is_REPLAYED_not_INVENTED(self):
        """The contract changed with #836, and it changed on purpose.

        `deparse` used to APPEND a terminal mark inferred from `fraztipo`. That is
        a GUESS, and guessing is what made this module emit `jarocento` and
        `enohavas`. Punctuation is now a real token in the AST, in its real
        position, so it is REPLAYED.

        Consequence: a sentence with no period deparses to no period. That is
        correct — `deparse` reproduces its input, it does not tidy it. Generation
        (which SHOULD supply punctuation) is `deparse_structural`.
        """
        assert deparse(parse("Mi amas vin.")).endswith('.')      # in -> out
        assert not deparse(parse("mi amas vin")).endswith('.')   # absent -> absent

    def test_question_ends_with_question_mark(self):
        ast = parse("Kiu fondis Esperanton?")
        result = deparse(ast)
        self.assertTrue(result.endswith('?'), f"Expected '?', got '{result[-1]}'")

    def test_imperative_ends_with_exclamation(self):
        ast = parse("Venu!")
        result = deparse(ast)
        self.assertTrue(result.endswith('!'), f"Expected '!', got '{result[-1]}'")

    def test_first_word_capitalised(self):
        ast = parse("mi manĝas panon")
        result = deparse(ast)
        self.assertTrue(result[0].isupper(), f"Expected capital first letter, got '{result[0]}'")


class TestDeparsePropraVorto(unittest.TestCase):

    def test_proper_noun_preserved_verbatim(self):
        ast = parse("Zamenhof fondis Esperanton")
        result = deparse(ast)
        self.assertIn('Zamenhof', result)
        self.assertIn('Esperanton', result)

    def test_proper_noun_uses_plena_vorto(self):
        word_ast = {
            "tipo": "vorto",
            "vortspeco": "propra_nomo",
            "plena_vorto": "Varsovio",
            "radiko": "Varsovi",
        }
        self.assertEqual(_reconstruct_word(word_ast), "Varsovio")


class TestDeparseCompoundWords(unittest.TestCase):

    def test_kunmetitaj_radikoj_reconstruction(self):
        """The AST must RECORD the linking vowel — it may not be GUESSED. (#833)

        This test used to assert that the deparser joins compound roots with a
        hard-coded 'o'. That is exactly the bug: the linking vowel is OPTIONAL in
        Esperanto. `hundodomo` has one; `mondmilito`, `jarcento`, `dufoje` and
        `enhavas` do not. Guessing turned them into `mondomilito`, `jarocento`,
        `duofoje`, `enohavas` — words that do not exist — and 40% of corpus
        sentences failed to round-trip with nothing to notice.

        `tigo` now carries the stem EXACTLY as it appeared. Where it is absent the
        deparser returns the surface rather than FABRICATING a linking vowel.
        """
        word_ast = {
            "tipo": "vorto",
            "vortspeco": "substantivo",
            "kunmetitaj_radikoj": ["libr", "vend"],
            "radiko": "vend",
            "sufiksoj": ["ist"],
            "tigo": "librovendist",       # RECORDED, not guessed
            "nombro": "singularo",
            "kazo": "nominativo",
        }
        self.assertEqual(_reconstruct_word(word_ast), "librovendisto")

    def test_a_compound_WITHOUT_a_linking_vowel(self):
        """`mondmilito` is mond+milit with NO linking vowel. The old code emitted
        `mondomilito`."""
        word_ast = {
            "tipo": "vorto",
            "vortspeco": "substantivo",
            "kunmetitaj_radikoj": ["mond", "milit"],
            "radiko": "milit",
            "sufiksoj": [],
            "tigo": "mondmilit",
            "nombro": "singularo",
            "kazo": "nominativo",
        }
        self.assertEqual(_reconstruct_word(word_ast), "mondmilito")

    def test_single_root_unaffected(self):
        word_ast = {
            "tipo": "vorto",
            "vortspeco": "substantivo",
            "kunmetitaj_radikoj": ["hund"],
            "radiko": "hund",
            "nombro": "singularo",
            "kazo": "nominativo",
        }
        result = _reconstruct_word(word_ast)
        self.assertEqual(result, "hundo")


class TestDeparseVerbMoods(unittest.TestCase):

    def test_kondicionalo(self):
        word_ast = {
            "tipo": "vorto",
            "vortspeco": "verbo",
            "radiko": "est",
            "modo": "kondicionalo",
        }
        self.assertEqual(_reconstruct_word(word_ast), "estus")

    def test_imperativo(self):
        word_ast = {
            "tipo": "vorto",
            "vortspeco": "verbo",
            "radiko": "vid",
            "modo": "imperativo",
        }
        self.assertEqual(_reconstruct_word(word_ast), "vidu")

    def test_infinitivo(self):
        word_ast = {
            "tipo": "vorto",
            "vortspeco": "verbo",
            "radiko": "kur",
            "modo": "infinitivo",
        }
        self.assertEqual(_reconstruct_word(word_ast), "kuri")

    def test_no_tempo_or_modo_defaults_to_infinitive(self):
        word_ast = {
            "tipo": "vorto",
            "vortspeco": "verbo",
            "radiko": "ir",
        }
        self.assertEqual(_reconstruct_word(word_ast), "iri")


class TestDeparseRelativeClause(unittest.TestCase):

    def test_round_trip_nominative_kiu(self):
        # Sentence without predicate adjectives so parser preserves word order.
        original = "La homo kiu vidas la hundon venas."
        ast = parse(original)
        result = deparse(ast)
        self.assertEqual(result.lower().rstrip('.'), original.lower().rstrip('.'))

    def test_round_trip_accusative_kiun(self):
        # Sentence without predicate adjectives so parser preserves word order.
        original = "La hundo kiun mi vidas kuras."
        ast = parse(original)
        result = deparse(ast)
        self.assertEqual(result.lower().rstrip('.'), original.lower().rstrip('.'))

    def test_relative_pronoun_not_duplicated(self):
        ast = parse("La homo kiu amas bonon estas feliĉa")
        result = deparse(ast)
        kiu_count = result.lower().count('kiu')
        self.assertEqual(kiu_count, 1, f"'kiu' appeared {kiu_count} times in: {result}")


class TestDeparseKeClause(unittest.TestCase):

    def test_ke_clause_round_trip(self):
        original = "mi scias ke vi amas lin."
        ast = parse(original)
        result = deparse(ast)
        self.assertIn('ke', result.lower())
        self.assertEqual(result.lower().rstrip('.'), original.lower().rstrip('.'))


if __name__ == '__main__':
    unittest.main()
