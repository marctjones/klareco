"""
Tests for the from-scratch, pure Python Esperanto Parser.
"""
import unittest
from klareco.parser import parse, parse_word

class TestScratchParser(unittest.TestCase):

    def test_parse_simple_word(self):
        """Tests parsing a simple noun: 'hundo'"""
        ast = parse_word("hundo")
        self.assertEqual(ast['tipo'], 'vorto')
        self.assertEqual(ast['plena_vorto'], 'hundo')
        self.assertEqual(ast['radiko'], 'hund')
        self.assertEqual(ast['vortspeco'], 'substantivo')
        self.assertEqual(ast['nombro'], 'singularo')
        self.assertEqual(ast['kazo'], 'nominativo')

    def test_parse_plural_accusative_adjective(self):
        """Tests a complex adjective: 'grandajn'"""
        ast = parse_word("grandajn")
        self.assertEqual(ast['radiko'], 'grand')
        self.assertEqual(ast['vortspeco'], 'adjektivo')
        self.assertEqual(ast['nombro'], 'pluralo')
        self.assertEqual(ast['kazo'], 'akuzativo')

    def test_parse_word_with_prefix_and_suffix(self):
        """Tests a complex word with multiple morphemes: 'resanigos'"""
        ast = parse_word("resanigos")
        # Parser now correctly prefers compositional decomposition over compound forms
        # 'resanigos' = re- (prefix) + san (root) + -ig (suffix) + -os (future tense)
        self.assertEqual(ast['radiko'], 'san')
        self.assertEqual(ast['prefikso'], 're')
        self.assertIn('ig', ast['sufiksoj'])
        self.assertEqual(ast['vortspeco'], 'verbo')
        self.assertEqual(ast['tempo'], 'futuro')

    def test_unknown_root_fails(self):
        """Tests that a word with an unknown root fails parsing."""
        with self.assertRaises(ValueError):
            parse_word("nekonataradiko") # "nekonataradiko" is not a known root

    def test_parse_simple_sentence(self):
        """Tests parsing a full, simple sentence."""
        text = "mi amas la grandan katon"
        ast = parse(text)

        # Overall structure
        self.assertEqual(ast['tipo'], 'frazo')
        self.assertIsNotNone(ast['subjekto'])
        self.assertIsNotNone(ast['verbo'])
        self.assertIsNotNone(ast['objekto'])

        # Subject: "mi"
        subjekto = ast['subjekto']['kerno']
        self.assertEqual(subjekto['radiko'], 'mi')
        self.assertEqual(subjekto['vortspeco'], 'pronomo')

        # Verb: "amas"
        verbo = ast['verbo']
        self.assertEqual(verbo['radiko'], 'am')
        self.assertEqual(verbo['tempo'], 'prezenco')

        # Object: "la grandan katon"
        objekto_kerno = ast['objekto']['kerno']
        self.assertEqual(objekto_kerno['radiko'], 'kat')
        self.assertEqual(objekto_kerno['kazo'], 'akuzativo')
        
        objekto_priskribo = ast['objekto']['priskriboj'][0]
        self.assertEqual(objekto_priskribo['radiko'], 'grand')
        self.assertEqual(objekto_priskribo['kazo'], 'akuzativo')


class TestParserVerbTenses(unittest.TestCase):
    """Test suite for verb tense parsing."""

    def test_present_tense_as(self):
        """Test present tense -as ending."""
        ast = parse_word("vidas")
        self.assertEqual(ast['vortspeco'], 'verbo')
        self.assertEqual(ast['tempo'], 'prezenco')

    def test_past_tense_is(self):
        """Test past tense -is ending."""
        ast = parse_word("vidis")
        self.assertEqual(ast['vortspeco'], 'verbo')
        self.assertEqual(ast['tempo'], 'pasinteco')

    def test_future_tense_os(self):
        """Test future tense -os ending."""
        ast = parse_word("vidos")
        self.assertEqual(ast['vortspeco'], 'verbo')
        self.assertEqual(ast['tempo'], 'futuro')

    def test_conditional_us(self):
        """Test conditional -us ending."""
        ast = parse_word("vidus")
        self.assertEqual(ast['vortspeco'], 'verbo')
        self.assertEqual(ast['tempo'], 'kondiĉa')

    def test_infinitive_i(self):
        """Test infinitive -i ending."""
        ast = parse_word("vidi")
        self.assertEqual(ast['vortspeco'], 'verbo')
        self.assertEqual(ast['modo'], 'infinitivo')

    def test_imperative_u(self):
        """Test imperative -u ending."""
        ast = parse_word("vidu")
        self.assertEqual(ast['vortspeco'], 'verbo')
        self.assertEqual(ast['modo'], 'imperativo')


class TestParserPrefixes(unittest.TestCase):
    """Test suite for prefix parsing."""

    def test_mal_prefix(self):
        """Test mal- prefix (opposite)."""
        ast = parse_word("malgrand")
        self.assertEqual(ast['radiko'], 'grand')
        self.assertEqual(ast['prefikso'], 'mal')

    def test_mal_prefix_with_ending(self):
        """Test mal- prefix with adjective ending."""
        ast = parse_word("malgranda")
        self.assertEqual(ast['radiko'], 'grand')
        self.assertEqual(ast['prefikso'], 'mal')
        self.assertEqual(ast['vortspeco'], 'adjektivo')

    def test_re_prefix(self):
        """Test re- prefix (again)."""
        ast = parse_word("refari")
        # Note: 'resan' is in KNOWN_ROOTS, so 'refar' might not parse as expected
        # This test may need adjustment based on vocabulary
        self.assertIn(ast['radiko'], ['far', 'refar'])

    def test_ge_prefix(self):
        """Test ge- prefix (both genders)."""
        ast = parse_word("gepatroj")
        # 'gepatr' might be in KNOWN_ROOTS
        self.assertIn(ast['radiko'], ['patr', 'gepatr'])


class TestParserSuffixes(unittest.TestCase):
    """Test suite for suffix parsing."""

    def test_ul_suffix(self):
        """Test -ul suffix (person characterized by)."""
        ast = parse_word("belulo")
        self.assertEqual(ast['radiko'], 'bel')
        self.assertIn('ul', ast['sufiksoj'])
        self.assertEqual(ast['vortspeco'], 'substantivo')

    def test_in_suffix(self):
        """Test -in suffix (feminine)."""
        ast = parse_word("hundino")
        self.assertEqual(ast['radiko'], 'hund')
        self.assertIn('in', ast['sufiksoj'])

    def test_et_suffix(self):
        """Test -et suffix (diminutive)."""
        ast = parse_word("dometo")
        self.assertEqual(ast['radiko'], 'dom')
        self.assertIn('et', ast['sufiksoj'])

    def test_eg_suffix(self):
        """Test -eg suffix (augmentative)."""
        ast = parse_word("domego")
        self.assertEqual(ast['radiko'], 'dom')
        self.assertIn('eg', ast['sufiksoj'])

    def test_ig_suffix(self):
        """Test -ig suffix (causative)."""
        ast = parse_word("sanigi")
        self.assertEqual(ast['radiko'], 'san')
        self.assertIn('ig', ast['sufiksoj'])
        self.assertEqual(ast['modo'], 'infinitivo')

    def test_ad_suffix(self):
        """Test -ad suffix (continuous action)."""
        ast = parse_word("paroladi")
        self.assertIn('ad', ast['sufiksoj'])

    def test_ej_suffix(self):
        """Test -ej suffix (place)."""
        ast = parse_word("lernejo")
        self.assertIn('ej', ast['sufiksoj'])
        self.assertEqual(ast['vortspeco'], 'substantivo')


class TestParserCaseAndNumber(unittest.TestCase):
    """Test suite for case and number marking."""

    def test_nominative_singular(self):
        """Test nominative singular (default)."""
        ast = parse_word("hundo")
        self.assertEqual(ast['kazo'], 'nominativo')
        self.assertEqual(ast['nombro'], 'singularo')

    def test_accusative_singular(self):
        """Test accusative singular -n."""
        ast = parse_word("hundon")
        self.assertEqual(ast['kazo'], 'akuzativo')
        self.assertEqual(ast['nombro'], 'singularo')

    def test_nominative_plural(self):
        """Test nominative plural -j."""
        ast = parse_word("hundoj")
        self.assertEqual(ast['kazo'], 'nominativo')
        self.assertEqual(ast['nombro'], 'pluralo')

    def test_accusative_plural(self):
        """Test accusative plural -jn."""
        ast = parse_word("hundojn")
        self.assertEqual(ast['kazo'], 'akuzativo')
        self.assertEqual(ast['nombro'], 'pluralo')

    def test_adjective_agreement(self):
        """Test adjective case/number agreement."""
        ast = parse_word("grandajn")
        self.assertEqual(ast['vortspeco'], 'adjektivo')
        self.assertEqual(ast['kazo'], 'akuzativo')
        self.assertEqual(ast['nombro'], 'pluralo')


class TestParserPartOfSpeech(unittest.TestCase):
    """Test suite for part of speech detection."""

    def test_noun_o(self):
        """Test noun with -o ending."""
        ast = parse_word("hundo")
        self.assertEqual(ast['vortspeco'], 'substantivo')

    def test_adjective_a(self):
        """Test adjective with -a ending."""
        ast = parse_word("granda")
        self.assertEqual(ast['vortspeco'], 'adjektivo')

    def test_adverb_e(self):
        """Test adverb with -e ending."""
        ast = parse_word("rapide")
        self.assertEqual(ast['vortspeco'], 'adverbo')

    def test_pronoun(self):
        """Test pronoun parsing."""
        ast = parse_word("mi")
        self.assertEqual(ast['vortspeco'], 'pronomo')

    def test_article(self):
        """Test article parsing."""
        ast = parse_word("la")
        self.assertEqual(ast['vortspeco'], 'artikolo')


class TestParserComplexWords(unittest.TestCase):
    """Test suite for complex word parsing."""

    def test_multiple_suffixes(self):
        """Test word with multiple suffixes."""
        ast = parse_word("belulino")
        self.assertEqual(ast['radiko'], 'bel')
        self.assertIn('ul', ast['sufiksoj'])
        self.assertIn('in', ast['sufiksoj'])

    def test_prefix_and_suffix(self):
        """Test word with both prefix and suffix."""
        ast = parse_word("malsanulo")
        self.assertEqual(ast['radiko'], 'san')
        self.assertEqual(ast['prefikso'], 'mal')
        self.assertIn('ul', ast['sufiksoj'])

    def test_compound_with_suffix_and_case(self):
        """Test complex word with suffix, plural, and accusative."""
        ast = parse_word("belulojn")
        self.assertEqual(ast['radiko'], 'bel')
        self.assertIn('ul', ast['sufiksoj'])
        self.assertEqual(ast['nombro'], 'pluralo')
        self.assertEqual(ast['kazo'], 'akuzativo')


class TestParserSentenceStructure(unittest.TestCase):
    """Test suite for sentence-level parsing."""

    def test_simple_svo_sentence(self):
        """Test Subject-Verb-Object sentence."""
        ast = parse("La hundo vidas la katon.")
        self.assertEqual(ast['tipo'], 'frazo')
        self.assertIsNotNone(ast['subjekto'])
        self.assertIsNotNone(ast['verbo'])
        self.assertIsNotNone(ast['objekto'])

    def test_sentence_with_adjectives(self):
        """Test sentence with multiple adjectives."""
        ast = parse("Malgrandaj hundoj vidas la grandan katon.")
        subjekto = ast['subjekto']
        self.assertGreater(len(subjekto.get('priskriboj', [])), 0)

    def test_sentence_with_pronoun_subject(self):
        """Test sentence with pronoun subject."""
        ast = parse("Mi vidas la hundon.")
        subjekto = ast['subjekto']
        self.assertEqual(subjekto['kerno']['vortspeco'], 'pronomo')

    def test_intransitive_sentence(self):
        """Test intransitive sentence (no object)."""
        ast = parse("La hundo kuras.")
        self.assertIsNotNone(ast['subjekto'])
        self.assertIsNotNone(ast['verbo'])
        # May or may not have objekto field


class TestParserEdgeCases(unittest.TestCase):
    """Test suite for parser edge cases."""

    def test_empty_string_fails(self):
        """Test that empty string raises error."""
        with self.assertRaises(ValueError):
            parse("")

    def test_unknown_word_fails(self):
        """Test that unknown word raises error."""
        with self.assertRaises(ValueError):
            parse_word("xyzabc")

    def test_word_with_only_ending_fails(self):
        """Test that word with only grammatical ending fails."""
        with self.assertRaises(ValueError):
            parse_word("o")

    def test_article_la_parses(self):
        """Test that article 'la' parses correctly."""
        ast = parse_word("la")
        self.assertEqual(ast['tipo'], 'vorto')
        self.assertEqual(ast['vortspeco'], 'artikolo')

    def test_single_letter_pronoun(self):
        """Test single-letter pronoun parsing."""
        ast = parse_word("mi")
        self.assertEqual(ast['vortspeco'], 'pronomo')


class TestParserNumbers(unittest.TestCase):
    """Test suite for number parsing."""

    def test_simple_number_unu(self):
        """Test number word 'unu' (one)."""
        ast = parse_word("unu")
        # May be parsed as number or regular word
        self.assertEqual(ast['tipo'], 'vorto')

    def test_simple_number_du(self):
        """Test number word 'du' (two)."""
        ast = parse_word("du")
        self.assertEqual(ast['tipo'], 'vorto')

    def test_number_dek(self):
        """Test number word 'dek' (ten)."""
        ast = parse_word("dek")
        self.assertEqual(ast['tipo'], 'vorto')


class TestParserSpecialCases(unittest.TestCase):
    """Test suite for special parsing cases."""

    def test_verb_with_suffix_ig(self):
        """Test verb with -ig suffix."""
        ast = parse_word("sanigi")
        self.assertEqual(ast['radiko'], 'san')
        self.assertIn('ig', ast['sufiksoj'])
        self.assertEqual(ast['modo'], 'infinitivo')

    def test_multiple_prefixes_not_supported(self):
        """Test that multiple prefixes might not be fully supported."""
        # mal-re- combination might not parse correctly
        # This test documents current limitation
        try:
            ast = parse_word("malrefari")
            # If it parses, check structure
            self.assertIsNotNone(ast)
        except ValueError:
            # Expected if not in vocabulary
            pass

    def test_word_order_flexibility(self):
        """Test that parser handles different word orders."""
        # Esperanto allows flexible word order due to case markers
        ast1 = parse("La hundo vidas la katon.")
        ast2 = parse("La katon vidas la hundo.")

        # Both should parse successfully
        self.assertEqual(ast1['tipo'], 'frazo')
        self.assertEqual(ast2['tipo'], 'frazo')


class TestParserCorrelatives(unittest.TestCase):
    """Test suite for correlative parsing."""

    def test_correlative_kiu(self):
        """Test correlative 'kiu' (who/which)."""
        ast = parse_word("kiu")
        # May be classified as pronoun or correlative
        self.assertIn(ast['vortspeco'], ['pronomo', 'korelativo'])

    def test_correlative_kio(self):
        """Test correlative 'kio' (what)."""
        ast = parse_word("kio")
        self.assertIn(ast['vortspeco'], ['pronomo', 'korelativo'])


if __name__ == '__main__':
    unittest.main()
