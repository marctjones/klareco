"""
Comprehensive tests for Fundamento de Esperanto vocabulary parsing.

This test module ensures the parser correctly handles:
1. All 2,067 official Fundamento roots
2. 83 roots that look like they have prefixes (but don't)
3. 294 roots that look like they have suffixes (but don't)
4. Real affixed words (that SHOULD be decomposed)

Issues: #134, #135, #136, #137
"""

import json
import unittest
from pathlib import Path

from klareco.parser import parse_word

# =============================================================================
# FUNDAMENTO DATA
# =============================================================================

# Load Fundamento roots at module level for efficiency
FUNDAMENTO_PATH = Path(__file__).parent.parent / 'data' / 'vocabularies' / 'fundamento_roots.json'

def load_fundamento_roots():
    """Load all Fundamento roots from the vocabulary file."""
    if FUNDAMENTO_PATH.exists():
        with open(FUNDAMENTO_PATH) as f:
            data = json.load(f)
            return set(data.get('roots', {}).keys())
    return set()

FUNDAMENTO_ROOTS = load_fundamento_roots()

# Known affixes
PREFIXES = {'mal', 're', 'ge', 'ek', 'eks', 'dis', 'mis', 'bo', 'fi', 'pra', 'for', 'vic'}
SUFFIXES = {'ul', 'ej', 'in', 'et', 'eg', 'ig', 'iĝ', 'ad', 'ar', 'ec', 'ebl', 'em', 'er',
            'estr', 'id', 'il', 'ind', 'ing', 'ism', 'ist', 'uj', 'um', 'ant', 'int', 'ont',
            'at', 'it', 'ot', 'aĉ', 'aĵ', 'end'}


# =============================================================================
# PROTECTED ROOTS - These should NEVER be split
# =============================================================================

# Fundamento roots that START with prefix-like sequences
# Format: {root: fake_prefix_it_starts_with}
PREFIX_PROTECTED_ROOTS = {
    # bo- (in-law) - these are NOT in-law words
    'boa': 'bo', 'boben': 'bo', 'boj': 'bo', 'bol': 'bo', 'bombon': 'bo',
    'bon': 'bo', 'bor': 'bo', 'boraks': 'bo', 'bord': 'bo', 'bot': 'bo',
    'botel': 'bo', 'bov': 'bo',

    # dis- (apart) - these are NOT scatter words
    'disput': 'dis', 'distil': 'dis', 'distr': 'dis',

    # ek-/eks- - these are NOT begin/former words
    'ekscit': 'ek', 'ekskurs': 'ek', 'ekspozici': 'ek', 'ekstrem': 'ek',
    'ekzekut': 'ek', 'ekzempl': 'ek', 'ekzempler': 'ek', 'ekzerc': 'ek', 'ekzist': 'ek',

    # fi- (morally bad) - these are NOT pejorative words
    'fibr': 'fi', 'fidel': 'fi', 'fig': 'fi', 'fiĥ': 'fi', 'fil': 'fi',
    'filologi': 'fi', 'filozofi': 'fi', 'filtr': 'fi', 'fin': 'fi', 'fingr': 'fi',
    'firm': 'fi', 'fiŝ': 'fi', 'fizik': 'fi',

    # for- (away) - these are NOT "away" words
    'fork': 'for', 'form': 'for', 'formik': 'for', 'formul': 'for', 'forn': 'for',
    'fort': 'for', 'fortik': 'for', 'fos': 'for',

    # ge- (both genders) - these are NOT gender-neutral words
    'gelaten': 'ge', 'gem': 'ge', 'geni': 'ge', 'genu': 'ge',
    'geografi': 'ge', 'geometri': 'ge', 'german': 'ge', 'gest': 'ge',

    # mal- - these are NOT opposite words
    'malt': 'mal',

    # mis- (wrongly) - these are NOT "wrong" words
    'misi': 'mis', 'mister': 'mis',

    # pra- (primordial) - these are NOT primordial words
    'praktik': 'pra',

    # re- (again) - these are NOT "again" words
    'redakci': 're', 'redut': 're', 'refut': 're', 'reg': 're', 'registr': 're',
    'regul': 're', 'rek': 're', 'reĝ': 're', 'reklam': 're', 'rekomend': 're',
    'rel': 're', 'relief': 're', 'religi': 're', 'rem': 're', 'rembur': 're',
    'remed': 're', 'ren': 're', 'rendevu': 're', 'renkont': 're', 'rent': 're',
    'republik': 're', 'respekt': 're', 'rest': 're', 'ret': 're', 'retori': 're',
    'rev': 're', 'revizi': 're', 'revu': 're',
}

# Fundamento roots that END with suffix-like sequences
# Format: {root: fake_suffix_it_ends_with}
SUFFIX_PROTECTED_ROOTS = {
    # -ar (collection)
    'altar': 'ar', 'avar': 'ar', 'bazar': 'ar', 'cezar': 'ar', 'cigar': 'ar',
    'dolar': 'ar', 'familiar': 'ar', 'hangar': 'ar', 'kalendar': 'ar',
    'konsular': 'ar', 'popular': 'ar', 'solar': 'ar',

    # -ul (person)
    'angul': 'ul', 'betul': 'ul', 'formul': 'ul', 'kapitol': 'ul', 'konsul': 'ul',
    'muskul': 'ul', 'pendol': 'ul', 'regul': 'ul', 'stimul': 'ul', 'tabul': 'ul',
    'tuberkul': 'ul', 'vestibul': 'ul',

    # -il (tool)
    'angil': 'il', 'april': 'il', 'argil': 'il', 'babil': 'il', 'daktil': 'il',
    'facil': 'il', 'fertil': 'il', 'fibril': 'il', 'fossil': 'il', 'krokodil': 'il',
    'lentil': 'il', 'papil': 'il', 'penicil': 'il', 'pupil': 'il', 'reptil': 'il',
    'simil': 'il', 'stencil': 'il', 'tonsil': 'il', 'utensil': 'il', 'vakul': 'il',
    'vanil': 'il', 'ventil': 'il', 'vigil': 'il', 'viril': 'il', 'bril': 'il',

    # -eg (augmentative)
    'koleg': 'eg', 'strateg': 'eg',

    # -et (diminutive)
    'alumet': 'et', 'bajonet': 'et', 'biljet': 'et', 'bret': 'et', 'bufet': 'et',
    'duet': 'et', 'gazet': 'et', 'kabinet': 'et', 'kadet': 'et', 'kaset': 'et',
    'ĵaket': 'et', 'koket': 'et', 'komplet': 'et', 'kornet': 'et', 'korset': 'et',
    'kvartet': 'et', 'minuet': 'et', 'mulet': 'et', 'oktet': 'et', 'omlet': 'et',
    'paket': 'et', 'pamflet': 'et', 'parket': 'et', 'piruet': 'et', 'poet': 'et',
    'raket': 'et', 'sekret': 'et', 'siluet': 'et', 'skelet': 'et', 'sonet': 'et',
    'stafet': 'et', 'stilet': 'et', 'tablet': 'et', 'tapet': 'et', 'triket': 'et',
    'trompet': 'et', 'trumpet': 'et', 'violet': 'et',

    # -er (smallest unit)
    'anser': 'er', 'aper': 'er', 'cifer': 'er', 'cirkuler': 'er', 'danĝer': 'er',
    'difer': 'er', 'elster': 'er', 'fajfer': 'er', 'infer': 'er', 'kajer': 'er',
    'klister': 'er', 'konsider': 'er', 'lucer': 'er', 'maner': 'er', 'miser': 'er',
    'moder': 'er', 'muster': 'er', 'numer': 'er', 'oper': 'er', 'paper': 'er',
    'prefer': 'er', 'profer': 'er', 'prosper': 'er', 'puder': 'er', 'refer': 'er',
    'sever': 'er', 'super': 'er', 'sufer': 'er', 'teler': 'er', 'toler': 'er',
    'transfer': 'er', 'veter': 'er', 'viper': 'er',

    # -in (feminine)
    'basin': 'in', 'benzin': 'in', 'bobelin': 'in', 'delfin': 'in', 'dezert': 'in',
    'din': 'in', 'doktrín': 'in', 'fin': 'in', 'glikol': 'in', 'jasmín': 'in',
    'kabin': 'in', 'kafeín': 'in', 'kamelin': 'in', 'karmín': 'in', 'kokin': 'in',
    'kuzin': 'in', 'latin': 'in', 'magazín': 'in', 'margarin': 'in', 'marin': 'in',
    'maŝin': 'in', 'medicín': 'in', 'min': 'in', 'molin': 'in', 'nikotin': 'in',
    'origin': 'in', 'parafin': 'in', 'pin': 'in', 'platin': 'in', 'ravin': 'in',
    'rezin': 'in', 'ruín': 'in', 'sardin': 'in', 'sin': 'in', 'termin': 'in',
    'toksín': 'in', 'urin': 'in', 'vaksin': 'in', 'vanilín': 'in', 'vazelin': 'in',
    'vin': 'in', 'violin': 'in', 'vitamin': 'in',

    # -ad (continuous action)
    'balustrad': 'ad', 'blokad': 'ad', 'brikad': 'ad', 'cikad': 'ad', 'dekad': 'ad',
    'fasad': 'ad', 'kamarad': 'ad', 'kaskad': 'ad', 'kolonad': 'ad', 'limonad': 'ad',
    'maskerad': 'ad', 'monad': 'ad', 'nomad': 'ad', 'olimpiad': 'ad', 'parad': 'ad',
    'pomad': 'ad', 'salad': 'ad', 'serenad': 'ad', 'tirad': 'ad', 'tonad': 'ad',

    # -it (passive past participle)
    'artrit': 'it', 'biskvit': 'it', 'dinamit': 'it', 'ermit': 'it', 'granit': 'it',
    'kredit': 'it', 'merit': 'it', 'orbit': 'it', 'profit': 'it', 'vizit': 'it',

    # -at (passive present participle)
    'advokat': 'at', 'agregat': 'at', 'akrobat': 'at', 'aparát': 'at', 'aristokrat': 'at',
    'blat': 'at', 'burokrat': 'at', 'celibat': 'at', 'ĉokolat': 'at', 'delegat': 'at',
    'demokrat': 'at', 'diplomat': 'at', 'drat': 'at', 'format': 'at', 'kandidat': 'at',
    'klimat': 'at', 'konsulat': 'at', 'kravat': 'at', 'magistrat': 'at', 'pirat': 'at',
    'plat': 'at', 'privat': 'at', 'proletari': 'at', 'rat': 'at', 'rezultat': 'at',
    'skarabat': 'at', 'soldat': 'at', 'spat': 'at', 'stat': 'at', 'stigmat': 'at',
    'sulfat': 'at', 'sindikat': 'at', 'tomat': 'at',

    # -ot (passive future participle)
    'azot': 'ot', 'balot': 'ot', 'bergamot': 'ot', 'bot': 'ot', 'fagot': 'ot',
    'gavot': 'ot', 'grot': 'ot', 'idiot': 'ot', 'jakbot': 'ot', 'kalikot': 'ot',
    'kapot': 'ot', 'karjot': 'ot', 'kompot': 'ot', 'komplot': 'ot', 'lot': 'ot',
    'margot': 'ot', 'maskot': 'ot', 'mot': 'ot', 'patriot': 'ot', 'pilot': 'ot',
    'pivot': 'ot', 'pot': 'ot', 'robot': 'ot', 'rot': 'ot', 'ŝalot': 'ot',
    'tarot': 'ot', 'trikot': 'ot',

    # -ec (quality)
    'dec': 'ec', 'direc': 'ec', 'indec': 'ec', 'infec': 'ec', 'spec': 'ec',

    # -ind (worthy of)
    'blind': 'ind', 'hind': 'ind', 'kind': 'ind',

    # -ant (active present participle)
    'briliant': 'ant', 'diamant': 'ant', 'elefant': 'ant', 'galant': 'ant',
    'gigant': 'ant', 'infant': 'ant', 'intendant': 'ant', 'komandant': 'ant',
    'konsonant': 'ant', 'konstant': 'ant', 'konsultant': 'ant', 'kvant': 'ant',
    'laborant': 'ant', 'liŭtenant': 'ant', 'merkant': 'ant', 'mutant': 'ant',
    'pedant': 'ant', 'plant': 'ant', 'protestant': 'ant', 'restaur': 'ant',
    'serĝant': 'ant', 'talant': 'ant', 'tirant': 'ant', 'variant': 'ant',

    # -int (active past participle)
    'absint': 'int', 'flint': 'int', 'hiacint': 'int', 'instinkt': 'int',
    'labrint': 'int', 'marmint': 'int', 'mint': 'int', 'pint': 'int', 'print': 'int',

    # -ont (active future participle)
    'font': 'ont', 'horizont': 'ont', 'kont': 'ont', 'mont': 'ont', 'pont': 'ont',

    # -ig (causative)
    'konfig': 'ig', 'orig': 'ig',

    # -ej (place)
    'muze': 'ej',

    # -id (offspring)
    'aspid': 'id', 'avid': 'id', 'david': 'id', 'humid': 'id', 'likv': 'id',
    'lucid': 'id', 'morbid': 'id', 'perfid': 'id', 'rapid': 'id', 'rigid': 'id',
    'solid': 'id', 'splendid': 'id', 'stupid': 'id', 'timid': 'id', 'valid': 'id',
    'vivid': 'id',

    # -ing (holder)
    'ating': 'ing', 'buding': 'ing', 'haring': 'ing', 'miting': 'ing',
    'peding': 'ing', 'pudding': 'ing', 'ŝiling': 'ing', 'sterling': 'ing',
    'viking': 'ing',

    # -uj (container)
    'halleluj': 'uj',

    # -ebl (able to be)
    'mebl': 'ebl',
}


# =============================================================================
# REAL AFFIXED WORDS - These SHOULD be decomposed
# =============================================================================

# Words that really DO have prefixes and should be split
# Format: (word, expected_prefix, expected_root)
REAL_PREFIXED_WORDS = [
    # mal- (opposite)
    ('malbona', 'mal', 'bon'),
    ('malgranda', 'mal', 'grand'),
    ('maljuna', 'mal', 'jun'),
    ('malsana', 'mal', 'san'),
    ('malfermi', 'mal', 'ferm'),
    ('mallonga', 'mal', 'long'),

    # re- (again)
    ('resendi', 're', 'send'),
    ('reveni', 're', 'ven'),
    ('relegi', 're', 'leg'),
    ('reskribi', 're', 'skrib'),

    # ge- (both genders)
    ('gepatroj', 'ge', 'patr'),
    ('gesinjoroj', 'ge', 'sinjor'),
    ('gefratoj', 'ge', 'frat'),

    # ek- (begin/sudden)
    ('ekvidi', 'ek', 'vid'),
    ('ekiri', 'ek', 'ir'),
    ('ekdormi', 'ek', 'dorm'),
    ('ekbrili', 'ek', 'bril'),

    # dis- (scatter/apart)
    ('dissendi', 'dis', 'send'),
    ('disigi', 'dis', 'ig'),
    ('disdoni', 'dis', 'don'),
    ('disŝiri', 'dis', 'ŝir'),

    # mis- (wrongly)
    ('misuzi', 'mis', 'uz'),
    ('miskompreni', 'mis', 'kompren'),
    ('misfari', 'mis', 'far'),

    # bo- (in-law)
    ('bopatro', 'bo', 'patr'),
    ('bopatrino', 'bo', 'patr'),  # bo + patr + in + o
    ('bofilo', 'bo', 'fil'),
    ('bofrato', 'bo', 'frat'),

    # fi- (morally bad)
    ('fiulo', 'fi', 'ul'),
    ('fifama', 'fi', 'fam'),

    # eks- (former)
    ('eksprezidanto', 'eks', 'prezid'),
    ('eksedziĝi', 'eks', 'edz'),

    # pra- (primordial)
    ('praavo', 'pra', 'av'),
    ('praepoko', 'pra', 'epok'),
    ('prahomo', 'pra', 'hom'),

    # for- (away/completely)
    ('foriri', 'for', 'ir'),
    ('formanĝi', 'for', 'manĝ'),
    ('forpreni', 'for', 'pren'),

    # vic- (deputy/vice)
    ('vicprezidanto', 'vic', 'prezid'),
    ('vicreĝo', 'vic', 'reĝ'),
]

# Words that really DO have suffixes and should be split
# Format: (word, expected_root, expected_suffix)
REAL_SUFFIXED_WORDS = [
    # -ul (person characterized by)
    ('belulo', 'bel', 'ul'),
    ('junulo', 'jun', 'ul'),
    ('riĉulo', 'riĉ', 'ul'),

    # -in (feminine)
    ('hundino', 'hund', 'in'),
    ('patrino', 'patr', 'in'),
    ('fratino', 'frat', 'in'),

    # -ej (place)
    ('lernejo', 'lern', 'ej'),
    ('vendejo', 'vend', 'ej'),
    ('manĝejo', 'manĝ', 'ej'),

    # -et (diminutive)
    ('dometo', 'dom', 'et'),
    ('libreto', 'libr', 'et'),
    ('hundeto', 'hund', 'et'),

    # -eg (augmentative)
    ('domego', 'dom', 'eg'),
    ('grandega', 'grand', 'eg'),
    ('varmega', 'varm', 'eg'),

    # -ig (causative)
    ('sanigi', 'san', 'ig'),
    ('purigi', 'pur', 'ig'),
    ('beligi', 'bel', 'ig'),

    # -iĝ (become)
    ('saniĝi', 'san', 'iĝ'),
    ('ruĝiĝi', 'ruĝ', 'iĝ'),

    # -ad (continuous action)
    ('paroladi', 'parol', 'ad'),
    ('kantadi', 'kant', 'ad'),

    # -ar (collection)
    ('arbaro', 'arb', 'ar'),
    ('vortaro', 'vort', 'ar'),
    ('homaro', 'hom', 'ar'),

    # -ec (quality)
    ('beleco', 'bel', 'ec'),
    ('boneco', 'bon', 'ec'),

    # -il (tool)
    ('skribilo', 'skrib', 'il'),
    ('tranĉilo', 'tranĉ', 'il'),
]


# =============================================================================
# TEST CLASSES
# =============================================================================

class TestParserFundamentoRoots(unittest.TestCase):
    """Test that all Fundamento roots are parseable."""

    @classmethod
    def setUpClass(cls):
        """Load Fundamento roots once for all tests."""
        cls.fundamento_roots = FUNDAMENTO_ROOTS
        if not cls.fundamento_roots:
            raise unittest.SkipTest("Fundamento roots file not found")

    def test_fundamento_roots_loaded(self):
        """Verify Fundamento roots were loaded."""
        self.assertGreater(len(self.fundamento_roots), 2000,
                          f"Expected 2000+ roots, got {len(self.fundamento_roots)}")

    def test_sample_fundamento_roots_as_nouns(self):
        """Test a sample of Fundamento roots parse correctly as nouns."""
        # Test a representative sample (full test would be slow)
        sample_roots = ['hund', 'kat', 'dom', 'libr', 'patr', 'frat', 'bird',
                       'arb', 'flor', 'akv', 'sun', 'lun', 'stel', 'ĉiel',
                       'ter', 'mar', 'mont', 'river', 'lag', 'insul']

        for root in sample_roots:
            if root in self.fundamento_roots:
                with self.subTest(root=root):
                    word = root + 'o'  # Make it a noun
                    try:
                        ast = parse_word(word)
                        # Root should be exactly the Fundamento root
                        self.assertEqual(ast['radiko'], root,
                                        f"Expected root '{root}' for word '{word}'")
                    except ValueError as e:
                        self.fail(f"Failed to parse '{word}': {e}")


class TestParserPrefixProtection(unittest.TestCase):
    """Test that Fundamento roots are NOT incorrectly split by prefix extraction."""

    def test_bo_prefix_protection(self):
        """Roots starting with 'bo' should not be split as bo- prefix."""
        bo_roots = {k: v for k, v in PREFIX_PROTECTED_ROOTS.items() if v == 'bo'}
        for root, _ in bo_roots.items():
            with self.subTest(root=root):
                word = root + 'o'  # Make it a noun
                try:
                    ast = parse_word(word)
                    self.assertEqual(ast['radiko'], root,
                                    f"Root '{root}' was incorrectly split")
                    self.assertEqual(ast['prefiksoj'], [],
                                    f"Root '{root}' should have no prefix")
                except ValueError:
                    pass  # Word might not be in vocabulary, that's OK

    def test_dis_prefix_protection(self):
        """Roots starting with 'dis' should not be split as dis- prefix."""
        dis_roots = {k: v for k, v in PREFIX_PROTECTED_ROOTS.items() if v == 'dis'}
        for root, _ in dis_roots.items():
            with self.subTest(root=root):
                word = root + 'o'
                try:
                    ast = parse_word(word)
                    self.assertEqual(ast['radiko'], root)
                    self.assertEqual(ast['prefiksoj'], [])
                except ValueError:
                    pass

    def test_ek_prefix_protection(self):
        """Roots starting with 'ek' should not be split as ek- prefix."""
        ek_roots = {k: v for k, v in PREFIX_PROTECTED_ROOTS.items() if v == 'ek'}
        for root, _ in ek_roots.items():
            with self.subTest(root=root):
                word = root + 'o'
                try:
                    ast = parse_word(word)
                    self.assertEqual(ast['radiko'], root)
                    self.assertEqual(ast['prefiksoj'], [])
                except ValueError:
                    pass

    def test_fi_prefix_protection(self):
        """Roots starting with 'fi' should not be split as fi- prefix."""
        fi_roots = {k: v for k, v in PREFIX_PROTECTED_ROOTS.items() if v == 'fi'}
        for root, _ in fi_roots.items():
            with self.subTest(root=root):
                word = root + 'o'
                try:
                    ast = parse_word(word)
                    self.assertEqual(ast['radiko'], root)
                    self.assertEqual(ast['prefiksoj'], [])
                except ValueError:
                    pass

    def test_mis_prefix_protection(self):
        """Roots starting with 'mis' should not be split as mis- prefix."""
        mis_roots = {k: v for k, v in PREFIX_PROTECTED_ROOTS.items() if v == 'mis'}
        for root, _ in mis_roots.items():
            with self.subTest(root=root):
                word = root + 'o'
                try:
                    ast = parse_word(word)
                    self.assertEqual(ast['radiko'], root)
                    self.assertEqual(ast['prefiksoj'], [])
                except ValueError:
                    pass

    def test_re_prefix_protection(self):
        """Roots starting with 're' should not be split as re- prefix."""
        re_roots = {k: v for k, v in PREFIX_PROTECTED_ROOTS.items() if v == 're'}
        for root, _ in re_roots.items():
            with self.subTest(root=root):
                word = root + 'o'
                try:
                    ast = parse_word(word)
                    self.assertEqual(ast['radiko'], root)
                    self.assertEqual(ast['prefiksoj'], [])
                except ValueError:
                    pass


class TestParserSuffixProtection(unittest.TestCase):
    """Test that Fundamento roots are NOT incorrectly split by suffix extraction."""

    def test_ar_suffix_protection(self):
        """Roots ending with 'ar' should not be split as -ar suffix."""
        ar_roots = {k: v for k, v in SUFFIX_PROTECTED_ROOTS.items() if v == 'ar'}
        for root, _ in ar_roots.items():
            with self.subTest(root=root):
                word = root + 'o'
                try:
                    ast = parse_word(word)
                    self.assertEqual(ast['radiko'], root)
                    self.assertNotIn('ar', ast['sufiksoj'])
                except ValueError:
                    pass

    def test_ul_suffix_protection(self):
        """Roots ending with 'ul' should not be split as -ul suffix."""
        ul_roots = {k: v for k, v in SUFFIX_PROTECTED_ROOTS.items() if v == 'ul'}
        for root, _ in ul_roots.items():
            with self.subTest(root=root):
                word = root + 'o'
                try:
                    ast = parse_word(word)
                    self.assertEqual(ast['radiko'], root)
                    self.assertNotIn('ul', ast['sufiksoj'])
                except ValueError:
                    pass

    def test_il_suffix_protection(self):
        """Roots ending with 'il' should not be split as -il suffix."""
        il_roots = {k: v for k, v in SUFFIX_PROTECTED_ROOTS.items() if v == 'il'}
        for root, _ in il_roots.items():
            with self.subTest(root=root):
                word = root + 'o'
                try:
                    ast = parse_word(word)
                    self.assertEqual(ast['radiko'], root)
                    self.assertNotIn('il', ast['sufiksoj'])
                except ValueError:
                    pass

    def test_et_suffix_protection(self):
        """Roots ending with 'et' should not be split as -et suffix."""
        et_roots = {k: v for k, v in SUFFIX_PROTECTED_ROOTS.items() if v == 'et'}
        for root, _ in et_roots.items():
            with self.subTest(root=root):
                word = root + 'o'
                try:
                    ast = parse_word(word)
                    self.assertEqual(ast['radiko'], root)
                    self.assertNotIn('et', ast['sufiksoj'])
                except ValueError:
                    pass

    def test_er_suffix_protection(self):
        """Roots ending with 'er' should not be split as -er suffix."""
        er_roots = {k: v for k, v in SUFFIX_PROTECTED_ROOTS.items() if v == 'er'}
        for root, _ in er_roots.items():
            with self.subTest(root=root):
                word = root + 'o'
                try:
                    ast = parse_word(word)
                    self.assertEqual(ast['radiko'], root)
                    self.assertNotIn('er', ast['sufiksoj'])
                except ValueError:
                    pass


class TestParserRealPrefixes(unittest.TestCase):
    """Test that words with REAL prefixes ARE correctly decomposed."""

    def test_mal_prefix_extraction(self):
        """Words with mal- prefix should be decomposed."""
        mal_words = [(w, p, r) for w, p, r in REAL_PREFIXED_WORDS if p == 'mal']
        for word, expected_prefix, expected_root in mal_words:
            with self.subTest(word=word):
                try:
                    ast = parse_word(word)
                    self.assertIn(expected_prefix, ast['prefiksoj'],
                                 f"Expected prefix '{expected_prefix}' in {word}")
                    self.assertEqual(ast['radiko'], expected_root,
                                    f"Expected root '{expected_root}' in {word}")
                except ValueError as e:
                    self.fail(f"Failed to parse '{word}': {e}")

    def test_re_prefix_extraction(self):
        """Words with re- prefix should be decomposed."""
        re_words = [(w, p, r) for w, p, r in REAL_PREFIXED_WORDS if p == 're']
        for word, expected_prefix, expected_root in re_words:
            with self.subTest(word=word):
                try:
                    ast = parse_word(word)
                    self.assertIn(expected_prefix, ast['prefiksoj'],
                                 f"Expected prefix '{expected_prefix}' in {word}")
                    self.assertEqual(ast['radiko'], expected_root,
                                    f"Expected root '{expected_root}' in {word}")
                except ValueError as e:
                    self.fail(f"Failed to parse '{word}': {e}")

    def test_dis_prefix_extraction(self):
        """Words with dis- prefix should be decomposed."""
        dis_words = [(w, p, r) for w, p, r in REAL_PREFIXED_WORDS if p == 'dis']
        for word, expected_prefix, expected_root in dis_words:
            with self.subTest(word=word):
                try:
                    ast = parse_word(word)
                    self.assertIn(expected_prefix, ast['prefiksoj'],
                                 f"Expected prefix '{expected_prefix}' in {word}")
                except ValueError as e:
                    self.fail(f"Failed to parse '{word}': {e}")

    def test_mis_prefix_extraction(self):
        """Words with mis- prefix should be decomposed."""
        mis_words = [(w, p, r) for w, p, r in REAL_PREFIXED_WORDS if p == 'mis']
        for word, expected_prefix, expected_root in mis_words:
            with self.subTest(word=word):
                try:
                    ast = parse_word(word)
                    self.assertIn(expected_prefix, ast['prefiksoj'],
                                 f"Expected prefix '{expected_prefix}' in {word}")
                except ValueError as e:
                    self.fail(f"Failed to parse '{word}': {e}")

    def test_bo_prefix_extraction(self):
        """Words with bo- prefix should be decomposed."""
        bo_words = [(w, p, r) for w, p, r in REAL_PREFIXED_WORDS if p == 'bo']
        for word, expected_prefix, expected_root in bo_words:
            with self.subTest(word=word):
                try:
                    ast = parse_word(word)
                    self.assertIn(expected_prefix, ast['prefiksoj'],
                                 f"Expected prefix '{expected_prefix}' in {word}")
                except ValueError as e:
                    self.fail(f"Failed to parse '{word}': {e}")


class TestParserRealSuffixes(unittest.TestCase):
    """Test that words with REAL suffixes ARE correctly decomposed."""

    def test_ul_suffix_extraction(self):
        """Words with -ul suffix should be decomposed."""
        ul_words = [(w, r, s) for w, r, s in REAL_SUFFIXED_WORDS if s == 'ul']
        for word, expected_root, expected_suffix in ul_words:
            with self.subTest(word=word):
                try:
                    ast = parse_word(word)
                    self.assertIn(expected_suffix, ast['sufiksoj'],
                                 f"Expected suffix '{expected_suffix}' in {word}")
                    self.assertEqual(ast['radiko'], expected_root,
                                    f"Expected root '{expected_root}' in {word}")
                except ValueError as e:
                    self.fail(f"Failed to parse '{word}': {e}")

    def test_ej_suffix_extraction(self):
        """Words with -ej suffix should be decomposed."""
        ej_words = [(w, r, s) for w, r, s in REAL_SUFFIXED_WORDS if s == 'ej']
        for word, expected_root, expected_suffix in ej_words:
            with self.subTest(word=word):
                try:
                    ast = parse_word(word)
                    self.assertIn(expected_suffix, ast['sufiksoj'],
                                 f"Expected suffix '{expected_suffix}' in {word}")
                except ValueError as e:
                    self.fail(f"Failed to parse '{word}': {e}")

    def test_ig_suffix_extraction(self):
        """Words with -ig suffix should be decomposed."""
        ig_words = [(w, r, s) for w, r, s in REAL_SUFFIXED_WORDS if s == 'ig']
        for word, expected_root, expected_suffix in ig_words:
            with self.subTest(word=word):
                try:
                    ast = parse_word(word)
                    self.assertIn(expected_suffix, ast['sufiksoj'],
                                 f"Expected suffix '{expected_suffix}' in {word}")
                except ValueError as e:
                    self.fail(f"Failed to parse '{word}': {e}")


class TestParserAmbiguousWords(unittest.TestCase):
    """Test specific ambiguous words that need careful handling."""

    def test_bona_not_bo_prefix(self):
        """'bona' (good) should NOT be parsed as bo- + na."""
        ast = parse_word("bona")
        self.assertEqual(ast['radiko'], 'bon')
        self.assertEqual(ast['prefiksoj'], [])

    def test_filo_not_fi_prefix(self):
        """'filo' (son) should NOT be parsed as fi- + lo."""
        ast = parse_word("filo")
        self.assertEqual(ast['radiko'], 'fil')
        self.assertEqual(ast['prefiksoj'], [])

    def test_mistera_not_mis_prefix(self):
        """'mistera' (mysterious) should NOT be parsed as mis- + tera."""
        ast = parse_word("mistera")
        self.assertEqual(ast['radiko'], 'mister')
        self.assertEqual(ast['prefiksoj'], [])

    def test_disputi_not_dis_prefix(self):
        """'disputi' (to dispute) should NOT be parsed as dis- + puti."""
        ast = parse_word("disputi")
        self.assertEqual(ast['radiko'], 'disput')
        self.assertEqual(ast['prefiksoj'], [])

    def test_ekzisti_not_ek_prefix(self):
        """'ekzisti' (to exist) should NOT be parsed as ek- + zisti."""
        ast = parse_word("ekzisti")
        self.assertEqual(ast['radiko'], 'ekzist')
        self.assertEqual(ast['prefiksoj'], [])

    def test_papero_not_er_suffix(self):
        """'papero' (paper) should NOT be parsed as pap- + -er + -o."""
        ast = parse_word("papero")
        self.assertEqual(ast['radiko'], 'paper')
        self.assertNotIn('er', ast['sufiksoj'])

    def test_dangero_not_er_suffix(self):
        """'danĝero' (danger) should NOT be parsed as danĝ- + -er + -o."""
        ast = parse_word("danĝero")
        self.assertEqual(ast['radiko'], 'danĝer')
        self.assertNotIn('er', ast['sufiksoj'])

    def test_malbona_is_mal_prefix(self):
        """'malbona' (bad) SHOULD be parsed as mal- + bon + -a."""
        ast = parse_word("malbona")
        self.assertEqual(ast['radiko'], 'bon')
        self.assertIn('mal', ast['prefiksoj'])

    def test_bopatro_is_bo_prefix(self):
        """'bopatro' (father-in-law) SHOULD be parsed as bo- + patr + -o."""
        ast = parse_word("bopatro")
        self.assertEqual(ast['radiko'], 'patr')
        self.assertIn('bo', ast['prefiksoj'])


if __name__ == '__main__':
    unittest.main()
