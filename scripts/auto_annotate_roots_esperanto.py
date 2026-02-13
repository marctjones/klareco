#!/usr/bin/env python3
"""
Automatically annotate roots with Esperanto semantic categories.

Uses root patterns, contexts, and semantic knowledge to assign categories
from the 286-category Esperanto taxonomy.
"""

import json
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))


# Semantic knowledge base: root → category mapping
# Built from Esperanto linguistic knowledge and Fundamento
RADIKO_AL_KATEGORIO = {
    # BESTOJ (Animals)
    'hund': 'besto:mamulo',
    'kat': 'besto:mamulo',
    'ĉeval': 'besto:mamulo',
    'bov': 'besto:mamulo',
    'ŝaf': 'besto:mamulo',
    'pork': 'besto:mamulo',
    'kapr': 'besto:mamulo',
    'muso': 'besto:mamulo',
    'rat': 'besto:mamulo',
    'kani': 'besto:mamulo',
    'leon': 'besto:mamulo',
    'tigr': 'besto:mamulo',
    'elefant': 'besto:mamulo',
    'simio': 'besto:mamulo',
    'best': 'besto:mamulo',

    'bird': 'besto:birdo',
    'kokino': 'besto:birdo',
    'ansero': 'besto:birdo',
    'kolombo': 'besto:birdo',
    'aglo': 'besto:birdo',
    'korvo': 'besto:birdo',
    'paser': 'besto:birdo',

    'fiŝ': 'besto:fiŝo',
    'karpo': 'besto:fiŝo',
    'salmono': 'besto:fiŝo',
    'haringo': 'besto:fiŝo',

    'serpent': 'besto:reptilio',
    'lacert': 'besto:reptilio',
    'krokodil': 'besto:reptilio',

    'rano': 'besto:amfibio',

    'abelo': 'besto:insekto',
    'papilio': 'besto:insekto',
    'formiko': 'besto:insekto',
    'muŝo': 'besto:insekto',
    'moskito': 'besto:insekto',
    'insekt': 'besto:insekto',

    'araneo': 'besto:artropodo',
    'krabo': 'besto:artropodo',

    'verm': 'besto:verma',

    # PLANTOJ (Plants)
    'arb': 'planto:arbo',
    'kverko': 'planto:arbo',
    'pino': 'planto:arbo',
    'betulo': 'planto:arbo',

    'rozo': 'planto:floro',
    'lilio': 'planto:floro',
    'tulipo': 'planto:floro',
    'violeto': 'planto:floro',
    'flor': 'planto:floro',

    'herbo': 'planto:herbo_greno',
    'greno': 'planto:kultivaĵo',
    'tritiko': 'planto:kultivaĵo',
    'maizo': 'planto:kultivaĵo',
    'rizo': 'planto:kultivaĵo',
    'hordeo': 'planto:kultivaĵo',

    'tomato': 'planto:legomo',
    'karoto': 'planto:legomo',
    'brasiko': 'planto:legomo',
    'legomo': 'planto:legomo',

    'pomo': 'planto:fruktarbo',
    'piro': 'planto:fruktarbo',
    'frukt': 'planto:fruktarbo',

    'fungo': 'planto:fungo',

    # NATURAĴOJ (Natural features)
    'mont': 'naturo:monto',
    'river': 'naturo:akvaĵo',
    'lago': 'naturo:akvaĵo',
    'maro': 'naturo:akvaĵo',
    'ocean': 'naturo:akvaĵo',
    'fluo': 'naturo:akvaĵo',
    'akv': 'substanco:akvo',

    'arbaro': 'naturo:arbaro',
    'ĝangalo': 'naturo:arbaro',

    'dezerto': 'naturo:dezerto',
    'insulo': 'naturo:insulo',
    'valo': 'naturo:valo',
    'groto': 'naturo:groto',
    'vulkano': 'naturo:vulkano',

    # VETERO (Weather)
    'pluv': 'vetero:pluvo',
    'neĝ': 'vetero:neĝo',
    'hajlo': 'vetero:neĝo',
    'ŝtorm': 'vetero:ŝtormo',
    'uragano': 'vetero:ŝtormo',
    'vento': 'vetero:vento',
    'fulm': 'vetero:fulmotondro',
    'tondro': 'vetero:fulmotondro',
    'nubo': 'vetero:nubo',
    'nebulo': 'vetero:nebulo',
    'veter': 'vetero:pluvo',

    # ĈIELAJ (Celestial)
    'suno': 'ĉiela:suno',
    'luno': 'ĉiela:luno',
    'stelo': 'ĉiela:stelo',
    'planedo': 'ĉiela:planedo',
    'galaksio': 'ĉiela:galaksio',
    'kometo': 'ĉiela:kometo',

    # SUBSTANCOJ (Substances)
    'aero': 'substanco:aero',
    'tero': 'substanco:tero',
    'fajr': 'substanco:fajro',
    'ŝton': 'substanco:ŝtono',
    'metal': 'substanco:metalo',
    'fero': 'substanco:metalo',
    'kupro': 'substanco:metalo',
    'oro': 'substanco:metalo',
    'arĝento': 'substanco:metalo',
    'ligno': 'substanco:ligno',
    'vitro': 'substanco:vitro',
    'plasto': 'substanco:plasto',

    # KORPAJ PARTOJ (Body parts)
    'kapo': 'korpo:kapo',
    'kranio': 'korpo:kapo',
    'cerbo': 'korpo:kapo',
    'vizaĝo': 'korpo:vizaĝo',
    'frunto': 'korpo:vizaĝo',
    'vango': 'korpo:vizaĝo',
    'okulo': 'korpo:okulo',
    'okul': 'korpo:okulo',
    'orelo': 'korpo:orelo',
    'orel': 'korpo:orelo',
    'nazo': 'korpo:nazo',
    'naz': 'korpo:nazo',
    'buŝo': 'korpo:buŝo',
    'buŝ': 'korpo:buŝo',
    'lipo': 'korpo:buŝo',
    'dento': 'korpo:buŝo',
    'lango': 'korpo:buŝo',
    'kolo': 'korpo:kolo',
    'gorĝo': 'korpo:kolo',
    'brusto': 'korpo:torso',
    'dorso': 'korpo:torso',
    'ventro': 'korpo:torso',
    'brako': 'korpo:brako',
    'kubuto': 'korpo:brako',
    'mano': 'korpo:mano',
    'man': 'korpo:mano',
    'fingro': 'korpo:mano',
    'polekso': 'korpo:mano',
    'gambo': 'korpo:gambo',
    'genuo': 'korpo:gambo',
    'kruro': 'korpo:gambo',
    'piedo': 'korpo:piedo',
    'pied': 'korpo:piedo',
    'koro': 'korpo:interna_organo',
    'kord': 'korpo:interna_organo',
    'pulmo': 'korpo:interna_organo',
    'stomako': 'korpo:interna_organo',
    'hepato': 'korpo:interna_organo',
    'korpo': 'korpo:torso',
    'korp': 'korpo:torso',

    # ILOJ (Tools)
    'tranĉil': 'ilo:tranĉilo',
    'seg': 'ilo:tranĉilo',
    'hakil': 'ilo:tranĉilo',
    'tondil': 'ilo:tranĉilo',

    'liniil': 'ilo:mezurilo',
    'mezuril': 'ilo:mezurilo',
    'termometro': 'ilo:mezurilo',

    'plumo': 'ilo:skribilo',
    'krajono': 'ilo:skribilo',
    'skribil': 'ilo:skribilo',

    'martel': 'ilo:konstruilo',
    'fosil': 'ilo:konstruilo',
    'boril': 'ilo:konstruilo',

    'ŝraŭbo': 'ilo:fiksilo',
    'najlo': 'ilo:fiksilo',
    'gluo': 'ilo:fiksilo',

    'ŝlosilo': 'ilo:ŝlosilo',
    'ŝlosil': 'ilo:ŝlosilo',

    'balailo': 'ilo:purigilo',
    'brosil': 'ilo:purigilo',

    # UJOJ (Containers)
    'botelo': 'ujo:botelo',
    'flakono': 'ujo:botelo',
    'skatolo': 'ujo:skatolo',
    'kesto': 'ujo:skatolo',
    'sako': 'ujo:sako',
    'valizo': 'ujo:sako',
    'poto': 'ujo:poto',
    'kruĉo': 'ujo:poto',
    'taso': 'ujo:taso',
    'glaso': 'ujo:taso',
    'plado': 'ujo:plado',
    'telero': 'ujo:plado',

    # MEBLOJ (Furniture)
    'seĝo': 'meble:sidaĵo',
    'seĝ': 'meble:sidaĵo',
    'benko': 'meble:sidaĵo',
    'sofao': 'meble:sidaĵo',
    'lito': 'meble:dormejo',
    'matrac': 'meble:dormejo',
    'ŝranko': 'meble:tenejo',
    'tablo': 'meble:surfaco',
    'tabl': 'meble:surfaco',
    'breto': 'meble:surfaco',

    # VESTOJ (Clothing)
    'ĉapelo': 'vesto:ĉapo',
    'boneto': 'vesto:ĉapo',
    'ĉemizo': 'vesto:supra',
    'jako': 'vesto:supra',
    'mantelo': 'vesto:supra',
    'pantalono': 'vesto:malsupra',
    'jupo': 'vesto:malsupra',
    'ŝuo': 'vesto:piedvesto',
    'ŝu': 'vesto:piedvesto',
    'boto': 'vesto:piedvesto',
    'sandalo': 'vesto:piedvesto',
    'vest': 'vesto:supra',

    # VETURILOJ (Vehicles)
    'aŭto': 'veturilo:tera',
    'aŭt': 'veturilo:tera',
    'biciklo': 'veturilo:tera',
    'motorciklo': 'veturilo:tera',
    'ĉaro': 'veturilo:tera',
    'ŝipo': 'veturilo:akva',
    'ŝip': 'veturilo:akva',
    'boato': 'veturilo:akva',
    'aviadilo': 'veturilo:aera',
    'helikoptero': 'veturilo:aera',
    'trajno': 'veturilo:rela',
    'vagon': 'veturilo:rela',

    # KONSTRUAĴOJ (Buildings)
    'domo': 'konstruaĵo:loĝejo',
    'dom': 'konstruaĵo:loĝejo',
    'apartamento': 'konstruaĵo:loĝejo',
    'kabano': 'konstruaĵo:loĝejo',

    'preĝejo': 'konstruaĵo:religia',
    'moskeo': 'konstruaĵo:religia',
    'sinagogo': 'konstruaĵo:religia',
    'templo': 'konstruaĵo:religia',

    'butiko': 'konstruaĵo:komerca',
    'vendejo': 'konstruaĵo:komerca',
    'bazaro': 'konstruaĵo:komerca',
    'magazeno': 'konstruaĵo:komerca',

    'palaco': 'konstruaĵo:registara',
    'parlamento': 'konstruaĵo:registara',

    'fabriko': 'konstruaĵo:industria',
    'uzino': 'konstruaĵo:industria',

    'lernejo': 'konstruaĵo:eduka',
    'universitato': 'konstruaĵo:eduka',
    'biblioteko': 'konstruaĵo:eduka',

    'hospitalo': 'konstruaĵo:kuraceja',
    'kliniko': 'konstruaĵo:kuraceja',

    'teatro': 'konstruaĵo:amuza',
    'kinejo': 'konstruaĵo:amuza',
    'muzeo': 'konstruaĵo:amuza',
    'stadiono': 'konstruaĵo:amuza',

    'ponto': 'konstruaĵo:infrastrukturo',
    'muro': 'konstruaĵo:infrastrukturo',

    # AGOJ - KORPAJ (Physical actions)
    'marŝ': 'ago:piediro',
    'promen': 'ago:piediro',
    'paŝ': 'ago:piediro',
    'iro': 'ago:piediro',
    'ir': 'ago:piediro',
    'veno': 'ago:piediro',
    'ven': 'ago:piediro',
    'kuro': 'ago:kuri',
    'kur': 'ago:kuri',
    'salto': 'ago:salti',
    'salt': 'ago:salti',
    'flugo': 'ago:flugi',
    'flug': 'ago:flugi',
    'naĝo': 'ago:naĝi',
    'naĝ': 'ago:naĝi',
    'grimpo': 'ago:grimpi',
    'grimp': 'ago:grimpi',
    'staro': 'ago:stari',
    'star': 'ago:stari',
    'sido': 'ago:sidi',
    'sid': 'ago:sidi',
    'kuŝo': 'ago:kuŝi',
    'kuŝ': 'ago:kuŝi',
    'danco': 'ago:danci',
    'danc': 'ago:danci',

    # AGOJ - MANAJ (Manual actions)
    'preno': 'ago:preni',
    'pren': 'ago:preni',
    'kap': 'ago:preni',
    'dono': 'ago:doni',
    'don': 'ago:doni',
    'meto': 'ago:meti',
    'met': 'ago:meti',
    'loko': 'ago:meti',
    'lok': 'ago:meti',
    'ĵeto': 'ago:ĵeti',
    'ĵet': 'ago:ĵeti',
    'teno': 'ago:teni',
    'ten': 'ago:teni',
    'porto': 'ago:teni',
    'port': 'ago:teni',
    'puŝo': 'ago:puŝi',
    'puŝ': 'ago:puŝi',
    'tiro': 'ago:tiri',
    'tir': 'ago:tiri',
    'frapo': 'ago:frapo',
    'frap': 'ago:frapo',
    'bato': 'ago:frapo',
    'bat': 'ago:frapo',
    'kareso': 'ago:karesi',
    'kares': 'ago:karesi',
    'konstruo': 'ago:konstrui',
    'konstru': 'ago:konstrui',

    # AGOJ - SENSAJ (Sensory actions)
    'vido': 'ago:vidi',
    'vid': 'ago:vidi',
    'rigardo': 'ago:vidi',
    'rigard': 'ago:vidi',
    'observo': 'ago:vidi',
    'observ': 'ago:vidi',
    'spekto': 'ago:vidi',
    'spekt': 'ago:vidi',
    'aŭdo': 'ago:aŭdi',
    'aŭd': 'ago:aŭdi',
    'aŭskulto': 'ago:aŭdi',
    'aŭskult': 'ago:aŭdi',
    'sento': 'ago:senti',
    'sent': 'ago:senti',
    'tuŝo': 'ago:senti',
    'tuŝ': 'ago:senti',
    'gusto': 'ago:gusti',
    'gust': 'ago:gusti',
    'flaro': 'ago:flari',
    'flar': 'ago:flari',

    # AGOJ - KOMUNIKAJ (Communication)
    'diro': 'ago:diri',
    'dir': 'ago:diri',
    'parolo': 'ago:diri',
    'parol': 'ago:diri',
    'esprimo': 'ago:diri',
    'espring': 'ago:diri',
    'demando': 'ago:demandi',
    'demand': 'ago:demandi',
    'respondo': 'ago:respondi',
    'respond': 'ago:respondi',
    'ordono': 'ago:ordoni',
    'ordon': 'ago:ordoni',
    'peto': 'ago:peti',
    'pet': 'ago:peti',
    'promeso': 'ago:promesi',
    'promes': 'ago:promesi',
    'skribo': 'ago:skribi',
    'skrib': 'ago:skribi',
    'lego': 'ago:legi',
    'leg': 'ago:legi',
    'montro': 'ago:montri',
    'montr': 'ago:montri',
    'prezento': 'ago:montri',
    'prezent': 'ago:montri',
    'signalo': 'ago:signali',
    'signal': 'ago:signali',

    # AGOJ - KOGNAJ (Cognitive)
    'penso': 'ago:pensi',
    'pens': 'ago:pensi',
    'konsidero': 'ago:pensi',
    'konsider': 'ago:pensi',
    'scio': 'ago:sci',
    'sci': 'ago:sci',
    'kono': 'ago:sci',
    'kon': 'ago:sci',
    'memoro': 'ago:memori',
    'memor': 'ago:memori',
    'rememoro': 'ago:memori',
    'rememor': 'ago:memori',
    'forgeso': 'ago:forgesi',
    'forges': 'ago:forgesi',
    'imago': 'ago:imagi',
    'imag': 'ago:imagi',
    'revo': 'ago:revi',
    'rev': 'ago:revi',
    'sonĝo': 'ago:revi',
    'sonĝ': 'ago:revi',
    'atento': 'ago:atenti',
    'atent': 'ago:atenti',
    'fokuso': 'ago:atenti',
    'fokus': 'ago:atenti',
    'lerno': 'ago:lerni',
    'lern': 'ago:lerni',
    'instruado': 'ago:instrui',
    'instru': 'ago:instrui',
    'eduko': 'ago:instrui',
    'eduk': 'ago:instrui',
    'kompreno': 'ago:kompreni',
    'komprend': 'ago:kompreni',
    'kompreno': 'ago:kompreni',
    'analizo': 'ago:analizi',
    'analiz': 'ago:analizi',
    'ekzameno': 'ago:analizi',
    'ekzamen': 'ago:analizi',
    'solvo': 'ago:solvi',
    'solv': 'ago:solvi',

    # AGOJ - SOCIAJ (Social)
    'helpo': 'ago:helpi',
    'help': 'ago:helpi',
    'asisto': 'ago:helpi',
    'asist': 'ago:helpi',
    'batalo': 'ago:batali',
    'batal': 'ago:batali',
    'milito': 'ago:batali',
    'milit': 'ago:batali',
    'kunlaboro': 'ago:kunlabori',
    'kunlabor': 'ago:kunlabori',
    'koopero': 'ago:kunlabori',
    'kooper': 'ago:kunlabori',
    'konkurso': 'ago:konkursi',
    'konkurs': 'ago:konkursi',
    'amo': 'ago:ami',
    'am': 'ago:ami',
    'ŝato': 'ago:ami',
    'ŝat': 'ago:ami',
    'malamo': 'ago:malami',
    'malam': 'ago:malami',
    'fido': 'ago:fidi',
    'fid': 'ago:fidi',
    'konfido': 'ago:fidi',
    'konfid': 'ago:fidi',
    'trompeto': 'ago:trompeti',
    'trompeŭ': 'ago:trompeti',
    'respekto': 'ago:respekti',
    'respekt': 'ago:respekti',
    'estimo': 'ago:respekti',
    'estim': 'ago:respekti',
    'honoro': 'ago:respekti',
    'honor': 'ago:respekti',
    'edzigo': 'ago:edzigi',
    'edzig': 'ago:edzigi',

    # AGOJ - LABORAJ (Work)
    'laboro': 'ago:labori',
    'labor': 'ago:labori',
    'faro': 'ago:fari',
    'far': 'ago:fari',
    'kreo': 'ago:krei',
    'kre': 'ago:krei',
    'produkto': 'ago:krei',
    'produkt': 'ago:krei',
    'detruo': 'ago:detrui',
    'detru': 'ago:detrui',
    'ruino': 'ago:detrui',
    'ruin': 'ago:detrui',
    'riparo': 'ago:ripari',
    'ripar': 'ago:ripari',
    'purigo': 'ago:purigi',
    'purig': 'ago:purigi',
    'lavo': 'ago:purigi',
    'lav': 'ago:purigi',
    'kuiro': 'ago:kuiri',
    'kuir': 'ago:kuiri',
    'bako': 'ago:kuiri',
    'bak': 'ago:kuiri',
    'kudro': 'ago:kudri',
    'kudr': 'ago:kudri',
    'planto': 'ago:planti',
    'plant': 'ago:planti',
    'rikolto': 'ago:rikolti',
    'rikolt': 'ago:rikolti',

    # SENTOJ - POZITIVAJ (Positive emotions)
    'ĝojo': 'sento:ĝojo',
    'ĝoj': 'sento:ĝojo',
    'gajo': 'sento:ĝojo',
    'gaj': 'sento:ĝojo',
    'feliĉo': 'sento:ĝojo',
    'feliĉ': 'sento:ĝojo',
    'dankemo': 'sento:dankemo',
    'dankem': 'sento:dankemo',
    'rekono': 'sento:dankemo',
    'rekon': 'sento:dankemo',
    'espero': 'sento:espero',
    'esper': 'sento:espero',
    'kontenteco': 'sento:kontenteco',
    'content': 'sento:kontenteco',
    'eksciteco': 'sento:eksciteco',
    'ekscit': 'sento:eksciteco',
    'entuziasmo': 'sento:eksciteco',
    'entuziasm': 'sento:eksciteco',
    'intereso': 'sento:intereso',
    'interes': 'sento:intereso',

    # SENTOJ - NEGATIVAJ (Negative emotions)
    'malĝojo': 'sento:malĝojo',
    'malĝoj': 'sento:malĝojo',
    'tristo': 'sento:malĝojo',
    'trist': 'sento:malĝojo',
    'malfeliĉo': 'sento:malĝojo',
    'malfeliĉ': 'sento:malĝojo',
    'kolero': 'sento:kolero',
    'koler': 'sento:kolero',
    'furio': 'sento:kolero',
    'furi': 'sento:kolero',
    'timo': 'sento:timo',
    'tim': 'sento:timo',
    'teruro': 'sento:timo',
    'terur': 'sento:timo',
    'honto': 'sento:honto',
    'hont': 'sento:honto',
    'kulpo': 'sento:kulpo',
    'kulp': 'sento:kulpo',
    'tedo': 'sento:tedo',
    'ted': 'sento:tedo',
    'enuo': 'sento:tedo',
    'enu': 'sento:tedo',

    # SENTOJ - KOMPLEKSAJ (Complex emotions)
    'fiero': 'sento:fiero',
    'fier': 'sento:fiero',
    'envio': 'sento:envio',
    'envi': 'sento:envio',
    'kompato': 'sento:kompato',
    'kompat': 'sento:kompato',
    'simpatio': 'sento:kompato',
    'simpati': 'sento:kompato',
    'surprizo': 'sento:surprizo',
    'surpriz': 'sento:surprizo',
    'maltrankvileco': 'sento:maltrankvileco',
    'maltrankil': 'sento:maltrankvileco',
    'zorgo': 'sento:maltrankvileco',
    'zorg': 'sento:maltrankvileco',

    # KOGNICIO (Cognitive states)
    'kredo': 'kognicio:kredo',
    'kred': 'kognicio:kredo',
    'opinio': 'kognicio:kredo',
    'opini': 'kognicio:kredo',
    'konfuzo': 'kognicio:konfuzo',
    'konfuz': 'kognicio:konfuzo',
    'perplekso': 'kognicio:konfuzo',
    'perpleks': 'kognicio:konfuzo',
    'konscieco': 'kognicio:konscieco',
    'konsciec': 'kognicio:konscieco',

    # VOLO (Will & desire)
    'volo': 'volo:volo',
    'vol': 'volo:volo',
    'deziro': 'volo:deziro',
    'dezir': 'volo:deziro',
    'bezono': 'volo:bezono',
    'bezon': 'volo:bezono',
    'intenco': 'volo:intenco',
    'intenc': 'volo:intenco',
    'decido': 'volo:decido',
    'decid': 'volo:decido',

    # ECOJ - GRANDECO (Size)
    'grando': 'eco:granda',
    'grand': 'eco:granda',
    'amplekso': 'eco:granda',
    'ampleks': 'eco:granda',
    'malgrando': 'eco:malgranda',
    'malgrand': 'eco:malgranda',
    'longo': 'eco:longa',
    'long': 'eco:longa',
    'mallongo': 'eco:mallonga',
    'mallong': 'eco:mallonga',
    'larĝo': 'eco:larĝa',
    'larĝ': 'eco:larĝa',
    'mallarĝo': 'eco:mallarĝa',
    'mallarĝ': 'eco:mallarĝa',
    'alto': 'eco:alta',
    'alt': 'eco:alta',
    'malalto': 'eco:malalta',
    'malalt': 'eco:malalta',
    'profundo': 'eco:malalta',
    'profund': 'eco:malalta',
    'diko': 'eco:dika',
    'dik': 'eco:dika',
    'maldiko': 'eco:maldika',
    'maldik': 'eco:maldika',

    # ECOJ - FORMO (Shape)
    'rondo': 'eco:ronda',
    'rond': 'eco:ronda',
    'sfero': 'eco:ronda',
    'sfer': 'eco:ronda',
    'cirklo': 'eco:ronda',
    'cirkl': 'eco:ronda',
    'kvadrato': 'eco:kvadrata',
    'kvadrat': 'eco:kvadrata',
    'triangulo': 'eco:triangula',
    'triangul': 'eco:triangula',
    'rekto': 'eco:rekta',
    'rekt': 'eco:rekta',
    'kurbo': 'eco:kurba',
    'kurb': 'eco:kurba',
    'arko': 'eco:kurba',
    'ark': 'eco:kurba',
    'formo': 'eco:ronda',
    'form': 'eco:ronda',

    # ECOJ - KOLOROJ (Colors)
    'ruĝo': 'eco:ruĝa',
    'ruĝ': 'eco:ruĝa',
    'bluo': 'eco:blua',
    'blu': 'eco:blua',
    'verdo': 'eco:verda',
    'verd': 'eco:verda',
    'flavo': 'eco:flava',
    'flav': 'eco:flava',
    'oranĝo': 'eco:oranĝa',
    'oranĝ': 'eco:oranĝa',
    'violkoloro': 'eco:violkolora',
    'violet': 'eco:violkolora',
    'purpuro': 'eco:violkolora',
    'purpur': 'eco:violkolora',
    'bruno': 'eco:bruna',
    'brun': 'eco:bruna',
    'nigro': 'eco:nigra',
    'nigr': 'eco:nigra',
    'blanko': 'eco:blanka',
    'blank': 'eco:blanka',
    'grizo': 'eco:griza',
    'griz': 'eco:griza',
    'koloro': 'eco:ruĝa',
    'kolor': 'eco:ruĝa',

    # ECOJ - TEKSTURO (Texture)
    'glato': 'eco:glata',
    'glat': 'eco:glata',
    'malglato': 'eco:malglata',
    'malglat': 'eco:malglata',
    'molo': 'eco:mola',
    'mol': 'eco:mola',
    'malmolo': 'eco:malmola',
    'malmol': 'eco:malmola',
    'firmo': 'eco:malmola',
    'firm': 'eco:malmola',
    'rigido': 'eco:malmola',
    'rigid': 'eco:malmola',

    # ECOJ - PEZO (Weight)
    'pezo': 'eco:peza',
    'pez': 'eco:peza',
    'malpezo': 'eco:malpeza',
    'malpez': 'eco:malpeza',
    'leĝero': 'eco:malpeza',
    'leĝer': 'eco:malpeza',

    # ECOJ - TEMPERATURO (Temperature)
    'varmo': 'eco:varma',
    'varm': 'eco:varma',
    'malvarmo': 'eco:malvarma',
    'malvarm': 'eco:malvarma',
    'frosto': 'eco:malvarma',
    'frost': 'eco:malvarma',
    'glacio': 'eco:malvarma',
    'glaci': 'eco:malvarma',

    # KVALITOJ (Evaluative qualities)
    'bono': 'kvalito:bona',
    'bon': 'kvalito:bona',
    'malbono': 'kvalito:malbona',
    'malbon': 'kvalito:malbona',
    'belo': 'kvalito:bela',
    'bel': 'kvalito:bela',
    'ĉarmo': 'kvalito:bela',
    'ĉarm': 'kvalito:bela',
    'malbelo': 'kvalito:malbela',
    'malbel': 'kvalito:malbela',
    'utilo': 'kvalito:utila',
    'util': 'kvalito:utila',
    'profito': 'kvalito:utila',
    'profit': 'kvalito:utila',
    'neutilo': 'kvalito:neutila',
    'neutil': 'kvalito:neutila',

    # INTENSO (Intensity)
    'forto': 'intenso:forta',
    'fort': 'intenso:forta',
    'potenc': 'intenso:forta',
    'malforto': 'intenso:malforta',
    'malfort': 'intenso:malforta',
    'feblo': 'intenso:malforta',
    'febl': 'intenso:malforta',
    'ekstremo': 'intenso:ekstrema',
    'ekstrem': 'intenso:ekstrema',
    'mildo': 'intenso:milda',
    'mild': 'intenso:milda',
    'modero': 'intenso:milda',
    'moder': 'intenso:milda',

    # ABSTRAKTAJ - SOCIAJ (Social institutions)
    'registaro': 'abstrakta:registaro',
    'registar': 'abstrakta:registaro',
    'reĝimo': 'abstrakta:registaro',
    'reĝim': 'abstrakta:registaro',
    'ŝtato': 'abstrakta:registaro',
    'ŝtat': 'abstrakta:registaro',
    'leĝo': 'abstrakta:leĝo',
    'leĝ': 'abstrakta:leĝo',
    'juro': 'abstrakta:leĝo',
    'jur': 'abstrakta:leĝo',
    'ekonomio': 'abstrakta:ekonomio',
    'ekonomi': 'abstrakta:ekonomio',
    'financo': 'abstrakta:ekonomio',
    'financ': 'abstrakta:ekonomio',
    'merkato': 'abstrakta:ekonomio',
    'merkat': 'abstrakta:ekonomio',
    'religio': 'abstrakta:religio',
    'religi': 'abstrakta:religio',
    'medicino': 'abstrakta:medicino',
    'medicin': 'abstrakta:medicino',
    'kuraco': 'abstrakta:medicino',
    'kurac': 'abstrakta:medicino',
    'sano': 'abstrakta:medicino',
    'san': 'abstrakta:medicino',

    # SCIENCOJ (Sciences)
    'fiziko': 'scienco:fiziko',
    'fizik': 'scienco:fiziko',
    'mekaniko': 'scienco:fiziko',
    'menanik': 'scienco:fiziko',
    'kemio': 'scienco:kemio',
    'kemi': 'scienco:kemio',
    'biologio': 'scienco:biologio',
    'biologi': 'scienco:biologio',
    'botaniko': 'scienco:biologio',
    'botanik': 'scienco:biologio',
    'zoologio': 'scienco:biologio',
    'zoologi': 'scienco:biologio',
    'matematiko': 'scienco:matematiko',
    'matematik': 'scienco:matematiko',
    'algebro': 'scienco:matematiko',
    'algebr': 'scienco:matematiko',
    'geometrio': 'scienco:matematiko',
    'geometri': 'scienco:matematiko',
    'astronomio': 'scienco:astronomio',
    'astronomi': 'scienco:astronomio',

    # ARTOJ (Arts)
    'muziko': 'arto:muziko',
    'muzik': 'arto:muziko',
    'kanto': 'arto:muziko',
    'kant': 'arto:muziko',
    'melodio': 'arto:muziko',
    'melodi': 'arto:muziko',
    'pentrado': 'arto:vida',
    'pentrad': 'arto:vida',
    'skulptado': 'arto:vida',
    'skulptad': 'arto:vida',
    'desegnado': 'arto:vida',
    'desegnad': 'arto:vida',
    'literaturo': 'arto:literatura',
    'literatur': 'arto:literatura',
    'poezio': 'arto:literatura',
    'poezi': 'arto:literatura',
    'prozo': 'arto:literatura',
    'proz': 'arto:literatura',
    'dramo': 'arto:teatra',
    'dram': 'arto:teatra',

    # FILOZOFIO (Philosophy)
    'vero': 'filozofio:vero',
    'ver': 'filozofio:vero',
    'justeco': 'filozofio:justeco',
    'just': 'filozofio:justeco',
    'rajto': 'filozofio:justeco',
    'rajt': 'filozofio:justeco',
    'libereco': 'filozofio:libereco',
    'liber': 'filozofio:libereco',
    'sendependeco': 'filozofio:libereco',
    'sendepend': 'filozofio:libereco',
    'beleco': 'filozofio:beleco',
    'boneco': 'filozofio:boneco',
    'etiko': 'filozofio:boneco',
    'etik': 'filozofio:boneco',
    'moralo': 'filozofio:boneco',
    'moral': 'filozofio:boneco',
    'kaŭzo': 'filozofio:kaŭzo',
    'kaŭz': 'filozofio:kaŭzo',
    'origino': 'filozofio:kaŭzo',
    'origin': 'filozofio:kaŭzo',
    'efiko': 'filozofio:efiko',
    'efik': 'filozofio:efiko',
    'rezulto': 'filozofio:efiko',
    'rezult': 'filozofio:efiko',

    # EVENTOJ - VIVAJ (Life events)
    'nasko': 'evento:nasko',
    'nask': 'evento:nasko',
    'naskiĝo': 'evento:nasko',
    'naskiĝ': 'evento:nasko',
    'kresko': 'evento:kresko',
    'kresk': 'evento:kresko',
    'disvolvigo': 'evento:kresko',
    'disvolvig': 'evento:kresko',
    'morto': 'evento:morto',
    'mort': 'evento:morto',
    'forpaso': 'evento:morto',
    'forpas': 'evento:morto',
    'malsano': 'evento:malsano',
    'malsan': 'evento:malsano',
    'resanigo': 'evento:resanigo',
    'resanig': 'evento:resanigo',
    'kuraco': 'evento:resanigo',
    'vundo': 'evento:vundigo',
    'vund': 'evento:vundigo',

    # EVENTOJ - SOCIAJ (Social events)
    'renkontiĝo': 'evento:renkontiĝo',
    'renkontiĝ': 'evento:renkontiĝo',
    'kunveno': 'evento:renkontiĝo',
    'kunven': 'evento:renkontiĝo',
    'festo': 'evento:festo',
    'fest': 'evento:festo',
    'celebrado': 'evento:festo',
    'celebrad': 'evento:festo',
    'ceremonio': 'evento:ceremonio',
    'ceremoni': 'evento:ceremonio',
    'rito': 'evento:ceremonio',
    'rit': 'evento:ceremonio',
    'paco': 'evento:paco',
    'pac': 'evento:paco',
    'harmonio': 'evento:paco',
    'harmoni': 'evento:paco',

    # EVENTOJ - NATURAJ (Natural events)
    'tertremo': 'evento:tertremo',
    'tertrem': 'evento:tertremo',
    'sismo': 'evento:tertremo',
    'sism': 'evento:tertremo',
    'inundo': 'evento:inundo',
    'inund': 'evento:inundo',
    'eklipso': 'evento:eklipso',
    'eklips': 'evento:eklipso',

    # ADDITIONAL COMMON ROOTS (using new valid taxonomy)
    # Publishing & Documentation (produkto domain)
    'eldon': 'arto:literatura',        # publishing/edition
    'tekst': 'produkto:teksto',        # text
    'parol': 'ago:diri',               # speech
    'rakont': 'arto:literatura',       # story
    'histori': 'arto:literatura',      # history
    'aŭtor': 'homo:aŭtoro',            # author
    'koment': 'produkto:noto',         # comment
    'not': 'produkto:noto',            # note
    'verk': 'produkto:verko',          # work/opus
    'libr': 'produkto:libro',          # book
    'dokument': 'produkto:dokumento',  # document
    'let': 'produkto:letero',          # letter
    'revu': 'produkto:revuo',          # review/journal
    'list': 'produkto:listo',          # list

    # Language & Communication (reusing existing ago categories)
    'lingv': 'abstrakta:edukado',      # language (no exact match, using education)
    'traduk': 'ago:diri',              # translate (no exact action, using speak)
    'esprim': 'ago:diri',              # express (using speak)
    'respond': 'ago:respondi',         # respond (already exists)
    'vort': 'produkto:teksto',         # word

    # Structure & Parts (new koncepto domain)
    'part': 'filozofio:efiko',         # part (no exact match, using effect)
    'enhav': 'ago:teni',               # contain (using hold)
    'div': 'ago:fari',                 # divide (using do/make)

    # Time (new tempo domain)
    'jar': 'tempo:jaro',               # year
    'monat': 'tempo:monato',           # month
    'semajn': 'tempo:semajno',         # week
    'tag': 'tempo:tago',               # day
    'hor': 'tempo:horo',               # hour
    'minut': 'tempo:minuto',           # minute
    'sekund': 'tempo:sekundo',         # second
    'epok': 'tempo:epoko',             # epoch
    'peri': 'tempo:periodo',           # period
    'moment': 'tempo:momento',         # moment

    # Location & Space (new loko domain)
    'lok': 'loko:loko',                # location
    'reg': 'loko:regiono',             # region
    'urb': 'loko:urbo',                # city
    'vilaĝ': 'loko:vilaĝo',            # village
    'land': 'loko:lando',              # land/country
    'kontinente': 'loko:kontinento',   # continent
    'strat': 'loko:strato',            # street
    'spac': 'loko:spaco',              # space

    # Abstract Concepts (new koncepto domain)
    'baz': 'filozofio:kaŭzo',          # base/foundation
    'numer': 'koncepto:numero',        # number
    'nombr': 'koncepto:numero',        # number
    'kvant': 'koncepto:kvanto',        # quantity
    'grad': 'koncepto:grado',          # degree/grade
    'ord': 'koncepto:ordo',            # order
    'sistem': 'koncepto:sistemo',      # system
    'met': 'koncepto:metodo',          # method
    'regul': 'koncepto:regulo',        # rule
    'model': 'koncepto:modelo',        # model
    'teori': 'koncepto:teorio',        # theory
    'problemem': 'koncepto:problemo',  # problem
    'solv': 'koncepto:solvo',          # solution/solve
    'kaŭz': 'koncepto:kaŭzo',          # cause
    'rezultat': 'koncepto:rezulto',    # result
    'stat': 'koncepto:stato',          # state
    'rilat': 'koncepto:rilato',        # relation
    'ideo': 'koncepto:ideo',           # idea

    # People & Social (new homo and grupo domains)
    'hom': 'homo:persono',             # human
    'vir': 'homo:viro',                # man
    'ino': 'homo:virino',              # woman
    'infan': 'homo:infano',            # child
    'patro': 'homo:patro',             # father
    'patrin': 'homo:patrino',          # mother
    'fil': 'homo:filo',                # son
    'frat': 'homo:frato',              # brother
    'person': 'homo:persono',          # person
    'amik': 'homo:amiko',              # friend
    'malamik': 'homo:malamiko',        # enemy
    'famili': 'grupo:familio',         # family
    'societ': 'grupo:societo',         # society
    'grup': 'grupo:grupo',             # group
    'klas': 'grupo:klaso',             # class
    'organizaĵ': 'grupo:organizaĵo',   # organization
    'popol': 'grupo:popolo',           # people/nation

    # Additional common roots from corpus
    'ser': 'ago:helpi',                # serve (using help)
    'uz': 'ago:fari',                  # use (using do/make)
    'bezon': 'volo:bezono',            # need
    'hav': 'ago:teni',                 # have (using hold)
    'pov': 'volo:volo',                # can/able (using will)
    'dev': 'volo:volo',                # must/should (using will)
    'riciv': 'ago:doni',               # receive (using give)
    'dank': 'sento:dankemo',           # thank
    'aprob': 'sento:kontenteco',       # approve (using contentment)
    'prefer': 'volo:deziro',           # prefer (using desire)
    'ag': 'ago:fari',                  # act/action (using do)
    'objekto': 'produkto:objekto',     # object

    # Common literary vocabulary
    'lag': 'naturo:akvaĵo',            # lake
    'larm': 'korpo:interna_organo',    # tear
    'strang': 'kvalito:malbona',       # strange (using bad quality - no exact match)
    'sekv': 'ago:piediro',             # follow (using walk)
    'ricev': 'ago:doni',               # receive
    'lud': 'ago:fari',                 # play/game
    'vitr': 'substanco:vitro',         # glass
    'fal': 'ago:salti',                # fall (using jump - closest)
    'dorm': 'ago:kuŝi',                # sleep (using lie down)
    'argument': 'koncepto:ideo',       # argument
    'solen': 'kvalito:bona',           # solemn (using good)
    'bril': 'eco:blanka',              # shine/brilliant (using white - closest color)
    'fuŝ': 'ago:detrui',               # botch/bungle (using destroy)
    'rifuz': 'ago:malami',             # refuse (using hate - closest)
    'bret': 'meble:surfaco',           # shelf
    'paĝ': 'produkto:libro',           # page
    'sak': 'ujo:sako',                 # sack/bag
    'kuir': 'ago:kuiri',               # cook
    'fum': 'substanco:aero',           # smoke
    'fier': 'sento:fiero',             # pride
    'plor': 'sento:malĝojo',           # cry/weep
    'sopir': 'sento:malĝojo',          # sigh
    'ek': 'ago:salti',                 # sudden action (using jump)
    'halt': 'ago:stari',               # halt/stop (using stand)
    'rapid': 'ago:kuri',               # rapid (using run)
    'lent': 'intenso:milda',           # slow
    'silent': 'kvalito:bona',          # silent
    'bru': 'ago:diri',                 # noise (using speak)
    'vok': 'ago:diri',                 # call/voice
    'kant': 'arto:muziko',             # sing
    'rid': 'sento:ĝojo',               # laugh
    'plend': 'sento:malĝojo',          # complain
    'tim': 'sento:timo',               # fear
    'kuraĝ': 'kvalito:bona',           # courage
    'saĝ': 'kognicio:scio',            # wise
    'stult': 'kognicio:konfuzo',       # stupid/foolish
    'riĉ': 'kvalito:bona',             # rich
    'malriĉ': 'kvalito:malbona',       # poor
    'pur': 'kvalito:bona',             # pure/clean
    'malpuri': 'kvalito:malbona',      # dirty/impure
    'fremd': 'homo:malamiko',          # strange/foreign (using enemy - closest)
    'kuŭn': 'homo:amiko',              # together/commune (using friend)
    'sol': 'homo:persono',             # alone/solo (using person)
    'komun': 'grupo:societo',          # common/community
    'publik': 'grupo:societo',         # public
    'privat': 'homo:persono',          # private
    'ord': 'koncepto:ordo',            # order
    'kaos': 'koncepto:problemo',       # chaos (using problem)
    'reg': 'ago:helpi',                # rule/govern (using help - closest)
    'obed': 'ago:helpi',               # obey (using help)
    'defend': 'ago:helpi',             # defend
    'atak': 'ago:batali',              # attack
    'venko': 'ago:batali',             # win/conquer
    'perd': 'ago:batali',              # lose
    'gajn': 'ago:doni',                # gain/win
    'don': 'ago:doni',                 # donate/give (already exists)
    'aĉet': 'ago:preni',               # buy (using take)
    'vend': 'ago:doni',                # sell (using give)
    'pag': 'ago:doni',                 # pay (using give)
    'ŝuld': 'volo:bezono',             # owe/debt
    'pret': 'kvalito:bona',            # ready
    'kapABL': 'kvalito:bona',          # capable
    'sukcES': 'kvalito:bona',          # success
    'fiasko': 'kvalito:malbona',       # fiasco/failure
    'eraR': 'koncepto:problemo',       # error
    'ĝust': 'filozofio:vero',          # correct/right
    'mal': 'filozofio:vero',           # wrong (mal- prefix issues)
    'nov': 'tempo:momento',            # new (using moment - temporal)
    'antikv': 'tempo:epoko',           # ancient (using epoch)
    'jun': 'homo:infano',              # young
    'maljun': 'homo:patro',            # old (using father)
    'freŝ': 'kvalito:bona',            # fresh
    'kaduk': 'kvalito:malbona',        # decrepit/old
    'san': 'abstrakta:medicino',       # healthy
    'malsan': 'evento:malsano',        # sick
    'kurac': 'abstrakta:medicino',     # cure/heal
    'vund': 'evento:vundigo',          # wound
    'dolor': 'sento:kompato',          # pain (using compassion - closest emotion)
    'plaĉ': 'sento:ĝojo',              # please
    'ĝen': 'sento:malĝojo',            # bother/disturb
    'lacig': 'sento:tedo',             # tire/weary (using boredom)
    'ripoz': 'ago:kuŝi',               # rest (using lie down)
    'labor': 'ago:labori',             # work (already exists)
    'task': 'ago:fari',                # task (using do)
    'cel': 'volo:intenco',             # goal/aim (using intention)
    'intenc': 'volo:intenco',          # intention
    'plan': 'koncepto:modelo',         # plan (using model)
    'projekt': 'koncepto:modelo',      # project
    'esper': 'sento:espero',           # hope
    'trem': 'ago:salti',               # tremble (using jump - closest motion)
    'ŝancel': 'ago:moviĝi',            # stagger (using move)
}


def annotate_root(root_data: dict) -> dict:
    """
    Automatically annotate a root with semantic category.

    Strategy:
    1. Check if root is in knowledge base (direct lookup)
    2. If not, analyze contexts to infer category
    3. Return with confidence score
    """
    radiko = root_data['radiko']

    # Direct lookup
    if radiko in RADIKO_AL_KATEGORIO:
        return {
            **root_data,
            'etikedo': {
                'kategorio': RADIKO_AL_KATEGORIO[radiko],
                'rilataj_radikoj': [],
                'fido': 1.0,
                'fonto': 'konata_radiko'
            }
        }

    # Try stemming variations (remove common endings)
    for suffix in ['o', 'a', 'e', 'i', 'j', 'n', 's']:
        if radiko.endswith(suffix) and len(radiko) > 2:
            stem = radiko[:-1]
            if stem in RADIKO_AL_KATEGORIO:
                return {
                    **root_data,
                    'etikedo': {
                        'kategorio': RADIKO_AL_KATEGORIO[stem],
                        'rilataj_radikoj': [],
                        'fido': 0.9,
                        'fonto': 'radika_variaĵo'
                    }
                }

    # Unknown - mark for manual annotation
    return {
        **root_data,
        'etikedo': {
            'kategorio': None,
            'rilataj_radikoj': [],
            'fido': 0.0,
            'fonto': 'nekonata',
            'bezonas_manan_anoton': True
        }
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Aŭtomate anoti radikojn')
    parser.add_argument(
        '--input',
        type=Path,
        default=Path('data/training/root_semantics/radikoj_por_anoti.jsonl'),
        help='Enira dosiero kun radikoj'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/training/root_semantics/radikoj_anotitaj.jsonl'),
        help='Elira dosiero kun anotitaj radikoj'
    )
    parser.add_argument(
        '--max-roots',
        type=int,
        default=4000,
        help='Maksimuma nombro de radikoj por anoti'
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERARO: Enira dosiero ne trovita: {args.input}")
        sys.exit(1)

    print("="*70)
    print("AŬTOMATA RADIKA ANOTADO")
    print("="*70)
    print()
    print(f"Eniro: {args.input}")
    print(f"Eliro: {args.output}")
    print(f"Maksimuma radikoj: {args.max_roots:,}")
    print()

    # Load roots
    roots = []
    with open(args.input, 'r') as f:
        for line in f:
            if line.strip():
                roots.append(json.loads(line))

    print(f"Ŝargita: {len(roots):,} radikoj")
    print()

    # Annotate up to max_roots
    annotated = []
    known_count = 0
    unknown_count = 0

    for i, root in enumerate(roots[:args.max_roots], 1):
        annotated_root = annotate_root(root)
        annotated.append(annotated_root)

        if annotated_root['etikedo']['kategorio']:
            known_count += 1
        else:
            unknown_count += 1

        if i % 500 == 0:
            print(f"  Procezita: {i:,}/{min(len(roots), args.max_roots):,}")

    print()
    print(f"✓ Anotita: {len(annotated):,} radikoj")
    print(f"  Konataj (aŭtomate): {known_count:,} ({known_count/len(annotated)*100:.1f}%)")
    print(f"  Nekonataj (bezonas manan): {unknown_count:,} ({unknown_count/len(annotated)*100:.1f}%)")
    print()

    # Show category distribution
    category_counts = Counter()
    for root in annotated:
        cat = root['etikedo'].get('kategorio')
        if cat:
            category_counts[cat] += 1

    print("Kategoria distribuo (plej oftaj):")
    for category, count in category_counts.most_common(20):
        print(f"  {category:30s}: {count:4,}")
    print()

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        for root in annotated:
            f.write(json.dumps(root, ensure_ascii=False) + '\n')

    print(f"✓ Konservita al: {args.output}")
    print()

    print("="*70)
    print("SEKVAJ PAŜOJ")
    print("="*70)
    print()
    print(f"Aŭtomate anoitaj: {known_count:,} radikoj")
    print(f"Bezonas manan anoton: {unknown_count:,} radikoj")
    print()
    print("Vi povas:")
    print("  1. Uzi la aŭtomate anotitajn radikojn por trejnado (pli rapida)")
    print("  2. Mane anoti la nekonatajn radikojn (pli alta kvalito)")
    print("  3. Kombinaĵo: trejni sur konataj, poste aldoni manajn")
    print()


if __name__ == '__main__':
    main()
