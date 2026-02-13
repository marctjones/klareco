"""
Radiko Semantiko (Root Semantics) Taxonomy for Klareco.

Pure Esperanto semantic categories for ROOT meanings.
These categories capture semantic information NOT available from grammar.

Philosophy:
- Grammar (deterministic) handles: vortspeco, afiksoj, gramatikaj roloj
- This taxonomy handles: What does the ROOT mean semantically?

Structure: ~270 categories organized by semantic domain
All labels in Esperanto (Pure Esperanto philosophy)
"""

from enum import Enum
from typing import Dict, List, Set


# ============================================================================
# VIVANTAJ ESTAĴOJ (Living Beings) - ~50 categories
# ============================================================================

class BestoTipo(Enum):
    """Bestoj (Animals) - semantic types by biological classification."""
    MAMULO = "besto:mamulo"              # hund, kat, ĉeval, bov, ŝaf, porko
    BIRDO = "besto:birdo"                # kokino, ansero, kolombo, aglo, korvo
    FIŜO = "besto:fiŝo"                  # karpo, salmono, haringo, tonuso
    REPTILIO = "besto:reptilio"          # serpento, lacerto, krokodilo
    AMFIBIO = "besto:amfibio"            # rano, salamandro
    INSEKTO = "besto:insekto"            # abelo, papilio, formiko, muŝo
    ARTROPODO = "besto:artropodo"        # araneo, skorpio, krabo
    MOLUSKO = "besto:molusko"            # limako, ostro, polipo
    VERMA = "besto:verma"                # tervermo, ringelvermo


class PlantoTipo(Enum):
    """Plantoj (Plants) - semantic types by plant classification."""
    ARBO = "planto:arbo"                 # kverko, pino, betulo, pomo-arbo
    FLORO = "planto:floro"               # rozo, lilio, tulipo, violeto
    HERBO_SPICO = "planto:herbo_spico"   # baziliko, mento, timiano, origan
    HERBO_GRENO = "planto:herbo_greno"   # herbo (grass), bambuo
    KULTIVAĴO = "planto:kultivaĵo"       # maizo, rizo, tritiko, hordeo
    LEGOMO = "planto:legomo"             # tomato, karoto, brasiko, salato
    FRUKTARBO = "planto:fruktarbo"       # pomoarbo, pirarbo, ĉerizarbo
    ARBUSTO = "planto:arbusto"           # rozo-arbusto, vinbero-arbusto
    VINO_GRIMPA = "planto:vino_grimpa"   # vino, hedera, kaprifolio
    FUNGO = "planto:fungo"               # fungo, ŝimpo, gisto


class NaturaĴoTipo(Enum):
    """Naturaĵoj (Natural entities) - non-living natural things."""
    MONTO = "naturo:monto"               # monto, pinto, altaĵo, montaro
    AKVAĴO = "naturo:akvaĵo"             # rivero, lago, maro, oceano, fluo
    ARBARO = "naturo:arbaro"             # arbaro, ĝangalo, taigo
    DEZERTO = "naturo:dezerto"           # dezerto, stepo, savano
    INSULO = "naturo:insulo"             # insulo, duoninsulo, atolio
    VALO = "naturo:valo"                 # valo, ravino, kanjono
    GROTO = "naturo:groto"               # groto, kaverno, kaveto
    VULKANO = "naturo:vulkano"           # vulkano, geizero, fumarolo
    GLACIO = "naturo:glacio"             # glaĉero, glacia_areo


class VeteroTipo(Enum):
    """Veteraĵoj (Weather phenomena)."""
    PLUVO = "vetero:pluvo"               # pluvo, pluvego, gutoj
    NEĜO = "vetero:neĝo"                 # neĝo, neĝero, hajlo
    ŜTORMO = "vetero:ŝtormo"             # ŝtormo, uragano, tifono, tornado
    VENTO = "vetero:vento"               # vento, ventego, brizo
    FULMOTONDRO = "vetero:fulmotondro"   # fulmo, tondro, ekbrilo
    NUBO = "vetero:nubo"                 # nubo, nebulo, bromo
    NEBULO = "vetero:nebulo"             # nebulo, densa nebulo


class ĈielaTipo(Enum):
    """Ĉielaj korpoj (Celestial bodies)."""
    SUNO = "ĉiela:suno"                  # suno, stelo
    LUNO = "ĉiela:luno"                  # luno, satelito
    PLANEDO = "ĉiela:planedo"            # planedo, Marso, Jupitero
    STELO = "ĉiela:stelo"                # stelo, nova stelo
    GALAKSIO = "ĉiela:galaksio"          # galaksio, nebulajo
    KOMETO = "ĉiela:kometo"              # kometo, asteroido, meteoro


class SubstancoTipo(Enum):
    """Substancoj (Substances)."""
    AKVO = "substanco:akvo"              # akvo, likvaĵo
    AERO = "substanco:aero"              # aero, gaso, vaporo
    TERO = "substanco:tero"              # tero, grundo, polvo, sablo
    FAJRO = "substanco:fajro"            # fajro, flamo, brulado
    ŜTONO = "substanco:ŝtono"            # ŝtono, roko, marmoro, granito
    METALO = "substanco:metalo"          # fero, kupro, oro, arĝento
    LIGNO = "substanco:ligno"            # ligno, planko, tabulo
    VITRO = "substanco:vitro"            # vitro, kristalo
    PLASTO = "substanco:plasto"          # plasto, polimero, rezino


class KorpaPartoPipo(Enum):
    """Korpaj partoj (Body parts)."""
    KAPO = "korpo:kapo"                  # kapo, kranio, cerbo
    VIZAĜO = "korpo:vizaĝo"              # vizaĝo, frunto, vango
    OKULO = "korpo:okulo"                # okulo, pupilo, retino
    ORELO = "korpo:orelo"                # orelo, aŭdsistemo
    NAZO = "korpo:nazo"                  # nazo, naztruo
    BUŜO = "korpo:buŝo"                  # buŝo, lipo, dento, lango
    KOLO = "korpo:kolo"                  # kolo, gorĝo, faringo
    TORSO = "korpo:torso"                # torso, brusto, dorso, ventro
    BRAKO = "korpo:brako"                # brako, kubuto, mano
    MANO = "korpo:mano"                  # mano, fingro, polekso, palmo
    GAMBO = "korpo:gambo"                # gambo, femuro, genuo, kruro
    PIEDO = "korpo:piedo"                # piedo, kalkano, piedingro
    INTERNA_ORGANO = "korpo:interna_organo"  # koro, pulmo, stomako, hepato


# ============================================================================
# ARTEFARITAĴOJ (Artifacts) - ~40 categories
# ============================================================================

class IloTipo(Enum):
    """Iloj (Tools) - by function."""
    TRANĈILO = "ilo:tranĉilo"            # tranĉilo, seg, hakilo, tondilo
    MEZURILO = "ilo:mezurilo"            # liniilo, mezurilo, termometro
    SKRIBILO = "ilo:skribilo"            # plumo, krajono, peno, markilo
    KONSTRUILO = "ilo:konstruilo"        # martel, fosilo, borilo, turnilo
    FIKSILO = "ilo:fiksilo"              # ŝraŭbo, najlo, boltilo, gluo
    ŜLOSILO = "ilo:ŝlosilo"              # ŝlosilo, pinĉilo, prenilo
    PURIGILO = "ilo:purigilo"            # balailo, lavilo, brosilo


class UjoTipo(Enum):
    """Ujoj (Containers)."""
    BOTELO = "ujo:botelo"                # botelo, flakono, karafo
    SKATOLO = "ujo:skatolo"              # skatolo, kesto, kofro
    SAKO = "ujo:sako"                    # sako, valizo, pakaĵo
    POTO = "ujo:poto"                    # poto, kruĉo, vazo
    TASO = "ujo:taso"                    # taso, glaso, pokalo
    PLADO = "ujo:plado"                  # plado, telero, pelvo


class MebleTipo(Enum):
    """Mebloj (Furniture)."""
    SIDAĴO = "meble:sidaĵo"              # seĝo, benko, sofao, tabureto
    DORMEJO = "meble:dormejo"            # lito, kuŝejo, matrac
    TENEJO = "meble:tenejo"              # ŝranko, tirkesto, bretaro
    SURFACO = "meble:surfaco"            # tablo, breto, pupitro


class VestoTipo(Enum):
    """Vestoj (Clothing)."""
    ĈAPO = "vesto:ĉapo"                  # ĉapelo, boneto, turban
    SUPRA = "vesto:supra"                # ĉemizo, jako, mantelo, veŝto
    MALSUPRA = "vesto:malsupra"          # pantalono, jupo, ŝorto
    PIEDVESTO = "vesto:piedvesto"        # ŝuo, boto, sandalo, pantoflo
    AKCESORAĴO = "vesto:akcesoraĵo"      # zono, kravato, glovo, ŝtrumpo


class VeturiloTipo(Enum):
    """Veturiloj (Vehicles)."""
    TERA = "veturilo:tera"               # aŭto, biciklo, motorciklo, ĉaro
    AKVA = "veturilo:akva"               # ŝipo, boato, vaporŝipo, submarŝipo
    AERA = "veturilo:aera"               # aviadilo, helikoptero, balono
    RELA = "veturilo:rela"               # trajno, vagonaro, metroo


class KonstruaĴoTipo(Enum):
    """Konstruaĵoj (Buildings/Structures)."""
    LOĜEJO = "konstruaĵo:loĝejo"         # domo, apartamento, kabano
    RELIGIA = "konstruaĵo:religia"       # preĝejo, moskeo, sinagogo, templo
    KOMERCA = "konstruaĵo:komerca"       # butiko, vendejo, bazaro, magazeno
    REGISTARA = "konstruaĵo:registara"   # palaco, parlamento, oficejo
    INDUSTRIA = "konstruaĵo:industria"   # fabriko, uzino, laborejo
    EDUKA = "konstruaĵo:eduka"           # lernejo, universitato, biblioteko
    KURACEJA = "konstruaĵo:kuraceja"     # hospitalo, kliniko, ambulanco
    AMUZA = "konstruaĵo:amuza"           # teatro, kinejo, muzeo, stadiono
    INFRASTRUKTURO = "konstruaĵo:infrastrukturo"  # ponto, tubo, digo, muro


class InstrumentoTipo(Enum):
    """Instrumentoj (Instruments)."""
    MUZIKA = "instrumento:muzika"        # piano, violono, gitaro, fluto
    SCIENCA = "instrumento:scienca"      # mikroskopo, teleskopo, spektroskopo
    KURACISTA = "instrumento:kuracista"  # stetoskopo, skalpelo, siringo
    OPTIKA = "instrumento:optika"        # lenso, prismo, okultuko


# ============================================================================
# AGOJ (Actions) - ~60 categories
# ============================================================================

class KorpaAgoTipo(Enum):
    """Korpaj agoj (Physical actions) - bodily movement."""
    PIEDIRO = "ago:piediro"              # marŝi, promeni, paŝi, iri
    KURI = "ago:kuri"                    # kuri, rapidi, sprinti
    SALTI = "ago:salti"                  # salti, eksalti, hopsalti
    FLUGI = "ago:flugi"                  # flugi, flugeti, sveni
    NAĜI = "ago:naĝi"                    # naĝi, dronfiŝi
    GRIMPI = "ago:grimpi"                # grimpi, supreniri
    STARI = "ago:stari"                  # stari, staradi, sin teni
    SIDI = "ago:sidi"                    # sidi, sidadi
    KUŜI = "ago:kuŝi"                    # kuŝi, kuŝadi, ripozi
    DANCI = "ago:danci"                  # danci, svingigi


class ManaAgoTipo(Enum):
    """Manaj agoj (Manual actions) - hand manipulation."""
    PRENI = "ago:preni"                  # preni, kapti, ekpreni
    DONI = "ago:doni"                    # doni, transdoni, liveri
    METI = "ago:meti"                    # meti, loki, starigi
    ĴETI = "ago:ĵeti"                    # ĵeti, lanĉi, flugigi
    TENI = "ago:teni"                    # teni, teni firme, porti
    PUŜI = "ago:puŝi"                    # puŝi, premi, sturmi
    TIRI = "ago:tiri"                    # tiri, treni, ŝiri
    FRAPO = "ago:frapo"                  # frapi, bati, marteli
    KARESI = "ago:karesi"                # karesi, tuŝeti, glate tuŝi
    KONSTRUI = "ago:konstrui"            # konstrui, bari, aranĝi


class SensaAgoTipo(Enum):
    """Sensaj agoj (Sensory actions) - perception."""
    VIDI = "ago:vidi"                    # vidi, rigardi, observi, spekti
    AŬDI = "ago:aŭdi"                    # aŭdi, aŭskulti
    SENTI = "ago:senti"                  # senti, tuŝi, palpi
    GUSTI = "ago:gusti"                  # gusti, gustumi, sensapori
    FLARI = "ago:flari"                  # flari, senodori, senti_odoron


class KomunikaAgoTipo(Enum):
    """Komunikaj agoj (Communication actions)."""
    DIRI = "ago:diri"                    # diri, paroli, eksprimi, prononci
    DEMANDI = "ago:demandi"              # demandi, informiĝi, esplori
    RESPONDI = "ago:respondi"            # respondi, rememori, kontraŭdiri
    ORDONI = "ago:ordoni"                # ordoni, komandi, postuli
    PETI = "ago:peti"                    # peti, mendigi, implori
    PROMESI = "ago:promesi"              # promesi, garantii, sindevigi
    SKRIBI = "ago:skribi"                # skribi, registri, noti
    LEGI = "ago:legi"                    # legi, tralegi, studlegi
    MONTRI = "ago:montri"                # montri, prezenti, elmontri
    SIGNALI = "ago:signali"              # signali, signi, gestadi


class KognaAgoTipo(Enum):
    """Kognaj agoj (Cognitive actions)."""
    PENSI = "ago:pensi"                  # pensi, konsideri, pripensi
    SCI = "ago:sci"                      # sci, koni, kompreni
    MEMORI = "ago:memori"                # memori, rememori, rekolekti
    FORGESI = "ago:forgesi"              # forgesi, ne-memori
    IMAGI = "ago:imagi"                  # imagi, bildi, figuri
    REVI = "ago:revi"                    # revi, sonĝi
    ATENTI = "ago:atenti"                # atenti, zorgi, fokusiĝi
    LERNI = "ago:lerni"                  # lerni, ellerni, edukiĝi
    INSTRUI = "ago:instrui"              # instrui, eduki, doktrinigi
    KOMPRENI = "ago:kompreni"            # kompreni, ekkompreni, kapti
    ANALIZI = "ago:analizi"              # analizi, ekzameni, dissekci
    SOLVI = "ago:solvi"                  # solvi, elfari, eltrovadi


class SociaAgoTipo(Enum):
    """Sociaj agoj (Social actions)."""
    HELPI = "ago:helpi"                  # helpi, asisti, apogi
    BATALI = "ago:batali"                # batali, militi, lukti
    KUNLABORI = "ago:kunlabori"          # kunlabori, kooperi, labori kune
    KONKURSI = "ago:konkursi"            # konkursi, rivali, lukti
    AMI = "ago:ami"                      # ami, ŝati, adori
    MALAMI = "ago:malami"                # malami, antipatiami
    FIDI = "ago:fidi"                    # fidi, konfidi, kredi al
    TROMPETI = "ago:trompeti"            # trompeti, malconfidi, perfidi
    RESPEKTI = "ago:respekti"            # respekti, estimi, honori
    EDZIGI = "ago:edzigi"                # edzigi, geedzigi, nutiĝi


class LaboraAgoTipo(Enum):
    """Laboraj agoj (Work actions)."""
    LABORI = "ago:labori"                # labori, okupiĝi
    FARI = "ago:fari"                    # fari, efektivigi, plenuim
    KREI = "ago:krei"                    # krei, produkti, estigigo
    DETRUI = "ago:detrui"                # detrui, ruinigi, neniigi
    RIPARI = "ago:ripari"                # ripari, fliki, rebonigi
    PURIGIRI = "ago:purigi"              # purigi, lavi, malpurigi
    KUIRI = "ago:kuiri"                  # kuiri, baki, rostadi, boligi
    KUDRI = "ago:kudri"                  # kudri, stopi, brodi
    PLANTI = "ago:planti"                # planti, semi, kultivi
    RIKOLTI = "ago:rikolti"              # rikolti, rikolti, enmeti


# ============================================================================
# SENTOJ KAJ MENSAJ STATOJ (Feelings & Mental States) - ~30 categories
# ============================================================================

class PozitivaSentoTipo(Enum):
    """Pozitivaj sentoj (Positive emotions)."""
    ĜOJO = "sento:ĝojo"                  # ĝojo, gaja, feliĉa
    AMO = "sento:amo"                    # amo, kara, aminda
    DANKEMO = "sento:dankemo"            # dankemo, rekona
    ESPERO = "sento:espero"              # espero, optimisma, fidanta
    KONTENTECO = "sento:kontenteco"      # kontenteco, kontenta, satiga
    EKSCITECO = "sento:eksciteco"        # eksciteco, entuziasma, vigla
    INTERESO = "sento:intereso"          # intereso, scivola, allogita


class NegativaSentoTipo(Enum):
    """Negativaj sentoj (Negative emotions)."""
    MALĜOJO = "sento:malĝojo"            # malĝojo, trista, malgaja, afliktita
    KOLERO = "sento:kolero"              # kolero, furioza, indigna
    TIMO = "sento:timo"                  # timo, timema, teruriga
    MALAMO = "sento:malamo"              # malamo, malamika, antipatio
    HONTO = "sento:honto"                # honto, hontema, embarasita
    KULPO = "sento:kulpo"                # kulpo, kulpa, pentrida
    MALARANKO = "sento:malaranko"        # malĝeno, ĝeno, malkontenta
    TEDO = "sento:tedo"                  # tedo, enuo, malamikeco


class KompleksaSentoTipo(Enum):
    """Kompleksaj sentoj (Complex emotions)."""
    FIERO = "sento:fiero"                # fiero, fiera, digna
    ENVIO = "sento:envio"                # envio, enviema, jealosa
    KOMPATO = "sento:kompato"            # kompato, simpatia, kompata
    SURPRIZO = "sento:surprizo"          # surprizo, miriga, neatendita
    MALTRANKVILECO = "sento:maltrankvileco"  # maltrankvileco, zorgema, timema


class KogniciaTipo(Enum):
    """Kognicaj statoj (Cognitive states)."""
    SCIO = "kognicio:scio"               # scio, konscio, klera
    KREDO = "kognicio:kredo"             # kredo, kredema, opinio
    KOMPRENO = "kognicio:kompreno"       # kompreno, komprenado, intelekto
    KONFUZO = "kognicio:konfuzo"         # konfuzo, perpleksa, miksita
    MEMORO = "kognicio:memoro"           # memoro, rememoro, rekolekto
    FORGESO = "kognicio:forgeso"         # forgeso, oblivio, ne-memoro
    ATENTO = "kognicio:atento"           # atento, koncentriĝo, fokuso
    KONSCIECO = "kognicio:konscieco"     # konscieco, vigla, maldorminta


class VolaTipo(Enum):
    """Volo kaj deziro (Will & desire)."""
    VOLO = "volo:volo"                   # volo, voli, intenci
    DEZIRO = "volo:deziro"               # deziro, dezirega, sopiri
    BEZONO = "volo:bezono"               # bezono, necesa, postuli
    INTENCO = "volo:intenco"             # intenco, celi, planisto
    DECIDO = "volo:decido"               # decido, determino, elekti


# ============================================================================
# ECOJ KAJ KVALITOJ (Properties & Qualities) - ~40 categories
# ============================================================================

class GrandecoTipo(Enum):
    """Grandeco (Size)."""
    GRANDA = "eco:granda"                # granda, ampleksa, vasta
    MALGRANDA = "eco:malgranda"          # malgranda, eta, maldika
    LONGA = "eco:longa"                  # longa, elongita, etenda
    MALLONGA = "eco:mallonga"            # mallonga, kurta, stumpa
    LARĜA = "eco:larĝa"                  # larĝa, ampleksa, spaca
    MALLARĜA = "eco:mallarĝa"            # mallarĝa, streta, angusta
    ALTA = "eco:alta"                    # alta, levita, supera
    MALALTA = "eco:malalta"              # malalta, profunda, subera
    DIKA = "eco:dika"                    # dika, ampleksa, korpulenta
    MALDIKA = "eco:maldika"              # maldika, svelta, streĉa


class FormoTipo(Enum):
    """Formo (Shape)."""
    RONDA = "eco:ronda"                  # ronda, sfera, cirkla
    KVADRATA = "eco:kvadrata"            # kvadrata, ortangula
    TRIANGULA = "eco:triangula"          # triangula, trilatera
    REKTA = "eco:rekta"                  # rekta, linia, senstaria
    KURBA = "eco:kurba"                  # kurba, arkita, fleksita


class KoloroTipo(Enum):
    """Koloroj (Colors)."""
    RUĜA = "eco:ruĝa"                    # ruĝa, skarlacho, rozkolora
    BLUA = "eco:blua"                    # blua, ĉielblua, marblua
    VERDA = "eco:verda"                  # verda, herbkolora
    FLAVA = "eco:flava"                  # flava, ora, citrona
    ORANĜA = "eco:oranĝa"                # oranĝa, oranĝkolora
    VIOLKOLORA = "eco:violkolora"        # violkolora, purpura
    BRUNA = "eco:bruna"                  # bruna, ĉokoladkolora
    NIGRA = "eco:nigra"                  # nigra, mallumakolora
    BLANKA = "eco:blanka"                # blanka, neĝkolora, pura
    GRIZA = "eco:griza"                  # griza, cindrokolora


class TeksturoTipo(Enum):
    """Teksturo (Texture)."""
    GLATA = "eco:glata"                  # glata, satena, slipema
    MALGLATA = "eco:malglata"            # malglata, aspra, rugita
    MOLA = "eco:mola"                    # mola, delikata, butersimila
    MALMOLA = "eco:malmola"              # malmola, firma, rigida, fortika
    VISKOZA = "eco:viskoza"              # viskoza, glua, gluanta


class PezoTipo(Enum):
    """Pezo (Weight)."""
    PEZA = "eco:peza"                    # peza, malfacilportebla
    MALPEZA = "eco:malpeza"              # malpeza, leĝera, senpeza


class TemperaturoTipo(Enum):
    """Temperaturo (Temperature)."""
    VARMA = "eco:varma"                  # varma, varmega, brulanta
    MALVARMA = "eco:malvarma"            # malvarma, malvarmega, glacia, frosta


class EvaluaTipo(Enum):
    """Evaluaj kvalitoj (Evaluative qualities)."""
    BONA = "kvalito:bona"                # bona, bonkvalita, perfekta
    MALBONA = "kvalito:malbona"          # malbona, misa, difekta
    BELA = "kvalito:bela"                # bela, belega, ĉarma
    MALBELA = "kvalito:malbela"          # malbela, naĝra, repuŝa
    UTILA = "kvalito:utila"              # utila, profitdona, praktika
    NEUTILA = "kvalito:neutila"          # neutila, senvalora, vana


class IntensoTipo(Enum):
    """Intenso (Intensity)."""
    FORTA = "intenso:forta"              # forta, potenca, vigora
    MALFORTA = "intenso:malforta"        # malforta, febla, senforta
    EKSTREMA = "intenso:ekstrema"        # ekstrema, tremega, ekscesa
    MILDA = "intenso:milda"              # milda, modera, trankvileta


# ============================================================================
# ABSTRAKTAJ KONCEPTOJ (Abstract Concepts) - ~30 categories
# ============================================================================

class SociaInstitucioTipo(Enum):
    """Sociaj institucioj (Social institutions)."""
    REGISTARO = "abstrakta:registaro"    # registaro, reĝimo, ŝtato
    LEĜO = "abstrakta:leĝo"              # leĝo, juro, regulamento
    EKONOMIO = "abstrakta:ekonomio"      # ekonomio, financo, merkato
    RELIGIO = "abstrakta:religio"        # religio, fido, kredo
    EDUKADO = "abstrakta:edukado"        # edukado, instruado, klerigado
    MEDICINO = "abstrakta:medicino"      # medicino, kuracado, sano


class SciencaTipo(Enum):
    """Sciencoj (Sciences)."""
    FIZIKO = "scienco:fiziko"            # fiziko, mekaniko, optiko
    KEMIO = "scienco:kemio"              # kemio, organika kemio
    BIOLOGIO = "scienco:biologio"        # biologio, botaniko, zoologio
    MATEMATIKO = "scienco:matematiko"    # matematiko, algebro, geometrio
    ASTRONOMIO = "scienco:astronomio"    # astronomio, kosmoologio


class ArtoTipo(Enum):
    """Artoj (Arts)."""
    MUZIKO = "arto:muziko"               # muziko, kanto, melodio
    VIDA = "arto:vida"                   # pentrado, skulptado, desegnado
    LITERATURA = "arto:literatura"       # literaturo, poezio, prozo
    TEATRA = "arto:teatra"               # teatro, dramo, aktoro
    DANCA = "arto:danca"                 # danco, baleo, koreografio


class FilozofioTipo(Enum):
    """Filozofiaj konceptoj (Philosophical concepts)."""
    VERO = "filozofio:vero"              # vero, verecо, aŭtentikeco
    JUSTECO = "filozofio:justeco"        # justeco, rajteco, egaleco
    LIBERECO = "filozofio:libereco"      # libereco, sendependeco, aŭtonomio
    BELECO = "filozofio:beleco"          # beleco, estetiko, harmonio
    BONECO = "filozofio:boneco"          # boneco, etiko, moralo
    KAŬZO = "filozofio:kaŭzo"            # kaŭzo, kaŭzado, origino
    EFIKO = "filozofio:efiko"            # efiko, rezulto, konsekvenco


# ============================================================================
# EVENTOJ KAJ PROCESOJ (Events & Processes) - ~20 categories
# ============================================================================

class VivaEventoTipo(Enum):
    """Vivaj eventoj (Life events)."""
    NASKO = "evento:nasko"               # nasko, naskiĝo
    KRESKO = "evento:kresko"             # kresko, disvolvigo, evoluego
    MORTO = "evento:morto"               # morto, forpaso, fino
    MALSANO = "evento:malsano"           # malsano, sufero, malsanuleco
    RESANIGO = "evento:resanigo"         # resanigo, kuracigo, sanigo
    VUNDIGO = "evento:vundigo"           # vundigo, damaĝo, lezio


class SociaEventoTipo(Enum):
    """Sociaj eventoj (Social events)."""
    RENKONTIĜO = "evento:renkontiĝo"     # renkontiĝo, kunveno, asembleo
    FESTO = "evento:festo"               # festo, celebrado, gajeco
    CEREMONIO = "evento:ceremonio"       # ceremonio, rito, solenaĵo
    MILITO = "evento:milito"             # milito, batalo, konfliktego
    PACO = "evento:paco"                 # paco, harmonio, trankvilo
    EDZIGO = "evento:edzigo"             # edzigo, geedzo, nupto


class NaturaEventoTipo(Enum):
    """Naturaj eventoj (Natural events)."""
    TERTREMO = "evento:tertremo"         # tertremo, sismo
    ŜTORMO = "evento:ŝtormo"             # ŝtormo, uragano
    INUNDO = "evento:inundo"             # inundo, superakvo, diluvio
    FAJRO = "evento:fajro"               # fajro, brulego, incendio
    EKLIPSO = "evento:eklipso"           # eklipso, suneklipso, luneklipso


# ============================================================================
# HOMOJ KAJ GRUPOJ (People & Groups) - ~25 categories
# ============================================================================

class HomoTipo(Enum):
    """Homoj kaj personoj (People and persons)."""
    PERSONO = "homo:persono"             # persono, individuo, homo
    VIRO = "homo:viro"                   # viro, virseksulo
    VIRINO = "homo:virino"               # virino, ino
    INFANO = "homo:infano"               # infano, bebo, junulo
    PATRO = "homo:patro"                 # patro, gepatroj
    PATRINO = "homo:patrino"             # patrino, panjo
    FILO = "homo:filo"                   # filo, gefiloj
    FRATO = "homo:frato"                 # frato, gefratoj
    AMIKO = "homo:amiko"                 # amiko, kamarado
    MALAMIKO = "homo:malamiko"           # malamiko, kontraŭulo
    AŬTORO = "homo:aŭtoro"               # aŭtoro, verkisto, kreinto


class GrupoTipo(Enum):
    """Grupoj kaj organizaĵoj (Groups and organizations)."""
    FAMILIO = "grupo:familio"            # familio, gento, parenco
    SOCIETO = "grupo:societo"            # societo, socio, komunumo
    GRUPO = "grupo:grupo"                # grupo, kolektivo, aro
    KLASO = "grupo:klaso"                # klaso, kategorio, nivelo
    ORGANIZAĴO = "grupo:organizaĵo"      # organizaĵo, asocio, instituto
    POPOLO = "grupo:popolo"              # popolo, nacio, etnio


# ============================================================================
# TEMPO KAJ SPACO (Time & Space) - ~20 categories
# ============================================================================

class TempoTipo(Enum):
    """Tempo kaj periodoj (Time and periods)."""
    JARO = "tempo:jaro"                  # jaro, jarcentо, epoko
    MONATO = "tempo:monato"              # monato, lunmonato
    SEMAJNO = "tempo:semajno"            # semajno, semajnfino
    TAGO = "tempo:tago"                  # tago, diurno
    HORO = "tempo:horo"                  # horo, horkloko
    MINUTO = "tempo:minuto"              # minuto, momento
    SEKUNDO = "tempo:sekundo"            # sekundo, momento
    EPOKO = "tempo:epoko"                # epoko, erao, periodo
    PERIODO = "tempo:periodo"            # periodo, fazo, ciklo
    MOMENTO = "tempo:momento"            # momento, instanco, okazo


class LokoTipo(Enum):
    """Lokoj kaj spacoj (Locations and spaces)."""
    LOKO = "loko:loko"                   # loko, ejo, pozicio
    REGIONO = "loko:regiono"             # regiono, provinco, zono
    URBO = "loko:urbo"                   # urbo, ĉefurbo, urbeto
    VILAĜO = "loko:vilaĝo"               # vilaĝo, komunumo
    LANDO = "loko:lando"                 # lando, regno, ŝtato
    KONTINENTO = "loko:kontinento"       # kontinento, mondoparto
    STRATO = "loko:strato"               # strato, vojo, aleo
    SPACO = "loko:spaco"                 # spaco, areo, amplekso


# ============================================================================
# PRODUKTOJ KAJ OBJEKTOJ (Products & Objects) - ~15 categories
# ============================================================================

class ProduktoTipo(Enum):
    """Produktoj kaj artefaritaĵoj (Products and artifacts)."""
    LIBRO = "produkto:libro"             # libro, volumo, folio
    TEKSTO = "produkto:teksto"           # teksto, skribaĵo, paĝo
    DOKUMENTO = "produkto:dokumento"     # dokumento, akto, skripto
    VERKO = "produkto:verko"             # verko, verkaro, kreado
    LETERO = "produkto:letero"           # letero, mesaĝo, korespondado
    REVUO = "produkto:revuo"             # revuo, gazeto, ĵurnalo
    NOTO = "produkto:noto"               # noto, rimarko, komento
    LISTO = "produkto:listo"             # listo, katalogo, indekso
    BILDO = "produkto:bildo"             # bildo, foto, desegno
    OBJEKTO = "produkto:objekto"         # objekto, aĵo, afero


# ============================================================================
# KONCEPTOJ KAJ IDEOJ (Concepts & Ideas) - ~25 categories
# ============================================================================

class KonceptoTipo(Enum):
    """Konceptoj kaj abstraktaj ideoj (Concepts and abstract ideas)."""
    IDEO = "koncepto:ideo"               # ideo, penso, koncepto
    NUMERO = "koncepto:numero"           # numero, cifero, nombro
    KVANTO = "koncepto:kvanto"           # kvanto, sumo, amplekso
    GRADO = "koncepto:grado"             # grado, nivelo, intens
    ORDO = "koncepto:ordo"               # ordo, ordiĝo, sinsekvo
    SISTEMO = "koncepto:sistemo"         # sistemo, strukturo, aranĝo
    METODO = "koncepto:metodo"           # metodo, maniero, vojo
    REGULO = "koncepto:regulo"           # regulo, leĝo, principo
    MODELO = "koncepto:modelo"           # modelo, ŝablono, paradigmo
    TEORIO = "koncepto:teorio"           # teorio, hipotezo, doktrino
    PROBLEMO = "koncepto:problemo"       # problemo, demando, afer
    SOLVO = "koncepto:solvo"             # solvo, respondo, solvaĵo
    KAŬZO = "koncepto:kaŭzo"             # kaŭzo, kialo, motivo
    REZULTO = "koncepto:rezulto"         # rezulto, efiko, konsekvenco
    STATO = "koncepto:stato"             # stato, kondiĉo, situacio
    RILATO = "koncepto:rilato"           # rilato, ligo, konekso


# ============================================================================
# MAPPING UTILITIES
# ============================================================================

ĈIUJ_KATEGORIOJ = [
    # Vivantaj estaĵoj
    *list(BestoTipo),
    *list(PlantoTipo),
    *list(NaturaĴoTipo),
    *list(VeteroTipo),
    *list(ĈielaTipo),
    *list(SubstancoTipo),
    *list(KorpaPartoPipo),
    # Artefaritaĵoj
    *list(IloTipo),
    *list(UjoTipo),
    *list(MebleTipo),
    *list(VestoTipo),
    *list(VeturiloTipo),
    *list(KonstruaĴoTipo),
    *list(InstrumentoTipo),
    # Agoj
    *list(KorpaAgoTipo),
    *list(ManaAgoTipo),
    *list(SensaAgoTipo),
    *list(KomunikaAgoTipo),
    *list(KognaAgoTipo),
    *list(SociaAgoTipo),
    *list(LaboraAgoTipo),
    # Sentoj kaj mensaj statoj
    *list(PozitivaSentoTipo),
    *list(NegativaSentoTipo),
    *list(KompleksaSentoTipo),
    *list(KogniciaTipo),
    *list(VolaTipo),
    # Ecoj kaj kvalitoj
    *list(GrandecoTipo),
    *list(FormoTipo),
    *list(KoloroTipo),
    *list(TeksturoTipo),
    *list(PezoTipo),
    *list(TemperaturoTipo),
    *list(EvaluaTipo),
    *list(IntensoTipo),
    # Abstraktaj konceptoj
    *list(SociaInstitucioTipo),
    *list(SciencaTipo),
    *list(ArtoTipo),
    *list(FilozofioTipo),
    # Eventoj kaj procesoj
    *list(VivaEventoTipo),
    *list(SociaEventoTipo),
    *list(NaturaEventoTipo),
    # Homoj kaj grupoj
    *list(HomoTipo),
    *list(GrupoTipo),
    # Tempo kaj spaco
    *list(TempoTipo),
    *list(LokoTipo),
    # Produktoj kaj objektoj
    *list(ProduktoTipo),
    # Konceptoj kaj ideoj
    *list(KonceptoTipo),
]

print(f"Total categories: {len(ĈIUJ_KATEGORIOJ)}")


def kategorio_al_teksto(kategorio: Enum) -> str:
    """Convert category enum to text string."""
    return kategorio.value


def teksto_al_kategorio(teksto: str) -> Enum:
    """Convert text string to category enum."""
    for kat in ĈIUJ_KATEGORIOJ:
        if kat.value == teksto:
            return kat
    return None
