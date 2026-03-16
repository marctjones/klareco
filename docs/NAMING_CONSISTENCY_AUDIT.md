# Naming Consistency Audit (v2.0 Schema)

Date: 2026-02-23
Issue: Mixed English/Esperanto naming violates "Pure Esperanto AI" principle

## Summary

The v2.0 schema mixes English and Esperanto inconsistently. For a project called **Klareco** (Pure Esperanto), all names should be in Esperanto.

## Findings by Category

### 1. NODE NAMES (Mixed)

**English:**
- `SourceCollection` → Should be `Fontaro` or `Kolektaĵo`
- `Document` → Should be `Dokumento`
- `Section` → Should be `Sekcio`
- `Paragraph` → Should be `Paragrafo`
- `Sentence` → Should be `Frazo` (but `Sentence` holds text, `Frazo` is AST)
- `Root` → Should be `Radiko`
- `Annotation` → Should be `Anotacio` or `Priskribo`
- `AnnotationType` → Should be `AnotacioTipo`
- `AnnotationValue` → Should be `AnotacioValoro`
- `AnnotationSet` → Should be `AnotacioAro`

**Esperanto (GOOD):**
- `AST` (acceptable - technical acronym)
- `Frazo` ✓
- `Vortgrupo` ✓
- `Vorto` ✓

### 2. RELATIONSHIP NAMES (Mostly English with Esperanto objects)

**Current (INCONSISTENT):**
```
SENTENCE_HAS_AST          (English_English_English)
AST_HAS_FRAZO             (English_English_Esperanto)
HAS_SUBJEKTO_VORTGRUPO    (English_Esperanto_Esperanto)
HAS_SUBJEKTO_VORTO        (English_Esperanto_Esperanto)
HAS_VERBO                 (English_Esperanto)
HAS_OBJEKTO_VORTGRUPO     (English_Esperanto_Esperanto)
HAS_OBJEKTO_VORTO         (English_Esperanto_Esperanto)
HAS_ALIAJ                 (English_Esperanto)
HAS_KERNO                 (English_Esperanto)
HAS_PRISKRIBO             (English_Esperanto)
HAS_KUNMETAJHO            (English_Esperanto)
HAS_ROOT                  (English_English)
IS_SYNONYM                (English_English)
IS_HYPERNYM               (English_English)
IS_ANTONYM                (English_English)
IN_COLLECTION             (English_English)
IN_DOCUMENT               (English_English)
IN_SECTION                (English_English)
IN_PARAGRAPH              (English_English)
PARENT_SECTION            (English_English)
NEXT_SECTION              (English_English)
NEXT_PARAGRAPH            (English_English)
NEXT_SENTENCE             (English_English)
WORD_HAS_ANNOTATION       (English_English_English)
VORTGRUPO_HAS_ANNOTATION  (Esperanto_English_English)
FRAZO_HAS_ANNOTATION      (Esperanto_English_English)
AST_HAS_ANNOTATION        (English_English_English)
SENTENCE_HAS_ANNOTATION   (English_English_English)
PARAGRAPH_HAS_ANNOTATION  (English_English_English)
SECTION_HAS_ANNOTATION    (English_English_English)
DOCUMENT_HAS_ANNOTATION   (English_English_English)
IN_ANNOTATION_SET         (English_English_English)
ANNOTATION_HAS_TYPE       (English_English_English)
ANNOTATION_HAS_VALUE      (English_English_English)
ANNOTATES_AST             (English_English)
ANNOTATION_TYPE_IS_ROOT   (English_English_English_English)
ANNOTATION_VALUE_IS_ROOT  (English_English_English_English)
```

### 3. PROPERTY NAMES (Mostly English)

**English properties on Esperanto nodes:**
- `Frazo.tipo` → OK (Esperanto)
- `Vorto.plena_vorto` → OK (Esperanto)
- `Vorto.radiko` → OK (Esperanto)
- `Vorto.vortspeco` → OK (Esperanto)
- `Vorto.nombro` → OK (Esperanto)
- `Vorto.kazo` → OK (Esperanto)
- `Vorto.tempo` → OK (Esperanto)
- `Vorto.modo` → OK (Esperanto)
- `Vorto.prefiksoj` → OK (Esperanto)
- `Vorto.sufiksoj` → OK (Esperanto)
- `Vorto.parse_status` → Should be `analizstato`
- `Vorto.parse_error` → Should be `analizeraro`
- `Vorto.proper_noun_category` → Should be `propranoma_kategorio`
- `Vorto.proper_noun_frequency` → Should be `propranoma_ofteco`
- `Vorto.korelativo_prefikso` → OK (Esperanto)
- `Vorto.korelativo_sufikso` → OK (Esperanto)
- `Vorto.korelativo_signifo` → OK (Esperanto)
- `Vorto.estas_kunmetita` → OK (Esperanto)
- `Vorto.kunmetitaj_radikoj` → OK (Esperanto)

**English properties on English nodes:**
- `Document.title` → Should be `titolo`
- `Document.author` → Should be `aŭtoro`
- `Document.year` → Should be `jaro`
- `Document.quality` → Should be `kvalito`
- `Document.metadata` → Should be `metadatenoj`
- `Section.section_name` → Should be `sekcio_nomo`
- `Section.section_level` → Should be `sekcio_nivelo`
- `Sentence.text` → Should be `teksto`
- `Sentence.sentence_order` → Should be `frazo_ordo`
- `Sentence.global_order` → Should be `tutmonda_ordo`

### 4. PROPERTY VALUES (Mixed)

**English values:**
- `fraztipo: "demando", "deklaro", "ordono"` → OK (Esperanto)
- `demandotipo: "ki", "ĉu"` → OK (Esperanto)
- `vortspeco: "verbo", "substantivo", "adjektivo"` → OK (Esperanto)
- `doc_type: "article", "book", "qa"` → Should be `"artikolo", "libro", "demandoj-respondoj"`
- `quality: "GOLD", "SILVER", "BRONZE"` → Should be `"ORO", "ARĜENTO", "BRONZO"`
- `parse_status: "success", "unknown_root", "failed"` → Should be `"sukceso", "nekonata_radiko", "malsukceso"`

### 5. CODE VARIABLE NAMES (Python - Mixed)

**Inconsistent in scripts:**
- `collection_id` (English)
- `doc_id` (English abbreviation)
- `frazo_id` (Esperanto)
- `vortgrupo_id` (Esperanto)
- `vorto_id` (Esperanto)
- `radiko` (Esperanto) vs `root` (English)

### 6. FILE NAMES (Mostly English)

**Schema files:**
- `kuzu_ast_schema.py` → OK (technical)

**Script files:**
- `index_corpus_v2.py` → OK
- `corpus_to_csv_v2.py` → OK
- `load_csv_to_kuzu_v2.py` → OK

## Proposed Naming Standard

### Option 1: Pure Esperanto (RECOMMENDED)

**Principle:** Use Esperanto for ALL domain concepts, English only for technical acronyms (AST, CSV, JSON).

**Node names:**
```
Fontaro (SourceCollection)
Dokumento (Document)
Sekcio (Section)
Paragrafo (Paragraph)
Frazo (Sentence - reuse existing)
AST (keep)
Frazo (keep)
Vortgrupo (keep)
Vorto (keep)
Radiko (Root)
Anotacio (Annotation)
AnotacioTipo (AnnotationType)
AnotacioValoro (AnnotationValue)
AnotacioAro (AnnotationSet)
```

**Relationship names:**
```
HAVAS_RADIKON (HAS_ROOT)
HAVAS_VERBON (HAS_VERBO)
HAVAS_SUBJEKTON (HAS_SUBJEKTO_VORTO)
HAVAS_OBJEKTON (HAS_OBJEKTO_VORTO)
HAVAS_ALIAJN (HAS_ALIAJ)
HAVAS_KERNON (HAS_KERNO)
HAVAS_PRISKRIBON (HAS_PRISKRIBO)
HAVAS_KUNMETAĴON (HAS_KUNMETAJHO)
ESTAS_SINONIMO (IS_SYNONYM)
ESTAS_HIPERONIMO (IS_HYPERNYM)
ESTAS_ANTONIMO (IS_ANTONYM)
EN_KOLEKTAĴO (IN_COLLECTION)
EN_DOKUMENTO (IN_DOCUMENT)
SEKVA_SEKCIO (NEXT_SECTION)
SEKVA_PARAGRAFO (NEXT_PARAGRAPH)
SEKVA_FRAZO (NEXT_SENTENCE)
```

**Alternative (more concise):**
```
RADIKO_DE (root of)
VERBO_DE (verb of)
SUBJEKTO_DE (subject of)
OBJEKTO_DE (object of)
ALIAJ_DE (others of)
```

### Option 2: Pragmatic Hybrid (NOT RECOMMENDED)

Keep English for technical infrastructure, Esperanto for linguistic concepts.

**Problems:**
- Violates "Pure Esperanto" principle
- Inconsistent and confusing
- Hard to document where to draw the line

## Recommendation

**Implement Option 1: Pure Esperanto**

Reasoning:
1. Aligns with project name "Klareco" (Pure Esperanto)
2. Consistent with AST node names (Frazo, Vortgrupo, Vorto)
3. Makes codebase self-documenting in Esperanto
4. Reinforces the core thesis: Esperanto-first architecture

## Implementation Plan

1. **Update schema definition** (`kuzu_ast_schema.py`)
   - Rename all node tables
   - Rename all relationship tables
   - Update property names

2. **Update CSV generation** (`corpus_to_csv_v2.py`)
   - Update CSV file names
   - Update column headers

3. **Update loader** (`load_csv_to_kuzu_v2_batched.py`)
   - Update table names
   - Update relationship names

4. **Regenerate CSVs** from corpus (15 minutes)

5. **Reload database** with new schema (30 minutes)

6. **Update tests** to use new names

7. **Update documentation**

## Breaking Changes

- CSV files incompatible with v2.0-alpha
- Kuzu queries need updating
- Code accessing graph needs updating

## Versioning

This should be **v2.1** (minor version) since it's a schema change.

Or keep as **v2.0-beta** (we're still in alpha anyway).

## Related Issues

- #592 (v2.0 implementation)
- Need new issue for naming consistency

## Questions for Discussion

1. Should technical acronyms stay in English (AST, CSV, JSON)?
   - **Recommendation:** Yes - these are universal technical terms

2. Should relationship names use verbs (HAVAS_) or prepositions (DE_, EN_)?
   - **Recommendation:** Verbs (HAVAS_, ESTAS_) - more explicit

3. Should property names be camelCase or snake_case?
   - **Recommendation:** snake_case (Python convention)

4. Should we use accusative case in relationship names?
   - `HAVAS_VERBON` (accusative - grammatically correct)
   - `HAVAS_VERBO` (nominative - simpler)
   - **Recommendation:** Nominative for simplicity

## Glossary

English → Esperanto translations:

| English | Esperanto | Notes |
|---------|-----------|-------|
| source collection | fontaro, kolektaĵo | "fonto" = source |
| document | dokumento | cognate |
| section | sekcio | cognate |
| paragraph | paragrafo | cognate |
| sentence | frazo | existing |
| word | vorto | existing |
| word group | vortgrupo | existing |
| root | radiko | existing |
| annotation | anotacio, priskribo | |
| has | havas | verb "to have" |
| is | estas | verb "to be" |
| in | en | preposition |
| of | de | preposition |
| next | sekva | adjective |
| parent | gepatra, patra | |
| quality | kvalito | cognate |
| metadata | metadatenoj | meta + datenoj (data) |
| title | titolo | cognate |
| author | aŭtoro | cognate |
| year | jaro | |
| text | teksto | cognate |
| order | ordo | cognate |
| success | sukceso | cognate |
| failed | malsukceso | mal- + sukceso |
| unknown | nekonata | ne- + konata |

## Decision

**PENDING USER APPROVAL**

Options:
A. Implement pure Esperanto naming (v2.1)
B. Keep current mixed naming (document as technical debt)
C. Different hybrid approach (specify)

User decision: _________
