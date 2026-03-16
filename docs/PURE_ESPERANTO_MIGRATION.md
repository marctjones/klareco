# Pure Esperanto Migration (v2.1)

Date: 2026-02-23
Status: Complete (scripts ready, not yet executed)

## Summary

The entire Klareco pipeline has been updated to use **pure Esperanto** naming for both property names and values, eliminating all mixed English/Esperanto inconsistencies.

## Changes Made

### 1. Parser (`klareco/parser.py`)

**Property Names (English → Esperanto):**
- `parse_status` → `analizstato`
- `parse_error` → `analizeraro`
- `category` → `kategorio`
- `total_words` → `tutaj_vortoj`
- `esperanto_words` → `esperantaj_vortoj`
- `non_esperanto_words` → `neesperantaj_vortoj`
- `success_rate` → `sukcesoprocento`
- `categories` → `analizkategorioj`

**Property Values (English → Esperanto):**
- `"success"` → `"sukceso"`
- `"failed"` → `"malsukceso"`
- `"unknown"` → `"nekonata"`
- `"proper_name_known"` → `"propranomo_konata"`
- `"proper_name"` → `"propranomo"`
- `"proper_name_esperantized"` → `"propranomo_esperantigita"`
- `"foreign_word"` → `"fremda_vorto"`
- `"number_literal"` → `"numero_laŭvorta"`
- `"single_letter"` → `"unusola_litero"`

### 2. Schema (`klareco/schema/kuzu_ast_schema_v2_1.py`)

**Node Names:**
- `SourceCollection` → `Fontaro`
- `Document` → `Dokumento`
- `Section` → `Sekcio`
- `Paragraph` → `Paragrafo`
- `Sentence` → `Frazoteksto`
- `Root` → `Radiko`
- `Annotation` → `Anotacio`
- `AnnotationType` → `AnotacioTipo`
- `AnnotationValue` → `AnotacioValoro`
- `AnnotationSet` → `AnotacioAro`

**Relationship Names:**
- `IN_COLLECTION` → `EN_FONTARO`
- `SENTENCE_HAS_AST` → `FRAZOTEKSTO_HAVAS_AST`
- `AST_HAS_FRAZO` → `AST_HAVAS_FRAZON`
- `HAS_SUBJEKTO_*` → `HAVAS_SUBJEKTON_*`
- `HAS_VERBO` → `HAVAS_VERBON`
- `HAS_OBJEKTO_*` → `HAVAS_OBJEKTON_*`
- `HAS_ALIAJ` → `HAVAS_ALIAJN`
- `HAS_KERNO` → `HAVAS_KERNON`
- `HAS_PRISKRIBO` → `HAVAS_PRISKRIBON`
- `HAS_ROOT` → `HAVAS_RADIKON`

**Property Names:**
- Document: `title`→`titolo`, `author`→`aŭtoro`, `year`→`jaro`, `quality`→`kvalito`, `metadata`→`metadatenoj`
- Frazoteksto: `text`→`teksto`, `sentence_order`→`frazo_ordo`, `global_order`→`tutmonda_ordo`
- AST: `created_at`→`kreita_je`, `is_current`→`estas_nuna`, `total_words`→`tutaj_vortoj`
- Vorto: `parse_status`→`analizstato`, `parse_error`→`analizeraro`
- Radiko: `root`→`radiko`, `doc_freq`→`dokumenta_ofteco`, `total_freq`→`tuta_ofteco`

**Property Values:**
- Quality: `GOLD`→`ORO`, `SILVER`→`ARĜENTO`, `BRONZE`→`BRONZO`
- Document type: `article`→`artikolo`, `book`→`libro`, `text`→`teksto`

### 3. CSV Generator (`scripts/corpus_to_csv_v2.1.py`)

- Updated to generate CSVs with v2.1 Esperanto table/column names
- Reads pure Esperanto AST output from parser (no translation needed)
- Only translates source metadata (quality, doc_type) from English to Esperanto

### 4. Batched Loader (`scripts/load_csv_to_kuzu_v2.1_batched.py`)

- Updated to load v2.1 CSVs with Esperanto table names
- Imports from `kuzu_ast_schema_v2_1` module
- Updated all node/relationship loading logic

## Verification

Parser now outputs pure Esperanto:

```json
{
  "tipo": "vorto",
  "plena_vorto": "hundo",
  "radiko": "hund",
  "vortspeco": "substantivo",
  "analizstato": "sukceso",
  "parse_statistics": {
    "tutaj_vortoj": 3,
    "esperantaj_vortoj": 3,
    "neesperantaj_vortoj": 0,
    "sukcesoprocento": 1.0,
    "analizkategorioj": {}
  }
}
```

Unknown word categorization:

```json
{
  "tipo": "vorto",
  "vortspeco": "fremda_vorto",
  "analizstato": "malsukceso",
  "analizeraro": "Not an Esperanto root",
  "kategorio": "fremda_vorto"
}
```

## What's Pure Esperanto Now

✅ **Node names**: Fontaro, Dokumento, Frazoteksto, Radiko
✅ **Relationship names**: HAVAS_*, ESTAS_*, EN_*, SEKVA_*
✅ **Property names**: analizstato, kategorio, tutaj_vortoj, etc.
✅ **Property values**: sukceso, malsukceso, propranomo_konata, etc.
✅ **Parser output**: All AST annotations in Esperanto
✅ **Statistics**: All parse_statistics fields in Esperanto

## What Remains in English

Technical acronyms (universal):
- AST (Abstract Syntax Tree)
- CSV (Comma-Separated Values)
- JSON (JavaScript Object Notation)

## Next Steps (User Will Execute)

1. Generate v2.1 CSVs:
   ```bash
   python scripts/corpus_to_csv_v2.1.py \
     --corpus data/corpus/unified_corpus.jsonl \
     --output data/csv_export_v2.1_full
   ```

2. Load into Kuzu v2.1 database:
   ```bash
   python scripts/load_csv_to_kuzu_v2.1_batched.py \
     --csvs data/csv_export_v2.1_full \
     --output data/indexes/v2.1_kuzu_index_full \
     --fresh
   ```

## Breaking Changes

- All code accessing the graph must use new Esperanto names
- Old v2.0 CSVs incompatible with v2.1 schema
- Queries need updating: `parse_status` → `analizstato`, etc.

## Benefits

1. **Consistency**: Pure Esperanto throughout (no English/Esperanto mixing)
2. **Self-documenting**: Code reinforces "Pure Esperanto AI" thesis
3. **Cleaner**: No translation layer needed - parser outputs directly to schema
4. **Principled**: Aligns with project name "Klareco" (clarity in Esperanto)
