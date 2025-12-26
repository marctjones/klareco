# Training Texts Organization

This directory contains authoritative Esperanto texts that are checked into git. These are the gold standard texts for Esperanto grammar, style, and learning.

**See also**: Wiki [[Training-Data-Strategy]] for complete corpus annotation and weighting specification.

## Directory Structure

```
texts/
├── README.md                 # This file
├── authoritative/            # Gold standard texts (checked into git)
│   ├── fundamento/           # Fundamento de Esperanto (Zamenhof, 1905)
│   │   ├── gramatiko.txt     # 16 grammar rules
│   │   ├── ekzercaro.txt     # 42 exercises
│   │   └── metadata.json     # Source annotation
│   ├── krestomatio/          # Fundamenta Krestomatio (Zamenhof, 1903)
│   │   ├── fundamenta_krestomatio.txt
│   │   └── metadata.json
│   └── gerda_malaperis/      # Gerda Malaperis (Piron, 1983)
│       ├── gerda_malaperis.txt
│       └── metadata.json
└── supplementary/            # Other quality texts (checked into git)
```

## Metadata Files

Each text directory includes a `metadata.json` with:
- **tier**: Authority level (1=highest)
- **weight**: Training weight multiplier
- **citation_format**: How to cite this source in corpus entries
- **training_notes**: Special handling instructions

## What Goes in `texts/` (Git-Tracked)

**Authoritative texts** that define correct Esperanto:
- Fundamento de Esperanto - The constitutional document of Esperanto
- Fundamenta Krestomatio - Zamenhof's curated anthology
- Gerda Malaperis - Modern learning standard

**Quality-controlled texts** with known provenance:
- Manually cleaned and verified texts
- Texts with clear copyright/public domain status

## What Goes in `data/` (Git-Ignored)

**Large generated datasets**:
- Parsed corpus files (corpus_with_sources_v*.jsonl)
- FAISS indices
- Training pairs
- Model outputs

**Downloaded raw data**:
- Wikipedia dumps
- Gutenberg books (before cleaning)
- Web scrapes

**Intermediate processing files**:
- Chunked corpora
- Vocabulary extractions
- Build logs

## Training Data Hierarchy

For training, texts should be weighted by authority:

| Tier | Source | Weight | Location |
|------|--------|--------|----------|
| 1 | Fundamento Ekzercaro | 10.0 | `texts/authoritative/fundamento/` |
| 2 | Fundamenta Krestomatio | 5.0 | `texts/authoritative/krestomatio/` |
| 3 | Gerda Malaperis | 3.0 | `texts/authoritative/gerda_malaperis/` |
| 4 | Plena Vortaro definitions | 2.0 | `data/grammar/` |
| 5 | Gutenberg translations | 1.5 | `data/clean_corpus/` |
| 6 | Wikipedia | 1.0 | `data/cleaned/` |

## File Naming Conventions

- Use lowercase with underscores: `fundamento_de_esperanto.txt`
- Include chapter numbers where applicable: `chapter_01.txt`
- Use `.txt` for cleaned plain text
- Keep original files in `original/` subdirectory if needed

## Quality Standards for `texts/`

All texts in this directory must:
1. Be cleaned of OCR artifacts
2. Use proper UTF-8 encoding with Esperanto characters (ĉ, ĝ, ĥ, ĵ, ŝ, ŭ)
3. Have consistent paragraph formatting
4. Be verified against authoritative sources where possible
5. Include source attribution in comments or metadata

## See Also

- Wiki: [[Fundamento-de-Esperanto]] - About the Fundamento
- Wiki: [[Fundamenta-Krestomatio]] - About the Krestomatio
- Wiki: [[Gerda-Malaperis]] - About Gerda Malaperis
- Wiki: [[Training-Data-Strategy]] - How texts are used in training
