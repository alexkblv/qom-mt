# QomL'aqtaqa — Qom–Spanish parallel corpus and MT baselines

First parallel corpus and neural machine translation baseline for **Qom (Toba, ISO 639-3
`tob`)**, an Indigenous language of the Guaycuruan family spoken in the Argentine Chaco.
Qom is absent from the major multilingual benchmarks (AmericasNLP 2021–2025, OPUS-100,
NLLB-200).

Affiliated with Universidad de Buenos Aires (UBA); supported in part by the Lacuna Fund
NLP 2024 grant *Corpus Lengua y Cultura Qom*.

---

## Repository layout

```
qom/
├── src/qomlaq/          # the pipeline: corpus, groups, splits, metrics, training, report
├── experiments/         # what to run: one file per experiment
├── notebooks/           # thin front-ends to the same operations as the CLI
├── translator/          # web demo (Gradio) that serves the models through qomlaq
├── tests/               # CPU-only test suite (no GPU, no corpus needed for unit tests)
├── artifacts/           # generated: manifests, split hashes, results, tables
├── corpus/              # raw sources (not distributed; see Notes)
└── legacy/              # the notebooks behind the AmericasNLP 2026 paper, with outputs
```

The corpus itself is not in this repository. Every artifact records a `corpus_sha256`, so
a result can always be traced to the exact corpus snapshot that produced it.

## Pipeline

All logic lives in `src/qomlaq/`, so there is exactly one implementation of ingestion,
QC, splitting, training and scoring. What gets run lives in `experiments/`: one TOML file
per experiment names a corpus version, its splits (each pinned to its hash), a model
profile, the directions, any training overrides and any extra evaluations. Run ids,
output paths and report tables all come from that file.
[`experiments/baseline.toml`](experiments/baseline.toml) is the annotated reference.

```bash
pip install -e ".[dev]"          # add the [train] extra on a machine with a GPU
qomlaq experiments               # list experiments
qomlaq build baseline            # corpus + splits, checked against the pinned hashes
qomlaq runs baseline             # the runs, and which are trained and scored
qomlaq train baseline base__stratified__es2qom
qomlaq evaluate baseline base__stratified__es2qom
qomlaq evaluate baseline --evaluation domain-shift
qomlaq report baseline           # tables under artifacts/tables/baseline/
pytest                           # corpus-backed tests skip if corpus/ is absent
```

Everything runs on any machine with the repository checked out. Training needs a GPU;
nothing else does. `QOMLAQ_DATA_DIR`, `QOMLAQ_ARTIFACTS_DIR` and `QOMLAQ_EXPERIMENTS_DIR`
move the inputs and outputs.

The notebooks are thin front-ends to the same functions, one per stage:

| Stage | Notebook | GPU | Output |
|---|---|---|---|
| Build corpus | `00_build_corpus.ipynb` | no | canonical table + `corpus_manifest.json` |
| Build splits | `01_make_splits.ipynb` | no | one verified split artifact per split in the experiment |
| Train | `10_train.ipynb` | yes | checkpoint + model card |
| Evaluate | `20_evaluate.ipynb` | yes | scores + per-row hypothesis dump |
| Ablation | `30_ablation.ipynb` | yes | several models on one shared held-out set |
| Report | `40_report.ipynb` | no | every table, regenerated from artifacts |

Four properties hold by construction:

- **Splits are artifacts, not values.** They are written under a path that embeds the
  corpus hash, and `load_split()` requires the caller to state the hash it expects. There
  is no code path from a raw dataframe to a score.
- **Splits are pinned.** Each experiment pins the hash of every split it uses. A build
  fails if a split comes out different, and training and evaluation refuse an unpinned
  split.
- **Metric names are derived from their parameters.** `chrf_display_name(word_order=2,
  beta=2)` returns `chrF2++`; no metric label is ever typed by hand.
- **Evaluation is guarded.** Scoring a checkpoint against pairs it trained on raises,
  using the training pair ids recorded in the model card.

### Adding an experiment

Copy `experiments/baseline.toml`, change what the new experiment is about, and run
`qomlaq build <name>`: it prints the hash of each new split so it can be pinned once
agreed.

- **A bigger model.** Add a profile to `qomlaq.config.MODELS` (the checkpoint and the batch
  settings it needs) and name it in the experiment.
- **More data.** Add the source to `qomlaq.config.SOURCES` and a new corpus version to
  `CORPORA`; older experiments keep building the corpus they were run on. Stratified
  splits pack each source separately, so a new source moves no unit of the existing ones.
- **Transfer.** `init_from` starts training from another checkpoint: a Hugging Face model
  or another experiment's run.
- **Comparisons.** An `[[evaluations]]` entry scores several runs, from this experiment or
  others, on one shared test set.

## Translator

[`translator/`](translator/) is a Gradio web demo for the private checkpoints, in both
directions. It uses the pipeline's own normalization, language tags and decoding, so the
demo translates exactly the way evaluation does. It has its own environment because it
pins `transformers` exactly; see its README.

## Corpus

Seven sources spanning oral narrative, educational, health, legal, literary and religious
registers. Segments are heterogeneous by nature — a segment may be a fragment, a sentence
or a paragraph, reflecting how each source was aligned.

| Source | Grouping unit |
|---|---|
| Arte verbal qom | fragment → section → chapter |
| Educación Sanitaria Intercultural | fragment |
| Materiales del Taller de Lengua y Cultura Toba | fragment → section |
| Las Aventuras de Copaic | whole document (held-out probe) |
| La Declaración Universal de los Derechos Humanos | `parte` |
| El Principito | chapter |
| La Biblia | book + chapter |

**Schema.** Sources are declared in `qomlaq.config.SOURCES` and grouped into named
corpus versions in `qomlaq.config.CORPORA`; these seven are `text-v1`. The xlsx sources use
`linea_qom` / `linea_es`; `El Principito.csv` uses `qom` / `espanol`; `La Biblia.csv` uses
`LNLE13` / `DHHS94`.

**Orthography.** Qom has no universally adopted written standard, and the corpus preserves
the variation in its sources (`ỹ`/`ȳ`, `ñ`, d/r, h/j, e/i). Preprocessing does *not*
normalize any of it. The one exception is the apostrophe, which writes the glottal stop:
sources disagree on the codepoint, so all variants are unified to U+0027 and a test
asserts that no other grapheme is touched.

**Filtering.** Within-source exact deduplication; both sides non-empty and at least two
tokens; token-length ratio (|ES|+1)/(|QOM|+1) within [0.15, 6.0]; both sides under 800
characters. Every stage's removals are recorded in the corpus manifest.

### Splits

Partitions are disjoint at the discourse-unit level: every fragment, chapter, article or
Bible chapter lies wholly within one partition. Groups are allocated largest-first by row
count, so realized proportions stay within ~0.5% of the 80/10/10 target rather than
drifting with group sizes.

Two split strategies are built per configuration — `stratified` (proportional within each
source document) and `random` (over the corpus as a whole). *Las Aventuras de Copaic* is
held out of training entirely, as a declared out-of-domain probe, and reported separately.

A `base_bible/stratified__verse` split is also built and labeled a **control**: it groups
the Bible verse by verse, so the effect of chapter grouping can be measured (93.6% of
held-out verses sit next to a training verse under verse-level grouping, 0.0% under
chapter grouping). The report stage refuses to place control runs in a results
table.

## Model

**Base:** `facebook/nllb-200-distilled-600M` (the `nllb-600m` profile, used by `baseline`)
**Proxy language tag:** `grn_Latn` (Guaraní) for Qom — NLLB has no `tob` code — and
`spa_Latn` for Spanish. Conditioning is via the tokenizer's `src_lang` plus a forced BOS
token for the target language.
**Optimizer:** Adafactor, lr 5e-4, effective batch 16 (2 × 8 accumulation), fp16.
**Schedule:** up to 10 epochs with early stopping on dev loss (patience 2).

## Evaluation

Primary metric is **chrF2++** (`sacrebleu`, `char_order=6, word_order=2, beta=2`) — chrF++
as defined by Popović (2017) and used by the AmericasNLP shared tasks. Plain chrF and BLEU
are reported alongside. Every score is stored with sacrebleu's own signature string and the
resolved library version, and `metric_signatures.tex` is generated from those signatures.

Scores are corpus-level over detokenized text. Each evaluation writes a per-row hypothesis
dump, so a different metric, a per-source breakdown or a lexical analysis is a CPU rescore
rather than another GPU run.

## Results

The published results are in the AmericasNLP 2026 paper (see Citation). This pipeline
writes each experiment's tables to `artifacts/tables/<experiment>/`; results will be
listed here as runs complete.

## Models

Trained checkpoints are currently private. Hub locations will be listed here when they are
published.

## Citation

```bibtex
@inproceedings{cotik2026qom,
  title     = {QomL'aqtaqa: A Qom--Spanish Parallel Corpus for Natural Language
               Processing with Machine Translation Evaluation},
  author    = {Cotik, Viviana and Korablev, Aleksei and C{\'u}neo, Paola and Laciana, Pablo},
  booktitle = {Proceedings of AmericasNLP 2026},
  year      = {2026}
}
```

## Acknowledgments

Corpus construction involved collaboration with linguists Paola Cúneo and Temis Tacconi.
Parallelization of *La Biblia* and *El Principito* was led by Pablo Laciana (UBA).

## Notes

- **Bible copyright.** The copyright status of *La Biblia* for training use is unresolved;
  results relying on Bible data should be read accordingly.
- **Data release.** The processed corpus is not yet released. A public release is planned
  for a future version incorporating ongoing recording, transcription and translation work.
