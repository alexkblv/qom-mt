# Qom–Spanish MT Baseline (NLLB-600M)

Fine-tuning of [facebook/nllb-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M) on a Qom (Toba)–Spanish parallel corpus. Both translation directions are trained and evaluated.

Qom (also known as Toba) is an endangered Guaycurú language spoken by ~80,000 people in Argentina's Chaco province. It has no prior representation in NLP datasets or major MT systems. There is no native NLLB language code for Qom; we use `grn_Latn` (Guaraní) as a proxy tag.

Part of a project to build the first Qom–Spanish parallel corpus and neural MT system, affiliated with [Universidad de Buenos Aires](https://www.uba.ar/). Targeting publication at [AmericasNLP 2026](https://americasnlp.github.io/).

---

## Notebooks

| Notebook | Split strategy |
|---|---|
| `qom-mt-nostrat.ipynb` | Random (non-stratified) split |
| `qom-mt-strat.ipynb` | Stratified split by source document |

---

## Results

### Random split (`qom-mt-nostrat.ipynb`)

Train: 1933 · Dev: 142 · Test: 257

| Direction | BLEU | ChrF | ChrF++ | ChrF2 | ChrF2++ |
|---|---|---|---|---|---|
| ES → QOM | 9.19 | 38.43 | 35.13 | 36.62 | 33.48 |
| QOM → ES | 8.42 | 29.88 | 27.98 | 28.53 | 26.68 |

### Stratified split (`qom-mt-strat.ipynb`)

Train: 1832 · Dev: 192 · Test: 308

| Direction | BLEU | ChrF | ChrF++ | ChrF2 | ChrF2++ |
|---|---|---|---|---|---|
| ES → QOM | 6.93 | 37.48 | 34.24 | 35.69 | 32.61 |
| QOM → ES | 8.26 | 31.11 | 29.29 | 28.87 | 27.31 |

The stratified split is the more rigorous evaluation. Random splits inflate scores because the test set ends up dominated by Arte verbal qom (the largest source document), which overlaps heavily with the training distribution. The stratified split enforces proportional representation of all four source documents across train/dev/test.

---

## Corpus

The parallel corpus (~2,300 sentence pairs after deduplication) is drawn from four source documents:

- Arte verbal qom
- Educación Sanitaria Intercultural
- Materiales del Taller de Lengua y Cultura Toba
- Las Aventuras de Copaic

The corpus is not included in this repository (private data).

---

## Setup

Notebooks run on Kaggle (T4×2 GPU; single GPU enforced via `CUDA_VISIBLE_DEVICES=0`).

**Requirements:** The notebooks install their own dependencies (`sacrebleu`, `evaluate`, `openpyxl`, `transformers`).

**Input:** A Kaggle dataset with the parallel corpus as `.xlsx` files, each containing `linea_qom` and `linea_es` columns.

---

## Model

- Base model: `facebook/nllb-200-distilled-600M`
- Optimizer: Adafactor
- Epochs: 10 · Effective batch size: 16
- Saved in fp16

Model weights are not included in this repository.
