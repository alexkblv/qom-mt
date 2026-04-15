# Qom–Spanish Neural Machine Translation

First parallel corpus and neural MT baseline for **Qom (Toba, ISO: `tob`)**, an endangered Guaycurú language spoken by ~80,000 people in the Argentine Chaco. Qom is absent from all major multilingual NLP benchmarks (AmericasNLP editions, OPUS-100, NLLB-200).

This work is being submitted to **AmericasNLP 2026** and is affiliated with the Universidad de Buenos Aires (UBA).

---

## Repository structure

```
qom/
├── corpus/                         # Raw and processed corpus files
├── corpus-statistics.ipynb         # Corpus statistics and analysis
├── qom-mt-ablation-inference.ipynb # Inference and ablation experiments
├── qom-mt-bible-es2qom.ipynb       # Bible-only baseline: ES → QOM
├── qom-mt-bible-qom2es.ipynb       # Bible-only baseline: QOM → ES
├── qom-mt-translations-eval.ipynb  # Translation evaluation
├── qom-mt-translations.ipynb       # Translation generation
├── qom-mt-v1-nostrat.ipynb         # V1 corpus, non-stratified split
├── qom-mt-v1-strat.ipynb           # V1 corpus, stratified split
├── qom-mt-v2-nostrat-es2qom.ipynb  # V2 corpus, non-stratified, ES → QOM
├── qom-mt-v2-nostrat-qom2es.ipynb  # V2 corpus, non-stratified, QOM → ES
├── qom-mt-v2-strat-es2qom.ipynb    # V2 corpus, stratified, ES → QOM
├── qom-mt-v2-strat-qom2es.ipynb    # V2 corpus, stratified, QOM → ES
├── qom-nllb-nonstrat/              # Saved model: non-stratified split
├── qom-nllb-strat/                 # Saved model: stratified split
└── README.md
```

---

## Corpus

### V1 (~2,277 pairs)
Four source documents, manually parallelized:

| Source | Pairs |
|---|---|
| Arte verbal qom | 1,829 |
| Materiales del Taller de Lengua y Cultura Toba | 267 |
| Las Aventuras de Copaic | 17 |
| Educación Sanitaria Intercultural | 169 |
| La Declaración Universal de los Derechos Humanos | 61 |

### V2 (~33,331 pairs)
V1 plus two large-scale additions aligned by the project team:

- **La Biblia** (`La Biblia.csv`): Qom version La'aqtaga Ñim Lo'onatac 'Enauacna 2013 (`LÑLE13`) aligned with Spanish version DHHS94. Bible pairs constitute ~91% of V2 training data.
- **El Principito** (`El Principito.csv`): Qom translation aligned with the Spanish original.

> **Note on V2 metrics:** La Biblia dominates both the training set and the test set (~89–94% of test pairs under random splits). This inflates V2 evaluation scores relative to V1. A Bible-only baseline experiment is included to quantify this effect. All V2 performance claims should be interpreted in light of data composition.

### Data schema
All corpus sources are normalized to two columns: `qom` and `linea_es` (or equivalent). Source-specific column names:
- xlsx sources: `linea_qom` / `linea_es`
- `El Principito.csv`: `qom` / `espanol`
- `La Biblia.csv`: `LÑLE13` / `DHHS94`

### Data loading
Sources are loaded via the `read_csv_source()` helper with a `DATA_DIR` path variable and normalized to a canonical `qom` / `es` schema before training.

---

## Model

**Base model:** `facebook/nllb-200-distilled-600M`  
**Proxy language tag:** `grn_Latn` (Guaraní) — chosen for typological similarity (agglutinative morphology, Guaraní–Spanish is the closest publicly available parallel corpus at ~30K pairs).  
**Optimizer:** Adafactor  
**Epochs:** 10  
**Effective batch size:** 16  
**Learning rate:** 5e-4  
**Precision:** FP16  
**Compute:** Kaggle (T4×2, restricted to single GPU via `CUDA_VISIBLE_DEVICES=0`)

---

## Experiments

Four experimental conditions, each with its own notebook:

| Condition | Split | Direction | Notebook |
|---|---|---|---|
| V2 stratified | Stratified by source doc | ES → QOM | `qom-mt-v2-strat-es2qom.ipynb` |
| V2 stratified | Stratified by source doc | QOM → ES | `qom-mt-v2-strat-qom2es.ipynb` |
| V2 non-stratified | Random | ES → QOM | `qom-mt-v2-nostrat-es2qom.ipynb` |
| V2 non-stratified | Random | QOM → ES | `qom-mt-v2-nostrat-qom2es.ipynb` |
| Bible-only baseline | Stratified | ES → QOM | `qom-mt-bible-es2qom.ipynb` |
| Bible-only baseline | Stratified | QOM → ES | `qom-mt-bible-qom2es.ipynb` |
| V1 stratified | Stratified by source doc | Both | `qom-mt-v1-strat.ipynb` |
| V1 non-stratified | Random | Both | `qom-mt-v1-nostrat.ipynb` |

**Stratified vs. random splits:** Stratified splits assign entire source documents proportionally across train/dev/test, preventing data leakage at the fragment level. They produce lower but more representative scores. Random splits are dominated by La Biblia and inflate metrics.

---

## Results

### V1 baseline

| Direction | BLEU | ChrF | ChrF++ | ChrF2 | ChrF2++ |
|---|---|---|---|---|---|
| ES → QOM | 0.0 | 7.70 | 6.16 | 5.12 | 4.10 |
| QOM → ES | 0.0 | 4.03 | 3.22 | 2.70 | 2.16 |

### V2 (random split)

| Direction | ChrF2++ |
|---|---|
| ES → QOM | 52.21 |
| QOM → ES | — (run cut off by Kaggle time limit) |

Primary metric: **ChrF2++** (aligned with AmericasNLP competition scoring).

---

## Models on Hugging Face

| Model | Hub ID |
|---|---|
| V2 stratified ES→QOM | `alexkblv/qom-nllb-strat-es2qom` |
| V2 stratified QOM→ES | `alexkblv/qom-nllb-strat-qom2es` |
| V2 non-stratified ES→QOM | `alexkblv/qom-nllb-nostrat-es2qom` |
| V2 non-stratified QOM→ES | `alexkblv/qom-nllb-nostrat-qom2es` |

---

## Known issues and caveats

- **Kaggle environment:** Use `eval_strategy` (not `evaluation_strategy`) and `processing_class` (not `tokenizer`) — transformers 4.46 breaking changes. Always set `CUDA_VISIBLE_DEVICES=0` to avoid DataParallel OOM. Clear checkpoints before saving the final model to avoid disk-full errors.
- **Session time limit:** Kaggle enforces a 12-hour session limit. Long QOM→ES runs may be cut off and must be resumed from the last checkpoint.
- **Bible copyright:** The copyright status of La Biblia for training use is unresolved. Results relying on Bible data should be interpreted accordingly.
- **BLEU = 0 in V1:** BLEU was not computed due to computational constraints; ChrF-family metrics are preferred for morphologically rich low-resource languages in any case.

---

## Citation

If you use this corpus or models, please cite:

```bibtex
@inproceedings{korablev2026qom,
  title     = {A Parallel Corpus and Neural {MT} Baseline for {Qom} ({Toba})},
  author    = {Korablev, Aleksei and Cotik, Viviana},
  booktitle = {Proceedings of AmericasNLP 2026},
  year      = {2026}
}
```

---

## Acknowledgments

This project is affiliated with the Universidad de Buenos Aires (UBA). Corpus construction involved collaboration with linguists Paola Cúneo and Temis Tacconi. Parallelization of La Biblia and El Principito was led by Pablo (UBA).
