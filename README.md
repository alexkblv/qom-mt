# Qom–Spanish Neural Machine Translation

First parallel corpus and neural MT baseline for **Qom (Toba, ISO: `tob`)**, an endangered Guaycurú language spoken by ~80,000 people in the Argentine Chaco. Qom is absent from all major multilingual NLP benchmarks (AmericasNLP editions, OPUS-100, NLLB-200).

This work is being submitted to **AmericasNLP 2026** and is affiliated with the Universidad de Buenos Aires (UBA).

---

## Repository structure

```
qom/
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
└── README.md
```

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
