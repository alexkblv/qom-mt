# Paper notebooks

The notebooks behind the AmericasNLP 2026 paper on the QomL'aqtaqa Qom–Spanish corpus. New
work uses the tested package in [`../src/qomlaq/`](../src/qomlaq/) and the drivers in
[`../notebooks/`](../notebooks/) instead.

They are kept **with their executed outputs exactly as originally run**. The cell outputs are
the record of each run, so nothing here should be re-executed, cleared, or reformatted.

## Contents

| Notebook | Role |
|---|---|
| `corpus-statistics.ipynb` | Corpus ingestion + QC statistics (source of the paper's corpus table) |
| `qom-mt-v1-strat.ipynb` | QomL-Base, stratified split, both directions |
| `qom-mt-v1-nostrat.ipynb` | QomL-Base, random split, both directions |
| `qom-mt-v2-strat-es2qom.ipynb` | QomL-Base+Bible, stratified, ES→QOM |
| `qom-mt-v2-strat-qom2es.ipynb` | QomL-Base+Bible, stratified, QOM→ES |
| `qom-mt-v2-nostrat-es2qom.ipynb` | QomL-Base+Bible, random, ES→QOM |
| `qom-mt-v2-nostrat-qom2es.ipynb` | QomL-Base+Bible, random, QOM→ES |
| `qom-mt-bible-es2qom.ipynb` | Bible-only baseline, ES→QOM |
| `qom-mt-bible-qom2es.ipynb` | Bible-only baseline, QOM→ES |
| `qom-mt-ablation-inference.ipynb` | Domain-shift ablation + inference |
| `qom-mt-translations.ipynb` | Translation generation across models |
| `qom-mt-translations-eval.ipynb` | Translation comparison + human-eval sample |

Naming note: in these notebooks `v1` / `v2` refer to **corpus configurations** (`v1` = QomL-Base,
`v2` = QomL-Base+Bible). The current pipeline calls these `base`, `base_bible` and `bible_only`.

## Environment

Kaggle, single NVIDIA T4, 12-hour session limit; `facebook/nllb-200-distilled-600M` fine-tuned
with `Seq2SeqTrainer`, Adafactor, lr 5e-4, effective batch 16 (2 × 8 accumulation), fp16,
10 epochs, seed 42. Qom is represented with the Guaraní proxy tag `grn_Latn`; Spanish is
`spa_Latn`.
