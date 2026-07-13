# MPM multi-omics prognostic biomarker feature-selection pipeline

Reliability-first discovery of prognostic **biomarker panels** for malignant **pleural**
mesothelioma, trained on **MESOMICS multi-omics** and validated by cross-cohort survival transfer.
Built from scratch, self-contained (isolated `.venv`, no dependency on root or a prior workspace).

## Run

```bash
bash run_all.sh          # stages 1 → 5, output in results/
```

Interpreter: `./.venv/bin/python` (isolated environment on /mnt/data). Fixed seed 42.

## What it does

1. **Build** a 120-sample MESOMICS multi-omics feature matrix — expression, recurrent CNV peaks,
   LOH, methylation (promoter/gene-body/enhancer), somatic driver alterations, SV burden (~25.5k features).
2. **In-house selection** — ensemble of repeated 2-part splits with χ² top-300 intersection
   (univariate) + epistatic pair test (bivariate), then bootstrap **stability LASSO-Cox**, then union.
3. **Third-party selectors** — SIS, pawph, Network-LASSO, RSF, XGBoost, DeepSurv, DeepKEGG, DeePathNet,
   and a transfer-learning DeepSurv.
4. **Ensemble/consensus** — cross-method vote aggregation (kept if it beats the best tool).
5. **Evaluate** — every panel trained on MESOMICS, transferred to TCGA (multi-omics), Bueno, NCI, Blum,
   French; ranked by external-transfer C-index.

See **`DESIGN.md`** for the full method and **`results/REPORT.md`** for results.

## Layout

```
mpm_multiomics_pipeline/
├── run_all.sh
├── DESIGN.md · README.md
├── .venv/                       isolated interpreter (on /mnt/data)
├── src/
│   ├── common.py                config, loaders, discretization, survival metrics
│   ├── 01_build_data.py         MESOMICS multi-omics train matrix + survival
│   ├── 02_build_validation.py   transferable TCGA methylation/alteration layers
│   ├── 03_featsel_inhouse.py    ensemble split + epistasis + stability LASSO-Cox
│   ├── 04_featsel_thirdparty.py 8 selectors + transfer-learning DeepSurv
│   ├── 05_consensus.py          ensemble / consensus panels
│   ├── 06_evaluate.py           cross-cohort transfer C-index
│   └── 07_report.py             figures + REPORT.md
├── data/processed/              built matrices + feature annotation
└── results/tables · figures · REPORT.md
```

## Cohorts (all pleural MPM)

| cohort | role | layers used in transfer |
|---|---|---|
| MESOMICS | **train** | all (EXPR, CNV, LOH, MET×3, ALT, SV) |
| TCGA-MESO (74 Hmeljak-2018) | validation | EXPR + METH + ALT (multi-omics) |
| Bueno 2016 / NCI 2023 / Blum 2019 / French | validation | EXPR |
