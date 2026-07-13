# MPM multi-omics feature-selection workflow — design

A reliability-first pipeline for discovering prognostic **biomarker panels** in malignant
**pleural** mesothelioma from only a few hundred samples. Trained on MESOMICS multi-omics,
validated by cross-cohort survival transfer. Built from scratch, self-contained.

## Goal

Choose features that are **reliable and accurate** enough to be biomarkers despite small n.
Every design choice trades raw fit for **reproducibility**: repeated-split ensembling,
bootstrap stability selection, cross-method consensus, and out-of-cohort validation.

## Data

**Training — MESOMICS (120 pleural patients, patient-level, one namespace of features):**

| layer | source | features | notes |
|---|---|---|---|
| `EXPR`  | MESOMICS_expression (FPKM, Ensembl→Hugo) | ~4,990 | log-FPKM |
| `CNV`   | TableS31-37 **S36 "AMP DEL genes"** | 6 | **gene-focused** driver CNV status (NF2, BAP1, CDKN2A, CDKN2B, MTAP, TERT); ordinal −2/−1/0/+1. MESOMICS reports 12 recurrent **deletion** peaks and no significant amplification peaks — this driver-gene view is the interpretable, TCGA-transferable representation. |
| `LOH`   | MESOMICS_loh | ~4,980 | gene-group LOH fraction |
| `METpro/METbod/METenh` | MESOMICS methylation ×3 | 15,000 | promoter / gene-body / enhancer M-values; each probe annotated to its gene (Illumina 450K → GENCODE v36) for interpretation and expression-surrogate transfer |
| `ALT`   | MESOMICS_alterations_drivers | 510 | binary somatic driver alteration (SNV/indel/CNV/SV on driver gene) |
| `SV`    | TableS41 structural variants | 6 | per-class SV burden (DEL/DUP/INV/TRA/…), log1p |

Total ≈ **25,500 features × 120 samples**, 103 events. Survival from S2 (Survival.Time/Censor);
histology (MME/MMB/MMS) → epithelioid vs non-epithelioid.

> miRNA is **not** available for MESOMICS (controlled), so it is not a training layer.
> Gene fusions are not gene-annotated in the open SV table, so SVs enter as per-class burden;
> gene-level fusion/SNV/CNV hits on driver genes are captured by the `ALT` layer.

**Validation cohorts (all pleural):** TCGA-MESO (74 Hmeljak-2018 patients, **multi-omics**:
expression + methylation450 + WXS driver alterations), Bueno 2016, NCI 2023 (pleural only),
Blum 2019, French E-MTAB-1719 — expression-only. Expression + survival come from the curated,
pleural-restricted processed matrices; TCGA methylation/alteration layers are rebuilt here so
multi-omics panels transfer, not just their expression subset.

## Stage 2 — in-house selection (the core method)

1. **Binary survival label.** Landmark = median OS among decedents (~13 mo); poor = died ≤ landmark,
   good = survived past it, uninformative early-censored dropped.
2. **Ensemble of repeated 2-part splits (×50).** Each split is event-stratified into two halves.
   - *Univariate:* χ² between every discretized feature and the label in each half; a feature scores
     a hit when it is in the **top-300 of BOTH halves**. Stability = hits / 50. (A single split gives
     only ~6 features by chance; the ensemble makes the screen reproducible.)
   - *Bivariate / epistasis:* over a compact per-split marginal pool (top-120), test every pair's
     **joint** genotype; a pair counts only if its joint χ² **beats both marginals** (true interaction)
     and is top-300 in both halves. Reproducible pairs → their constituent features. This is how the
     workflow handles **highly correlated features** — it keeps pairs that carry joint signal.
3. **Stability LASSO-Cox (bootstrap ×200).** Run separately on the univariate set and the bivariate
   set: L1 CoxNet on each bootstrap resample, keep features selected in ≥ 50% of resamples.
4. **Union** of the two stability panels = **Inhouse-Union**.

Discretization for χ²: expression/methylation/SV → tertiles; CNV peaks → loss/neutral/gain by sign;
LOH → present > 0.7; ALT → binary.

## Stage 3 — third-party selectors

All start from a shared pre-screen **POOL** (top-800 by univariate Cox concordance) for fairness,
each emitting a top-20 panel: **SIS**, **pawph** (weighted L1-Cox), **Network-LASSO** (correlation-graph
smoothing), **RSF**, **XGBoost** (survival:cox), **DeepSurv**, **DeepKEGG** (KEGG-pathway-masked NN),
**DeePathNet** (pathway-module NN with attention), plus **DeepSurv-TL** — a transfer-learning variant
pretrained on pooled external expression (Bueno+TCGA) and fine-tuned on MESOMICS, kept only if it helps.

## Stage 3b — ensemble / consensus

Rank-aggregate every panel: a feature's consensus score = number of independent methods selecting it
(+ borda rank weight). Produces `Consensus-VoteN` and `Consensus-TopK`. The ensemble is **kept only if
it beats the best individual third-party tool** on external transfer (Stage 4 decides).

## Stage 4 — evaluation (two modes)

Only MESOMICS and TCGA have full multi-omics; the other cohorts are expression-only. So each panel
is scored two ways:

1. **Native multi-omics transfer** — z-score the panel's actual features on MESOMICS, fit ridge CoxPH,
   transfer the fixed coefficients to each cohort that measures those features (expression everywhere;
   methylation + CNV + alterations also on TCGA). Faithful, but methylation/CNV features have no
   coverage on the expression-only cohorts. Columns `C_<cohort>`.
2. **Gene-expression surrogate** — collapse the panel to its **gene set** (methylation probe → annotated
   gene; CNV/ALT already gene-level), fit a Cox on the training cohort's *expression* of those genes,
   and transfer to every cohort's expression. Full coverage on all cohorts, so a multi-omics panel can
   be validated on Bueno/NCI/Blum/French. Columns `Cexpr_<cohort>`; **`Cexpr_mean` is the headline.**

Plus 5-fold out-of-fold internal C-index on the training cohort. Ranking in
`results/tables/panel_evaluation.tsv`; heatmaps in `results/figures/`.

> The **swap run** (`../mpm_multiomics_pipeline_swap`) repeats the entire pipeline with **TCGA** as the
> training cohort (its own multi-omics: expression, gene-level CNV, methylation, **miRNA**, alterations)
> and MESOMICS as multi-omics validation, then `compare_runs.py` reports which features are selected by
> BOTH runs — a robustness check on the choice of training data.

## Reproducibility

Isolated venv in `./.venv` (built on /mnt/data, root untouched). Fixed seed 42. `bash run_all.sh`.
