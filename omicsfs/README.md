# omicsfs — reliability-first feature selection for omics survival data

A small, dependency-light library that selects **reproducible** prognostic features from
high-dimensional omics data with time-to-event outcomes. It is the standalone, cohort-agnostic
version of the in-house selector developed in the MPM multi-omics pipeline.

Built for the **small-n / large-p** regime, where a single train/test split is noisy and easy to
overfit. Instead of trusting one fit, `omicsfs` keeps only features that **reproduce across
resamples**:

1. **Repeated event-stratified K-part screening** — in each random split, a feature must rank in the
   top of *every* part (univariate χ² vs a landmark poor/good survival label); aggregated over many
   splits.
2. **Epistasis hubs** *(optional)* — genes that recur in the strongest interacting pairs whose joint
   χ² beats both marginals.
3. **Bootstrap stability LASSO-Cox** — keep features selected in ≥ a fraction of bootstrapped
   elastic-net CoxNet fits (mild L2 stabilizes selection under collinearity).

The returned panel is the **union** of the stability-selected univariate and epistasis sets.

> `omicsfs` is a feature **selector**, not a survival model. Fit a Cox / RSF / DeepSurv on the
> selected features afterwards.

## Install

```bash
pip install numpy pandas scikit-survival     # dependencies
# then copy the omicsfs/ folder next to your code, or add this repo to PYTHONPATH
```

## Usage

```python
import pandas as pd
from omicsfs import OmicsSurvivalSelector

# X: samples x features DataFrame (any mix of continuous or discrete omics)
# t: survival time per sample   e: event indicator (1 = event, 0 = censored)
sel = OmicsSurvivalSelector(random_state=0).fit(X, durations=t, events=e)

sel.selected_features_     # -> list of reproducible features (the panel)
sel.univariate_panel_      # -> stability-selected reproducible-marginal features
sel.bivariate_panel_       # -> stability-selected epistasis-hub features
X_sel = sel.transform(X)   # -> X restricted to the selected panel
```

Runnable demo (recovers planted signal from synthetic data):

```bash
python -m omicsfs.example
```

## Key parameters

| parameter | default | meaning |
|---|---|---|
| `parts` | 2 | event-stratified parts per split; stricter as it grows |
| `n_splits` | 50 | repeated random K-part splits (screening ensemble) |
| `screen_tau` | 0.5 | keep a feature reproducible in ≥ this fraction of splits |
| `bivariate` | True | also screen epistatic interaction hubs (slower) |
| `n_boot` | 200 | bootstrap resamples for stability LASSO-Cox |
| `stab_thresh` | 0.5 | keep a feature selected in ≥ this fraction of bootstraps |
| `l1_ratio` | 0.9 | elastic-net mix for CoxNet (mostly L1, mild L2) |
| `discretize_kind` | `"auto"` | tertile-bin continuous features; keep small-cardinality codes |
| `landmark` | `None` | survival landmark for the poor/good label (default: median OS of decedents) |

## Notes & caveats

- **Discretization** for the χ² stage is generic (`auto`: tertiles for continuous features, integer
  codes for ≤5-level features). Domain-specific discretization can improve screening.
- **Small samples**: designed for it, but selection is only as good as the signal; validate panels
  on an independent cohort.
- **Determinism**: fully controlled by `random_state`.

## Citation

Developed as part of an MPM multi-omics prognostic biomarker study —
https://github.com/Xiao-Zhong/multi-omics-feature-selection
