"""omicsfs — reliability-first feature selection for high-dimensional omics survival data.

A standalone, cohort-agnostic version of the in-house selector from the MPM multi-omics pipeline.
Designed for the small-n / large-p time-to-event regime where a single train/test split is noisy:
it prizes *reproducibility* over a single fit by combining

  1. repeated event-stratified K-part screening  — a feature must rank top in ALL K parts,
     aggregated over many random splits (univariate chi-square vs a landmark survival label);
  2. epistasis hubs                              — genes recurring in the strongest super-marginal
     interacting pairs (optional);
  3. bootstrap stability LASSO-Cox               — keep features selected in >= a fraction of
     bootstrapped elastic-net CoxNet fits.

The returned panel is the UNION of the stability-selected univariate and epistasis sets.

Usage
-----
    from omicsfs import OmicsSurvivalSelector
    sel = OmicsSurvivalSelector(random_state=0).fit(X, durations=t, events=e)
    sel.selected_features_          # -> list of reproducible features (the panel)
    X_sel = sel.transform(X)        # -> X restricted to the panel

X is a samples x features DataFrame (any mix of continuous or discrete omics); durations/events are
array-likes of survival time and 0/1 event. This is a feature SELECTOR, not a survival model — fit a
Cox / RSF / DeepSurv on `X_sel` afterwards.

Requires: numpy, pandas, scipy, scikit-survival.
"""
from .selector import OmicsSurvivalSelector, discretize, landmark_binary_label

__all__ = ["OmicsSurvivalSelector", "discretize", "landmark_binary_label"]
__version__ = "0.1.0"
