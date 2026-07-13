"""Core implementation of the reliability-first omics survival feature selector.

Cohort-agnostic port of the MPM pipeline's in-house method (src/03_featsel_inhouse.py),
depending only on numpy / pandas / scikit-survival.
"""
import numpy as np
import pandas as pd

# ===================================================================== preprocessing
def _zscore(df):
    mu = df.mean(0); sd = df.std(0).replace(0, np.nan)
    return df.sub(mu, 1).div(sd, 1)

def discretize(X, kind="auto", n_bins=3):
    """Map each feature to small integer categories for chi-square screening.

    kind="auto": features with <= 5 unique values are treated as already-categorical (kept as
    rounded int codes — e.g. mutation 0/1/2, CNV loss/neutral/gain); continuous features are cut
    into `n_bins` quantile bins (default tertiles). kind="quantile" forces quantile binning;
    kind="none" passes numeric values through unchanged.
    """
    cols = {}
    for c in X.columns:
        s = pd.to_numeric(X[c], errors="coerce")
        k = s.notna()
        if kind == "none":
            cols[c] = s
        elif kind == "auto" and s[k].nunique() <= 5:
            cols[c] = s.round()
        elif k.sum() < n_bins * 2:
            cols[c] = s * np.nan
        else:
            try:
                q = pd.qcut(s[k], n_bins, labels=False, duplicates="drop")
            except Exception:
                cols[c] = s * np.nan
                continue
            col = pd.Series(np.nan, index=s.index); col[k] = q.astype(float)
            cols[c] = col
    return pd.DataFrame(cols, index=X.index)[list(X.columns)]

def landmark_binary_label(durations, events, landmark=None):
    """Dichotomize survival into poor(1)/good(0) for chi-square screening.
      poor = event==1 and time <= landmark   (died early)
      good = time  > landmark                 (survived past landmark; censored ok)
      drop = censored before landmark (uninformative, -> NaN)
    landmark defaults to the median survival time among decedents.
    """
    t = np.asarray(durations, float); e = np.asarray(events, int)
    if landmark is None:
        ev_t = t[e == 1]
        landmark = float(np.median(ev_t)) if len(ev_t) else float(np.median(t))
    y = np.full(len(t), np.nan)
    y[(e == 1) & (t <= landmark)] = 1
    y[t > landmark] = 0
    return y, landmark

# ===================================================================== chi-square
def _chi2_stat(cat, label):
    """Pearson chi2 between an int-coded feature and a binary label (NaNs dropped)."""
    cat = np.asarray(cat, float); label = np.asarray(label, float)
    m = ~(np.isnan(cat) | np.isnan(label))
    if m.sum() < 12:
        return 0.0
    a = cat[m].astype(np.int64); b = label[m].astype(np.int64)
    a -= a.min()
    na, nb = a.max() + 1, b.max() + 1
    if na < 2 or nb < 2:
        return 0.0
    ct = np.zeros((na, nb)); np.add.at(ct, (a, b), 1)
    rs = ct.sum(1, keepdims=True); cs = ct.sum(0, keepdims=True); tot = ct.sum()
    exp = rs @ cs / tot
    if (exp <= 0).any():
        ct = ct[(rs.ravel() > 0)]; rs = ct.sum(1, keepdims=True); exp = rs @ cs / tot
        if (exp <= 0).any() or ct.shape[0] < 2:
            return 0.0
    return float(((ct - exp) ** 2 / exp).sum())

# ===================================================================== screening ensemble
def _kpart_indices(events, rng, K):
    """Event-stratified partition of samples into K parts; returns list of K index arrays."""
    e = np.asarray(events, int); n = len(e)
    parts = [[] for _ in range(K)]
    for ev in (0, 1):
        ids = rng.permutation(np.where(e == ev)[0])
        for k in range(K):
            parts[k].extend(ids[k::K].tolist())
    return [np.array(sorted(p)) for p in parts]

def _pair_top(D, rows, lab, pool_idx, marg, topn, hub_pairs, hub_top):
    """Epistatic pair + interaction-hub screen. A pair qualifies only if its joint chi2 beats
    BOTH marginals; hub genes are drawn from the strongest such pairs by weighted degree."""
    y = lab[rows]; res = []
    for ii in range(len(pool_idx)):
        a = pool_idx[ii]; ca = D[rows, a]
        for jj in range(ii + 1, len(pool_idx)):
            b = pool_idx[jj]; cb = D[rows, b]
            m = ~(np.isnan(ca) | np.isnan(cb))
            if m.sum() < 12:
                continue
            jc = np.full(len(ca), np.nan)
            jc[m] = ca[m].astype(int) * 10 + cb[m].astype(int)
            st = _chi2_stat(jc, y)
            if st > marg[a] and st > marg[b]:
                res.append(((a, b), st))
    res.sort(key=lambda x: x[1], reverse=True)
    pair_set = set(p for p, _ in res[:topn])
    deg = {}
    for (a, b), st in res[:hub_pairs]:
        deg[a] = deg.get(a, 0.0) + st; deg[b] = deg.get(b, 0.0) + st
    hub_set = set(sorted(deg, key=lambda g: deg[g], reverse=True)[:hub_top])
    return pair_set, hub_set

def _screen_ensemble(D, lab, feats, events, K, repeats, seed, p):
    """Repeated K-part screening. A univariate 'hit' = top-`top_uni` in ALL K parts; an
    interaction-hub 'hit' = hub gene in ALL K parts. Returns stability = hits / repeats."""
    F = len(feats)
    uni_hits = np.zeros(F); marg_score = np.zeros(F); hub_hits = {}
    rng = np.random.default_rng(seed)
    for _ in range(repeats):
        idxs = _kpart_indices(events, rng, K)
        S = [np.array([_chi2_stat(D[ix, i], lab[ix]) for i in range(F)]) for ix in idxs]
        marg_score += np.mean(S, axis=0)
        tops = [set(np.argsort(s)[::-1][:p["top_uni"]]) for s in S]
        for i in set.intersection(*tops):
            uni_hits[i] += 1
        if p["bivariate"]:
            pool_idx = list(np.argsort(np.sum(S, axis=0))[::-1][:p["biv_pool"]])
            hub_sets = []
            for ix, s in zip(idxs, S):
                _, hs = _pair_top(D, ix, lab, pool_idx, s, p["top_biv"], p["hub_pairs"], p["hub_top"])
                hub_sets.append(hs)
            if hub_sets and all(len(h) for h in hub_sets):
                for gi in set.intersection(*hub_sets):
                    hub_hits[gi] = hub_hits.get(gi, 0) + 1
    uni_stab = pd.Series(uni_hits / repeats, index=feats).sort_values(ascending=False)
    hub_stab = pd.Series({feats[i]: c / repeats for i, c in hub_hits.items()}
                         ).sort_values(ascending=False) if hub_hits else pd.Series(dtype=float)
    return uni_stab, hub_stab

# ===================================================================== stability LASSO-Cox
def _stability_lasso(X, durations, events, feats, rng, p):
    from sksurv.linear_model import CoxnetSurvivalAnalysis
    from sksurv.util import Surv
    feats = [f for f in feats if f in X.columns]
    if len(feats) < 2:
        return pd.Series(0.0, index=feats)
    Z = _zscore(X[feats]).fillna(0.0)
    t = np.asarray(durations, float); e = np.asarray(events, int); n = len(Z)
    counts = pd.Series(0.0, index=feats)
    for _ in range(p["n_boot"]):
        idx = rng.integers(0, n, n)
        Xb, tb, eb = Z.values[idx], t[idx], e[idx]
        if eb.sum() < 5:
            continue
        y = Surv.from_arrays(event=eb.astype(bool), time=np.clip(tb, 1e-3, None))
        try:
            mdl = CoxnetSurvivalAnalysis(l1_ratio=p["l1_ratio"], n_alphas=40,
                                         alpha_min_ratio=0.05, max_iter=20000, normalize=False)
            mdl.fit(Xb, y)
            coefs = mdl.coef_
            nz = (np.abs(coefs) > 1e-8).sum(axis=0)
            j = np.argmin(np.abs(nz - p["target_k"]))
            counts[feats] += (np.abs(coefs[:, j]) > 1e-8).astype(float)
        except Exception:
            continue
    return (counts / p["n_boot"]).sort_values(ascending=False)

# ===================================================================== public API
class OmicsSurvivalSelector:
    """Reliability-first feature selection for high-dimensional omics survival data.

    Parameters
    ----------
    parts : int                number of event-stratified parts per split (2-4); stricter as it grows
    n_splits : int             repeated random K-part splits for the screening ensemble
    top_uni : int              top univariate features (per part) that must reproduce in ALL parts
    screen_tau : float         keep a feature if reproducible in >= this fraction of splits
    bivariate : bool           also screen epistatic interaction hubs
    biv_pool, top_biv, hub_pairs, hub_top : int   epistasis-screen sizes (see paper/DESIGN.md)
    n_boot : int               bootstrap resamples for stability LASSO-Cox
    stab_thresh : float        keep a feature selected in >= this fraction of bootstraps
    target_k : int             nonzero-coef target that sets each bootstrap's penalty
    l1_ratio : float           elastic-net mix for CoxNet (0.9 = mostly-L1, mild L2 for stability)
    discretize_kind : str      "auto" | "quantile" | "none"  (feature discretization for chi-square)
    landmark : float or None   survival landmark for the poor/good label (default: median OS of decedents)
    random_state : int
    verbose : bool

    Attributes (after fit)
    ----------------------
    selected_features_ : list          the panel = union of the two stability-selected sets
    univariate_panel_ : list           stability-selected reproducible-marginal features
    bivariate_panel_  : list           stability-selected epistasis-hub features (empty if bivariate=False)
    univariate_stability_, hub_stability_ : pd.Series    screening reproducibility scores
    stability_scores_ : dict of pd.Series                bootstrap selection frequencies
    landmark_ : float
    """

    def __init__(self, parts=2, n_splits=50, top_uni=300, screen_tau=0.5, bivariate=True,
                 biv_pool=120, top_biv=300, hub_pairs=50, hub_top=40, n_boot=200,
                 stab_thresh=0.5, target_k=12, l1_ratio=0.9, discretize_kind="auto",
                 landmark=None, random_state=42, verbose=True):
        self.parts = parts; self.n_splits = n_splits; self.top_uni = top_uni
        self.screen_tau = screen_tau; self.bivariate = bivariate; self.biv_pool = biv_pool
        self.top_biv = top_biv; self.hub_pairs = hub_pairs; self.hub_top = hub_top
        self.n_boot = n_boot; self.stab_thresh = stab_thresh; self.target_k = target_k
        self.l1_ratio = l1_ratio; self.discretize_kind = discretize_kind
        self.landmark = landmark; self.random_state = random_state; self.verbose = verbose

    def _log(self, *a):
        if self.verbose:
            print(*a, flush=True)

    def fit(self, X, durations, events):
        """X: (samples x features) DataFrame; durations/events: array-likes aligned to X's rows."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame (samples x features)")
        durations = np.asarray(durations, float); events = np.asarray(events, int)
        if len(durations) != len(X) or len(events) != len(X):
            raise ValueError("durations and events must have one entry per row of X")
        p = dict(top_uni=self.top_uni, bivariate=self.bivariate, biv_pool=self.biv_pool,
                 top_biv=self.top_biv, hub_pairs=self.hub_pairs, hub_top=self.hub_top,
                 n_boot=self.n_boot, target_k=self.target_k, l1_ratio=self.l1_ratio)

        label, self.landmark_ = landmark_binary_label(durations, events, self.landmark)
        self._log(f"[label] landmark={self.landmark_:.2f} | "
                  f"poor={int(np.nansum(label==1))} good={int(np.nansum(label==0))} "
                  f"drop={int(np.isnan(label).sum())}")

        feats = list(X.columns)
        D = discretize(X, kind=self.discretize_kind).values
        self._log(f"[ensemble] {self.n_splits} repeated {self.parts}-part splits on "
                  f"{X.shape[0]} samples x {X.shape[1]} features ...")
        uni_stab, hub_stab = _screen_ensemble(D, label, feats, events, self.parts,
                                              self.n_splits, self.random_state, p)
        self.univariate_stability_ = uni_stab; self.hub_stability_ = hub_stab

        uni_set = sorted(uni_stab[uni_stab >= self.screen_tau].index)
        if len(uni_set) < 8:
            uni_set = sorted(uni_stab.head(40).index)
        biv_set = sorted(hub_stab[hub_stab >= self.screen_tau].index)
        if len(biv_set) < 3 and len(hub_stab):
            biv_set = sorted(hub_stab.head(30).index)
        self._log(f"[screen] reproducible univariate={len(uni_set)} epistasis-hub={len(biv_set)}")

        rng = np.random.default_rng(self.random_state + 1)
        freq_uni = _stability_lasso(X, durations, events, uni_set, rng, p)
        freq_biv = (_stability_lasso(X, durations, events, biv_set,
                                     np.random.default_rng(self.random_state + 2), p)
                    if biv_set else pd.Series(dtype=float))
        self.stability_scores_ = {"univariate": freq_uni, "bivariate": freq_biv}

        self.univariate_panel_ = sorted(freq_uni[freq_uni >= self.stab_thresh].index)
        self.bivariate_panel_ = sorted(freq_biv[freq_biv >= self.stab_thresh].index) if len(freq_biv) else []
        self.selected_features_ = sorted(set(self.univariate_panel_) | set(self.bivariate_panel_))
        self._log(f"[panels] univariate={len(self.univariate_panel_)} "
                  f"bivariate={len(self.bivariate_panel_)} union={len(self.selected_features_)}")
        return self

    def transform(self, X):
        return X[[f for f in self.selected_features_ if f in X.columns]]

    def fit_transform(self, X, durations, events):
        return self.fit(X, durations, events).transform(X)
