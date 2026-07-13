#!/usr/bin/env python
"""STAGE 2 - in-house reliability-first feature selection on MESOMICS multi-omics.

Pipeline (designed for reliability at small n):
  A. Split the cohort into 2 event-stratified parts.
  B. UNIVARIATE: in each part, chi-square between every (discretized) feature and a
     binary survival label; keep the top-300 by chi2 in each part; take the INTERSECTION
     -> features whose marginal prognostic signal reproduces across both halves.
  C. BIVARIATE (epistasis): over a compact pool, test every gene-pair's JOINT genotype
     against survival; a pair qualifies only if its joint chi2 beats both marginals. From
     the few STRONGEST such pairs per part, take their genes as interaction-HUBS and keep
     hubs present in ALL parts -> reproducible interaction drivers, robust to collinear
     partner-substitution yet strict (the exact-pair intersection is near-empty at this n;
     'any gene in a top pair' would saturate the pool, so hubs come from few top pairs).
  D. STABILITY LASSO-Cox: for the univariate set and the bivariate set separately, run
     bootstrapped elastic-net CoxNet (l1_ratio=0.9, 200x) and keep features selected in
     >= 50% of resamples. The mild L2 term stabilizes selection under feature collinearity.
  E. UNION of the two stability panels = the in-house multi-omics panel.

Outputs: results/tables/inhouse_*.tsv, results/tables/inhouse_panels.json
"""
import os, sys, json, itertools, warnings, argparse
import numpy as np, pandas as pd
from scipy.stats import chi2_contingency
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
warnings.filterwarnings("ignore")

TOP_UNI = 300         # top features per part (univariate)
BIV_POOL = 120        # marginal pool feeding the bivariate search
TOP_BIV = 300         # top pairs per part (bivariate)
HUB_PAIRS = 50        # draw interaction-hub genes from only this many strongest pairs (strict)
HUB_TOP = 40          # safety cap on hub genes per part (strength-weighted degree)
N_SPLITS = 50         # repeated event-stratified 2-part splits (screening ensemble)
SCREEN_TAU = 0.5      # keep feature/pair if reproducible in >= this fraction of splits
N_BOOT = 200          # bootstrap resamples for stability LASSO-Cox
STAB_THRESH = 0.5     # keep features selected in >= this fraction of bootstraps
TARGET_K = 12         # nonzero-coef target that defines each bootstrap's alpha

# ----------------------------------------------------------------- chi-square
def chi2_stat(cat, label):
    """Fast chi2 statistic between an int-coded feature and a binary label (numpy arrays,
    NaNs dropped). Builds the contingency table with np.add.at and applies Pearson chi2."""
    cat = np.asarray(cat, float); label = np.asarray(label, float)
    m = ~(np.isnan(cat) | np.isnan(label))
    if m.sum() < 12:
        return 0.0
    a = cat[m].astype(np.int64); b = label[m].astype(np.int64)
    a -= a.min()
    na, nb = a.max() + 1, b.max() + 1
    if na < 2 or nb < 2:
        return 0.0
    ct = np.zeros((na, nb))
    np.add.at(ct, (a, b), 1)
    rs = ct.sum(1, keepdims=True); cs = ct.sum(0, keepdims=True); tot = ct.sum()
    exp = rs @ cs / tot
    if (exp <= 0).any():
        ct = ct[(rs.ravel() > 0)]; rs = ct.sum(1, keepdims=True)
        exp = rs @ cs / tot
        if (exp <= 0).any() or ct.shape[0] < 2:
            return 0.0
    return float(((ct - exp) ** 2 / exp).sum())

def univariate_rank(disc, label, feats):
    lab = np.asarray(label, float)
    D = disc[feats].values
    s = {f: chi2_stat(D[:, i], lab) for i, f in enumerate(feats)}
    return pd.Series(s).sort_values(ascending=False)

# ----------------------------------------------------------------- bivariate epistasis
def joint_code(ci, cj):
    m = (~pd.isna(ci)) & (~pd.isna(cj))
    out = np.full(len(ci), np.nan)
    out[m] = ci[m].astype(int) * 10 + cj[m].astype(int)
    return out

def _pair_top(D, rows, lab, pool_idx, marg, topn, hub_pairs=HUB_PAIRS, hub_top=HUB_TOP):
    """Epistatic pair + interaction-hub screen among pool_idx in a data subset `rows`.
    A pair qualifies only if its joint chi2 beats BOTH marginals (epistasis emphasis).
    Returns (pair_set, hub_set):
      pair_set : the top-`topn` (a,b) index tuples by joint chi2 (interpretable pairs).
      hub_set  : genes drawn from only the `hub_pairs` STRONGEST pairs, ranked by
                 strength-weighted degree and capped at `hub_top`. This is the unit
                 intersected across parts for reproducible-interaction selection:
                 robust to collinear partner-substitution, yet strict (few top pairs)
                 so it does not saturate the marginal pool the way 'any gene in a top
                 pair' would."""
    y = lab[rows]
    res = []
    for ii in range(len(pool_idx)):
        a = pool_idx[ii]; ca = D[rows, a]
        for jj in range(ii + 1, len(pool_idx)):
            b = pool_idx[jj]; cb = D[rows, b]
            m = ~(np.isnan(ca) | np.isnan(cb))
            if m.sum() < 12:
                continue
            jc = np.full(len(ca), np.nan)
            jc[m] = ca[m].astype(int) * 10 + cb[m].astype(int)
            st = chi2_stat(jc, y)
            if st > marg[a] and st > marg[b]:
                res.append(((a, b), st))
    res.sort(key=lambda x: x[1], reverse=True)
    pair_set = set(p for p, _ in res[:topn])
    deg = {}
    for (a, b), st in res[:hub_pairs]:           # hubs from the few strongest pairs only
        deg[a] = deg.get(a, 0.0) + st
        deg[b] = deg.get(b, 0.0) + st
    hub_set = set(sorted(deg, key=lambda g: deg[g], reverse=True)[:hub_top])
    return pair_set, hub_set

def bivariate_rank(disc, label, pool, marg):
    """Return Series indexed by 'featA|featB' of joint chi2 for pairs that beat both
    marginals (epistasis emphasis)."""
    res = {}
    lab = label
    cols = {f: disc[f].values for f in pool}
    for fa, fb in itertools.combinations(pool, 2):
        jc = joint_code(cols[fa], cols[fb])
        st = chi2_stat(jc, lab)
        if st > marg.get(fa, 0) and st > marg.get(fb, 0):
            res[f"{fa}|{fb}"] = st
    return pd.Series(res).sort_values(ascending=False) if res else pd.Series(dtype=float)

# ----------------------------------------------------------------- stability LASSO-Cox
def stability_lasso(Xraw, surv, feats, rng):
    from sksurv.linear_model import CoxnetSurvivalAnalysis
    from sksurv.util import Surv
    feats = [f for f in feats if f in Xraw.columns]
    if len(feats) < 2:
        return pd.Series(0.0, index=feats)
    X = C.zscore_cols(Xraw[feats]).fillna(0.0)
    t = surv["months"].values.astype(float)
    e = surv["event"].values.astype(int)
    n = len(X)
    counts = pd.Series(0.0, index=feats)
    for b in range(N_BOOT):
        idx = rng.integers(0, n, n)
        Xb, tb, eb = X.values[idx], t[idx], e[idx]
        if eb.sum() < 5:
            continue
        y = Surv.from_arrays(event=eb.astype(bool), time=np.clip(tb, 1e-3, None))
        try:
            mdl = CoxnetSurvivalAnalysis(l1_ratio=0.9, n_alphas=40, alpha_min_ratio=0.05,
                                         max_iter=20000, normalize=False)
            mdl.fit(Xb, y)
            coefs = mdl.coef_                       # features x alphas
            nz = (np.abs(coefs) > 1e-8).sum(axis=0)
            j = np.argmin(np.abs(nz - TARGET_K))   # alpha giving ~TARGET_K features
            sel = np.abs(coefs[:, j]) > 1e-8
            counts[feats] += sel.astype(float)
        except Exception:
            continue
    return (counts / N_BOOT).sort_values(ascending=False)

# ----------------------------------------------------------------- K-part screening ensemble
def kpart_indices(surv, rng, K):
    """Event-stratified partition of the cohort into K parts; returns list of K index arrays."""
    parts = [[] for _ in range(K)]
    for ev, grp in surv.groupby("event"):
        ids = rng.permutation(grp.index.values)
        for k in range(K):
            parts[k] += list(ids[k::K])
    return [[surv.index.get_loc(x) for x in p] for p in parts]

def screen_ensemble(Dall, labarr, feats, surv, K, repeats, seed):
    """Repeated K-part screening. A feature is a univariate 'hit' when it is in the top-TOP_UNI
    of ALL K parts (stricter as K grows). For interactions we track two units, both requiring
    the joint chi2 to beat both marginals: (i) exact pairs top-TOP_BIV in ALL K parts
    (interpretable but very strict), and (ii) interaction-HUB genes top-HUB_TOP by
    strength-weighted degree in ALL K parts (the set used downstream; reproducible across
    collinear partner-substitution). Returns uni/pair/hub stability = hits / repeats."""
    uni_hits = np.zeros(len(feats)); marg_score = np.zeros(len(feats))
    pair_hits = {}; hub_hits = {}
    rng = np.random.default_rng(seed)
    for _ in range(repeats):
        idxs = kpart_indices(surv, rng, K)
        S = [np.array([chi2_stat(Dall[ix, i], labarr[ix]) for i in range(len(feats))]) for ix in idxs]
        marg_score += np.mean(S, axis=0)
        tops = [set(np.argsort(s)[::-1][:TOP_UNI]) for s in S]
        inter = set.intersection(*tops)
        for i in inter:
            uni_hits[i] += 1
        pool_idx = list(np.argsort(np.sum(S, axis=0))[::-1][:BIV_POOL])
        pair_sets = []; hub_sets = []
        for ix, s in zip(idxs, S):
            ps, hs = _pair_top(Dall, ix, labarr, pool_idx, s, TOP_BIV)
            pair_sets.append(ps); hub_sets.append(hs)
        for pr in set.intersection(*pair_sets):
            pair_hits[pr] = pair_hits.get(pr, 0) + 1
        if hub_sets and all(len(h) for h in hub_sets):
            for gi in set.intersection(*hub_sets):
                hub_hits[gi] = hub_hits.get(gi, 0) + 1
    uni_stab = pd.Series(uni_hits / repeats, index=feats).sort_values(ascending=False)
    marg = pd.Series(marg_score / repeats, index=feats)
    pair_stab = pd.Series({f"{feats[a]}|{feats[b]}": c / repeats for (a, b), c in pair_hits.items()}
                          ).sort_values(ascending=False) if pair_hits else pd.Series(dtype=float)
    hub_stab = pd.Series({feats[i]: c / repeats for i, c in hub_hits.items()}
                         ).sort_values(ascending=False) if hub_hits else pd.Series(dtype=float)
    return uni_stab, marg, pair_stab, hub_stab

# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="in-house K-part stability feature selection")
    ap.add_argument("--parts", type=int, default=2, help="number of parts per split (2/3/4)")
    ap.add_argument("--repeats", type=int, default=N_SPLITS, help="repeated random splits")
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--tag", default="", help="suffix for output files (e.g. _k3)")
    args = ap.parse_args()
    K, repeats, seed, tag = args.parts, args.repeats, args.seed, args.tag

    X, surv = C.load_train()
    disc = pd.read_csv(os.path.join(C.PROC, "features_mesomics_disc.tsv.gz"),
                       sep="\t", index_col=0).loc[X.index]

    label, landmark = C.binary_survival_label(surv["months"].values, surv["event"].values)
    label = pd.Series(label, index=surv.index)
    print(f"[label] landmark={landmark:.1f} mo | poor={int((label==1).sum())} good={int((label==0).sum())} drop={int(label.isna().sum())}")

    feats = list(X.columns)
    Dall = disc[feats].values
    labarr = label.values

    # ---- ENSEMBLE of repeated event-stratified K-part splits ----
    # feature/pair is 'stable' if reproducibly top-ranked in ALL K parts, aggregated over splits.
    print(f"[ensemble] {repeats} repeated {K}-part splits (seed={seed}) univariate + bivariate ...")
    uni_stab, marg, pair_stab, hub_stab = screen_ensemble(Dall, labarr, feats, surv, K, repeats, seed)
    uni_stab.to_csv(os.path.join(C.TABLES, f"inhouse_univariate_stability{tag}.tsv"), sep="\t")

    uni_stable = sorted(uni_stab[uni_stab >= SCREEN_TAU].index)
    if len(uni_stable) < 8:                          # fall back to top-40 by stability
        uni_stable = sorted(uni_stab.head(40).index)
    print(f"[univariate] reproducible (>= {SCREEN_TAU}) = "
          f"{int((uni_stab>=SCREEN_TAU).sum())} -> using {len(uni_stable)} features")

    # bivariate feature set = reproducible interaction-HUB genes (top-HUB_PAIRS strongest
    # super-marginal pairs, hub genes present in ALL parts, stable across splits).
    biv_feats = sorted(hub_stab[hub_stab >= SCREEN_TAU].index)
    if len(biv_feats) < 3 and len(hub_stab):
        biv_feats = sorted(hub_stab.head(30).index)
    biv_pairs = sorted(pair_stab[pair_stab >= SCREEN_TAU].index)   # kept for interpretability
    pair_stab.head(200).to_csv(os.path.join(C.TABLES, f"inhouse_bivariate_pairs{tag}.tsv"), sep="\t")
    hub_stab.to_csv(os.path.join(C.TABLES, f"inhouse_bivariate_hubs{tag}.tsv"), sep="\t")
    print(f"[bivariate] interaction-hub features (>= {SCREEN_TAU}) = {len(biv_feats)}; "
          f"strict reproducible pairs={len(biv_pairs)}")

    # D. stability LASSO-Cox on each set
    print(f"[stability] bootstrapped CoxNet ({N_BOOT}x) on univariate set ...")
    freq_uni = stability_lasso(X, surv, uni_stable, np.random.default_rng(seed + 1))
    print(f"[stability] bootstrapped CoxNet ({N_BOOT}x) on bivariate set ...")
    freq_biv = stability_lasso(X, surv, biv_feats, np.random.default_rng(seed + 2))
    freq_uni.to_csv(os.path.join(C.TABLES, f"inhouse_stability_univariate{tag}.tsv"), sep="\t")
    freq_biv.to_csv(os.path.join(C.TABLES, f"inhouse_stability_bivariate{tag}.tsv"), sep="\t")

    panel_uni = sorted(freq_uni[freq_uni >= STAB_THRESH].index)
    panel_biv = sorted(freq_biv[freq_biv >= STAB_THRESH].index)
    panel_union = sorted(set(panel_uni) | set(panel_biv))
    print(f"[panels] univariate={len(panel_uni)} bivariate={len(panel_biv)} union={len(panel_union)}")

    panels = {
        "Inhouse-Univariate": panel_uni,
        "Inhouse-Bivariate": panel_biv,
        "Inhouse-Union": panel_union,
    }
    json.dump(panels, open(os.path.join(C.TABLES, f"inhouse_panels{tag}.json"), "w"), indent=2)
    meta = dict(parts=K, repeats=repeats, seed=seed, landmark=landmark,
                uni_stable=len(uni_stable), biv_pairs=len(biv_pairs), biv_feats=len(biv_feats),
                panel_sizes={k: len(v) for k, v in panels.items()})
    json.dump(meta, open(os.path.join(C.TABLES, f"inhouse_meta{tag}.json"), "w"), indent=2)
    print("[done]", json.dumps(meta["panel_sizes"]))

if __name__ == "__main__":
    main()
