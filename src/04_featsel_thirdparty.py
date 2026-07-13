#!/usr/bin/env python
"""STAGE 3 - third-party feature selectors on the MESOMICS multi-omics matrix.

All selectors start from a shared pre-screen POOL (top univariate prognostic features)
for a fair comparison, and each emits a ranked importance -> top-K panel:

  SIS            marginal Cox |z| screening (Fan & Lv sure independence screening)
  LASSO-Cox      plain L1-penalized Cox
  RSF            Random Survival Forest, hyperparameter-tuned (OOB), permutation importance
  XGBoost        gradient-boosted Cox (survival:cox), CV-tuned, gain importance
  DeepSurv       native pycox DeepSurv (external/run_pycox_deepsurv.py), permutation importance

Only faithful, verified methods are used as survival comparators. Earlier unverified additions were
removed: in-house PAWPH-inspired and correlation-graph Cox heuristics (not faithful to any published
method), and the non-survival DeepKEGG / DeePathNet models (recurrence classification / drug-response,
not time-to-event). DeepSurv now comes from the genuine pycox package.

Outputs: results/tables/tp_<method>_importance.tsv, results/tables/thirdparty_panels.json
"""
import os, sys, json, warnings
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
warnings.filterwarnings("ignore")

POOL_N = 800
TOP_K = 20
SEED = C.SEED
np.random.seed(SEED)

# ----------------------------------------------------------------- shared prep
def marginal_cox_concordance(X, surv):
    """Per-feature univariate Cox concordance (prognostic strength) -> screening score."""
    from lifelines import CoxPHFitter
    t = surv["months"].values; e = surv["event"].values
    sc = {}
    Xz = C.zscore_cols(X).fillna(0.0)
    for f in X.columns:
        v = Xz[f].values
        if np.std(v) == 0:
            sc[f] = 0.0; continue
        c = C.cindex(v, t, e)
        sc[f] = abs(c - 0.5) if not np.isnan(c) else 0.0
    return pd.Series(sc).sort_values(ascending=False)

def surv_y(surv):
    from sksurv.util import Surv
    return Surv.from_arrays(event=surv["event"].values.astype(bool),
                            time=np.clip(surv["months"].values.astype(float), 1e-3, None))

# ----------------------------------------------------------------- selectors
def sel_SIS(X, surv, pool):
    from lifelines import CoxPHFitter
    t = surv["months"].values; e = surv["event"].values
    Xz = C.zscore_cols(X[pool]).fillna(0.0)
    z = {}
    for f in pool:
        try:
            df = pd.DataFrame({"x": Xz[f].values, "time": t, "event": e})
            r = CoxPHFitter().fit(df, "time", "event").summary.loc["x"]
            z[f] = abs(float(r["z"]))
        except Exception:
            z[f] = 0.0
    return pd.Series(z).sort_values(ascending=False)

def sel_LASSOCox(X, surv, pool):
    """Plain L1-penalized Cox (lasso) — a standard, unimpeachable penalized-Cox baseline with
    no network/graph structure. Rank features by |beta| at a moderate penalty (~25 nonzero)."""
    from sksurv.linear_model import CoxnetSurvivalAnalysis
    Xz = C.zscore_cols(X[pool]).fillna(0.0)
    y = surv_y(surv)
    try:
        mdl = CoxnetSurvivalAnalysis(l1_ratio=1.0, n_alphas=30, alpha_min_ratio=0.05,
                                     max_iter=20000).fit(Xz.values, y)
        coefs = mdl.coef_; nz = (np.abs(coefs) > 1e-8).sum(0)
        j = np.argmin(np.abs(nz - 25))
        imp = pd.Series(np.abs(coefs[:, j]), index=pool)
    except Exception:
        imp = pd.Series(0.0, index=pool)
    return imp.sort_values(ascending=False)

def sel_RSF(X, surv, pool):
    from sksurv.ensemble import RandomSurvivalForest
    Xz = C.zscore_cols(X[pool]).fillna(0.0).values
    y = surv_y(surv)
    # hyperparameter tuning via out-of-bag concordance (honest: OOB uses only training data)
    grid = [(msl, mf) for msl in (3, 8, 15) for mf in ("sqrt", "log2")]
    rsf, best_oob, best_hp = None, -np.inf, None
    for msl, mf in grid:
        m = RandomSurvivalForest(n_estimators=300, min_samples_leaf=msl, max_features=mf,
                                 oob_score=True, bootstrap=True, n_jobs=-1,
                                 random_state=SEED).fit(Xz, y)
        if m.oob_score_ > best_oob:
            best_oob, rsf, best_hp = m.oob_score_, m, (msl, mf)
    print(f"      [tuned RSF] min_samples_leaf={best_hp[0]} max_features={best_hp[1]} "
          f"oob_c={best_oob:.3f}")
    base = rsf.score(Xz, y)
    rng = np.random.default_rng(SEED)
    imp = {}
    for i, f in enumerate(pool):
        Xp = Xz.copy(); Xp[:, i] = rng.permutation(Xp[:, i])
        imp[f] = base - rsf.score(Xp, y)
    return pd.Series(imp).sort_values(ascending=False)

def sel_XGB(X, surv, pool):
    import xgboost as xgb
    Xz = C.zscore_cols(X[pool]).fillna(0.0).values
    t = surv["months"].values.astype(float); e = surv["event"].values
    ylabel = np.where(e == 1, t, -t)               # xgb cox: negative = censored
    dtrain = xgb.DMatrix(Xz, label=ylabel, feature_names=list(pool))
    base = dict(objective="survival:cox", eval_metric="cox-nloglik", subsample=0.8, seed=SEED)
    # tune depth / min_child_weight / eta and #rounds by 4-fold CV on cox-nloglik (early stopping)
    grid = [dict(max_depth=d, min_child_weight=w, eta=lr)
            for d in (2, 3) for w in (3, 6) for lr in (0.03, 0.1)]
    best, best_score, best_rounds = base, np.inf, 200
    for g in grid:
        params = {**base, **g}
        cv = xgb.cv(params, dtrain, num_boost_round=500, nfold=4,
                    early_stopping_rounds=30, seed=SEED, verbose_eval=False)
        score = cv["test-cox-nloglik-mean"].min()
        if score < best_score:
            best_score, best, best_rounds = score, params, len(cv)
    print(f"      [tuned XGB] depth={best['max_depth']} mcw={best['min_child_weight']} "
          f"eta={best['eta']} rounds={best_rounds} cv_nloglik={best_score:.3f}")
    bst = xgb.train(best, dtrain, num_boost_round=best_rounds)
    g = bst.get_score(importance_type="gain")
    return pd.Series({f: g.get(f, 0.0) for f in pool}).sort_values(ascending=False)

# ----------------------------------------------------------------- driver
# Survival comparators must be faithful implementations of an established, verified method on the
# SAME (time-to-event) task. DeepSurv is provided by the NATIVE pycox package (run below in its own
# venv). Unverified homemade heuristics (PAWPH-inspired weighted Cox, correlation-graph Cox) and the
# non-survival DeepKEGG / DeePathNet models were removed rather than scored as comparators.
SELECTORS = [
    ("SIS", sel_SIS),                       # marginal Cox screening (Fan & Lv sure independence screening)
    ("LASSO-Cox", sel_LASSOCox),            # plain L1-penalized Cox
    ("RSF", sel_RSF), ("XGBoost", sel_XGB),
]

def main():
    X, surv = C.load_train()
    print(f"[data] {X.shape[0]} samples x {X.shape[1]} features, {int(surv.event.sum())} events")
    if os.path.exists(os.path.join(C.TABLES, "tp_pool.tsv")):
        pool = pd.read_csv(os.path.join(C.TABLES, "tp_pool.tsv"), sep="\t", index_col=0).index.tolist()
    else:
        print("[pool] univariate Cox concordance pre-screen ...")
        conc = marginal_cox_concordance(X, surv)
        pool = conc.head(POOL_N).index.tolist()
        conc.head(POOL_N).to_frame("concordance").to_csv(os.path.join(C.TABLES, "tp_pool.tsv"), sep="\t")
    pool = [f for f in pool if f in X.columns]
    print(f"[pool] {len(pool)} features "
          f"(layers: {pd.Series([p.split(':')[0] for p in pool]).value_counts().to_dict()})")

    panels = {}
    for name, fn in SELECTORS:
        print(f"[run] {name} ...", flush=True)
        try:
            imp = fn(X, surv, pool)
            imp.to_csv(os.path.join(C.TABLES, f"tp_{name}_importance.tsv"), sep="\t")
            panels[name] = imp.head(TOP_K).index.tolist()
            print(f"      top: {panels[name][:5]}")
        except Exception as ex:
            print(f"      FAILED {name}: {type(ex).__name__}: {ex}")
            panels[name] = []

    # native DeepSurv (real havakv/pycox) — runs in its own isolated venv, gives the faithful
    # DeepSurv comparator instead of an in-house torch reimplementation.
    print("[run] DeepSurv (native pycox) ...", flush=True)
    try:
        import subprocess
        ext_dir = os.path.join(C.ROOT, "external")
        pycox_py = os.path.join(ext_dir, ".venv_pycox", "bin", "python")
        script = os.path.join(ext_dir, "run_pycox_deepsurv.py")
        if os.path.exists(pycox_py) and os.path.exists(script):
            env = dict(os.environ, TMPDIR=os.path.join(ext_dir, "tmp"))
            subprocess.run([pycox_py, script], check=True, env=env,
                           stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            imp = pd.read_csv(os.path.join(C.TABLES, "tp_DeepSurv-pycox_importance.tsv"),
                              sep="\t", index_col=0)
            panels["DeepSurv"] = imp.index[:TOP_K].tolist()
            print(f"      native DeepSurv panel: {panels['DeepSurv'][:5]}")
        else:
            print("      SKIP: pycox env not found at", pycox_py)
    except Exception as ex:
        print(f"      FAILED DeepSurv(native): {type(ex).__name__}: {ex}")

    json.dump(panels, open(os.path.join(C.TABLES, "thirdparty_panels.json"), "w"), indent=2)
    print("[done] panels:", {k: len(v) for k, v in panels.items()})

if __name__ == "__main__":
    main()
