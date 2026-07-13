#!/usr/bin/env python
"""STAGE 4 - unified cross-cohort evaluation for every panel.

Common harness (identical footing for in-house, third-party and ensemble panels):
  1. TRAIN: z-score the panel features on MESOMICS, fit a ridge CoxPH -> risk model.
  2. INTERNAL: 5-fold out-of-fold C-index on MESOMICS (honest internal estimate).
  3. TRANSFER: apply the fixed coefficients to each validation cohort using only the panel
     features measurable there (EXPR everywhere; MET/ALT also on TCGA), z-scored within the
     cohort. Report C-index and coverage per cohort.
Headline metric = mean transfer C-index across external cohorts (coverage-weighted).

Outputs: results/tables/panel_evaluation.tsv
"""
import os, sys, json, warnings
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
warnings.filterwarnings("ignore")

MIN_COHORT_N = 15

def load_panels():
    panels = {}
    for fn in ("inhouse_panels.json", "thirdparty_panels.json", "consensus_panels.json"):
        p = os.path.join(C.TABLES, fn)
        if os.path.exists(p):
            for k, v in json.load(open(p)).items():
                if v:
                    panels[k] = v
    return panels

def fit_cox(Xz, surv, feats):
    from lifelines import CoxPHFitter
    feats = [f for f in feats if f in Xz.columns]
    df = Xz[feats].copy(); df["time"] = surv["months"].values; df["event"] = surv["event"].values
    cph = CoxPHFitter(penalizer=0.1, l1_ratio=0.0).fit(df, "time", "event")
    return cph.params_.reindex(feats).fillna(0.0)

def internal_cv_risk(X, surv, feats, k=5):
    """Return the out-of-fold risk vector (used both for the point C-index and its bootstrap CI)."""
    from lifelines import CoxPHFitter
    from sklearn.model_selection import StratifiedKFold
    feats = [f for f in feats if f in X.columns]
    risk = np.full(len(X), np.nan)
    if len(feats) == 0:
        return risk
    y = surv["event"].values
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=C.SEED)
    idx = np.arange(len(X))
    for tr, te in skf.split(idx, y):
        Xtr = C.zscore_cols(X.iloc[tr][feats]).fillna(0.0)
        mu = X.iloc[tr][feats].mean(); sd = X.iloc[tr][feats].std().replace(0, np.nan)
        Xte = ((X.iloc[te][feats] - mu) / sd).fillna(0.0)
        df = Xtr.copy(); df["time"] = surv["months"].values[tr]; df["event"] = y[tr]
        try:
            cph = CoxPHFitter(penalizer=0.1, l1_ratio=0.0).fit(df, "time", "event")
            risk[te] = (Xte * cph.params_.reindex(feats).fillna(0.0)).sum(axis=1).values
        except Exception:
            pass
    return risk

def bootstrap_cindex_ci(risk, t, e, n_boot=1000, seed=C.SEED):
    """95% percentile-bootstrap CI for the C-index of a fixed risk vector (resample patients).
    At n~120 the CIs are wide and overlap across panels, so ranks alone are not significant."""
    risk = np.asarray(risk, float); t = np.asarray(t, float); e = np.asarray(e)
    keep = ~np.isnan(risk)
    risk, t, e = risk[keep], t[keep], e[keep]
    n = len(risk)
    if n < 10 or np.std(risk) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    cs = []
    for _ in range(n_boot):
        b = rng.integers(0, n, n)
        if e[b].sum() < 2 or np.std(risk[b]) == 0:
            continue
        c = C.cindex(risk[b], t[b], e[b])
        if not np.isnan(c):
            cs.append(c)
    if len(cs) < 50:
        return (np.nan, np.nan)
    return (float(np.percentile(cs, 2.5)), float(np.percentile(cs, 97.5)))

def transfer(cohort, feats, coef):
    surv = C.load_val_survival(cohort).dropna(subset=["months", "event"])
    M = C.resolve_features(cohort, feats)
    if M.empty:
        return np.nan, 0, 0
    M = M.reindex(surv.index).dropna(how="all")
    surv = surv.reindex(M.index)
    avail = [f for f in M.columns if f in coef.index]
    if len(avail) == 0 or len(surv) < MIN_COHORT_N:
        return np.nan, len(avail), len(surv)
    Mz = C.zscore_cols(M[avail]).fillna(0.0)
    risk = (Mz * coef.reindex(avail).values).sum(axis=1).values
    c = C.cindex(risk, surv["months"].values, surv["event"].values)
    return c, len(avail), len(surv)

def panel_genes(feats, ann):
    """Map a panel to its underlying gene set (methylation probe -> annotated gene)."""
    gs = []
    for f in feats:
        if C.feat_layer(f).startswith("MET") and f in ann.index:
            g = str(ann.loc[f, "gene"])
        else:
            g = C.feat_gene(f)
        if g and g != "nan" and g != "":
            gs.append(g)
    return list(dict.fromkeys(gs))

def gene_surrogate(X, surv, feats, ann):
    """Validate a multi-omics panel on EXPRESSION-ONLY cohorts by collapsing it to its gene
    set and scoring each gene's EXPRESSION: train a Cox on the training cohort's expression of
    those genes, transfer to every cohort's expression. Gives full coverage everywhere, which
    native multi-omics transfer cannot for expression-only cohorts (Bueno/NCI/Blum/French)."""
    from lifelines import CoxPHFitter
    genes = panel_genes(feats, ann)
    tr = [g for g in genes if f"EXPR:{g}" in X.columns]
    out = {}
    if len(tr) < 2:
        return {coh: (np.nan, 0) for coh in C.VAL_COHORTS}, tr
    Xtr = C.zscore_cols(X[[f"EXPR:{g}" for g in tr]]).fillna(0.0)
    Xtr.columns = tr
    df = Xtr.copy(); df["time"] = surv["months"].values; df["event"] = surv["event"].values
    coef = CoxPHFitter(penalizer=0.1, l1_ratio=0.0).fit(df, "time", "event").params_.reindex(tr).fillna(0.0)
    for coh in C.VAL_COHORTS:
        sv = C.load_val_survival(coh).dropna(subset=["months", "event"])
        ex = C.load_val_expr(coh)
        av = [g for g in tr if g in ex.index]
        if len(av) < 2 or len(sv) < MIN_COHORT_N:
            out[coh] = (np.nan, len(av)); continue
        M = ex.loc[av].T.reindex(sv.index)
        risk = (C.zscore_cols(M).fillna(0.0) * coef.reindex(av).values).sum(axis=1).values
        out[coh] = (C.cindex(risk, sv["months"].values, sv["event"].values), len(av))
    return out, tr

def main():
    X, surv = C.load_train()
    Xz = C.zscore_cols(X).fillna(0.0)
    ann = C.load_feature_annotation()
    panels = load_panels()
    print(f"[panels] evaluating {len(panels)}: {list(panels)}")

    rows = []
    for name, feats in panels.items():
        feats = [f for f in feats if f in X.columns]
        coef = fit_cox(Xz, surv, feats)
        risk_cv = internal_cv_risk(X, surv, feats)
        c_cv = C.cindex(risk_cv, surv["months"].values, surv["event"].values)
        ci_lo, ci_hi = bootstrap_cindex_ci(risk_cv, surv["months"].values, surv["event"].values)
        row = {"panel": name, "size": len(feats), "C_internal_cv": c_cv,
               "C_internal_cv_lo": ci_lo, "C_internal_cv_hi": ci_hi}
        # (a) native multi-omics transfer (uses real features; only MESOMICS<->TCGA multi-omics)
        ext = []
        for coh in C.VAL_COHORTS:
            c, cov, n = transfer(coh, feats, coef)
            row[f"C_{coh}"] = c
            row[f"cov_{coh}"] = f"{cov}/{len(feats)}"
            if not np.isnan(c):
                ext.append(c)
        row["C_native_mean"] = float(np.mean(ext)) if ext else np.nan
        # (b) gene-expression surrogate (full coverage on every cohort, incl. expression-only)
        surro, tr_genes = gene_surrogate(X, surv, feats, ann)
        sur = []
        for coh in C.VAL_COHORTS:
            c, cov = surro[coh]
            row[f"Cexpr_{coh}"] = c
            if not np.isnan(c):
                sur.append(c)
        row["n_genes"] = len(tr_genes)
        row["Cexpr_mean"] = float(np.mean(sur)) if sur else np.nan
        row["layers"] = ",".join(f"{k}:{v}" for k, v in
                                 pd.Series([C.feat_layer(f) for f in feats]).value_counts().items())
        rows.append(row)
        print(f"  {name:22s} n={len(feats):2d} cv={c_cv:.3f} "
              f"[{ci_lo:.3f}-{ci_hi:.3f}] "
              f"native_mean={row['C_native_mean']:.3f} expr_surrogate_mean={row['Cexpr_mean']:.3f}")

    df = pd.DataFrame(rows).sort_values("Cexpr_mean", ascending=False)
    cols = (["panel", "size", "n_genes", "C_internal_cv", "C_internal_cv_lo", "C_internal_cv_hi",
             "Cexpr_mean", "C_native_mean"]
            + [f"C_{c}" for c in C.VAL_COHORTS] + [f"Cexpr_{c}" for c in C.VAL_COHORTS]
            + [f"cov_{c}" for c in C.VAL_COHORTS] + ["layers"])
    df = df[[c for c in cols if c in df.columns]]
    df.to_csv(os.path.join(C.TABLES, "panel_evaluation.tsv"), sep="\t", index=False)
    print("\n[ranking] by gene-expression-surrogate mean C-index (internal CV C-index with 95% bootstrap CI):")
    print(df[["panel", "size", "n_genes", "C_internal_cv", "C_internal_cv_lo",
              "C_internal_cv_hi", "Cexpr_mean", "C_native_mean"]].to_string(index=False))

if __name__ == "__main__":
    main()
