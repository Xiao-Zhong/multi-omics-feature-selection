#!/usr/bin/env python
"""Shared config, loaders, discretization and survival metrics.

Design: TRAIN cohort = MESOMICS multi-omics (expression, CNV, LOH, methylation
[promoter/genebody/enhancer], somatic driver alterations, structural-variant burden).
Validation cohorts (TCGA multi-omics + expression cohorts Bueno/NCI/Blum/French).

Feature naming convention (one namespace across omics layers):
    EXPR:<gene>      log-FPKM gene expression
    CNV:<gene>       ASCAT-style total copy ratio (~2 = diploid)
    LOH:<gene>       loss-of-heterozygosity fraction (0..1)
    METpro:<cg>      promoter methylation (M-value)
    METbod:<cg>      gene-body methylation (M-value)
    METenh:<cg>      enhancer methylation (M-value)
    ALT:<gene>       binary somatic driver alteration (SNV/indel/CNV/SV on driver gene)
    SV:<type>        structural-variant burden per class (DEL/DUP/INV/TRA/total, log1p)
    MUT:TMB          tumor mutational burden (mutations/Mb, log1p)
    MUT:<SBS>        COSMIC SNV mutational-signature relative exposure (0..1 proportion)

All processed matrices are written samples x features (rows = samples).
"""
import os, re, json, gzip
import numpy as np, pandas as pd

# ----------------------------------------------------------------- paths
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
TABLES = os.path.join(ROOT, "results", "tables")
FIGS = os.path.join(ROOT, "results", "figures")
HACK = "/mnt/data/hackathon/xiao"                 # sibling raw dataset workspace
MESO = os.path.join(HACK, "meso_mesomics")
MOFA = os.path.join(MESO, "mofa_matrices")
TCGA74 = os.path.join(HACK, "meso_tcga", "mpm74")
# validated processed expression+survival for the external cohorts (provenance below)
CURATED = os.path.join(HACK, "mpm_biomarker_pipeline", "data", "processed")
PROBEMAP = os.path.join(HACK, "meso_tcga", "raw", "gencode.v36.annotation.gtf.gene.probemap")

for d in (PROC, TABLES, FIGS):
    os.makedirs(d, exist_ok=True)

SEED = 42

# ----------------------------------------------------------------- annotation
_E2H = None
def ens2hugo():
    global _E2H
    if _E2H is None:
        m = {}
        with open(PROBEMAP) as f:
            next(f)
            for line in f:
                c = line.split("\t")
                m[c[0].split(".")[0]] = c[1]
        _E2H = m
    return _E2H

def meso_pid(col):
    """MOFA matrix column (MESO_001_T / MESO_002_T2) -> canonical patient id."""
    m = re.search(r"MESO_\d+", str(col))
    return "MESOMICS_" + m.group(0) if m else None

# ----------------------------------------------------------------- survival metrics
def zscore_cols(df):
    """z-score every column (feature) across samples; robust to constant/NA columns."""
    mu = df.mean(axis=0)
    sd = df.std(axis=0).replace(0, np.nan)
    return df.sub(mu, axis=1).div(sd, axis=1)

def cindex(risk, t, e):
    from sksurv.metrics import concordance_index_censored
    risk = np.asarray(risk, float); t = np.asarray(t, float); e = np.asarray(e)
    k = ~np.isnan(risk)
    if k.sum() < 5 or np.std(risk[k]) == 0:
        return np.nan
    return concordance_index_censored(e[k].astype(bool), t[k], risk[k])[0]

def binary_survival_label(t, e, landmark=None):
    """Dichotomize survival into poor(1)/good(0) prognosis for chi-square screening.
    poor  = event==1 and time <= landmark   (died early)
    good  = time > landmark                  (survived past landmark, censored ok)
    drop  = censored before landmark (uninformative).
    landmark defaults to the median OS among decedents."""
    t = np.asarray(t, float); e = np.asarray(e, int)
    if landmark is None:
        ev_t = t[e == 1]
        landmark = float(np.median(ev_t)) if len(ev_t) else float(np.median(t))
    y = np.full(len(t), np.nan)
    y[(e == 1) & (t <= landmark)] = 1
    y[t > landmark] = 0
    return y, landmark

# ----------------------------------------------------------------- discretization
def discretize(series, kind):
    """Map a continuous/categorical feature to small integer categories for chi-square.
    Returns an int-coded pd.Series with NaN preserved."""
    s = pd.to_numeric(series, errors="coerce")
    if kind == "binary":                     # ALT layer already 0/1
        return s.round().clip(0, 1)
    if kind == "loh":                        # LOH fraction -> LOH present if > 0.7
        return (s > 0.7).astype(float).where(s.notna())
    if kind == "cnvdel":                     # MESOMICS deletion peak: actual copy change ->
        # 0 none (t>-0.1) / 1 hemizygous (-1.3<t<=-0.1) / 2 homozygous (t<=-1.3), paper thresholds
        out = pd.Series(np.where(s > -0.1, 0, np.where(s <= -1.3, 2, 1)), index=s.index, dtype=float)
        return out.where(s.notna())
    if kind == "cnvpeak":                    # gene-level copy amplitude (sign) -> loss/neutral/gain
        out = pd.Series(np.where(s < -0.3, 0, np.where(s > 0.3, 2, 1)), index=s.index, dtype=float)
        return out.where(s.notna())
    # default: tertiles (expression, methylation, SV burden)
    k = s.notna()
    if k.sum() < 6:
        return s * np.nan
    try:
        q = pd.qcut(s[k], 3, labels=False, duplicates="drop")
    except Exception:
        return s * np.nan
    out = pd.Series(np.nan, index=s.index)
    out[k] = q.astype(float)
    return out

LAYER_KIND = {"EXPR": "tertile", "CNV": "cnvpeak", "LOH": "loh",
              "METpro": "tertile", "METbod": "tertile", "METenh": "tertile",
              "ALT": "binary", "SV": "tertile", "MUT": "tertile"}

def feat_layer(name):
    return name.split(":", 1)[0]

def feat_gene(name):
    """Best-effort gene symbol behind a feature (for cross-omics/cross-cohort mapping)."""
    layer, rest = name.split(":", 1)
    return rest

# ----------------------------------------------------------------- loaders (processed)
def load_train():
    """MESOMICS multi-omics feature matrix (samples x features) + survival."""
    X = pd.read_csv(os.path.join(PROC, "features_mesomics.tsv.gz"), sep="\t", index_col=0)
    surv = pd.read_csv(os.path.join(PROC, "survival_mesomics.tsv"), sep="\t", index_col=0)
    surv = surv.loc[surv.index.intersection(X.index)]
    X = X.loc[surv.index]
    return X, surv

def load_feature_annotation():
    return pd.read_csv(os.path.join(PROC, "feature_annotation.tsv"), sep="\t", index_col=0)

# ----------------------------------------------------------------- validation cohorts
# FRENCH (E-MTAB-1719) excluded: n=26 with 25/26 events (underpowered, C-index CI ~+/-0.15) and
# cross-platform (Affymetrix microarray vs RNA-seq training) -> near-chance for every panel.
# Alignment verified correct (sex concordance 28/29), so this is structural, not a fixable bug.
# Recovered genome-wide expression is kept on disk; re-add "FRENCH" here to restore it.
VAL_COHORTS = ["TCGA", "BUENO", "NCI", "BLUM"]
_CUR_KEY = {"TCGA": "tcga", "BUENO": "bueno", "NCI": "nci_external",
            "BLUM": "blum_external", "FRENCH": "french_external"}
_val_cache = {}

def load_val_survival(cohort):
    return pd.read_csv(os.path.join(CURATED, f"survival_{_CUR_KEY[cohort]}.tsv"),
                       sep="\t", index_col=0)

def load_val_expr(cohort):
    if ("expr", cohort) not in _val_cache:
        df = pd.read_csv(os.path.join(CURATED, f"expression_{_CUR_KEY[cohort]}.tsv.gz"),
                         sep="\t", index_col=0)
        df = df[~df.index.duplicated()]
        _val_cache[("expr", cohort)] = df
    return _val_cache[("expr", cohort)]

def _load_tcga_extra():
    if "tcga_meth" not in _val_cache:
        _val_cache["tcga_meth"] = pd.read_csv(os.path.join(PROC, "val_tcga_METH.tsv.gz"),
                                              sep="\t", index_col=0)
        _val_cache["tcga_alt"] = pd.read_csv(os.path.join(PROC, "val_tcga_ALT.tsv.gz"),
                                             sep="\t", index_col=0)
        cnvp = os.path.join(PROC, "val_tcga_CNV.tsv.gz")
        _val_cache["tcga_cnv"] = (pd.read_csv(cnvp, sep="\t", index_col=0)
                                  if os.path.exists(cnvp) else pd.DataFrame())
    return _val_cache["tcga_meth"], _val_cache["tcga_alt"], _val_cache["tcga_cnv"]

def resolve_features(cohort, features):
    """Return a (samples x available-features) DataFrame in the panel namespace for a
    validation cohort, pulling EXPR from expression, MET/ALT from TCGA extra layers.
    Features not measurable in the cohort are simply omitted (reported as coverage)."""
    expr = load_val_expr(cohort)                       # gene x sample
    cols = {}
    meth = alt = cnv = None
    if cohort == "TCGA":
        meth, alt, cnv = _load_tcga_extra()
    for f in features:
        layer, key = f.split(":", 1)
        if layer == "EXPR":
            if key in expr.index:
                cols[f] = expr.loc[key]
        elif layer.startswith("MET") and meth is not None:
            if key in meth.index:
                cols[f] = meth.loc[key]
        elif layer == "CNV" and cnv is not None and not cnv.empty:
            if key in cnv.index:
                cols[f] = cnv.loc[key]
        elif layer == "ALT" and alt is not None:
            if key in alt.index:
                cols[f] = alt.loc[key]
            else:                                       # gene sequenced, not mutated -> 0
                cols[f] = pd.Series(0.0, index=alt.columns)
    if not cols:
        return pd.DataFrame()
    return pd.DataFrame(cols)                           # samples x features
