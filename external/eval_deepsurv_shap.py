#!/usr/bin/env python
"""Head-to-head: DeepSurv permutation panel vs SHAP panel — internal CV + cross-cohort transfer,
using the pipeline's own evaluation functions (identical methodology)."""
import os, sys, json, importlib.util
import numpy as np, pandas as pd
SRC = "/mnt/data/hackathon/xiao/mpm_multiomics_pipeline/src"
sys.path.insert(0, SRC)
import common as C
spec = importlib.util.spec_from_file_location("ev", os.path.join(SRC, "06_evaluate.py"))
ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)

def main():
    X, surv = C.load_train(); ann = C.load_feature_annotation()
    perm = json.load(open(os.path.join(C.TABLES, "thirdparty_panels.json")))["DeepSurv"]
    shap = json.load(open("/mnt/data/hackathon/xiao/mpm_multiomics_pipeline/external/deepsurv_shap_panels.json"))["shap"]
    Xz = C.zscore_cols(X).fillna(0.0)
    rows = []
    for name, feats in [("DeepSurv-permutation", perm), ("DeepSurv-SHAP", shap)]:
        feats = [f for f in feats if f in X.columns]
        risk = ev.internal_cv_risk(X, surv, feats)
        c_cv = C.cindex(risk, surv["months"].values, surv["event"].values)
        lo, hi = ev.bootstrap_cindex_ci(risk, surv["months"].values, surv["event"].values)
        surro, tr = ev.gene_surrogate(X, surv, feats, ann)
        per = {coh: (round(v[0], 3) if not np.isnan(v[0]) else None) for coh, v in surro.items()}
        vals = [v[0] for v in surro.values() if not np.isnan(v[0])]
        rows.append({"panel": name, "size": len(feats), "n_genes": len(tr),
                     "C_internal_cv": round(c_cv, 3), "CI": f"{lo:.3f}-{hi:.3f}",
                     "Cexpr_mean": round(float(np.mean(vals)), 3) if vals else None, **per})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    ov = len(set(perm) & set(shap))
    print(f"\npanel overlap: {ov}/20 shared features")

if __name__ == "__main__":
    main()
