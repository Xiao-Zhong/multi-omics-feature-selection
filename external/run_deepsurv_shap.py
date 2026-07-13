#!/usr/bin/env python
"""SHAP vs permutation feature importance for the SAME tuned DeepSurv (pycox) model.
Trains one net (best config), then derives two top-K panels — permutation and SHAP
(GradientExplainer) — so the only difference is the attribution method. Writes:
  results/tables/tp_DeepSurv-SHAP_importance.tsv
  external/deepsurv_shap_panels.json  {permutation, shap, overlap}
"""
import os, sys, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
SRC = "/mnt/data/hackathon/xiao/mpm_multiomics_pipeline/src"
sys.path.insert(0, SRC)
import common as C
import torch, torchtuples as tt, shap
from pycox.models import CoxPH

SEED = C.SEED; TOP_K = 20
np.random.seed(SEED); torch.manual_seed(SEED)
BEST = {"nodes": [64, 32], "lr": 0.01, "dropout": 0.1}   # chosen by the inner-CV sweep

def main():
    X, surv = C.load_train()
    t = surv["months"].values.astype("float32"); e = surv["event"].values.astype("float32")
    pool = pd.read_csv(os.path.join(C.TABLES, "tp_pool.tsv"), sep="\t", index_col=0).index.tolist()
    pool = [f for f in pool if f in X.columns]
    Xz = C.zscore_cols(X[pool]).fillna(0.0)
    Xall = Xz.values.astype("float32")

    # ---- train the (same) tuned model on full data ----
    net = tt.practical.MLPVanilla(len(pool), num_nodes=BEST["nodes"], out_features=1,
                                  batch_norm=True, dropout=BEST["dropout"], output_bias=False)
    model = CoxPH(net, tt.optim.Adam(lr=BEST["lr"], weight_decay=1e-4))
    rng = np.random.default_rng(SEED); perm = rng.permutation(len(Xall))
    nval = int(0.15 * len(Xall)); vi, ti = perm[:nval], perm[nval:]
    model.fit(Xall[ti], (t[ti], e[ti]), batch_size=32, epochs=512,
              callbacks=[tt.callbacks.EarlyStopping(patience=20)], verbose=False,
              val_data=(Xall[vi], (t[vi], e[vi])))
    base_c = C.cindex(model.predict(Xall).squeeze(-1), t, e)
    print(f"[model] tuned DeepSurv trained; in-sample C={base_c:.4f}")

    # ---- (1) permutation importance ----
    pr = np.random.default_rng(SEED); perm_imp = {}
    for i, f in enumerate(pool):
        Xp = Xall.copy(); Xp[:, i] = pr.permutation(Xp[:, i])
        c = C.cindex(model.predict(Xp).squeeze(-1), t, e)
        perm_imp[f] = (base_c - c) if not np.isnan(c) else 0.0
    perm_imp = pd.Series(perm_imp).sort_values(ascending=False)
    perm_panel = perm_imp.head(TOP_K).index.tolist()

    # ---- (2) SHAP (GradientExplainer on the torch net) ----
    net.eval()
    bg_idx = rng.choice(len(Xall), size=min(60, len(Xall)), replace=False)
    background = torch.tensor(Xall[bg_idx], dtype=torch.float32)
    explainer = shap.GradientExplainer(net, background)
    sv = explainer.shap_values(torch.tensor(Xall, dtype=torch.float32))
    sv = np.array(sv)
    if sv.ndim == 3:                       # (n, features, outputs) -> drop output axis
        sv = sv[..., 0]
    shap_imp = pd.Series(np.mean(np.abs(sv), axis=0), index=pool).sort_values(ascending=False)
    shap_panel = shap_imp.head(TOP_K).index.tolist()
    shap_imp.to_csv(os.path.join(C.TABLES, "tp_DeepSurv-SHAP_importance.tsv"), sep="\t")

    shared = sorted(set(perm_panel) & set(shap_panel))
    print(f"[panels] permutation vs SHAP overlap: {len(shared)}/{TOP_K}")
    print(f"  perm top8: {perm_panel[:8]}")
    print(f"  shap top8: {shap_panel[:8]}")
    json.dump({"permutation": perm_panel, "shap": shap_panel, "n_shared": len(shared),
               "shared": shared},
              open("/mnt/data/hackathon/xiao/mpm_multiomics_pipeline/external/deepsurv_shap_panels.json", "w"), indent=2)
    print("[done]")

if __name__ == "__main__":
    main()
