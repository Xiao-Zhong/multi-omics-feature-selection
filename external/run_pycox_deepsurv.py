#!/usr/bin/env python
"""Native pycox DeepSurv (havakv/pycox) on the MESOMICS multi-omics matrix.

Same data / same 800-feature pool / same C-index (C.cindex) as the in-house selectors.
Architecture + learning-rate are chosen by inner 5-fold CV (honest: training data only),
then the top-K permutation-importance panel is derived from the best configuration. Produces:
  - results/tables/tp_DeepSurv-pycox_importance.tsv
  - external/pycox_deepsurv_report.json  (perf + chosen hyperparameters + panel overlap)
"""
import os, sys, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

SRC = "/mnt/data/hackathon/xiao/mpm_multiomics_pipeline/src"
sys.path.insert(0, SRC)
import common as C

import torch, torchtuples as tt
from pycox.models import CoxPH
from sklearn.model_selection import KFold

SEED = C.SEED
np.random.seed(SEED); torch.manual_seed(SEED)
TOP_K = 20

# small architecture / learning-rate / dropout sweep, tuned by inner-CV C-index
GRID = [
    {"nodes": [32, 32], "lr": 0.01,  "dropout": 0.1},
    {"nodes": [64, 32], "lr": 0.01,  "dropout": 0.1},
    {"nodes": [32, 32], "lr": 0.001, "dropout": 0.1},
    {"nodes": [32, 32], "lr": 0.01,  "dropout": 0.3},
]

def make_model(in_features, hp):
    net = tt.practical.MLPVanilla(in_features, num_nodes=hp["nodes"], out_features=1,
                                  batch_norm=True, dropout=hp["dropout"], output_bias=False)
    return CoxPH(net, tt.optim.Adam(lr=hp["lr"], weight_decay=1e-4))

def fit(model, x_tr, t_tr, e_tr, x_val, t_val, e_val):
    cb = [tt.callbacks.EarlyStopping(patience=20)]
    model.fit(x_tr, (t_tr, e_tr), batch_size=32, epochs=512, callbacks=cb,
              verbose=False, val_data=(x_val, (t_val, e_val)))
    return model

def cv_cindex(Xp, t, e, hp, in_features):
    """5-fold out-of-fold C-index for a hyperparameter setting (per-fold z-score + early stop)."""
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.full(len(Xp), np.nan)
    for fold, (tr, te) in enumerate(kf.split(Xp)):
        rng = np.random.default_rng(SEED + fold)
        perm = rng.permutation(tr); nval = max(8, int(0.15 * len(tr)))
        vi, ti = perm[:nval], perm[nval:]
        mu = Xp.iloc[ti].mean(0); sd = Xp.iloc[ti].std(0).replace(0, np.nan)
        z = lambda idx: Xp.iloc[idx].sub(mu, 1).div(sd, 1).fillna(0.0).values.astype("float32")
        torch.manual_seed(SEED + fold)
        m = fit(make_model(in_features, hp), z(ti), t[ti], e[ti], z(vi), t[vi], e[vi])
        oof[te] = m.predict(z(te)).squeeze(-1)
    return C.cindex(oof, t, e)

def main():
    X, surv = C.load_train()
    t = surv["months"].values.astype("float32"); e = surv["event"].values.astype("float32")
    print(f"[data] {X.shape[0]} samples x {X.shape[1]} feats, {int(e.sum())} events")
    pool = pd.read_csv(os.path.join(C.TABLES, "tp_pool.tsv"), sep="\t", index_col=0).index.tolist()
    pool = [f for f in pool if f in X.columns]
    Xp = X[pool]
    print(f"[pool] {len(pool)} features")

    # ---------- tune architecture/lr by inner-CV C-index ----------
    scored = []
    for hp in GRID:
        c = cv_cindex(Xp, t, e, hp, len(pool))
        scored.append((c, hp))
        print(f"[sweep] nodes={hp['nodes']} lr={hp['lr']} dropout={hp['dropout']} -> OOF C={c:.4f}")
    cv_c, best = max(scored, key=lambda x: x[0])
    print(f"[best]  {best} OOF C-index={cv_c:.4f}")

    # ---------- full-data fit with best config for the feature panel ----------
    Xz = C.zscore_cols(Xp).fillna(0.0)
    Xall = Xz.values.astype("float32")
    rng = np.random.default_rng(SEED); perm = rng.permutation(len(Xall))
    nval = int(0.15 * len(Xall)); vi, ti = perm[:nval], perm[nval:]
    torch.manual_seed(SEED)
    model = fit(make_model(len(pool), best), Xall[ti], t[ti], e[ti], Xall[vi], t[vi], e[vi])
    base_c = C.cindex(model.predict(Xall).squeeze(-1), t, e)
    print(f"[perf] in-sample C-index = {base_c:.4f}")

    pr = np.random.default_rng(SEED)
    imp = {}
    for i, f in enumerate(pool):
        Xperm = Xall.copy(); Xperm[:, i] = pr.permutation(Xperm[:, i])
        c = C.cindex(model.predict(Xperm).squeeze(-1), t, e)
        imp[f] = (base_c - c) if not np.isnan(c) else 0.0
    imp = pd.Series(imp).sort_values(ascending=False)
    imp.to_csv(os.path.join(C.TABLES, "tp_DeepSurv-pycox_importance.tsv"), sep="\t")
    panel = imp.head(TOP_K).index.tolist()
    print(f"[panel] top-{TOP_K}: {panel[:8]} ...")

    report = {"chosen_hp": best, "cv_oof_cindex": float(cv_c), "insample_cindex": float(base_c),
              "sweep": [{"hp": hp, "oof_cindex": float(c)} for c, hp in scored], "top_panel": panel}
    json.dump(report, open("/mnt/data/hackathon/xiao/mpm_multiomics_pipeline/external/pycox_deepsurv_report.json", "w"), indent=2)
    print("[done] wrote tp_DeepSurv-pycox_importance.tsv + pycox_deepsurv_report.json")

if __name__ == "__main__":
    main()
