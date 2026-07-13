#!/usr/bin/env python
"""Evaluate the IN-HOUSE DeepSurv (src/04) under the same 5-fold OOF CV as pycox,
so native-vs-reimplementation is compared on identical data/splits/metric."""
import os, sys, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
SRC = "/mnt/data/hackathon/xiao/mpm_multiomics_pipeline/src"
sys.path.insert(0, SRC)
import common as C
import torch
from sklearn.model_selection import KFold
SEED = C.SEED
np.random.seed(SEED); torch.manual_seed(SEED)

def cox_ph_loss(risk, t, e):
    order = torch.argsort(t, descending=True)
    risk, e = risk[order], e[order]
    logcum = torch.logcumsumexp(risk, dim=0)
    return -((risk - logcum) * e).sum() / (e.sum() + 1e-8)

def train_net(Xt, t, e, p, epochs=250, lr=1e-3, wd=1e-3):
    torch.manual_seed(SEED)
    net = torch.nn.Sequential(torch.nn.Linear(p, 32), torch.nn.ReLU(),
                              torch.nn.Dropout(0.3), torch.nn.Linear(32, 1))
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    tt = torch.tensor(t, dtype=torch.float32); te = torch.tensor(e, dtype=torch.float32)
    for _ in range(epochs):
        net.train(); opt.zero_grad()
        loss = cox_ph_loss(net(Xt).squeeze(-1), tt, te)
        loss.backward(); opt.step()
    return net

def main():
    X, surv = C.load_train()
    t = surv["months"].values.astype(float); e = surv["event"].values.astype(float)
    pool = pd.read_csv(os.path.join(C.TABLES, "tp_pool.tsv"), sep="\t", index_col=0).index.tolist()
    pool = [f for f in pool if f in X.columns]
    Xp = X[pool]
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.full(len(Xp), np.nan)
    for tr, te_idx in kf.split(Xp):
        mu = Xp.iloc[tr].mean(0); sd = Xp.iloc[tr].std(0).replace(0, np.nan)
        def z(idx): return torch.tensor(Xp.iloc[idx].sub(mu,1).div(sd,1).fillna(0.0).values, dtype=torch.float32)
        net = train_net(z(tr), t[tr], e[tr], len(pool))
        net.eval()
        with torch.no_grad():
            oof[te_idx] = net(z(te_idx)).squeeze(-1).numpy()
    c = C.cindex(oof, t, e)
    print(f"[in-house DeepSurv] 5-fold OOF C-index = {c:.4f}")
    json.dump({"inhouse_deepsurv_oof_cindex": float(c)},
              open("/mnt/data/hackathon/xiao/mpm_multiomics_pipeline/external/inhouse_deepsurv_cv.json","w"), indent=2)

if __name__ == "__main__":
    main()
