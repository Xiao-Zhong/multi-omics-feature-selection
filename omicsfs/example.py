"""Minimal runnable example: recover prognostic features from synthetic omics + survival data.

    python -m omicsfs.example        (from the repo root)
"""
import numpy as np, pandas as pd
from omicsfs import OmicsSurvivalSelector

def make_data(n=150, n_informative=8, n_noise=400, seed=0):
    rng = np.random.default_rng(seed)
    Xinfo = rng.normal(size=(n, n_informative))
    Xnoise = rng.normal(size=(n, n_noise))                     # pure noise features
    beta = rng.choice([-1.2, -0.8, 0.8, 1.2], size=n_informative)
    lp = Xinfo @ beta
    u = rng.uniform(size=n)
    t_event = -np.log(u) / np.exp(lp - lp.mean())              # exponential survival ~ exp(risk)
    t_cens = rng.exponential(scale=np.mean(t_event) * 1.5, size=n)
    time = np.minimum(t_event, t_cens)
    event = (t_event <= t_cens).astype(int)
    cols = [f"INFO_{i}" for i in range(n_informative)] + [f"NOISE_{i}" for i in range(n_noise)]
    X = pd.DataFrame(np.hstack([Xinfo, Xnoise]), columns=cols)
    return X, time, event

def main():
    X, time, event = make_data()
    print(f"data: {X.shape[0]} samples x {X.shape[1]} features, {int(event.sum())} events\n")
    sel = OmicsSurvivalSelector(n_splits=20, n_boot=100, bivariate=False, random_state=0).fit(
        X, durations=time, events=event)
    panel = sel.selected_features_
    hits = [f for f in panel if f.startswith("INFO_")]
    fps = [f for f in panel if f.startswith("NOISE_")]
    print(f"\nselected panel ({len(panel)}): {panel}")
    print(f"informative recovered: {len(hits)}/8   false positives (noise): {len(fps)}")

if __name__ == "__main__":
    main()
