#!/usr/bin/env python
"""Compare the in-house K-part screening for K = 2, 3, 4.

Reports, per K:
  - per-part sample / event counts (statistical-validity check: fewer samples per part as K grows)
  - stable-feature and panel sizes
  - internal CV + gene-expression-surrogate + native transfer C-index of the K's union panel
Recommends the K that keeps chi-square valid (adequate per-part n/events) AND validates best.
"""
import os, sys, json, importlib.util
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

def _imp(mod, path):
    spec = importlib.util.spec_from_file_location(mod, os.path.join(os.path.dirname(__file__), path))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

EV = _imp("evalmod", "06_evaluate.py")

def part_sizes(surv, label, K):
    """min per-part usable (non-null label) sample count and min poor/good event balance."""
    rng = np.random.default_rng(C.SEED)
    parts = [[] for _ in range(K)]
    for ev, grp in surv.groupby("event"):
        ids = rng.permutation(grp.index.values)
        for k in range(K):
            parts[k] += list(ids[k::K])
    sizes, valid, mincell = [], [], []
    for p in parts:
        lab = label.loc[p].dropna()
        sizes.append(len(p)); valid.append(len(lab))
        vc = lab.value_counts()
        mincell.append(int(vc.min()) if len(vc) else 0)
    return min(sizes), min(valid), min(mincell)

def main():
    X, surv = C.load_train()
    ann = C.load_feature_annotation()
    Xz = C.zscore_cols(X).fillna(0.0)
    label, _ = C.binary_survival_label(surv["months"].values, surv["event"].values)
    label = pd.Series(label, index=surv.index)

    rows = []
    for K, tag in [(2, ""), (3, "_k3"), (4, "_k4")]:
        pj = os.path.join(C.TABLES, f"inhouse_panels{tag}.json")
        mj = os.path.join(C.TABLES, f"inhouse_meta{tag}.json")
        if not os.path.exists(pj):
            print(f"[skip] K={K}: no panels ({pj})"); continue
        panels = json.load(open(pj)); meta = json.load(open(mj)) if os.path.exists(mj) else {}
        feats = [f for f in panels.get("Inhouse-Union", []) if f in X.columns]
        minn, minvalid, mincell = part_sizes(surv, label, K)
        coef = EV.fit_cox(Xz, surv, feats)
        cv = EV.internal_cv(X, surv, feats)
        sur, tr = EV.gene_surrogate(X, surv, feats, ann)
        sur_mean = np.nanmean([sur[c][0] for c in C.VAL_COHORTS]) if feats else np.nan
        nat = [EV.transfer(c, feats, coef)[0] for c in C.VAL_COHORTS]
        nat_mean = np.nanmean([x for x in nat if not np.isnan(x)]) if feats else np.nan
        rows.append({
            "K": K, "per_part_n": minn, "per_part_valid_label": minvalid,
            "per_part_min_class": mincell,
            "uni_stable": meta.get("uni_stable"), "biv_pairs": meta.get("biv_pairs"),
            "union_size": len(feats), "n_genes": len(tr),
            "C_internal_cv": round(cv, 3), "Cexpr_mean": round(sur_mean, 3),
            "C_native_mean": round(nat_mean, 3),
        })
        print(f"[K={K}] per-part n≈{minn} (valid-label {minvalid}, min class {mincell}) | "
              f"union={len(feats)} | cv={cv:.3f} surrogate={sur_mean:.3f} native={nat_mean:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(C.TABLES, "inhouse_kpart_comparison.tsv"), sep="\t", index=False)
    print("\n" + df.to_string(index=False))
    # recommendation: require >=10 samples/part with >=8 in the minority class for valid chi-square,
    # then pick the highest surrogate C-index among statistically-adequate K.
    ok = df[(df["per_part_valid_label"] >= 20) & (df["per_part_min_class"] >= 8)]
    pool = ok if len(ok) else df
    best = pool.sort_values("Cexpr_mean", ascending=False).iloc[0]
    print(f"\n[recommendation] K={int(best['K'])} "
          f"(adequate per-part statistics AND best surrogate C-index {best['Cexpr_mean']}).")
    json.dump({"recommended_parts": int(best["K"])},
              open(os.path.join(C.TABLES, "inhouse_kpart_recommendation.json"), "w"), indent=2)

if __name__ == "__main__":
    main()
