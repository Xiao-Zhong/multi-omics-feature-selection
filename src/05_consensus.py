#!/usr/bin/env python
"""STAGE 3b - ENSEMBLE / consensus feature selection.

Rank-aggregation across every panel produced so far (in-house + third-party). A feature's
consensus score = number of independent methods that selected it (optionally weighted by
each method's rank). Reliable biomarkers are those chosen by MANY independent methods.

Produces:
  Consensus-Vote-N : features selected by >= N methods (N chosen to give a compact panel)
  Consensus-TopK   : top-K features by weighted vote (borda-style rank aggregation)
These are evaluated alongside the individual methods in Stage 4; if the ensemble wins on
cross-cohort transfer it is kept as the recommended panel.

Outputs: results/tables/consensus_votes.tsv, results/tables/consensus_panels.json
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

TOPK = 20

def load_all_panels():
    panels = {}
    for fn in ("inhouse_panels.json", "thirdparty_panels.json"):
        p = os.path.join(C.TABLES, fn)
        if os.path.exists(p):
            panels.update(json.load(open(p)))
    # drop empty and drop the in-house *union* (it is itself an aggregate) from voters
    voters = {k: v for k, v in panels.items() if v and k != "Inhouse-Union"}
    return panels, voters

def main():
    panels, voters = load_all_panels()
    print(f"[voters] {len(voters)} panels: {list(voters)}")

    votes = {}           # feature -> count
    borda = {}           # feature -> summed (K - rank) weight
    for name, feats in voters.items():
        k = len(feats)
        for r, f in enumerate(feats):
            votes[f] = votes.get(f, 0) + 1
            borda[f] = borda.get(f, 0.0) + (k - r) / k
    tab = pd.DataFrame({"votes": pd.Series(votes), "borda": pd.Series(borda)})
    ann = C.load_feature_annotation()
    tab["layer"] = [C.feat_layer(f) for f in tab.index]
    tab = tab.sort_values(["votes", "borda"], ascending=False)
    tab.to_csv(os.path.join(C.TABLES, "consensus_votes.tsv"), sep="\t")
    print("[votes] top:\n", tab.head(12).to_string())

    # Vote-N: pick the smallest N>=2 that yields a usable (>=8-feature) panel
    cons = {}
    for N in (4, 3, 2):
        sel = tab.index[tab["votes"] >= N].tolist()
        if len(sel) >= 8:
            cons[f"Consensus-Vote{N}"] = sel
            break
    else:
        cons["Consensus-Vote2"] = tab.index[tab["votes"] >= 2].tolist()
    cons["Consensus-TopK"] = tab.head(TOPK).index.tolist()

    json.dump(cons, open(os.path.join(C.TABLES, "consensus_panels.json"), "w"), indent=2)
    print("[done] consensus panels:", {k: len(v) for k, v in cons.items()})

if __name__ == "__main__":
    main()
