#!/usr/bin/env python
"""STAGE 6 - biological interpretation: are the selected features meaningful or noise?

For each panel (and the cross-method consensus set) we:
  1. collapse to a gene set (methylation probe -> annotated gene);
  2. KEGG pathway over-representation (hypergeometric ORA vs the measured gene universe, BH-FDR);
  3. overlap with a curated MPM driver / prognostic gene set and KEGG cancer pathways;
  4. a 'biological support' score = fraction of panel genes that sit in an enriched pathway,
     a cancer pathway, or the known-MPM set — high support argues the panel is not random noise.

Outputs: results/tables/biology_enrichment.tsv, biology_gene_support.tsv,
         biology_summary.json, results/figures/fig_enrichment.png
"""
import os, sys, json
import numpy as np, pandas as pd
from scipy.stats import hypergeom
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

# established MPM driver / prognostic genes (Hmeljak 2018, Bueno 2016, MESOMICS 2023, reviews)
MPM_GENES = {
    "BAP1", "NF2", "CDKN2A", "CDKN2B", "MTAP", "TP53", "SETD2", "LATS1", "LATS2", "SETDB1",
    "TERT", "DDX3X", "RDX", "WT1", "MSLN", "TYMS", "PDGFRB", "CCNE1", "CDK4", "CDK6",
    "SF3B1", "ULK2", "TRAF7", "NF1", "PBRM1", "SMARCA4", "MMP", "VIM", "CALB2",
}
CANCER_KEGG = ("hsa05200", "hsa05202", "hsa04010", "hsa04151", "hsa04115", "hsa04110",
               "hsa04310", "hsa04350", "hsa04520", "hsa04068", "hsa04210")  # cancer/MAPK/PI3K/p53/cellcycle/Wnt/TGFb/adherens/FoxO/apoptosis

def load_kegg():
    sets = {}
    with open(os.path.join(C.PROC, "kegg_hsa.gmt")) as f:
        for line in f:
            c = line.rstrip("\n").split("\t")
            sets[c[0]] = set(g for g in c[2:] if g and g != "NA")
    return sets

def gene_of(f, ann):
    if C.feat_layer(f).startswith("MET") and f in ann.index:
        g = str(ann.loc[f, "gene"])
    else:
        g = C.feat_gene(f)
    return "" if g in ("nan", "") else g

def panel_genes(feats, ann):
    return list(dict.fromkeys(g for g in (gene_of(f, ann) for f in feats) if g))

def ora(genes, sets, universe):
    """Hypergeometric over-representation of `genes` in each KEGG set (restricted to universe)."""
    G = set(genes) & universe
    M = len(universe); N = len(G)
    rows = []
    if N == 0:
        return pd.DataFrame(columns=["pathway", "overlap", "set_size", "genes", "p"])
    for name, gs in sets.items():
        s = gs & universe
        k = len(G & s)
        if k < 2:
            continue
        p = hypergeom.sf(k - 1, M, len(s), N)
        rows.append({"pathway": name, "overlap": k, "set_size": len(s),
                     "genes": ",".join(sorted(G & s)), "p": p})
    df = pd.DataFrame(rows).sort_values("p") if rows else pd.DataFrame(
        columns=["pathway", "overlap", "set_size", "genes", "p"])
    if len(df):
        m = len(df)
        df["q"] = (df["p"].values * m / (np.arange(len(df)) + 1)).clip(0, 1)
        df["q"] = df["q"][::-1].cummin()[::-1]
    return df

def main():
    ann = C.load_feature_annotation()
    sets = load_kegg()
    universe = set(g for g in ann["gene"].dropna().astype(str) if g and g != "nan")
    universe |= set(C.feat_gene(f) for f in ann.index if C.feat_layer(f) == "EXPR")
    cancer_sets = [gs for name, gs in sets.items() if name.split("_")[0] in CANCER_KEGG]
    cancer_genes = (set().union(*cancer_sets) if cancer_sets else set()) & universe
    print(f"[universe] {len(universe)} genes | cancer-pathway genes {len(cancer_genes)}")

    panels = {}
    for fn in ("inhouse_panels.json", "thirdparty_panels.json", "consensus_panels.json"):
        p = os.path.join(C.TABLES, fn)
        if os.path.exists(p):
            for k, v in json.load(open(p)).items():
                if v:
                    panels[k] = v

    enr_rows, summary, support_rows = [], {}, []
    for name, feats in panels.items():
        genes = panel_genes(feats, ann)
        df = ora(genes, sets, universe)
        sig = df[df["q"] < 0.10] if len(df) else df
        enr_genes = set()
        for g in sig["genes"]:
            enr_genes |= set(g.split(","))
        mpm = sorted(set(genes) & MPM_GENES)
        canc = sorted(set(genes) & cancer_genes)
        supported = set(genes) & (enr_genes | cancer_genes | MPM_GENES)
        support = len(supported) / max(1, len(genes))
        summary[name] = {
            "n_genes": len(genes), "n_enriched_pathways": int((df["q"] < 0.10).sum()) if len(df) else 0,
            "top_pathways": sig.head(5)["pathway"].tolist(),
            "min_q": float(df["q"].min()) if len(df) else np.nan,
            "mpm_known": mpm, "n_cancer_pathway_genes": len(canc),
            "biological_support": round(support, 3),
        }
        for _, r in sig.head(8).iterrows():
            enr_rows.append({"panel": name, **r[["pathway", "overlap", "set_size", "p", "q", "genes"]].to_dict()})
        support_rows.append({"panel": name, "n_genes": len(genes),
                             "biological_support": round(support, 3),
                             "n_enriched_pathways": summary[name]["n_enriched_pathways"],
                             "mpm_known": ",".join(mpm)})

    pd.DataFrame(enr_rows).to_csv(os.path.join(C.TABLES, "biology_enrichment.tsv"), sep="\t", index=False)
    sup = pd.DataFrame(support_rows).sort_values("biological_support", ascending=False)
    sup.to_csv(os.path.join(C.TABLES, "biology_gene_support.tsv"), sep="\t", index=False)
    json.dump(summary, open(os.path.join(C.TABLES, "biology_summary.json"), "w"), indent=2)

    # figure: biological support per panel
    d = sup.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, max(3, 0.34 * len(d))))
    colors = ["#c0392b" if p.startswith(("Inhouse", "Consensus")) else "#2c6fbb" for p in d["panel"]]
    ax.barh(d["panel"], d["biological_support"], color=colors)
    ax.set_xlabel("biological support (fraction of panel genes in enriched / cancer / known-MPM sets)")
    ax.set_title("Biological plausibility of panels  (higher = less likely noise)")
    fig.tight_layout(); fig.savefig(os.path.join(C.FIGS, "fig_biology_support.png"), dpi=130)
    plt.close(fig)

    print("[biology] support by panel:")
    print(sup.to_string(index=False))
    print("[done] biology tables + fig_biology_support.png")

if __name__ == "__main__":
    main()
