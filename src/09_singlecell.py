#!/usr/bin/env python
"""STAGE 7 - single-cell validation of the selected genes.

Uses the Obacz 2024 pleura scRNA atlas (GSE243446, 63,748 cells, 10 FinalCellType classes
incl. Mesothelial = MPM cell-of-origin, Fibroblast, Endothelial, immune). For every gene in
the selected panels we compute per-cell-type mean expression and % of cells expressing, to
check the prognostic genes are actually expressed in a plausible cell type (not analysis noise)
and to reveal which compartment (tumor-like mesothelial vs stroma vs immune) drives each gene.

Outputs: results/tables/singlecell_celltype_expr.tsv, singlecell_gene_specificity.tsv,
         results/figures/fig_singlecell_heatmap.png
"""
import os, sys, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

SC = "/mnt/data/hackathon/xiao/meso_obacz2024_pleura_scrna/processed"
COUNTS = os.path.join(SC, "GSE243446_all_counts.txt.gz")
META = os.path.join(SC, "GSE243446_all_cells_metadata.csv.gz")

def panel_gene_set(ann):
    genes = {}
    for fn in ("inhouse_panels.json", "thirdparty_panels.json", "consensus_panels.json"):
        p = os.path.join(C.TABLES, fn)
        if not os.path.exists(p):
            continue
        for pan, feats in json.load(open(p)).items():
            for f in feats:
                lay, key = f.split(":", 1)
                g = str(ann.loc[f, "gene"]) if (lay.startswith("MET") and f in ann.index) else key
                if g and g != "nan":
                    genes.setdefault(g, set()).add(pan)
    return genes

def main():
    ann = C.load_feature_annotation()
    genes = panel_gene_set(ann)
    gset = set(genes)
    print(f"[genes] {len(gset)} panel genes to locate in scRNA")

    md = pd.read_csv(META, usecols=["Unnamed: 0", "FinalCellType"]).set_index("Unnamed: 0")
    ct = md["FinalCellType"]
    print(f"[scRNA] {len(ct)} cells | {ct.nunique()} cell types")

    # extract ONLY the panel-gene rows (+ header) in one awk decompression pass — the matrix
    # is 40k genes x 63k cells, far too wide for a pandas full parse.
    import subprocess, tempfile
    gf = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, dir=C.PROC)
    gf.write("\n".join(sorted(gset)) + "\n"); gf.close()
    out = tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False, dir=C.PROC); out.close()
    cmd = (f"zcat {COUNTS} | awk -F'\\t' 'FNR==NR{{g[$1]=1;next}} FNR==1||($1 in g)' "
           f"{gf.name} - > {out.name}")
    subprocess.run(cmd, shell=True, check=True, executable="/bin/bash")
    M = pd.read_csv(out.name, sep="\t", index_col=0)       # genes x cells (raw counts)
    os.unlink(gf.name); os.unlink(out.name)
    M = M[~M.index.duplicated()]
    cells = [c for c in M.columns if c in ct.index]
    M = M[cells]; types = ct.loc[cells]
    print(f"[matrix] {M.shape[0]} genes x {M.shape[1]} cells matched")

    # log1p(CP10k) normalization per cell, then per-cell-type mean + % expressing
    lib = M.sum(axis=0).replace(0, np.nan)
    norm = np.log1p(M.div(lib, axis=1) * 1e4)
    mean_ct = norm.T.groupby(types).mean().T                # genes x celltype
    pct_ct = (M > 0).T.groupby(types).mean().T * 100
    mean_ct.to_csv(os.path.join(C.TABLES, "singlecell_celltype_expr.tsv"), sep="\t")

    # per-gene specificity: dominant cell type + a specificity score (max share of z)
    z = mean_ct.sub(mean_ct.mean(axis=1), axis=0).div(mean_ct.std(axis=1).replace(0, np.nan), axis=0)
    spec = pd.DataFrame({
        "gene": mean_ct.index,
        "top_celltype": mean_ct.idxmax(axis=1).values,
        "top_mean_expr": mean_ct.max(axis=1).round(3).values,
        "pct_expressing_top": [pct_ct.loc[g, mean_ct.loc[g].idxmax()].round(1) for g in mean_ct.index],
        "n_panels": [len(genes[g]) for g in mean_ct.index],
        "expressed": (mean_ct.max(axis=1) > 0.1).values,
    }).sort_values(["n_panels", "top_mean_expr"], ascending=False)
    spec.to_csv(os.path.join(C.TABLES, "singlecell_gene_specificity.tsv"), sep="\t", index=False)

    # heatmap: selected genes x cell type, both hierarchically clustered so similar cell
    # types group together (stromal-like vs immune-like fall into separate clusters).
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import pdist
    COMPART = {"T cells": "immune", "B cells": "immune", "Plasma": "immune", "Mast cells": "immune",
               "Neutrophils": "immune", "Dendritic": "immune", "NK cells": "immune",
               "Macrophage": "immune", "Monocyte": "immune", "Myeloid": "immune",
               "Mesothelial": "tumour", "Fibroblast": "stromal", "Endothelial": "stromal",
               "Pericyte": "stromal", "SmoothMuscle": "stromal"}
    CCOL = {"immune": "#1f6fb0", "stromal": "#5a7d1e", "tumour": "#c0392b", "other": "#555"}

    def leaf_order(mat):
        if mat.shape[0] < 3:
            return list(range(mat.shape[0]))
        d = np.nan_to_num(pdist(mat, metric="correlation"))
        return list(leaves_list(linkage(d, method="average")))

    top = spec.head(35)["gene"].tolist()            # rows ordered by n_panels (most-selected first)
    hm = z.loc[top]
    # columns: GROUP BY COMPARTMENT (immune | stromal | tumour), alphabetical within each block,
    # with white separators between blocks. Rows kept in selection-frequency order.
    COMP_RANK = {"immune": 0, "stromal": 1, "tumour": 2, "other": 3}
    col_order = sorted(hm.columns, key=lambda c: (COMP_RANK[COMPART.get(c, "other")], c))
    hm = hm[col_order]
    fig, ax = plt.subplots(figsize=(1.7 + 0.7 * hm.shape[1], 1.7 + 0.30 * hm.shape[0]))
    im = ax.imshow(hm.values, cmap="RdBu_r", vmin=-1.5, vmax=1.5, aspect="auto")
    ax.set_xticks(range(hm.shape[1])); ax.set_xticklabels(hm.columns, rotation=40, ha="right", fontsize=8)
    for lab in ax.get_xticklabels():                                # colour labels by compartment
        lab.set_color(CCOL.get(COMPART.get(lab.get_text(), "other")))
        lab.set_fontweight("bold")
    ax.set_yticks(range(hm.shape[0])); ax.set_yticklabels(hm.index, fontsize=7)
    comps = [COMPART.get(c, "other") for c in hm.columns]           # white lines between blocks
    for i in range(1, len(comps)):
        if comps[i] != comps[i - 1]:
            ax.axvline(i - 0.5, color="white", lw=2.5)
    ax.set_title("scRNA cell-type expression of selected genes (z-scored)\n"
                 "cell types grouped by compartment (blue immune | green stromal | red tumour); "
                 "genes clustered into modules", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.025, label="z (row)")
    fig.tight_layout(); fig.savefig(os.path.join(C.FIGS, "fig_singlecell_heatmap.png"), dpi=130)
    plt.close(fig)

    n_expr = int(spec["expressed"].sum())
    print(f"[result] {n_expr}/{len(spec)} panel genes detectably expressed in pleura scRNA")
    print(spec.head(12).to_string(index=False))
    summ = {"n_genes": len(spec), "n_expressed": n_expr,
            "celltype_of_top_genes": spec.head(20).set_index("gene")["top_celltype"].to_dict()}
    json.dump(summ, open(os.path.join(C.TABLES, "singlecell_summary.json"), "w"), indent=2)
    print("[done] single-cell validation")

if __name__ == "__main__":
    main()
