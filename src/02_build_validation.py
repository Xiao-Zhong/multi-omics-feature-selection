#!/usr/bin/env python
"""STAGE 1b - build transferable validation layers.

Expression + survival for every validation cohort come from the curated processed
matrices (gene-symbol x sample), all pleural-restricted. In addition we precompute the
two non-expression layers that can transfer from MESOMICS panels onto TCGA:
  - TCGA methylation450 subset to the cg probes used as MESOMICS MET features
  - TCGA driver-alteration binary (gene x sample) from the WXS mutation calls
These let multi-omics panels (EXPR + MET + ALT features) be scored on TCGA, not just
their expression subset. Bueno/NCI/Blum/French transfer via expression only.

Writes: data/processed/val_tcga_METH.tsv.gz, val_tcga_ALT.tsv.gz, validation_manifest.json
"""
import os, sys, json, gzip
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

def tcga_id(barcode):
    return "TCGA_" + "-".join(str(barcode).split("-")[:3])

def build_tcga_meth(cg_set):
    path = os.path.join(C.TCGA74, "methylation450.mpm74.tsv.gz")
    keep = []
    for chunk in pd.read_csv(path, sep="\t", index_col=0, chunksize=50000):
        keep.append(chunk[chunk.index.isin(cg_set)])
    meth = pd.concat(keep)
    meth.columns = [tcga_id(c) for c in meth.columns]
    meth = meth.loc[:, ~meth.columns.duplicated()]
    meth.to_csv(os.path.join(C.PROC, "val_tcga_METH.tsv.gz"), sep="\t", compression="gzip")
    return meth.shape

def build_tcga_cnv(cnv_genes):
    """TCGA ASCAT3 gene-level copy number (centered at 2 -> amplitude) for the driver genes
    used as CNV features, so gene-focused CNV panels transfer to TCGA."""
    E2H = C.ens2hugo()
    df = pd.read_csv(os.path.join(C.TCGA74, "cnv_ascat3.mpm74.tsv.gz"), sep="\t", index_col=0)
    df.index = [E2H.get(str(i).split(".")[0]) for i in df.index]
    df = df[[i in cnv_genes for i in df.index]]
    df = df.groupby(level=0).mean() - 2.0
    df.columns = [tcga_id(c) for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]
    df.to_csv(os.path.join(C.PROC, "val_tcga_CNV.tsv.gz"), sep="\t", compression="gzip")
    return df.shape

def build_tcga_alt(alt_genes):
    m = pd.read_csv(os.path.join(C.TCGA74, "somaticmutation_wxs.mpm74.tsv"), sep="\t")
    m["pid"] = [tcga_id(s) for s in m["sample"]]
    m = m[m["gene"].isin(alt_genes)]
    tab = (m.groupby(["gene", "pid"]).size().unstack(fill_value=0) > 0).astype(int)
    tab.to_csv(os.path.join(C.PROC, "val_tcga_ALT.tsv.gz"), sep="\t", compression="gzip")
    return tab.shape

def main():
    ann = C.load_feature_annotation()
    cg_set = set(f.split(":", 1)[1] for f in ann.index[ann.layer.str.startswith("MET")])
    alt_genes = set(f.split(":", 1)[1] for f in ann.index[ann.layer == "ALT"])
    cnv_genes = set(f.split(":", 1)[1] for f in ann.index[ann.layer == "CNV"])

    print(f"[1] TCGA methylation subset over {len(cg_set)} cg probes ...")
    ms = build_tcga_meth(cg_set)
    print(f"    TCGA METH {ms[0]} probes x {ms[1]} samples")

    print(f"[2] TCGA driver-alteration binary over {len(alt_genes)} genes ...")
    as_ = build_tcga_alt(alt_genes)
    print(f"    TCGA ALT  {as_[0]} genes x {as_[1]} samples")

    print(f"[2b] TCGA gene-level CNV over {len(cnv_genes)} driver genes ...")
    cs = build_tcga_cnv(cnv_genes)
    print(f"    TCGA CNV  {cs[0]} genes x {cs[1]} samples")

    manifest = {
        "expression_source": C.CURATED,
        "cohorts": {
            "TCGA":   {"layers": ["EXPR", "METH", "ALT"], "note": "74 Hmeljak-2018 pleural"},
            "BUENO":  {"layers": ["EXPR"]},
            "NCI":    {"layers": ["EXPR"], "note": "pleural only"},
            "BLUM":   {"layers": ["EXPR"]},
            "FRENCH": {"layers": ["EXPR"], "note": "tiny gene panel"},
        },
    }
    json.dump(manifest, open(os.path.join(C.PROC, "validation_manifest.json"), "w"), indent=2)
    print("[done] validation layers built")

if __name__ == "__main__":
    main()
