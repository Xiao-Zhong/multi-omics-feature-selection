#!/usr/bin/env python
"""STAGE 1 - build harmonized MESOMICS multi-omics TRAIN matrix + survival.

Layers (all pleural MPM, patient-level):
  EXPR   MESOMICS_expression.tsv.gz         (Ensembl log-FPKM -> Hugo)
  CNV    TableS31-37 S33 'Peaks per samples' (24 recurrent GISTIC peaks, amplitude)
  LOH    MESOMICS_loh.tsv.gz                 (gene-group LOH fraction)
  METpro/METbod/METenh  MESOMICS_methylation_*.tsv.gz (M-values, cg probes)
  ALT    MESOMICS_alterations_drivers.tsv.gz (510 driver genes, binary any-alteration SNV/indel/CNV/SV)
  SV     TableS41 SVs                        (per-class structural-variant burden, log1p)
  MUT    TableS45 SNB and Signatures         (TMB log1p + COSMIC SNV-signature exposures)

Writes:
  data/processed/features_mesomics.tsv.gz     samples x features (raw scale)
  data/processed/features_mesomics_disc.tsv.gz samples x features (int categories for chi-square)
  data/processed/survival_mesomics.tsv        sample, months, event, nonepi
  data/processed/feature_annotation.tsv       feature, layer, gene
  data/processed/build_summary.json
"""
import os, re, sys, json, gzip
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

E2H = C.ens2hugo()

def _read_mofa(fname):
    """MOFA matrix (features x samples) -> DataFrame samples x features (patient ids)."""
    df = pd.read_csv(os.path.join(C.MOFA, fname), sep="\t", index_col=0)
    df.columns = [C.meso_pid(c) for c in df.columns]
    df = df.loc[:, [c is not None for c in df.columns]]
    return df.T                                   # samples x features

def _ens_group_to_hugo(idx):
    """LOH/CNV row index may be comma-joined Ensembl ids -> single readable gene tag."""
    genes = []
    for e in str(idx).split(","):
        g = E2H.get(e.split(".")[0])
        if g and g not in genes:
            genes.append(g)
    return "|".join(genes[:3]) if genes else str(idx).split(",")[0].split(".")[0]

def build_expression():
    df = _read_mofa("MESOMICS_expression.tsv.gz")          # samples x ENSG
    cols = {}
    for e in df.columns:
        g = E2H.get(str(e).split(".")[0])
        if g:
            cols.setdefault(g, []).append(e)
    out = pd.DataFrame({f"EXPR:{g}": df[v].mean(axis=1) for g, v in cols.items()})
    return out

def build_loh():
    df = _read_mofa("MESOMICS_loh.tsv.gz")
    out = df.copy()
    out.columns = [f"LOH:{_ens_group_to_hugo(c)}" for c in df.columns]
    out = out.T.groupby(level=0).mean().T                  # collapse tag collisions
    return out

def build_methylation():
    frames = []
    for tag, fname in [("METpro", "MESOMICS_methylation_promoter.tsv.gz"),
                       ("METbod", "MESOMICS_methylation_genebody.tsv.gz"),
                       ("METenh", "MESOMICS_methylation_enhancer.tsv.gz")]:
        df = _read_mofa(fname)
        df.columns = [f"{tag}:{c}" for c in df.columns]
        frames.append(df)
    return pd.concat(frames, axis=1)

def build_alterations():
    df = _read_mofa("MESOMICS_alterations_drivers.tsv.gz")
    out = df.copy()
    out.columns = [f"ALT:{E2H.get(str(c).split('.')[0], str(c).split('.')[0])}" for c in df.columns]
    out = out.T.groupby(level=0).max().T                    # any-alteration if collision
    return out

_CNV_CODE = {
    "no cn change": 0.0, "nloh": 0.0, "possible false negative": np.nan,
    "amplification": 1.0,
    "heterozygous deletion": -1.0, "cut heterozygous deletion": -1.0,
    "homozygous deletion": -2.0, "cut homozygous deletion": -2.0,
    "homozygous deletion/heterozygous deletion": -1.5,
}

def build_cnv_genes():
    """S36 'AMP DEL genes' -> gene-focused CNV status for the canonical MPM drivers
    (NF2, BAP1, CDKN2A, CDKN2B, MTAP, TERT). Ordinal copy code: -2 homozygous del,
    -1 heterozygous del, 0 no change / nLOH, +1 amplification. Transfers to TCGA gene CNV.
    MESOMICS reports 12 recurrent DELETION peaks and no significant amplification peaks;
    this driver-gene view is the interpretable, cross-cohort-comparable representation."""
    d = pd.read_excel(os.path.join(C.MESO, "TableS31-37_CNVs.xlsx"),
                      sheet_name="S36 AMP DEL genes", header=1)
    d = d.dropna(subset=["Cohort"])
    d["pid"] = [C.meso_pid(s) for s in d["Cohort"]]
    d = d.dropna(subset=["pid"])
    cols = {"NF2": ["NF2"], "BAP1": ["BAP1"], "CDKN2A/2B": ["CDKN2A", "CDKN2B"],
            "MTAP": ["MTAP"], "TERT": ["TERT"]}
    out = {}
    for src, genes in cols.items():
        if src not in d.columns:
            continue
        code = d[src].astype(str).str.strip().str.lower().map(_CNV_CODE)
        for g in genes:
            out[f"CNV:{g}"] = code.values
    mat = pd.DataFrame(out, index=d["pid"].values)
    mat = mat[~mat.index.duplicated(keep="first")]
    return mat                                             # samples x genes

def build_sv_burden():
    """S41 SV calls -> per-class burden (log1p count) per sample."""
    d = pd.read_excel(os.path.join(C.MESO, "TableS41-42_SVs.xlsx"),
                      sheet_name="S41 SVs", header=1)
    d = d[[c for c in d.columns if str(c) != "nan"]]
    d["pid"] = [C.meso_pid(s) for s in d["sample"]]
    d = d.dropna(subset=["pid"])
    tab = d.groupby(["pid", "Type"]).size().unstack(fill_value=0)
    tab["total"] = tab.sum(axis=1)
    tab = np.log1p(tab)
    tab.columns = [f"SV:{c}" for c in tab.columns]
    return tab

def build_mut():
    """S45 'SNB and Signatures' -> mutation-burden + COSMIC SNV-signature layer.
    Two stacked sub-tables in one sheet:
      * TMB block (header at sheet row 2): per-sample total_perMB (mutations/Mb).
      * Reconstructed COSMIC SNV signatures (header at sheet row 181): per-sample
        signature counts across SBS1/2/3/4/5/13/30/31/35/40.
    Features:
      MUT:TMB    log1p(mutations/Mb)                 (~115 samples)
      MUT:<SBS>  relative exposure = count / sample total   (~46 samples, WGS subset)
    Multi-region samples (T1/T2) collapse to one patient by mean. Signatures with
    <10 non-null or no variance are dropped (uninformative for chi-square screening)."""
    path = os.path.join(C.MESO, "TableS44-46_SNVs.xlsx")
    SH = "S45 SNB and Signatures"

    # --- TMB: leading contiguous MESO rows under the TMB header (sheet row index 2) ---
    tmb = pd.read_excel(path, sheet_name=SH, header=2)
    is_meso = tmb["Sample"].astype(str).str.startswith("MESO")
    n = int((~is_meso).values.argmax()) if (~is_meso).any() else len(tmb)
    tmb = tmb.iloc[:n].copy()
    tmb["pid"] = [C.meso_pid(s) for s in tmb["Sample"]]
    tmb = tmb.dropna(subset=["pid"])
    tmb_feat = (np.log1p(pd.to_numeric(tmb["total_perMB"], errors="coerce"))
                .groupby(tmb["pid"]).mean().rename("MUT:TMB"))

    # --- COSMIC reconstructed signatures: counts -> per-sample relative exposure ---
    cos = pd.read_excel(path, sheet_name=SH, header=181)
    cos = cos[cos["Samples"].astype(str).str.startswith("MESO")].copy()
    sbs = [c for c in cos.columns if str(c).startswith("SBS")]
    cos["pid"] = [C.meso_pid(s) for s in cos["Samples"]]
    cos = cos.dropna(subset=["pid"])
    cnt = cos.groupby("pid")[sbs].mean()
    prop = cnt.div(cnt.sum(axis=1).replace(0, np.nan), axis=0)
    prop.columns = [f"MUT:{c}" for c in prop.columns]

    out = pd.concat([tmb_feat, prop], axis=1)
    keep = [c for c in out.columns
            if out[c].notna().sum() >= 10 and out[c].std(skipna=True) > 0]
    return out[keep]                                       # samples x features

def load_cg2gene():
    """cg probe -> first gene symbol, from the persisted Illumina 450K GENCODE manifest."""
    path = os.path.join(C.PROC, "hm450_manifest.tsv.gz")
    if not os.path.exists(path):
        return {}
    m = pd.read_csv(path, sep="\t", usecols=["probeID", "geneNames"])
    m = m.dropna(subset=["geneNames"])
    return {p: str(g).split(";")[0] for p, g in zip(m["probeID"], m["geneNames"])}

def build_survival():
    d = pd.read_excel(os.path.join(C.MESO, "TableS2-3_SamplesOverview.xlsx"),
                      sheet_name="S2 Sample overview", header=2)
    rows = {}
    for _, r in d.iterrows():
        sid = r.get("ID_MESOMICS")
        if not isinstance(sid, str) or not sid.startswith("MESO_"):
            continue
        pid = "MESOMICS_" + sid
        mo = pd.to_numeric(pd.Series([r.get("Survival.Time")]), errors="coerce")[0]
        cen = str(r.get("Survival.Censor")).strip().lower()
        ev = 1 if cen in ("dead", "1", "deceased") else (0 if cen in ("alive", "0", "living") else np.nan)
        typ = str(r.get("Type", "")).strip().upper()
        nonepi = 0 if typ == "MME" else (1 if typ in ("MMB", "MMS") else np.nan)
        if pd.notna(mo) and pd.notna(ev):
            rows[pid] = dict(months=float(mo), event=int(ev), nonepi=nonepi)
    return pd.DataFrame(rows).T

def main():
    print("[1] loading omics layers ...")
    layers = {
        "EXPR": build_expression(),
        "CNV": build_cnv_genes(),
        "LOH": build_loh(),
        "MET": build_methylation(),
        "ALT": build_alterations(),
        "SV": build_sv_burden(),
        "MUT": build_mut(),
    }
    for k, v in layers.items():
        print(f"    {k:5s} {v.shape[0]:4d} samples x {v.shape[1]:6d} features")

    surv = build_survival()
    print(f"[2] survival: {len(surv)} patients, {int(surv.event.sum())} events")

    # union of samples that have survival; align everything to it
    samples = sorted(surv.index)
    X = pd.concat([lay.reindex(samples) for lay in layers.values()], axis=1)
    # drop all-NA features and features with < 3 non-null distinct values handled later
    X = X.loc[:, X.notna().sum(axis=0) >= 10]
    print(f"[3] combined feature matrix: {X.shape[0]} samples x {X.shape[1]} features")

    # per-cohort z-scored raw matrix stays raw here (models z-score at use time)
    ann = pd.DataFrame({"feature": X.columns})
    ann["layer"] = [C.feat_layer(f) for f in X.columns]
    ann["gene"] = [C.feat_gene(f) for f in X.columns]
    # annotate methylation probes with their gene (Illumina 450K -> GENCODE v36) for
    # interpretation; selection still runs at probe level (standard, highest resolution)
    cg2gene = load_cg2gene()
    met_mask = ann["layer"].str.startswith("MET")
    ann.loc[met_mask, "gene"] = [cg2gene.get(g, "") for g in ann.loc[met_mask, "gene"]]
    ann = ann.set_index("feature")

    # discretized matrix for chi-square
    print("[4] discretizing for chi-square ...")
    disc = {}
    for f in X.columns:
        disc[f] = C.discretize(X[f], C.LAYER_KIND.get(C.feat_layer(f), "tertile"))
    disc = pd.DataFrame(disc, index=X.index)

    X.to_csv(os.path.join(C.PROC, "features_mesomics.tsv.gz"), sep="\t", compression="gzip")
    disc.to_csv(os.path.join(C.PROC, "features_mesomics_disc.tsv.gz"), sep="\t", compression="gzip")
    surv.to_csv(os.path.join(C.PROC, "survival_mesomics.tsv"), sep="\t")
    ann.to_csv(os.path.join(C.PROC, "feature_annotation.tsv"), sep="\t")

    summ = dict(n_samples=int(X.shape[0]), n_features=int(X.shape[1]),
                n_events=int(surv.event.sum()),
                layer_counts=ann.layer.value_counts().to_dict())
    json.dump(summ, open(os.path.join(C.PROC, "build_summary.json"), "w"), indent=2)
    print("[done]", json.dumps(summ["layer_counts"]))

if __name__ == "__main__":
    main()
