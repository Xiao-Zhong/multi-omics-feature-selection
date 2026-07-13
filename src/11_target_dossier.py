#!/usr/bin/env python
"""STAGE 9 - TARGET DOSSIER: turn the consensus feature panel into drug-discovery target cards.

For each selected gene it reports what a translational / immunogenetics reader needs to act:
  * DIRECTION of effect (univariate Cox HR on the real feature) -> inhibit vs restore
  * METHYLATION mechanism (probe<->expression correlation -> silencing hypothesis)
  * CELL-OF-ORIGIN + COMPARTMENT (tumour / immune / stromal) from the pleura scRNA
  * DRUGGABILITY class + modality (small molecule vs antibody/CAR-T) + tractability tier
  * IMMUNE role flag (immuno-oncology relevance)
  * EVIDENCE tier (consensus votes + cross-cohort + single-cell + literature)
  * a one-line THERAPEUTIC HYPOTHESIS

Outputs: results/tables/target_dossier.tsv, results/figures/fig_target_priority.png
"""
import os, sys, json, warnings
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
warnings.filterwarnings("ignore")

# ---- cell type -> tumour/immune/stromal compartment ----
IMMUNE_CT = {"T cells", "B cells", "Plasma", "Mast cells", "Neutrophils", "Dendritic",
             "Macrophage", "Myeloid", "NK cells", "Monocyte"}
TUMOUR_CT = {"Mesothelial"}
def compartment(ct):
    if ct in IMMUNE_CT: return "immune"
    if ct in TUMOUR_CT: return "tumour"
    if ct in ("Fibroblast", "Endothelial", "Pericyte", "SmoothMuscle"): return "stromal"
    return "other"

# ---- offline druggability: curated + gene-family heuristics ----
# tier 0 (hard) .. 3 (established target class). modality: SM=small molecule, Ab=antibody/CAR-T
DRUG = {
    "UBE2C": ("ubiquitin-conjugating enzyme (E2)", "SM", 2),
    "CAPN2": ("calpain-2 protease", "SM", 2),
    "SLC22A5": ("OCTN2 carnitine transporter", "SM", 2),
    "DKK2": ("secreted Wnt antagonist", "Ab", 2),
    "LRIG1": ("membrane EGFR/RTK regulator", "Ab", 2),
    "B3GALT4": ("glycosyltransferase", "SM", 1),
    "LVRN": ("laeverin/aminopeptidase Q", "SM", 1),
    "CPXM2": ("carboxypeptidase X-2", "SM", 1),
    "CHL1": ("L1-family cell-adhesion (surface)", "Ab", 1),
    "DSCAML1": ("Ig-superfamily surface adhesion", "Ab", 1),
    "LRRN3": ("LRR transmembrane", "Ab", 1),
    "PPP1R18": ("PP1 regulatory subunit (scaffold)", "SM", 1),
    "CALY": ("calcyon vesicular protein", "SM", 0),
    "COL7A1": ("collagen VII (ECM structural)", "-", 0),
    "ACTL10": ("actin-like (structural)", "-", 0),
    "CHL1": ("L1-family cell-adhesion (surface)", "Ab", 1),
}
def druggability(gene):
    if gene in DRUG:
        return DRUG[gene]
    if gene is None:
        return ("unannotated CpG (EPIC probe)", "-", 0)
    fam = [
        ("SLC", ("solute-carrier transporter", "SM", 2)),
        ("COL", ("collagen / ECM structural", "-", 0)),
        ("CTS", ("cathepsin protease", "SM", 2)),
        ("CAPN", ("calpain protease", "SM", 2)),
        ("MMP", ("matrix metalloprotease", "SM", 2)),
        ("HLA", ("MHC antigen presentation", "Ab", 2)),
        ("CD", ("cell-surface immune marker", "Ab", 2)),
        ("IL", ("interleukin / cytokine", "Ab", 2)),
        ("CXCL", ("chemokine", "Ab", 2)),
        ("CCL", ("chemokine", "Ab", 2)),
        ("UBE2", ("ubiquitin-conjugating enzyme", "SM", 2)),
        ("KDM", ("histone demethylase", "SM", 3)),
        ("HDAC", ("histone deacetylase", "SM", 3)),
    ]
    for pfx, val in fam:
        if gene.startswith(pfx):
            return val
    return ("unclassified", "SM", 1)

def cox_hr(v, t, e):
    from lifelines import CoxPHFitter
    try:
        r = CoxPHFitter().fit(pd.DataFrame({"x": v, "time": t, "event": e}),
                              "time", "event").summary.loc["x"]
        return float(r["exp(coef)"]), float(r["z"])
    except Exception:
        return np.nan, np.nan

def main():
    X, surv = C.load_train(); ann = C.load_feature_annotation()
    t = surv["months"].values; e = surv["event"].values
    T = os.path.join(C.TABLES, "")

    # target set: consensus (cross-method) + best in-house panel
    panels = {}
    for fn in ("consensus_panels.json", "inhouse_panels.json"):
        p = os.path.join(C.TABLES, fn)
        if os.path.exists(p): panels.update(json.load(open(p)))
    feats = list(dict.fromkeys(panels.get("Consensus-TopK", []) + panels.get("Inhouse-Union", [])))

    # side tables
    sc = pd.read_csv(T + "singlecell_gene_specificity.tsv", sep="\t").set_index("gene") \
        if os.path.exists(T + "singlecell_gene_specificity.tsv") else pd.DataFrame()
    votes = pd.read_csv(T + "consensus_votes.tsv", sep="\t", index_col=0) \
        if os.path.exists(T + "consensus_votes.tsv") else pd.DataFrame()
    lit = pd.read_csv(T + "literature_support.tsv", sep="\t").set_index("gene") \
        if os.path.exists(T + "literature_support.tsv") else pd.DataFrame()

    def gene_of(f):
        lay = C.feat_layer(f)
        g = str(ann.loc[f, "gene"]) if (lay.startswith("MET") and f in ann.index) else C.feat_gene(f)
        return None if g in ("nan", "", "None") else g

    # group features by gene (keep None-gene probes as their own rows)
    bygene = {}
    for f in feats:
        g = gene_of(f); bygene.setdefault(g if g else f"<probe>{f}", []).append(f)

    rows = []
    for key, fs in bygene.items():
        gene = None if key.startswith("<probe>") else key
        rep = fs[0]                                   # representative feature
        # direction from the strongest feature of this gene
        best_hr, best_z, best_f = np.nan, 0.0, rep
        for f in fs:
            v = C.zscore_cols(X[[f]]).fillna(0.0)[f].values
            hr, z = cox_hr(v, t, e)
            if abs(z) >= abs(best_z): best_hr, best_z, best_f = hr, z, f
        direction = "inhibit (high=worse)" if best_hr > 1 else "restore (protective)"
        effect = abs(np.log2(best_hr)) if best_hr == best_hr and best_hr > 0 else np.nan

        # methylation silencing: correlate a MET probe with the gene's expression (training)
        sil = ""
        if gene and f"EXPR:{gene}" in X.columns:
            metf = [f for f in fs if C.feat_layer(f).startswith("MET")]
            if metf:
                r = np.corrcoef(X[metf[0]].fillna(X[metf[0]].mean()),
                                X[f"EXPR:{gene}"].fillna(0.0))[0, 1]
                if r < -0.2: sil = f"methylation-silenced (r={r:.2f})"
                elif r > 0.2: sil = f"methylation-activated (r={r:.2f})"
                else: sil = f"meth~expr decoupled (r={r:.2f})"

        ct = str(sc.loc[gene, "top_celltype"]) if (gene in sc.index) else "-"
        comp = compartment(ct)
        dclass, modality, tier = druggability(gene)
        n_votes = int(votes.reindex(fs)["votes"].max()) if (len(votes) and any(f in votes.index for f in fs)) else 1
        lit_s = str(lit.loc[gene, "literature_support"]) if (gene in lit.index) else ""
        immuno = comp == "immune" or dclass.split()[0] in ("MHC", "chemokine", "interleukin",
                                                            "cell-surface")
        # evidence tier
        ev = "A" if (n_votes >= 3 and lit_s) else "B" if (n_votes >= 3 or lit_s) else "C"
        # therapeutic hypothesis
        modtxt = {"SM": "small molecule", "Ab": "antibody / CAR-T", "-": "low tractability"}[modality]
        hyp = (f"{direction.split()[0].capitalize()} — {dclass} ({modtxt}); "
               f"{comp} compartment ({ct})" + (f"; {sil}" if sil else ""))

        rows.append({
            "gene": gene if gene else key.replace("<probe>", ""),
            "layers": ",".join(sorted(set(C.feat_layer(f) for f in fs))),
            "direction": direction, "HR": round(best_hr, 2) if best_hr == best_hr else None,
            "effect_size": round(effect, 2) if effect == effect else None,
            "cell_of_origin": ct, "compartment": comp,
            "druggable_class": dclass, "modality": modality, "tractability_tier": tier,
            "immuno_oncology": "yes" if immuno else "",
            "mechanism": sil, "consensus_votes": n_votes, "literature": lit_s,
            "evidence_tier": ev, "therapeutic_hypothesis": hyp,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(["evidence_tier", "tractability_tier", "effect_size"],
                        ascending=[True, False, False])
    df.to_csv(os.path.join(C.TABLES, "target_dossier.tsv"), sep="\t", index=False)
    print(f"[dossier] {len(df)} targets ({(df.gene.notna()).sum()} gene-mapped)")
    print(df[["gene", "direction", "HR", "compartment", "druggable_class",
              "tractability_tier", "immuno_oncology", "evidence_tier"]].to_string(index=False))

    # ---- prioritisation figure: effect size vs druggability, coloured by compartment ----
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    d = df[df["effect_size"].notna() & df["gene"].notna()].copy()
    cmap = {"tumour": "#c0392b", "immune": "#2471a3", "stromal": "#7d8a2e", "other": "#95a5a6"}
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    rng = np.random.default_rng(C.SEED)
    for comp, sub in d.groupby("compartment"):
        jit = sub["tractability_tier"] + rng.uniform(-0.12, 0.12, len(sub))
        ax.scatter(sub["effect_size"], jit, s=40 + 30 * sub["consensus_votes"],
                   c=cmap.get(comp, "#888"), alpha=0.8, edgecolor="k", linewidth=0.4, label=comp)
    for _, r in d.iterrows():
        ax.annotate(r["gene"], (r["effect_size"], r["tractability_tier"]),
                    fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("prognostic effect size  |log2 HR|  (bigger = stronger)")
    ax.set_ylabel("druggability tractability tier (0 hard - 3 established)")
    ax.set_title("Target prioritisation: prognostic effect vs druggability\n"
                 "(colour = compartment, size = consensus votes)")
    ax.legend(title="compartment", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(C.FIGS, "fig_target_priority.png"), dpi=130)
    print("[done] target_dossier.tsv + fig_target_priority.png")

if __name__ == "__main__":
    main()
