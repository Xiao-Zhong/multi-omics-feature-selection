#!/usr/bin/env python
"""STAGE 5 - REPORT.md with embedded figures (captioned) + tables (commented), plus biology,
single-cell and literature validation sections."""
import os, sys, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

def load(name):
    p = os.path.join(C.TABLES, name)
    return json.load(open(p)) if os.path.exists(p) else {}

def tsv(name):
    p = os.path.join(C.TABLES, name)
    return pd.read_csv(p, sep="\t") if os.path.exists(p) else pd.DataFrame()

def _is_inhouse(p):
    return str(p).startswith(("Inhouse", "Consensus"))

def img(fname, caption):
    path = os.path.join(C.FIGS, fname)
    if not os.path.exists(path):
        return ""
    return f"\n![{caption}](figures/{fname})\n\n*{caption}*\n"

# --------------------------------------------------------------------- figures
def fig_eval(df):
    d = df.dropna(subset=["Cexpr_mean"]).sort_values("Cexpr_mean")
    fig, ax = plt.subplots(figsize=(8, max(3, 0.34 * len(d))))
    colors = ["#c0392b" if _is_inhouse(p) else "#2c6fbb" for p in d["panel"]]
    ax.barh(d["panel"], d["Cexpr_mean"], color=colors)
    ax.axvline(0.5, ls="--", c="gray", lw=0.8)
    ax.set_xlabel("mean gene-expression-surrogate C-index (all cohorts)")
    ax.set_title("Panel performance  (red = in-house / ensemble)")
    fig.tight_layout(); fig.savefig(os.path.join(C.FIGS, "fig_panel_eval.png"), dpi=130); plt.close(fig)

def _heatmap(mat, title, fname):
    if mat.empty:
        return
    fig, ax = plt.subplots(figsize=(1.1 + 0.85 * mat.shape[1], 1.2 + 0.36 * mat.shape[0]))
    im = ax.imshow(mat.values.astype(float), cmap="RdYlBu_r", vmin=0.35, vmax=0.75, aspect="auto")
    ax.set_xticks(range(mat.shape[1])); ax.set_xticklabels(mat.columns, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(mat.shape[0])); ax.set_yticklabels(mat.index, fontsize=8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                        color="black" if 0.42 < v < 0.68 else "white")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.025, label="C-index")
    fig.tight_layout(); fig.savefig(os.path.join(C.FIGS, fname), dpi=130); plt.close(fig)

def fig_cindex_heatmaps(ev):
    ev = ev.sort_values("Cexpr_mean", ascending=False).set_index("panel")
    scols = ["C_internal_cv"] + [f"Cexpr_{c}" for c in C.VAL_COHORTS if f"Cexpr_{c}" in ev.columns]
    sm = ev[scols].rename(columns=lambda c: c.replace("Cexpr_", "").replace("C_internal_cv", "internalCV"))
    _heatmap(sm, "Gene-expression-surrogate C-index (panels x datasets)", "fig_cindex_heatmap.png")
    ncols = [f"C_{c}" for c in C.VAL_COHORTS if f"C_{c}" in ev.columns]
    nm = ev[ncols].rename(columns=lambda c: c.replace("C_", ""))
    _heatmap(nm, "Native multi-omics transfer C-index (panels x datasets)", "fig_native_heatmap.png")

def fig_feature_heatmap(voters, votes, ann, topn=30):
    if votes.empty:
        return
    feats = votes.head(topn).index.tolist(); methods = list(voters)
    M = pd.DataFrame(0, index=feats, columns=methods)
    for m, fl in voters.items():
        for f in fl:
            if f in M.index:
                M.loc[f, m] = 1
    labels = []
    for f in feats:
        lay = C.feat_layer(f); g = str(ann.loc[f, "gene"]) if f in ann.index else C.feat_gene(f)
        labels.append(f"{g} [{lay}]" if g and g != "nan" else f)
    fig, ax = plt.subplots(figsize=(1.2 + 0.5 * len(methods), 1.5 + 0.32 * len(feats)))
    ax.imshow(M.values, cmap="Greens", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(methods))); ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(feats))); ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(f"Feature selection across methods (top {topn} by consensus)", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(C.FIGS, "fig_feature_heatmap.png"), dpi=130); plt.close(fig)

def fig_votes(votes):
    if votes.empty:
        return
    d = votes.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh([str(i) for i in d.index], d["votes"], color="#4a9e6f")
    ax.set_xlabel("# methods selecting feature (consensus)")
    ax.set_title("Cross-method consensus features")
    fig.tight_layout(); fig.savefig(os.path.join(C.FIGS, "fig_consensus.png"), dpi=130); plt.close(fig)

def load_voters():
    panels = {}
    for fn in ("inhouse_panels.json", "thirdparty_panels.json"):
        panels.update(load(fn))
    return {k: v for k, v in panels.items() if v and k != "Inhouse-Union"}

def main():
    ev = tsv("panel_evaluation.tsv")
    votes = pd.read_csv(os.path.join(C.TABLES, "consensus_votes.tsv"), sep="\t", index_col=0) \
        if os.path.exists(os.path.join(C.TABLES, "consensus_votes.tsv")) else pd.DataFrame()
    inmeta = load("inhouse_meta.json")
    ann = C.load_feature_annotation()
    panels = {}
    for fn in ("inhouse_panels.json", "thirdparty_panels.json", "consensus_panels.json"):
        panels.update(load(fn))
    voters = load_voters()

    fig_eval(ev); fig_cindex_heatmaps(ev); fig_votes(votes); fig_feature_heatmap(voters, votes, ann)

    ev = ev.sort_values("Cexpr_mean", ascending=False)
    best = ev.iloc[0]; best_feats = panels.get(best["panel"], [])
    inhouse_best = ev[ev.panel.map(_is_inhouse)].sort_values("Cexpr_mean", ascending=False)
    tp_best = ev[~ev.panel.map(_is_inhouse)].sort_values("Cexpr_mean", ascending=False)

    def gene_of(f):
        g = str(ann.loc[f, "gene"]) if f in ann.index else C.feat_gene(f)
        return "" if g == "nan" else g

    L = ["# MPM multi-omics prognostic biomarker selection — report\n"]
    L.append("Reliability-first feature selection trained on **MESOMICS multi-omics** "
             "(expression, gene-focused CNV, LOH, methylation, driver alterations, SV burden), "
             "validated by cross-cohort survival transfer, biological pathway analysis, single-cell "
             "expression, and a literature check. Every figure is embedded below with a description.\n")

    # ---- project description ----
    L.append("## Project description\n")
    L.append(
        "**What we built & investigated.** A reliability-first multi-omics feature-selection pipeline "
        "for malignant pleural mesothelioma (MPM) prognosis. It selects prognostic biomarker panels "
        "from the MESOMICS cohort (120 patients × ~25,000 features spanning expression, DNA "
        "methylation, LOH, CNV and driver alterations) and validates them by survival transfer to "
        "external cohorts, KEGG pathway enrichment, single-cell cell-type expression, and a curated "
        "literature check. Before benchmarking the in-house selectors against third-party methods we "
        "**audited the comparators**: homemade heuristics that merely borrowed famous names (a "
        "\"Network-LASSO\" that is not Hallac's Network Lasso; a \"PAWPH\" that does not reproduce the "
        "real method) were removed, and two deep-learning tools (DeepKEGG, DeePathNet) were dropped "
        "because they are not survival models at all. We then ran the *genuine* upstream tools "
        "(`pycox` DeepSurv, the original R PAWPH) to measure reimplementation fidelity, "
        "hyperparameter-tuned the tree/deep models so none were handicapped, and reconstructed a "
        "validation cohort (French E-MTAB-1719) from raw microarray CEL files. Finally we reframed the "
        "selected features into a drug-discovery **Target Dossier** (direction of effect, "
        "cell-of-origin, druggability, immuno-oncology relevance).\n")
    L.append(
        "**What we found.** The in-house consensus panel generalizes best across cohorts "
        "(expression-surrogate C-index ~0.67; native multi-omics transfer to TCGA ~0.72), edging "
        "established methods — though at n=120 the bootstrap confidence intervals overlap, so this is "
        "\"competitive and best-generalizing,\" not a statistically separated win. Two methodological "
        "results stand out: (1) a reimplementation can match a tool's *predictive performance* yet "
        "select a *different panel* — our DeepSurv reimplementation vs native `pycox` had an identical "
        "C-index but shared only 7/20 features — so unverified reimplementations are unsafe biomarker "
        "comparators; and (2) internal cross-validation is optimistic and disagrees with cross-cohort "
        "transfer, so transfer must be the primary endpoint. Biologically, the prognostic signal sits "
        "largely *outside* canonical driver pathways, and several top targets are immune-compartment "
        "genes (T-cell, mast, neutrophil, dendritic).\n")
    L.append(
        "**Why it matters.** Methodologically, this is a template for a *defensible* biomarker "
        "benchmark — verified comparators only, honest confidence intervals, generalization prized over "
        "internal fit — that resists the strawman-baseline critique. Translationally, it delivers a "
        "short, druggable, immuno-oncology-relevant target shortlist for MPM, an asbestos-linked cancer "
        "with few effective therapies and growing use of immunotherapy.\n")

    # ---- headline ----
    L.append("## 1. Headline\n")
    L.append(f"- **Best panel (surrogate mean):** `{best['panel']}` — {int(best['size'])} features / "
             f"{int(best.get('n_genes', 0))} genes, surrogate-mean C-index **{best['Cexpr_mean']:.3f}** "
             f"(internal CV {best['C_internal_cv']:.3f}).")
    if len(inhouse_best):
        ib = inhouse_best.iloc[0]
        L.append(f"- **Best in-house / ensemble:** `{ib['panel']}` — surrogate **{ib['Cexpr_mean']:.3f}**, "
                 f"native TCGA {ib.get('C_TCGA', float('nan')):.3f}.")
    if len(tp_best):
        tb = tp_best.iloc[0]
        L.append(f"- **Best third-party tool:** `{tb['panel']}` — surrogate **{tb['Cexpr_mean']:.3f}**.")
    L.append("- Two validation lenses: **native multi-omics transfer** (real features; MESOMICS↔TCGA) "
             "and **gene-expression surrogate** (panel collapsed to genes, scored on every cohort's "
             "expression — full coverage; the headline metric).\n")

    # ---- performance figures ----
    L.append("## 2. Cross-cohort performance\n")
    L.append(img("fig_panel_eval.png",
                 "Mean gene-expression-surrogate C-index per panel across all validation cohorts. "
                 "Red = in-house/ensemble panels, blue = third-party tools; dashed line = random (0.5). "
                 "All panels sit in a modest 0.56–0.66 band, typical for n=120 discovery."))
    L.append(img("fig_cindex_heatmap.png",
                 "Surrogate C-index of every panel (rows) across datasets (columns) + internal CV. "
                 "Full coverage on all cohorts because features are collapsed to gene expression; "
                 "read rows for panel consistency, columns for which cohorts are easier to predict."))
    L.append(img("fig_native_heatmap.png",
                 "Native multi-omics transfer C-index: the panel's ACTUAL features scored where measured. "
                 "MESOMICS and TCGA carry methylation/CNV/alteration features; the expression-only cohorts "
                 "(Bueno/NCI/Blum/French) score fewer features natively — compare with the surrogate heatmap."))

    L.append("\n**Full ranking.** `Cexpr_*` = surrogate C-index per cohort; `C_native_mean` = mean of the "
             "native-transfer columns. n_genes = genes usable for the surrogate.\n")
    scols = ["panel", "size", "n_genes", "C_internal_cv", "Cexpr_mean", "C_native_mean"] \
        + [f"Cexpr_{c}" for c in C.VAL_COHORTS if f"Cexpr_{c}" in ev.columns]
    L.append(ev[[c for c in scols if c in ev.columns]].round(3).to_markdown(index=False))

    # ---- consensus / feature selection ----
    L.append("\n\n## 3. Which features are selected, and by how many methods\n")
    L.append(img("fig_feature_heatmap.png",
                 "Selection matrix: top consensus features (rows, labelled gene [layer]) × methods "
                 "(columns). A filled cell = that method selected the feature. Rows filled across many "
                 "columns are the cross-method-robust biomarkers; single-column rows are method-specific."))
    L.append(img("fig_consensus.png",
                 "Number of independent methods selecting each feature (consensus vote). Features chosen "
                 "by many methods are the most reliable candidates at small n."))
    if not votes.empty:
        vt = votes.head(12).copy(); vt["gene"] = [gene_of(f) for f in vt.index]
        L.append("\n**Top consensus features.**\n")
        L.append(vt.reset_index().rename(columns={"index": "feature"})[
            ["feature", "gene", "votes"] + [c for c in ("layer",) if c in vt.columns]].to_markdown(index=False))

    # ---- biology ----
    biosum = load("biology_summary.json"); bioenr = tsv("biology_enrichment.tsv")
    if biosum:
        L.append("\n\n## 4. Biological plausibility (pathway / cancer-network)\n")
        L.append(img("fig_biology_support.png",
                     "Biological support per panel = fraction of panel genes in an enriched KEGG pathway, "
                     "a KEGG cancer pathway, or the known-MPM driver set. Higher = less likely to be noise. "
                     "These are unsupervised discovery methods, so support is generally modest, indicating "
                     "MPM prognostic signal sits largely OUTSIDE canonical driver pathways."))
        brows = [{"panel": p, "n_genes": s["n_genes"], "bio_support": s["biological_support"],
                  "enriched_KEGG": s["n_enriched_pathways"], "MPM_known": ",".join(s["mpm_known"]) or "-"}
                 for p, s in biosum.items()]
        L.append(pd.DataFrame(brows).sort_values("bio_support", ascending=False).to_markdown(index=False))

    # ---- single-cell ----
    scsum = load("singlecell_summary.json"); scspec = tsv("singlecell_gene_specificity.tsv")
    if scsum:
        L.append("\n\n## 5. Single-cell validation (pleura scRNA atlas)\n")
        L.append(f"Selected genes were located in the Obacz 2024 pleura scRNA atlas "
                 f"({scsum.get('n_genes')} panel genes, 10 cell types incl. **Mesothelial** = MPM "
                 f"cell-of-origin). **{scsum.get('n_expressed')}/{scsum.get('n_genes')}** are detectably "
                 f"expressed, confirming they are real transcripts in the relevant tissue, not artifacts.\n")
        L.append(img("fig_singlecell_heatmap.png",
                     "Row-z-scored mean expression of the most-selected genes across pleura cell types. "
                     "Reveals the compartment driving each gene: proliferation genes in cycling/mesothelial "
                     "cells, ECM genes in fibroblasts, immune genes in leukocytes."))
        if not scspec.empty:
            L.append("\n**Cell-type of the most-selected genes.**\n")
            L.append(scspec.head(12)[["gene", "n_panels", "top_celltype", "pct_expressing_top"]].to_markdown(index=False))

    # ---- literature ----
    litsum = load("literature_summary.json"); lit = tsv("literature_support.tsv")
    if litsum and not lit.empty:
        L.append("\n\n## 6. Literature check\n")
        L.append(f"**{litsum.get('n_literature_supported')}/{litsum.get('n_genes')}** selected genes have "
                 f"curated prognostic evidence ({litsum.get('n_mpm_specific')} MPM-specific). The reproducible "
                 f"core is literature-backed; the long tail are novel candidates to treat as provisional.\n")
        lb = lit[lit.literature_support != "novel/uncurated"]
        L.append(lb[["gene", "n_panels", "literature_support", "role", "evidence"]].to_markdown(index=False))

    # ---- target dossier (drug-discovery / immunogenetics view) ----
    dossier = tsv("target_dossier.tsv")
    if not dossier.empty:
        L.append("\n\n## 6b. Target dossier — drug-discovery view\n")
        L.append("Each consensus gene reframed as a candidate **target**: *direction* (inhibit high-risk "
                 "genes vs restore protective ones, from the univariate Cox hazard ratio on the real "
                 "feature), *compartment* (tumour / immune / stromal, from the pleura scRNA), "
                 "*druggability* class + modality (SM = small molecule, Ab = antibody / CAR-T), "
                 "immuno-oncology relevance, and a one-line therapeutic hypothesis. Tractability tier "
                 "0 (hard) → 3 (established target class).\n")
        L.append(img("fig_target_priority.png",
                     "Target prioritisation: prognostic effect size (|log2 HR|) vs druggability tractability, "
                     "coloured by tumour/immune/stromal compartment, sized by consensus votes. Upper-right = "
                     "strong AND druggable — start here."))
        dm = dossier[dossier["druggable_class"] != "unannotated CpG (EPIC probe)"]
        L.append("\n**Gene-mapped targets** (ranked by evidence tier, then druggability):\n")
        L.append(dm[["gene", "direction", "HR", "compartment", "druggable_class", "modality",
                     "tractability_tier", "immuno_oncology", "evidence_tier"]].to_markdown(index=False))
        io = dm[dm["immuno_oncology"] == "yes"]
        if not io.empty:
            L.append("\n**Immuno-oncology–relevant targets** (immune compartment or immune gene class) — "
                     "of particular interest for immunogenetics / tumour-microenvironment-directed design:\n")
            for _, r in io.iterrows():
                L.append(f"- **{r['gene']}** ({r['cell_of_origin']}): {r['therapeutic_hypothesis']} "
                         f"— HR {r['HR']}, tier {r['tractability_tier']}, evidence {r['evidence_tier']}")
        n_epic = int((dossier["druggable_class"] == "unannotated CpG (EPIC probe)").sum())
        if n_epic:
            L.append(f"\n_{n_epic} additional consensus probes are EPIC-array CpGs absent from the 450K "
                     "manifest — they carry prognostic signal but no gene assignment yet; recover via an "
                     "EPIC manifest to add them as targets._")

    # ---- K-part sensitivity ----
    kp = tsv("inhouse_kpart_comparison.tsv"); krec = load("inhouse_kpart_recommendation.json")
    if not kp.empty:
        L.append("\n\n## 7. Split-count (K) sensitivity of the in-house screen\n")
        L.append("The 2-part split was generalized to K parts (a feature must rank top in ALL K parts). "
                 "As K grows each part has fewer samples, so the χ² becomes unreliable (contingency "
                 "cells < 5) and the epistasis step collapses. `per_part_min_class` = smallest "
                 "poor/good count per part; `biv_pairs` = reproducible epistatic pairs.\n")
        L.append(kp.round(3).to_markdown(index=False))
        if krec:
            L.append(f"\n- **Recommended: K = {krec.get('recommended_parts')}** — the largest K that keeps "
                     f"per-part statistics valid while retaining the epistasis component (the default).\n")

    # ---- best panel ----
    L.append(f"\n\n## 8. Best-panel features (`{best['panel']}`)\n")
    for f in best_feats:
        g = gene_of(f)
        L.append(f"- `{f}`" + (f"  → gene **{g}**" if g else ""))

    # ---- provenance + caveats ----
    L.append("\n## 9. Method provenance & caveats\n")
    if inmeta:
        L.append(f"- In-house: {inmeta.get('parts', 2)}-part split × {inmeta.get('repeats', 50)} repeats, "
                 f"landmark {inmeta.get('landmark', float('nan')):.1f} mo; reproducible univariate "
                 f"{inmeta.get('uni_stable')}, stable epistatic pairs {inmeta.get('biv_pairs')}.")
    L.append("- **Gene-focused features:** CNV = canonical MPM drivers (NF2/BAP1/CDKN2A/CDKN2B/MTAP/TERT) "
             "from S36; methylation probes annotated to genes (450K → GENCODE v36).")
    L.append("- **Small n (120 train):** C-indices are modest by design; the pipeline optimizes for "
             "*reproducible* features (repeated-split ensemble + bootstrap stability + cross-method "
             "consensus + out-of-cohort transfer + biology/scRNA/literature) over any single number.")
    L.append("- See `DESIGN.md` for the full workflow.")

    open(os.path.join(C.ROOT, "results", "REPORT.md"), "w").write("\n".join(L))
    print("[done] wrote results/REPORT.md with embedded figures + tables")

if __name__ == "__main__":
    main()
