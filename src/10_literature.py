#!/usr/bin/env python
"""STAGE 8 - literature check of the selected genes.

Cross-references every selected gene against a curated, cited knowledge base of MPM / cancer
prognostic evidence (compiled from PubMed/PMC, July 2026). Flags each selected gene as
literature-supported (MPM-specific / pan-cancer) or a novel candidate with no curated evidence,
so biologically grounded features can be separated from possible data-analysis artifacts.

Outputs: results/tables/literature_support.tsv, literature_summary.json
"""
import os, sys, json
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

# curated evidence base (gene -> role, prognostic evidence, MPM-specific?, citations)
KB = {
    "CDC20": dict(role="APC/C co-activator; mitotic/cell-cycle (G2M, E2F target)",
                  evidence="Pan-cancer poor-prognosis proliferation marker (meta-analysis); part of an MPM fibroblast-differentiation survival signature.",
                  mpm=True, refs=["https://www.frontiersin.org/articles/10.3389/fonc.2022.1017864/full",
                                  "https://link.springer.com/article/10.1186/s13578-023-01180-7"]),
    "MYBL2": dict(role="B-Myb proliferation transcription factor (cell cycle)",
                  evidence="Prognostic proliferation TF; interacts with CDC20; in an MPM fibroblast-differentiation survival-prediction network.",
                  mpm=True, refs=["https://link.springer.com/article/10.1186/s13578-023-01180-7"]),
    "COL7A1": dict(role="Type VII collagen; basement-membrane / ECM",
                   evidence="Prognostic in clear-cell RCC, gastric and lung adenocarcinoma (high expression = worse outcome). Not MPM-specific.",
                   mpm=False, refs=["https://pmc.ncbi.nlm.nih.gov/articles/PMC8426344/",
                                    "https://pmc.ncbi.nlm.nih.gov/articles/PMC11868062/"]),
    "CDKN2A": dict(role="p16INK4a/p14ARF tumor suppressor, 9p21",
                   evidence="Homozygous 9p21 deletion in up to ~74% of MPM; deleted cases have markedly worse OS (~10 vs ~34 mo).",
                   mpm=True, refs=["https://pubmed.ncbi.nlm.nih.gov/20081810/",
                                   "https://pmc.ncbi.nlm.nih.gov/articles/PMC7693432/"]),
    "CDKN2B": dict(role="p15INK4b tumor suppressor, 9p21 (co-deleted with CDKN2A)",
                   evidence="Co-deleted in the 9p21 focal loss with CDKN2A/MTAP; poor-prognosis marker in MPM.",
                   mpm=True, refs=["https://pmc.ncbi.nlm.nih.gov/articles/PMC7693432/"]),
    "MTAP": dict(role="Methylthioadenosine phosphorylase, 9p21 (co-deleted with CDKN2A)",
                 evidence="9p21 co-deletion; MTAP IHC loss is a validated MPM diagnostic/prognostic surrogate for CDKN2A FISH; worse survival; MTAP-pathway therapeutic target.",
                 mpm=True, refs=["https://pubmed.ncbi.nlm.nih.gov/20081810/",
                                 "https://pmc.ncbi.nlm.nih.gov/articles/PMC7693432/"]),
    "CCNE1": dict(role="Cyclin E1; G1/S cell-cycle",
                  evidence="Cell-cycle/proliferation prognostic marker across cancers; amplification linked to poor outcome.",
                  mpm=False, refs=["https://www.frontiersin.org/journals/oncology"]),
    "BAP1": dict(role="BRCA1-associated deubiquitinase tumor suppressor",
                 evidence="Most frequently mutated MPM driver; BAP1 loss is an established prognostic/ predictive biomarker.",
                 mpm=True, refs=["https://pmc.ncbi.nlm.nih.gov/articles/PMC12839189/"]),
    "NF2": dict(role="Merlin tumor suppressor (Hippo pathway)",
                evidence="Recurrent MPM driver; Hippo-pathway inactivation, prognostic relevance.",
                mpm=True, refs=["https://pmc.ncbi.nlm.nih.gov/articles/PMC12839189/"]),
    "TERT": dict(role="Telomerase reverse transcriptase",
                 evidence="TERT promoter alterations in MPM; associated with aggressive disease.",
                 mpm=True, refs=["https://pmc.ncbi.nlm.nih.gov/articles/PMC12839189/"]),
    # HLF: intentionally NOT in the KB -> flagged 'novel/uncurated' (no established MPM/cancer
    # prognostic evidence found), despite being a cross-run-robust selection.
}

def gene_of(f, ann):
    lay, key = f.split(":", 1)
    g = str(ann.loc[f, "gene"]) if (lay.startswith("MET") and f in ann.index) else key
    return "" if g == "nan" else g

def selected_genes():
    ann = C.load_feature_annotation()
    g2p = {}
    for fn in ("inhouse_panels.json", "thirdparty_panels.json", "consensus_panels.json"):
        p = os.path.join(C.TABLES, fn)
        if not os.path.exists(p):
            continue
        for pan, feats in json.load(open(p)).items():
            for f in feats:
                g = gene_of(f, ann)
                if g:
                    g2p.setdefault(g, set()).add(pan)
    return g2p

def main():
    g2p = selected_genes()
    rows = []
    for g, pans in sorted(g2p.items(), key=lambda x: -len(x[1])):
        kb = KB.get(g)
        rows.append({
            "gene": g, "n_panels": len(pans),
            "literature_support": ("MPM-specific" if kb and kb["mpm"]
                                   else "pan-cancer" if kb else "novel/uncurated"),
            "role": kb["role"] if kb else "",
            "evidence": kb["evidence"] if kb else "",
            "refs": "; ".join(kb["refs"]) if kb else "",
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(C.TABLES, "literature_support.tsv"), sep="\t", index=False)

    n_sup = int((df["literature_support"] != "novel/uncurated").sum())
    n_mpm = int((df["literature_support"] == "MPM-specific").sum())
    summ = {"n_genes": len(df), "n_literature_supported": n_sup, "n_mpm_specific": n_mpm,
            "supported_genes": df[df.literature_support != "novel/uncurated"]["gene"].tolist()}
    json.dump(summ, open(os.path.join(C.TABLES, "literature_summary.json"), "w"), indent=2)
    print(f"[literature] {n_sup}/{len(df)} selected genes have curated prognostic evidence "
          f"({n_mpm} MPM-specific)")
    print(df[df.literature_support != "novel/uncurated"][["gene", "n_panels", "literature_support", "role"]].to_string(index=False))
    print("[done] literature check")

if __name__ == "__main__":
    main()
