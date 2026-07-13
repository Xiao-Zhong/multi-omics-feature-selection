#!/usr/bin/env bash
# MPM multi-omics feature-selection pipeline — full reproducible run.
# Self-contained isolated environment lives in ./.venv (built on /mnt/data, not root).
set -e
cd "$(dirname "$0")"
PY=./.venv/bin/python
export TMPDIR="$PWD/env/tmp"; mkdir -p "$TMPDIR"

echo "== Stage 1  build MESOMICS multi-omics train matrix =="
$PY src/01_build_data.py
echo "== Stage 1b build transferable validation layers =="
$PY src/02_build_validation.py
echo "== Stage 2  in-house selection (ensemble split + epistasis + stability LASSO) =="
$PY src/03_featsel_inhouse.py
echo "== Stage 3  third-party selectors (+ transfer-learning DeepSurv) =="
$PY src/04_featsel_thirdparty.py
echo "== Stage 3b ensemble / consensus panels =="
$PY src/05_consensus.py
echo "== Stage 4  cross-cohort evaluation (native + gene-expression surrogate) =="
$PY src/06_evaluate.py
echo "== Stage 6  biology: KEGG pathway ORA + MPM driver / cancer-network check =="
$PY src/08_biology.py
echo "== Stage 7  single-cell validation (pleura scRNA cell-type expression) =="
$PY src/09_singlecell.py
echo "== Stage 8  literature check (curated prognostic evidence) =="
$PY src/10_literature.py
echo "== Stage 9  target dossier (druggability + immuno-oncology) =="
$PY src/11_target_dossier.py
echo "== Stage 5  report + figures (embedded) =="
$PY src/07_report.py
echo "== DONE == see results/REPORT.md"
