#!/usr/bin/env bash
cd /mnt/data/hackathon/xiao/mpm_multiomics_pipeline
PY=./.venv/bin/python
export TMPDIR="$PWD/env/tmp"; mkdir -p "$TMPDIR"
set -e
echo "[$(date +%H:%M:%S)] Stage 3 selectors"; $PY src/04_featsel_thirdparty.py
echo "[$(date +%H:%M:%S)] Stage 3b consensus"; $PY src/05_consensus.py
echo "[$(date +%H:%M:%S)] Stage 4 evaluate"; $PY src/06_evaluate.py
echo "[$(date +%H:%M:%S)] Stage 6 biology"; $PY src/08_biology.py
echo "[$(date +%H:%M:%S)] Stage 7 singlecell"; $PY src/09_singlecell.py
echo "[$(date +%H:%M:%S)] Stage 8 literature"; $PY src/10_literature.py
echo "[$(date +%H:%M:%S)] Stage 5 report"; $PY src/07_report.py
echo "[$(date +%H:%M:%S)] DONE"
